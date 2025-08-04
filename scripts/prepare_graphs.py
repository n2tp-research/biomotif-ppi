#!/usr/bin/env python3
"""
Script to pre-compute and cache graph structures for the GNN module.
This avoids expensive graph construction during training.
"""

import os
import sys
import h5py
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
import torch.nn.functional as F

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def construct_protein_graph(
    embeddings: np.ndarray,
    properties: np.ndarray,
    config: dict
) -> dict:
    """
    Construct graph structure for a single protein.
    
    Args:
        embeddings: ESM-2 embeddings [seq_len, embedding_dim]
        properties: Physicochemical properties [seq_len, prop_dim]
        config: Configuration dictionary
        
    Returns:
        Dictionary containing graph data
    """
    seq_len = embeddings.shape[0]
    
    # Convert to tensors
    embeddings = torch.from_numpy(embeddings).float()
    properties = torch.from_numpy(properties).float()
    
    # Extract secondary structure propensities (last 2 dimensions)
    alpha_prop = properties[:, -2]
    beta_prop = properties[:, -1]
    
    edge_list = []
    
    # Always connect sequential neighbors
    for i in range(seq_len - 1):
        edge_list.append([i, i + 1])
        edge_list.append([i + 1, i])
    
    # Sample long-range connections based on features
    # Limit connections to keep graph sparse but informative
    num_long_range = min(seq_len * 5, 500)
    
    if seq_len > 10:
        # Pre-compute all pairwise similarities for efficiency
        # Use embedding similarity as proxy for contact probability
        embeddings_norm = F.normalize(embeddings, dim=-1)
        
        # Sample random pairs
        for _ in range(num_long_range):
            i = np.random.randint(0, max(1, seq_len - 5))
            j = i + np.random.randint(5, min(50, seq_len - i))
            
            if j < seq_len:
                # Compute similarity
                similarity = torch.dot(embeddings_norm[i], embeddings_norm[j]).item()
                
                # Add structural preference
                ss_bonus = alpha_prop[i] * alpha_prop[j] + beta_prop[i] * beta_prop[j]
                combined_score = similarity + 0.3 * ss_bonus.item()
                
                # Add edge based on combined score
                if combined_score > 0.5:
                    edge_list.append([i, j])
                    edge_list.append([j, i])
    
    # Convert to edge index tensor
    if len(edge_list) > 0:
        edge_index = np.array(edge_list, dtype=np.int64).T
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)
    
    # Compute graph statistics
    num_nodes = seq_len
    num_edges = edge_index.shape[1] // 2  # Undirected edges
    
    # Compute degree for each node
    degrees = np.zeros(num_nodes)
    if edge_index.shape[1] > 0:
        unique, counts = np.unique(edge_index[0], return_counts=True)
        degrees[unique] = counts
    
    avg_degree = np.mean(degrees) if num_nodes > 0 else 0
    max_degree = np.max(degrees) if num_nodes > 0 else 0
    
    return {
        'edge_index': edge_index,
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'degrees': degrees,
        'avg_degree': avg_degree,
        'max_degree': max_degree
    }


def prepare_graphs(config: dict, update_existing: bool = True):
    """
    Pre-compute graph structures for all protein pairs in the dataset.
    
    Args:
        config: Configuration dictionary
        update_existing: If True, update existing cache with missing entries
    """
    
    # Load embedding and property caches
    model_name = config['esm']['model_name'].replace('/', '_')
    embedding_cache_path = os.path.join(
        config['data']['cache_dir'],
        f'bernett_gold_ppi_{model_name}_embeddings.h5'
    )
    properties_cache_path = os.path.join(
        config['data']['cache_dir'],
        'bernett_gold_ppi_properties.h5'
    )
    
    if not os.path.exists(embedding_cache_path):
        print(f"ERROR: Embedding cache not found at {embedding_cache_path}")
        print("Please run: python scripts/prepare_data.py")
        return
    
    if not os.path.exists(properties_cache_path):
        print(f"ERROR: Properties cache not found at {properties_cache_path}")
        print("Please run: python scripts/prepare_properties.py")
        return
    
    print("Loading embeddings and properties...")
    embeddings_h5 = h5py.File(embedding_cache_path, 'r')
    properties_h5 = h5py.File(properties_cache_path, 'r')
    
    # Prepare graph cache path
    graph_cache_path = os.path.join(
        config['data']['cache_dir'],
        'bernett_gold_ppi_graphs.h5'
    )
    
    # Check existing cache
    existing_keys = set()
    if os.path.exists(graph_cache_path) and update_existing:
        print(f"Found existing graph cache at {graph_cache_path}")
        with h5py.File(graph_cache_path, 'r') as h5f:
            existing_keys = set(h5f.keys())
            print(f"Existing cached graphs: {len(existing_keys)}")
    
    # Process all unique proteins
    all_protein_ids = set()
    splits = ['train', 'valid', 'test']
    
    print("\nCollecting protein IDs...")
    for split in splits:
        dataset = load_dataset(
            config['data']['dataset_name'],
            split=split,
            cache_dir=config['data']['cache_dir']
        )
        
        for idx in range(len(dataset)):
            id_a = f"{split}_{idx}_A"
            id_b = f"{split}_{idx}_B"
            all_protein_ids.add(id_a)
            all_protein_ids.add(id_b)
    
    print(f"Total unique proteins: {len(all_protein_ids)}")
    
    # Determine which proteins need processing
    if update_existing and existing_keys:
        proteins_to_process = all_protein_ids - existing_keys
        print(f"New proteins to process: {len(proteins_to_process)}")
    else:
        proteins_to_process = all_protein_ids
        print(f"Processing all {len(proteins_to_process)} proteins")
    
    if not proteins_to_process:
        print("No new proteins to process. Graph cache is up to date.")
        return
    
    # Open cache file for writing
    mode = 'a' if (os.path.exists(graph_cache_path) and update_existing) else 'w'
    
    with h5py.File(graph_cache_path, mode) as h5f:
        # Process each protein
        for protein_id in tqdm(proteins_to_process, desc="Constructing graphs"):
            try:
                # Load embeddings and properties
                if protein_id not in embeddings_h5:
                    print(f"Warning: No embeddings for {protein_id}")
                    continue
                if protein_id not in properties_h5:
                    print(f"Warning: No properties for {protein_id}")
                    continue
                
                embeddings = embeddings_h5[protein_id][:]
                properties = properties_h5[protein_id][:]
                
                # Construct graph
                graph_data = construct_protein_graph(embeddings, properties, config)
                
                # Create group for this protein
                if protein_id in h5f:
                    del h5f[protein_id]
                
                grp = h5f.create_group(protein_id)
                
                # Store graph data
                grp.create_dataset('edge_index', data=graph_data['edge_index'], 
                                   compression='gzip', compression_opts=4)
                grp.create_dataset('degrees', data=graph_data['degrees'],
                                   compression='gzip', compression_opts=4)
                
                # Store statistics as attributes
                grp.attrs['num_nodes'] = graph_data['num_nodes']
                grp.attrs['num_edges'] = graph_data['num_edges']
                grp.attrs['avg_degree'] = graph_data['avg_degree']
                grp.attrs['max_degree'] = graph_data['max_degree']
                
            except Exception as e:
                print(f"Error processing {protein_id}: {e}")
                continue
    
    # Close input files
    embeddings_h5.close()
    properties_h5.close()
    
    print(f"\nGraphs cached successfully at {graph_cache_path}")
    
    # Verify cache
    with h5py.File(graph_cache_path, 'r') as h5f:
        print(f"Total cached graphs: {len(h5f.keys())}")
        
        # Show sample statistics
        sample_key = list(h5f.keys())[0]
        sample_grp = h5f[sample_key]
        print(f"\nSample graph statistics for {sample_key}:")
        print(f"  Nodes: {sample_grp.attrs['num_nodes']}")
        print(f"  Edges: {sample_grp.attrs['num_edges']}")
        print(f"  Avg degree: {sample_grp.attrs['avg_degree']:.2f}")
        print(f"  Max degree: {sample_grp.attrs['max_degree']}")


def main():
    parser = argparse.ArgumentParser(description='Pre-compute graph structures for GNN')
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Path to configuration file')
    parser.add_argument('--update', action='store_true',
                        help='Update existing cache with missing entries')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate all graphs (ignore existing cache)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create cache directory if needed
    os.makedirs(config['data']['cache_dir'], exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(config['reproducibility']['seed'])
    torch.manual_seed(config['reproducibility']['seed'])
    
    # Prepare graphs
    update_existing = args.update and not args.force
    prepare_graphs(config, update_existing=update_existing)
    
    print("\nGraph preparation complete!")


if __name__ == '__main__':
    main()