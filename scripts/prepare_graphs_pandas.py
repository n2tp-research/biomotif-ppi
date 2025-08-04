#!/usr/bin/env python3
"""
Script to pre-compute graph structures using pandas and pandarallel.
Cleaner and more efficient than multiprocessing.
"""

import os
import sys
import h5py
import yaml
import torch
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset
import torch.nn.functional as F
from pandarallel import pandarallel

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def construct_graph_for_protein(row, config):
    """
    Construct graph for a single protein.
    
    Args:
        row: DataFrame row with embeddings and properties
        config: Configuration dictionary
    
    Returns:
        Dictionary with graph data
    """
    try:
        embeddings = row['embeddings']
        properties = row['properties']
        
        if embeddings is None or properties is None:
            return None
        
        seq_len = embeddings.shape[0]
        
        # Convert to tensors
        embeddings = torch.from_numpy(embeddings).float()
        properties = torch.from_numpy(properties).float()
        
        # Extract secondary structure propensities
        alpha_prop = properties[:, -2]
        beta_prop = properties[:, -1]
        
        edge_list = []
        
        # Sequential neighbors
        for i in range(seq_len - 1):
            edge_list.append([i, i + 1])
            edge_list.append([i + 1, i])
        
        # Long-range connections
        num_long_range = min(seq_len * 5, 500)
        
        if seq_len > 10:
            embeddings_norm = F.normalize(embeddings, dim=-1)
            
            # Sample connections
            for _ in range(num_long_range):
                i = np.random.randint(0, max(1, seq_len - 5))
                j = i + np.random.randint(5, min(50, seq_len - i))
                
                if j < seq_len:
                    similarity = torch.dot(embeddings_norm[i], embeddings_norm[j]).item()
                    ss_bonus = alpha_prop[i] * alpha_prop[j] + beta_prop[i] * beta_prop[j]
                    combined_score = similarity + 0.3 * ss_bonus.item()
                    
                    if combined_score > 0.5:
                        edge_list.append([i, j])
                        edge_list.append([j, i])
        
        # Convert to edge index
        if len(edge_list) > 0:
            edge_index = np.array(edge_list, dtype=np.int64).T
        else:
            edge_index = np.zeros((2, 0), dtype=np.int64)
        
        # Compute statistics
        num_nodes = seq_len
        num_edges = edge_index.shape[1] // 2
        
        degrees = np.zeros(num_nodes)
        if edge_index.shape[1] > 0:
            unique, counts = np.unique(edge_index[0], return_counts=True)
            degrees[unique] = counts
        
        return {
            'edge_index': edge_index,
            'degrees': degrees,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'avg_degree': np.mean(degrees),
            'max_degree': np.max(degrees) if num_nodes > 0 else 0
        }
        
    except Exception as e:
        print(f"Error constructing graph: {e}")
        return None


def prepare_graphs(config: dict, update_existing: bool = True, num_workers: int = -1):
    """
    Pre-compute graph structures using pandas and pandarallel.
    
    Args:
        config: Configuration dictionary
        update_existing: If True, update existing cache with missing entries
        num_workers: Number of parallel workers (-1 for all CPUs)
    """
    
    # Initialize pandarallel
    pandarallel.initialize(
        nb_workers=num_workers,
        progress_bar=True,
        verbose=1
    )
    
    print(f"Initialized Pandarallel with {num_workers if num_workers > 0 else 'all'} workers")
    
    # Set random seed
    np.random.seed(config['reproducibility']['seed'])
    torch.manual_seed(config['reproducibility']['seed'])
    
    # Load caches
    model_name = config['esm']['model_name'].replace('/', '_')
    embedding_cache_path = os.path.join(
        config['data']['cache_dir'],
        f'bernett_gold_ppi_{model_name}_embeddings.h5'
    )
    properties_cache_path = os.path.join(
        config['data']['cache_dir'],
        'bernett_gold_ppi_properties.h5'
    )
    graph_cache_path = os.path.join(
        config['data']['cache_dir'],
        'bernett_gold_ppi_graphs.h5'
    )
    
    if not os.path.exists(embedding_cache_path):
        print(f"ERROR: Embedding cache not found at {embedding_cache_path}")
        return
    
    if not os.path.exists(properties_cache_path):
        print(f"ERROR: Properties cache not found at {properties_cache_path}")
        return
    
    # Load protein IDs
    print("\nCollecting protein IDs...")
    protein_ids = []
    
    for split in ['train', 'valid', 'test']:
        dataset = load_dataset(
            config['data']['dataset_name'],
            split=split,
            cache_dir=config['data']['cache_dir']
        )
        
        for idx in range(len(dataset)):
            protein_ids.append(f"{split}_{idx}_A")
            protein_ids.append(f"{split}_{idx}_B")
    
    # Create DataFrame
    df = pd.DataFrame({'protein_id': protein_ids})
    df = df.drop_duplicates()
    print(f"Total unique proteins: {len(df)}")
    
    # Check existing cache
    if os.path.exists(graph_cache_path) and update_existing:
        print(f"Found existing graph cache at {graph_cache_path}")
        with h5py.File(graph_cache_path, 'r') as h5f:
            existing_ids = set(h5f.keys())
            print(f"Existing cached graphs: {len(existing_ids)}")
            
        # Filter to only new proteins
        df = df[~df['protein_id'].isin(existing_ids)]
        print(f"New proteins to process: {len(df)}")
    
    if len(df) == 0:
        print("No new proteins to process. Graph cache is up to date.")
        return
    
    # Load embeddings and properties for proteins
    print("\nLoading embeddings and properties...")
    
    embeddings_dict = {}
    properties_dict = {}
    
    with h5py.File(embedding_cache_path, 'r') as emb_h5:
        with h5py.File(properties_cache_path, 'r') as prop_h5:
            for protein_id in df['protein_id']:
                if protein_id in emb_h5 and protein_id in prop_h5:
                    embeddings_dict[protein_id] = emb_h5[protein_id][:]
                    properties_dict[protein_id] = prop_h5[protein_id][:]
    
    # Add to DataFrame
    df['embeddings'] = df['protein_id'].map(embeddings_dict)
    df['properties'] = df['protein_id'].map(properties_dict)
    
    # Filter out proteins without data
    initial_count = len(df)
    df = df.dropna(subset=['embeddings', 'properties'])
    if len(df) < initial_count:
        print(f"Warning: {initial_count - len(df)} proteins missing embeddings/properties")
    
    # Add sequence length for statistics
    df['seq_length'] = df['embeddings'].apply(lambda x: x.shape[0] if x is not None else 0)
    
    print(f"\nSequence length statistics:")
    print(df['seq_length'].describe())
    
    # Construct graphs in parallel
    print("\nConstructing graphs in parallel...")
    df['graph_data'] = df.parallel_apply(
        lambda row: construct_graph_for_protein(row, config), 
        axis=1
    )
    
    # Check for failures
    failed_count = df['graph_data'].isna().sum()
    if failed_count > 0:
        print(f"Warning: {failed_count} graphs failed construction")
        df = df.dropna(subset=['graph_data'])
    
    # Save to HDF5
    mode = 'a' if (os.path.exists(graph_cache_path) and update_existing) else 'w'
    
    print(f"\nSaving {len(df)} graphs to cache...")
    with h5py.File(graph_cache_path, mode) as h5f:
        for _, row in df.iterrows():
            protein_id = row['protein_id']
            graph_data = row['graph_data']
            
            if graph_data is not None:
                try:
                    if protein_id in h5f:
                        del h5f[protein_id]
                    
                    grp = h5f.create_group(protein_id)
                    
                    # Store graph data
                    grp.create_dataset('edge_index', 
                                      data=graph_data['edge_index'],
                                      compression='gzip', 
                                      compression_opts=4)
                    grp.create_dataset('degrees', 
                                      data=graph_data['degrees'],
                                      compression='gzip', 
                                      compression_opts=4)
                    
                    # Store statistics
                    grp.attrs['num_nodes'] = graph_data['num_nodes']
                    grp.attrs['num_edges'] = graph_data['num_edges']
                    grp.attrs['avg_degree'] = graph_data['avg_degree']
                    grp.attrs['max_degree'] = graph_data['max_degree']
                    
                except Exception as e:
                    print(f"Error saving {protein_id}: {e}")
    
    print(f"\nGraphs cached successfully at {graph_cache_path}")
    
    # Show statistics
    with h5py.File(graph_cache_path, 'r') as h5f:
        print(f"Total cached graphs: {len(h5f.keys())}")
    
    # Graph statistics from DataFrame
    if 'graph_data' in df.columns:
        df['num_edges'] = df['graph_data'].apply(
            lambda x: x['num_edges'] if x is not None else 0
        )
        df['avg_degree'] = df['graph_data'].apply(
            lambda x: x['avg_degree'] if x is not None else 0
        )
        
        print("\n" + "="*60)
        print("Graph Statistics:")
        print("="*60)
        print(f"Average edges per graph: {df['num_edges'].mean():.2f}")
        print(f"Average degree: {df['avg_degree'].mean():.2f}")
        print(f"Min/Max edges: {df['num_edges'].min()}/{df['num_edges'].max()}")


def main():
    parser = argparse.ArgumentParser(description='Pre-compute graph structures with pandas')
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Path to configuration file')
    parser.add_argument('--update', action='store_true',
                        help='Update existing cache with missing entries')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate all graphs (ignore existing cache)')
    parser.add_argument('--num-workers', type=int, default=-1,
                        help='Number of parallel workers (-1 for all CPUs)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create cache directory if needed
    os.makedirs(config['data']['cache_dir'], exist_ok=True)
    
    # Prepare graphs
    update_existing = args.update and not args.force
    prepare_graphs(config, update_existing=update_existing, num_workers=args.num_workers)
    
    print("\nGraph preparation complete!")


if __name__ == '__main__':
    main()