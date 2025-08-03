#!/usr/bin/env python3
"""
Script to prepare data for BioMotif-PPI training.
Downloads dataset from Hugging Face and prepares sequence files.
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from typing import Dict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.esm_embeddings import ESMEmbeddingGenerator


def download_sequences_from_uniprot(protein_ids: list, output_file: str):
    """
    Download sequences from UniProt (placeholder function).
    In practice, you would implement UniProt API calls here.
    """
    print("Note: This is a placeholder function.")
    print("In production, implement UniProt API calls to fetch sequences.")
    print(f"Would download {len(protein_ids)} sequences to {output_file}")
    
    # Create dummy sequences for demonstration
    sequences = {}
    for protein_id in tqdm(protein_ids[:100], desc="Creating dummy sequences"):
        # Generate random sequence of appropriate length
        import random
        length = random.randint(100, 500)
        sequence = ''.join(random.choices('ACDEFGHIKLMNPQRSTVWY', k=length))
        sequences[protein_id] = sequence
    
    return sequences


def prepare_bernett_gold_dataset(config_path: str, output_dir: str, generate_embeddings: bool = True, batch_size: int = None):
    """
    Prepare the Bernett Gold PPI dataset for training.
    Downloads dataset and generates ESM-2 embeddings.
    """
    # Load config
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = config['data']['cache_dir']
    os.makedirs(cache_dir, exist_ok=True)
    
    print("Loading Bernett Gold PPI dataset from Hugging Face...")
    print("Note: Sequences are included in the dataset (SeqA and SeqB fields)")
    
    # Collect all sequences
    all_sequences = {}
    total_samples = 0
    stats = {}
    
    for split in ['train', 'valid', 'test']:
        print(f"\nProcessing {split} split...")
        dataset = load_dataset(
            config['data']['dataset_name'],
            split=split,
            cache_dir=output_dir
        )
        
        num_samples = len(dataset)
        total_samples += num_samples
        stats[f'{split}_samples'] = num_samples
        
        # Collect sequences with unique IDs matching dataset.py
        for idx, item in enumerate(tqdm(dataset, desc=f"Collecting from {split}")):
            seq_a = item['SeqA']
            seq_b = item['SeqB']
            
            # Create IDs that match dataset.py
            id_a = f"{split}_{idx}_A"
            id_b = f"{split}_{idx}_B"
            
            all_sequences[id_a] = seq_a
            all_sequences[id_b] = seq_b
        
        # Show sample
        if num_samples > 0:
            sample = dataset[0]
            print(f"Sample keys: {list(sample.keys())}")
            print(f"SeqA length: {len(sample['SeqA'])}")
            print(f"SeqB length: {len(sample['SeqB'])}")
            print(f"Label: {sample['labels']}")
    
    stats['total_samples'] = total_samples
    stats['total_sequences'] = len(all_sequences)
    
    # Save statistics
    stats_file = os.path.join(output_dir, 'dataset_stats.txt')
    with open(stats_file, 'w') as f:
        f.write("Bernett Gold PPI Dataset Statistics\n")
        f.write("===================================\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nDataset statistics saved to {stats_file}")
    print(f"Total sequences to embed: {len(all_sequences)}")
    
    # Generate ESM embeddings if requested
    if generate_embeddings:
        print("\n" + "="*50)
        print("Generating ESM-2 embeddings...")
        print("="*50)
        
        # Check device
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        # Override batch size if specified
        if batch_size is not None:
            original_batch_size = config['esm']['batch_size']
            config['esm']['batch_size'] = batch_size
            print(f"Using batch size: {batch_size} (overriding config: {original_batch_size})")
        else:
            print(f"Using batch size from config: {config['esm']['batch_size']}")
        
        # Initialize embedding generator
        esm_generator = ESMEmbeddingGenerator(config, device=device)
        
        # Generate embeddings
        cache_path = esm_generator.generate_embeddings(
            all_sequences,
            "bernett_gold_ppi",
            force_regenerate=True
        )
        
        print(f"\nEmbeddings saved to: {cache_path}")
        print(f"Cache file size: {os.path.getsize(cache_path) / (1024**3):.2f} GB")
    else:
        print("\nSkipping embedding generation (use --generate-embeddings to enable)")
    
    print("\nData preparation completed!")


def main():
    parser = argparse.ArgumentParser(description='Prepare data for BioMotif-PPI')
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='./data',
                        help='Output directory for prepared data')
    parser.add_argument('--generate-embeddings', action='store_true',
                        help='Generate ESM-2 embeddings (default: True)')
    parser.add_argument('--no-generate-embeddings', dest='generate_embeddings', 
                        action='store_false',
                        help='Skip ESM-2 embedding generation')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size for ESM-2 embedding generation (default: use config)')
    parser.set_defaults(generate_embeddings=True)
    
    args = parser.parse_args()
    
    prepare_bernett_gold_dataset(args.config, args.output_dir, args.generate_embeddings, args.batch_size)


if __name__ == '__main__':
    main()