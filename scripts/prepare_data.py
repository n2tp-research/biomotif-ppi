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


def prepare_bernett_gold_dataset(config_path: str, output_dir: str):
    """
    Prepare the Bernett Gold PPI dataset for training.
    """
    # Load config
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading Bernett Gold PPI dataset from Hugging Face...")
    
    # Collect all unique protein IDs
    all_protein_ids = set()
    
    for split in ['train', 'valid', 'test']:
        print(f"Processing {split} split...")
        dataset = load_dataset(
            config['data']['dataset_name'],
            split=split,
            cache_dir=output_dir
        )
        
        for item in tqdm(dataset, desc=f"Collecting protein IDs from {split}"):
            all_protein_ids.add(item['SeqA'])
            all_protein_ids.add(item['SeqB'])
    
    print(f"Found {len(all_protein_ids)} unique protein IDs")
    
    # Check if sequence file already exists
    sequence_file = os.path.join(
        output_dir,
        f"sequences_{config['data']['uniprot_release']}.pkl"
    )
    
    if os.path.exists(sequence_file):
        print(f"Sequence file already exists at {sequence_file}")
        with open(sequence_file, 'rb') as f:
            sequences = pickle.load(f)
        print(f"Loaded {len(sequences)} sequences")
    else:
        print("\nDownloading sequences from UniProt...")
        print("Note: This is using dummy sequences for demonstration.")
        print("In production, implement proper UniProt downloading.")
        
        sequences = download_sequences_from_uniprot(
            list(all_protein_ids),
            sequence_file
        )
        
        # Save sequences
        print(f"Saving sequences to {sequence_file}")
        with open(sequence_file, 'wb') as f:
            pickle.dump(sequences, f)
    
    # Create dataset statistics
    stats = {
        'total_proteins': len(all_protein_ids),
        'proteins_with_sequences': len(sequences),
        'missing_sequences': len(all_protein_ids) - len(sequences)
    }
    
    # Save statistics
    stats_file = os.path.join(output_dir, 'dataset_stats.txt')
    with open(stats_file, 'w') as f:
        f.write("Bernett Gold PPI Dataset Statistics\n")
        f.write("===================================\n\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nDataset statistics saved to {stats_file}")
    print("Data preparation completed!")


def main():
    parser = argparse.ArgumentParser(description='Prepare data for BioMotif-PPI')
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='./data',
                        help='Output directory for prepared data')
    
    args = parser.parse_args()
    
    prepare_bernett_gold_dataset(args.config, args.output_dir)


if __name__ == '__main__':
    main()