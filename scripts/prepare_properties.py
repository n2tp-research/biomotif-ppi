#!/usr/bin/env python3
"""
Script to pre-compute and cache physicochemical properties for all protein sequences.
This avoids recalculating properties during training.
"""

import os
import sys
import h5py
import yaml
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from data.properties import PhysicochemicalEncoder


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_properties(config: dict, update_existing: bool = True):
    """
    Pre-compute physicochemical properties for all sequences in the dataset.
    
    Args:
        config: Configuration dictionary
        update_existing: If True, update existing cache with missing entries
    """
    
    # Initialize property encoder
    print("Initializing physicochemical encoder...")
    prop_encoder = PhysicochemicalEncoder(config)
    
    # Prepare cache path
    model_name = config['esm']['model_name'].replace('/', '_')
    cache_path = os.path.join(
        config['data']['cache_dir'],
        f'bernett_gold_ppi_properties.h5'
    )
    
    # Load dataset splits
    splits = ['train', 'valid', 'test']
    all_sequences = {}
    
    print("\nLoading sequences from dataset...")
    for split in splits:
        dataset = load_dataset(
            config['data']['dataset_name'],
            split=split,
            cache_dir=config['data']['cache_dir']
        )
        
        for idx, item in enumerate(tqdm(dataset, desc=f"Processing {split}")):
            # Create unique IDs matching the dataset loader
            id_a = f"{split}_{idx}_A"
            id_b = f"{split}_{idx}_B"
            
            # Store sequences
            all_sequences[id_a] = item['SeqA']
            all_sequences[id_b] = item['SeqB']
    
    print(f"\nTotal unique sequences: {len(all_sequences)}")
    
    # Check existing cache
    existing_keys = set()
    if os.path.exists(cache_path) and update_existing:
        print(f"Found existing cache at {cache_path}")
        with h5py.File(cache_path, 'r') as h5f:
            existing_keys = set(h5f.keys())
            print(f"Existing cached properties: {len(existing_keys)}")
    
    # Determine which sequences need processing
    if update_existing and existing_keys:
        sequences_to_process = {k: v for k, v in all_sequences.items() if k not in existing_keys}
        print(f"New sequences to process: {len(sequences_to_process)}")
    else:
        sequences_to_process = all_sequences
        print(f"Processing all {len(sequences_to_process)} sequences")
    
    if not sequences_to_process:
        print("No new sequences to process. Cache is up to date.")
        return
    
    # Open cache file for writing
    mode = 'a' if (os.path.exists(cache_path) and update_existing) else 'w'
    
    with h5py.File(cache_path, mode) as h5f:
        # Process sequences
        for protein_id, sequence in tqdm(sequences_to_process.items(), desc="Computing properties"):
            try:
                # Compute properties with secondary structure
                properties = prop_encoder.encode_sequence_with_ss(sequence)
                
                # Store in HDF5
                if protein_id in h5f:
                    del h5f[protein_id]  # Remove old entry if exists
                
                h5f.create_dataset(
                    protein_id,
                    data=properties.numpy(),
                    compression='gzip',
                    compression_opts=4
                )
                
            except Exception as e:
                print(f"Error processing {protein_id}: {e}")
                continue
    
    print(f"\nProperties cached successfully at {cache_path}")
    
    # Verify cache
    with h5py.File(cache_path, 'r') as h5f:
        print(f"Total cached properties: {len(h5f.keys())}")
        
        # Show sample entry
        sample_key = list(h5f.keys())[0]
        sample_data = h5f[sample_key][:]
        print(f"Sample property shape: {sample_data.shape}")
        print(f"Property dimensions: {sample_data.shape[-1]} features per residue")


def main():
    parser = argparse.ArgumentParser(description='Pre-compute physicochemical properties')
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Path to configuration file')
    parser.add_argument('--update', action='store_true',
                        help='Update existing cache with missing entries')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate all properties (ignore existing cache)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create cache directory if needed
    os.makedirs(config['data']['cache_dir'], exist_ok=True)
    
    # Prepare properties
    update_existing = args.update and not args.force
    prepare_properties(config, update_existing=update_existing)
    
    print("\nProperty preparation complete!")


if __name__ == '__main__':
    main()