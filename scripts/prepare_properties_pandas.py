#!/usr/bin/env python3
"""
Script to pre-compute and cache physicochemical properties using pandas and pandarallel.
Much cleaner and faster than multiprocessing.
"""

import os
import sys
import h5py
import yaml
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_dataset
from pandarallel import pandarallel

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from data.properties import PhysicochemicalEncoder


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def compute_properties_for_sequence(sequence, encoder):
    """
    Compute properties for a single sequence.
    
    Args:
        sequence: Protein sequence string
        encoder: PhysicochemicalEncoder instance
    
    Returns:
        Numpy array of properties
    """
    try:
        properties = encoder.encode_sequence_with_ss(sequence)
        return properties.numpy()
    except Exception as e:
        print(f"Error processing sequence: {e}")
        return None


def prepare_properties(config: dict, update_existing: bool = True, num_workers: int = -1):
    """
    Pre-compute physicochemical properties using pandas and pandarallel.
    
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
    
    # Initialize property encoder
    print("Initializing physicochemical encoder...")
    prop_encoder = PhysicochemicalEncoder(config)
    
    # Prepare cache path
    cache_path = os.path.join(
        config['data']['cache_dir'],
        'bernett_gold_ppi_properties.h5'
    )
    
    # Load dataset into pandas DataFrame
    print("\nLoading sequences into DataFrame...")
    data_records = []
    
    for split in ['train', 'valid', 'test']:
        dataset = load_dataset(
            config['data']['dataset_name'],
            split=split,
            cache_dir=config['data']['cache_dir']
        )
        
        for idx, item in enumerate(dataset):
            # Add both sequences
            data_records.append({
                'protein_id': f"{split}_{idx}_A",
                'sequence': item['SeqA'],
                'split': split,
                'pair_idx': idx,
                'chain': 'A'
            })
            data_records.append({
                'protein_id': f"{split}_{idx}_B",
                'sequence': item['SeqB'],
                'split': split,
                'pair_idx': idx,
                'chain': 'B'
            })
    
    # Create DataFrame
    df = pd.DataFrame(data_records)
    print(f"Loaded {len(df)} sequences into DataFrame")
    
    # Remove duplicates (if any)
    df = df.drop_duplicates(subset=['protein_id'])
    print(f"Unique sequences: {len(df)}")
    
    # Check existing cache
    existing_ids = set()
    if os.path.exists(cache_path) and update_existing:
        print(f"Found existing cache at {cache_path}")
        with h5py.File(cache_path, 'r') as h5f:
            existing_ids = set(h5f.keys())
            print(f"Existing cached properties: {len(existing_ids)}")
        
        # Filter to only new sequences
        df = df[~df['protein_id'].isin(existing_ids)]
        print(f"New sequences to process: {len(df)}")
    
    if len(df) == 0:
        print("No new sequences to process. Cache is up to date.")
        return
    
    # Add sequence statistics
    df['seq_length'] = df['sequence'].str.len()
    print(f"\nSequence length statistics:")
    print(df['seq_length'].describe())
    
    # Compute properties in parallel using pandarallel
    print("\nComputing properties in parallel...")
    df['properties'] = df['sequence'].parallel_apply(
        lambda seq: compute_properties_for_sequence(seq, prop_encoder)
    )
    
    # Check for failures
    failed_count = df['properties'].isna().sum()
    if failed_count > 0:
        print(f"Warning: {failed_count} sequences failed property computation")
        df = df.dropna(subset=['properties'])
    
    # Save to HDF5
    mode = 'a' if (os.path.exists(cache_path) and update_existing) else 'w'
    
    print(f"\nSaving {len(df)} property arrays to cache...")
    with h5py.File(cache_path, mode) as h5f:
        for _, row in df.iterrows():
            protein_id = row['protein_id']
            properties = row['properties']
            
            if properties is not None:
                try:
                    if protein_id in h5f:
                        del h5f[protein_id]
                    
                    h5f.create_dataset(
                        protein_id,
                        data=properties,
                        compression='gzip',
                        compression_opts=4
                    )
                except Exception as e:
                    print(f"Error saving {protein_id}: {e}")
    
    print(f"\nProperties cached successfully at {cache_path}")
    
    # Verify cache and show statistics
    with h5py.File(cache_path, 'r') as h5f:
        total_cached = len(h5f.keys())
        print(f"Total cached properties: {total_cached}")
        
        if total_cached > 0:
            # Show sample
            sample_key = list(h5f.keys())[0]
            sample_data = h5f[sample_key][:]
            print(f"\nSample property shape: {sample_data.shape}")
            print(f"Property dimensions: {sample_data.shape[-1]} features per residue")
    
    # Show processing statistics
    print("\n" + "="*60)
    print("Processing Statistics:")
    print("="*60)
    
    stats_df = df.groupby('split').agg({
        'protein_id': 'count',
        'seq_length': ['mean', 'min', 'max']
    }).round(2)
    
    print(stats_df)


def main():
    parser = argparse.ArgumentParser(description='Pre-compute physicochemical properties with pandas')
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Path to configuration file')
    parser.add_argument('--update', action='store_true',
                        help='Update existing cache with missing entries')
    parser.add_argument('--force', action='store_true',
                        help='Regenerate all properties (ignore existing cache)')
    parser.add_argument('--num-workers', type=int, default=-1,
                        help='Number of parallel workers (-1 for all CPUs)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create cache directory if needed
    os.makedirs(config['data']['cache_dir'], exist_ok=True)
    
    # Prepare properties
    update_existing = args.update and not args.force
    prepare_properties(config, update_existing=update_existing, num_workers=args.num_workers)
    
    print("\nProperty preparation complete!")


if __name__ == '__main__':
    main()