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
from multiprocessing import Pool, cpu_count
from functools import partial

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))

from data.properties import PhysicochemicalEncoder


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def compute_properties_batch(batch_data, config):
    """
    Compute properties for a batch of sequences.
    
    Args:
        batch_data: Tuple of (protein_ids, sequences)
        config: Configuration dictionary
    
    Returns:
        List of (protein_id, properties) tuples
    """
    protein_ids, sequences = batch_data
    
    # Initialize encoder for this process
    from data.properties import PhysicochemicalEncoder
    prop_encoder = PhysicochemicalEncoder(config)
    
    results = []
    for protein_id, sequence in zip(protein_ids, sequences):
        try:
            # Compute properties with secondary structure
            properties = prop_encoder.encode_sequence_with_ss(sequence)
            results.append((protein_id, properties.numpy()))
        except Exception as e:
            print(f"Error processing {protein_id}: {e}")
            results.append((protein_id, None))
    
    return results


def prepare_properties(config: dict, update_existing: bool = True, num_workers: int = None):
    """
    Pre-compute physicochemical properties for all sequences in the dataset.
    
    Args:
        config: Configuration dictionary
        update_existing: If True, update existing cache with missing entries
        num_workers: Number of parallel workers (default: CPU count - 1)
    """
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    print(f"Using {num_workers} parallel workers for property computation")
    
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
    
    # Prepare batches for parallel processing
    protein_ids = list(sequences_to_process.keys())
    sequences = list(sequences_to_process.values())
    
    # Create batches
    batch_size = max(1, len(protein_ids) // (num_workers * 10))  # 10 batches per worker
    batches = []
    for i in range(0, len(protein_ids), batch_size):
        batch_ids = protein_ids[i:i+batch_size]
        batch_seqs = sequences[i:i+batch_size]
        batches.append((batch_ids, batch_seqs))
    
    print(f"Processing {len(sequences_to_process)} sequences in {len(batches)} batches")
    
    # Process batches in parallel
    compute_func = partial(compute_properties_batch, config=config)
    
    with Pool(num_workers) as pool:
        # Process with progress bar
        results = []
        with tqdm(total=len(sequences_to_process), desc="Computing properties") as pbar:
            for batch_results in pool.imap(compute_func, batches):
                results.extend(batch_results)
                pbar.update(len(batch_results))
    
    # Open cache file for writing
    mode = 'a' if (os.path.exists(cache_path) and update_existing) else 'w'
    
    print("\nSaving properties to cache...")
    with h5py.File(cache_path, mode) as h5f:
        # Store results
        success_count = 0
        for protein_id, properties in tqdm(results, desc="Writing to HDF5"):
            if properties is not None:
                try:
                    # Store in HDF5
                    if protein_id in h5f:
                        del h5f[protein_id]  # Remove old entry if exists
                    
                    h5f.create_dataset(
                        protein_id,
                        data=properties,
                        compression='gzip',
                        compression_opts=4
                    )
                    success_count += 1
                except Exception as e:
                    print(f"Error saving {protein_id}: {e}")
            else:
                print(f"Skipping {protein_id} (computation failed)")
    
    print(f"\nSuccessfully processed {success_count}/{len(results)} sequences")
    print(f"Properties cached at {cache_path}")
    
    # Verify cache
    with h5py.File(cache_path, 'r') as h5f:
        print(f"Total cached properties: {len(h5f.keys())}")
        
        # Show sample entry
        if len(h5f.keys()) > 0:
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
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count - 1)')
    
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