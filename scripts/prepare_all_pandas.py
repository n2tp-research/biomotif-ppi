#!/usr/bin/env python3
"""
Unified data preparation script using pandas and pandarallel.
Prepares embeddings, properties, and graphs efficiently.
"""

import os
import sys
import subprocess
import argparse
import pandas as pd
from pandarallel import pandarallel


def main():
    parser = argparse.ArgumentParser(description='Prepare all data caches using pandas/pandarallel')
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Path to configuration file')
    parser.add_argument('--skip-embeddings', action='store_true',
                        help='Skip ESM-2 embedding generation')
    parser.add_argument('--skip-properties', action='store_true',
                        help='Skip physicochemical property computation')
    parser.add_argument('--skip-graphs', action='store_true',
                        help='Skip graph structure pre-computation')
    parser.add_argument('--update', action='store_true',
                        help='Update existing caches with missing entries only')
    parser.add_argument('--num-workers', type=int, default=-1,
                        help='Number of parallel workers (-1 for all CPUs)')
    
    args = parser.parse_args()
    
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Initialize pandarallel once
    print(f"Initializing Pandarallel with {args.num_workers if args.num_workers > 0 else 'all'} workers")
    pandarallel.initialize(
        nb_workers=args.num_workers,
        progress_bar=True,
        verbose=1
    )
    
    # Step 1: Generate ESM-2 embeddings (still uses original script as it's already optimized)
    if not args.skip_embeddings:
        print("\n" + "="*60)
        print("Step 1: Generating ESM-2 embeddings...")
        print("="*60)
        
        cmd = [sys.executable, os.path.join(scripts_dir, 'prepare_data.py'), 
               '--config', args.config]
        
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print("Error generating embeddings!")
            return 1
        
        print("ESM-2 embeddings generated successfully!")
    else:
        print("Skipping ESM-2 embedding generation.")
    
    # Step 2: Compute physicochemical properties with pandas
    if not args.skip_properties:
        print("\n" + "="*60)
        print("Step 2: Computing physicochemical properties with pandas...")
        print("="*60)
        
        cmd = [sys.executable, os.path.join(scripts_dir, 'prepare_properties_pandas.py'), 
               '--config', args.config,
               '--num-workers', str(args.num_workers)]
        if args.update:
            cmd.append('--update')
        
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print("Error computing properties!")
            return 1
        
        print("Physicochemical properties computed successfully!")
    else:
        print("Skipping physicochemical property computation.")
    
    # Step 3: Pre-compute graph structures with pandas
    if not args.skip_graphs:
        print("\n" + "="*60)
        print("Step 3: Pre-computing graph structures with pandas...")
        print("="*60)
        
        cmd = [sys.executable, os.path.join(scripts_dir, 'prepare_graphs_pandas.py'), 
               '--config', args.config,
               '--num-workers', str(args.num_workers)]
        if args.update:
            cmd.append('--update')
        
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print("Error pre-computing graphs!")
            return 1
        
        print("Graph structures pre-computed successfully!")
    else:
        print("Skipping graph structure pre-computation.")
    
    # Summary
    print("\n" + "="*60)
    print("Data Preparation Summary")
    print("="*60)
    
    # Create summary DataFrame
    summary_data = []
    
    if not args.skip_embeddings:
        summary_data.append({
            'Step': 'ESM-2 Embeddings',
            'Status': '✓ Complete',
            'Method': 'PyTorch (GPU)'
        })
    
    if not args.skip_properties:
        summary_data.append({
            'Step': 'Physicochemical Properties',
            'Status': '✓ Complete',
            'Method': 'Pandas + Pandarallel'
        })
    
    if not args.skip_graphs:
        summary_data.append({
            'Step': 'Graph Structures',
            'Status': '✓ Complete',
            'Method': 'Pandas + Pandarallel'
        })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
    
    print("\n" + "="*60)
    print("All data preparation complete!")
    print("You can now run training with: python train.py")
    print("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())