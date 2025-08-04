#!/usr/bin/env python3
"""
Script to prepare all required data caches for training.
Runs both ESM-2 embedding generation and physicochemical property computation.
"""

import os
import sys
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(description='Prepare all data caches for training')
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
    
    args = parser.parse_args()
    
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Step 1: Generate ESM-2 embeddings
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
    
    # Step 2: Compute physicochemical properties
    if not args.skip_properties:
        print("\n" + "="*60)
        print("Step 2: Computing physicochemical properties...")
        print("="*60)
        
        cmd = [sys.executable, os.path.join(scripts_dir, 'prepare_properties.py'), 
               '--config', args.config]
        if args.update:
            cmd.append('--update')
        
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print("Error computing properties!")
            return 1
        
        print("Physicochemical properties computed successfully!")
    else:
        print("Skipping physicochemical property computation.")
    
    # Step 3: Pre-compute graph structures
    if not args.skip_graphs:
        print("\n" + "="*60)
        print("Step 3: Pre-computing graph structures...")
        print("="*60)
        
        cmd = [sys.executable, os.path.join(scripts_dir, 'prepare_graphs.py'), 
               '--config', args.config]
        if args.update:
            cmd.append('--update')
        
        result = subprocess.run(cmd, capture_output=False, text=True)
        if result.returncode != 0:
            print("Error pre-computing graphs!")
            return 1
        
        print("Graph structures pre-computed successfully!")
    else:
        print("Skipping graph structure pre-computation.")
    
    print("\n" + "="*60)
    print("All data preparation complete!")
    print("You can now run training with: python train.py")
    print("="*60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())