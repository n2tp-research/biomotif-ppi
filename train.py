#!/usr/bin/env python3
"""
Main training script for BioMotif-PPI model.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import random
from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.dataset import PPIDataset, collate_fn
from data.esm_embeddings import ESMEmbeddingGenerator
from data.properties import PhysicochemicalEncoder
from models.biomotif_ppi import create_model
from training.trainer import BioMotifTrainer
from utils.metrics import evaluate_model, MetricsCalculator


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Set deterministic algorithms if specified
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def prepare_data(config: dict):
    """Prepare datasets and data loaders."""
    print("Preparing datasets...")
    
    # Initialize property encoder
    prop_encoder = PhysicochemicalEncoder(config)
    
    # Check for cached embeddings
    model_name = config['esm']['model_name'].replace('/', '_')
    embedding_cache_path = os.path.join(
        config['data']['cache_dir'],
        f'bernett_gold_ppi_{model_name}_embeddings.h5'
    )
    
    if not os.path.exists(embedding_cache_path):
        print(f"ERROR: Embedding cache not found at {embedding_cache_path}")
        print("Please run: python scripts/generate_embeddings.py")
        return None, None, None
    
    print(f"Using cached embeddings from {embedding_cache_path}")
    
    # Create datasets (sequences are loaded from HuggingFace directly)
    train_dataset = PPIDataset(
        split=config['data']['train_split'],
        config=config,
        sequence_dict=None,  # Will use sequences from dataset
        embedding_cache=embedding_cache_path,
        transform=lambda x: add_properties(x, prop_encoder)
    )
    
    val_dataset = PPIDataset(
        split=config['data']['val_split'],
        config=config,
        sequence_dict=None,
        embedding_cache=embedding_cache_path,
        transform=lambda x: add_properties(x, prop_encoder)
    )
    
    test_dataset = PPIDataset(
        split=config['data']['test_split'],
        config=config,
        sequence_dict=None,
        embedding_cache=embedding_cache_path,
        transform=lambda x: add_properties(x, prop_encoder)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['data']['train_batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        collate_fn=collate_fn,
        prefetch_factor=config['data']['prefetch_factor']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['data']['val_batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        collate_fn=collate_fn,
        prefetch_factor=config['data']['prefetch_factor']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['data']['test_batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        collate_fn=collate_fn,
        prefetch_factor=config['data']['prefetch_factor']
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def add_properties(batch_item: dict, prop_encoder: PhysicochemicalEncoder) -> dict:
    """Add physicochemical properties to batch item."""
    # Encode properties for both sequences
    batch_item['properties_a'] = prop_encoder.encode_sequence_with_ss(batch_item['sequence_a'])
    batch_item['properties_b'] = prop_encoder.encode_sequence_with_ss(batch_item['sequence_b'])
    return batch_item


def main():
    parser = argparse.ArgumentParser(description='Train BioMotif-PPI model')
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only run evaluation on test set')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    config['hardware']['device'] = args.device
    
    # Set seed
    seed = args.seed if args.seed is not None else config['reproducibility']['seed']
    set_seed(seed)
    print(f"Using random seed: {seed}")
    
    # Create directories
    os.makedirs(config['paths']['checkpoint_root'], exist_ok=True)
    os.makedirs(config['paths']['log_root'], exist_ok=True)
    os.makedirs(config['paths']['results_root'], exist_ok=True)
    
    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(config)
    if train_loader is None:
        print("Failed to prepare data. Exiting.")
        return
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load checkpoint if specified
    if args.resume:
        print(f"Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Resumed from epoch {checkpoint['epoch']}")
    
    # Evaluation only mode
    if args.eval_only:
        print("Running evaluation on test set...")
        model.to(args.device)
        model.eval()
        
        # Evaluate
        metrics = evaluate_model(model, test_loader, args.device)
        print("\nTest Set Results:")
        print(metrics)
        
        # Save results
        results_path = os.path.join(
            config['paths']['results_root'],
            f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        with open(results_path, 'w') as f:
            f.write(str(metrics))
        print(f"\nResults saved to {results_path}")
        
        return
    
    # Create trainer
    print("Initializing trainer...")
    trainer = BioMotifTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device
    )
    
    # Load checkpoint if resuming
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train model
    print("\nStarting training...")
    trainer.train()
    
    # Final evaluation on test set
    print("\nRunning final evaluation on test set...")
    best_checkpoint = os.path.join(
        config['logging']['checkpoint_dir'],
        'best_model.pt'
    )
    
    if os.path.exists(best_checkpoint):
        print(f"Loading best model from {best_checkpoint}")
        checkpoint = torch.load(best_checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    test_metrics = evaluate_model(model, test_loader, args.device)
    
    print("\nFinal Test Set Results:")
    print(test_metrics)
    
    # Save final results
    results_path = os.path.join(
        config['paths']['results_root'],
        f"final_results_{config['experiment']['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )
    
    with open(results_path, 'w') as f:
        f.write(f"Experiment: {config['experiment']['name']}\n")
        f.write(f"Description: {config['experiment']['description']}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Test Set Results:\n")
        f.write(str(test_metrics))
        f.write(f"\n\nBest validation epoch: {checkpoint['epoch']}")
        f.write(f"\nBest validation {config['validation']['monitor_metric']}: {checkpoint['best_val_metric']:.4f}")
    
    print(f"\nResults saved to {results_path}")
    print("Training completed successfully!")


if __name__ == '__main__':
    main()