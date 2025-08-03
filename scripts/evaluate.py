#!/usr/bin/env python3
"""
Script to evaluate trained BioMotif-PPI model and generate plots.
"""

import os
import sys
import argparse
import torch
import yaml
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.dataset import PPIDataset, collate_fn
from data.properties import PhysicochemicalEncoder
from models.biomotif_ppi import create_model
from utils.metrics import MetricsCalculator, evaluate_model
from torch.utils.data import DataLoader


def evaluate_checkpoint(checkpoint_path: str, config_path: str, split: str = 'test'):
    """
    Evaluate a model checkpoint on specified dataset split.
    """
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("Loading model...")
    model = create_model(config)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Prepare data
    print(f"Preparing {split} dataset...")
    
    # Load sequences
    sequence_file = os.path.join(
        config['data']['cache_dir'],
        f"sequences_{config['data']['uniprot_release']}.pkl"
    )
    
    if not os.path.exists(sequence_file):
        print(f"ERROR: Sequence file not found at {sequence_file}")
        print("Please run prepare_data.py first.")
        return
    
    import pickle
    with open(sequence_file, 'rb') as f:
        sequence_dict = pickle.load(f)
    
    # Initialize property encoder
    prop_encoder = PhysicochemicalEncoder(config)
    
    # Determine split name
    split_map = {
        'train': config['data']['train_split'],
        'val': config['data']['val_split'],
        'test': config['data']['test_split']
    }
    
    # Create dataset
    dataset = PPIDataset(
        split=split_map[split],
        config=config,
        sequence_dict=sequence_dict,
        embedding_cache=None,  # Will load from cache if available
        transform=lambda x: add_properties(x, prop_encoder)
    )
    
    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=config['data']['test_batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        collate_fn=collate_fn
    )
    
    print(f"Evaluating on {len(dataset)} samples...")
    
    # Evaluate
    metrics_calc = MetricsCalculator()
    
    with torch.no_grad():
        for batch in dataloader:
            # Skip if no embeddings
            if 'embeddings_a' not in batch:
                print("WARNING: No embeddings found in batch. Skipping...")
                continue
                
            # Move to device
            embeddings_a = batch['embeddings_a'].to(device)
            embeddings_b = batch['embeddings_b'].to(device)
            properties_a = batch['properties_a'].to(device)
            properties_b = batch['properties_b'].to(device)
            labels = batch['labels'].to(device)
            
            mask_a = batch.get('mask_a', None)
            mask_b = batch.get('mask_b', None)
            if mask_a is not None:
                mask_a = mask_a.to(device)
            if mask_b is not None:
                mask_b = mask_b.to(device)
            
            # Forward pass
            outputs = model(
                embeddings_a, embeddings_b,
                properties_a, properties_b,
                mask_a, mask_b
            )
            
            probabilities = outputs['probabilities'].squeeze()
            predictions = (probabilities >= 0.5).float()
            
            # Update metrics
            metrics_calc.update(predictions, probabilities, labels)
    
    # Compute metrics
    metrics = metrics_calc.compute()
    print(f"\n{split.upper()} Set Results:")
    print(metrics)
    
    # Find optimal threshold
    print("\nFinding optimal threshold...")
    best_threshold, best_metrics = metrics_calc.find_optimal_threshold(metric='f1')
    print(f"Best threshold: {best_threshold:.3f}")
    print(f"Metrics at best threshold:")
    print(best_metrics)
    
    # Generate plots
    output_dir = Path(config['paths']['results_root']) / f"evaluation_{split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating plots in {output_dir}...")
    
    # ROC curve
    fig = metrics_calc.plot_roc_curve(output_dir / 'roc_curve.png')
    
    # PR curve
    fig = metrics_calc.plot_pr_curve(output_dir / 'pr_curve.png')
    
    # Confusion matrix
    fig = metrics_calc.plot_confusion_matrix(metrics, output_dir / 'confusion_matrix.png')
    
    # Threshold analysis
    fig = metrics_calc.plot_threshold_analysis(save_path=output_dir / 'threshold_analysis.png')
    
    # Save results
    results_file = output_dir / 'metrics.txt'
    with open(results_file, 'w') as f:
        f.write(f"Evaluation Results for {split} set\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Epoch: {checkpoint['epoch']}\n\n")
        f.write("Metrics at threshold 0.5:\n")
        f.write(str(metrics))
        f.write(f"\n\nOptimal threshold: {best_threshold:.3f}\n")
        f.write("Metrics at optimal threshold:\n")
        f.write(str(best_metrics))
    
    print(f"\nResults saved to {results_file}")


def add_properties(batch_item: dict, prop_encoder: PhysicochemicalEncoder) -> dict:
    """Add physicochemical properties to batch item."""
    batch_item['properties_a'] = prop_encoder.encode_sequence_with_ss(batch_item['sequence_a'])
    batch_item['properties_b'] = prop_encoder.encode_sequence_with_ss(batch_item['sequence_b'])
    return batch_item


def main():
    parser = argparse.ArgumentParser(description='Evaluate BioMotif-PPI model')
    parser.add_argument('checkpoint', type=str,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Path to configuration file')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    
    args = parser.parse_args()
    
    evaluate_checkpoint(args.checkpoint, args.config, args.split)


if __name__ == '__main__':
    main()