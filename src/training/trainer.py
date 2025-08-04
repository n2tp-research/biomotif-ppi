import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler
from torch import autocast
from torch.utils.data import DataLoader
from typing import Dict, Optional, Tuple, Any
import numpy as np
from tqdm import tqdm
import os
import json
from datetime import datetime
import wandb
from torch.utils.tensorboard import SummaryWriter


class BioMotifTrainer:
    """
    Trainer class for BioMotif-PPI model.
    Implements mixed precision training, gradient checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda'
    ):
        """
        Args:
            model: BioMotif-PPI model
            config: Configuration dictionary
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Training configuration
        self.train_config = config['training']
        self.max_epochs = self.train_config['max_epochs']
        self.gradient_clip = self.train_config['gradient_clip_norm']
        self.early_stopping_patience = self.train_config['early_stopping_patience']
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision training
        self.use_amp = self.train_config['use_amp']
        if self.use_amp:
            self.scaler = GradScaler(
                init_scale=self.train_config['amp_init_scale'],
                growth_interval=self.train_config.get('amp_growth_interval', 2000)
            )
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([self.train_config['pos_weight']]).to(device)
        )
        
        # Logging
        self._setup_logging()
        
        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = -float('inf')
        self.patience_counter = 0
        
        # Memory management
        self.clear_cache_interval = config['hardware']['empty_cache_interval']
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer from config."""
        opt_config = self.train_config['optimizer']
        
        if opt_config['type'] == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_config['lr'],
                betas=opt_config['betas'],
                weight_decay=opt_config['weight_decay'],
                eps=opt_config['eps'],
                foreach=opt_config.get('foreach', True)
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_config['type']}")
            
        return optimizer
    
    def _create_scheduler(self) -> Any:
        """Create learning rate scheduler from config."""
        sched_config = self.train_config['scheduler']
        
        if sched_config['type'] == 'CosineAnnealingWarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=sched_config['T_0'],
                T_mult=sched_config['T_mult'],
                eta_min=sched_config['eta_min']
            )
        else:
            raise ValueError(f"Unsupported scheduler: {sched_config['type']}")
            
        return scheduler
    
    def _setup_logging(self):
        """Setup logging with WandB and TensorBoard."""
        log_config = self.config['logging']
        
        # Create log directories
        os.makedirs(log_config['checkpoint_dir'], exist_ok=True)
        os.makedirs(log_config['tensorboard_dir'], exist_ok=True)
        
        # Initialize TensorBoard
        if log_config['use_tensorboard']:
            self.tb_writer = SummaryWriter(log_config['tensorboard_dir'])
        else:
            self.tb_writer = None
            
        # Initialize WandB
        if log_config['use_wandb']:
            wandb.init(
                project=log_config['wandb_project'],
                entity=log_config.get('wandb_entity', None),
                config=self.config,
                name=self.config['experiment']['name'],
                tags=self.config['experiment']['tags']
            )
            wandb.watch(self.model, log='all', log_freq=100)
        
        self.log_interval = log_config['log_every_n_steps']
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'num_samples': 0
        }
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._batch_to_device(batch)
            
            # Forward pass with mixed precision
            with autocast(device_type='cuda' if self.device == 'cuda' else 'cpu', enabled=self.use_amp):
                outputs = self.model(
                    batch['embeddings_a'],
                    batch['embeddings_b'],
                    batch['properties_a'],
                    batch['properties_b'],
                    batch.get('mask_a', None),
                    batch.get('mask_b', None)
                )
                
                loss = self.criterion(outputs['logits'].squeeze(), batch['labels'])
                
                # Add regularization losses if any
                reg_loss = self._compute_regularization_loss()
                total_loss = loss + reg_loss
            
            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.use_amp:
                self.scaler.scale(total_loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip
                )
                self.optimizer.step()
            
            # Update metrics
            with torch.no_grad():
                predictions = (outputs['probabilities'] > 0.5).float().squeeze()
                accuracy = (predictions == batch['labels']).float().mean()
                
                epoch_metrics['loss'] += loss.item() * batch['labels'].size(0)
                epoch_metrics['accuracy'] += accuracy.item() * batch['labels'].size(0)
                epoch_metrics['num_samples'] += batch['labels'].size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': accuracy.item(),
                'grad_norm': grad_norm.item()
            })
            
            # Logging
            if self.global_step % self.log_interval == 0:
                self._log_training_step(
                    loss.item(),
                    accuracy.item(),
                    grad_norm.item(),
                    outputs
                )
            
            # Memory management
            if batch_idx % self.clear_cache_interval == 0:
                torch.cuda.empty_cache()
            
            self.global_step += 1
        
        # Compute epoch averages
        epoch_metrics['loss'] /= epoch_metrics['num_samples']
        epoch_metrics['accuracy'] /= epoch_metrics['num_samples']
        
        # Step scheduler
        self.scheduler.step()
        
        return epoch_metrics
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc_roc': 0.0,
            'num_samples': 0
        }
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = self._batch_to_device(batch)
                
                with autocast(device_type='cuda' if self.device == 'cuda' else 'cpu', enabled=self.use_amp):
                    outputs = self.model(
                        batch['embeddings_a'],
                        batch['embeddings_b'],
                        batch['properties_a'],
                        batch['properties_b'],
                        batch.get('mask_a', None),
                        batch.get('mask_b', None)
                    )
                    
                    loss = self.criterion(outputs['logits'].squeeze(), batch['labels'])
                
                # Collect predictions
                probabilities = outputs['probabilities'].squeeze()
                predictions = (probabilities > 0.5).float()
                
                all_predictions.append(predictions.cpu())
                all_labels.append(batch['labels'].cpu())
                all_probabilities.append(probabilities.cpu())
                
                # Update metrics
                val_metrics['loss'] += loss.item() * batch['labels'].size(0)
                val_metrics['num_samples'] += batch['labels'].size(0)
        
        # Compute metrics
        all_predictions = torch.cat(all_predictions)
        all_labels = torch.cat(all_labels)
        all_probabilities = torch.cat(all_probabilities)
        
        val_metrics['loss'] /= val_metrics['num_samples']
        val_metrics.update(self._compute_metrics(all_predictions, all_labels, all_probabilities))
        
        return val_metrics
    
    def train(self):
        """Main training loop."""
        print(f"Starting training for {self.max_epochs} epochs")
        
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Log epoch metrics
            self._log_epoch_metrics(train_metrics, val_metrics)
            
            # Check for improvement
            monitor_metric = self.config['validation']['monitor_metric']
            current_metric = val_metrics[monitor_metric.replace('val_', '')]
            
            if current_metric > self.best_val_metric:
                self.best_val_metric = current_metric
                self.patience_counter = 0
                self._save_checkpoint(is_best=True)
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
                
            # Regular checkpoint
            if (epoch + 1) % self.config['logging']['checkpoint_frequency'] == 0:
                self._save_checkpoint(is_best=False)
        
        print("Training completed!")
        self._cleanup()
    
    def _batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch
    
    def _compute_regularization_loss(self) -> torch.Tensor:
        """Compute additional regularization losses."""
        reg_loss = 0.0
        
        # L2 regularization is handled by optimizer weight_decay
        # Add any custom regularization here
        
        return reg_loss
    
    def _compute_metrics(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        probabilities: torch.Tensor
    ) -> Dict[str, float]:
        """Compute classification metrics."""
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score,
            f1_score, roc_auc_score, matthews_corrcoef
        )
        
        predictions = predictions.numpy()
        labels = labels.numpy()
        probabilities = probabilities.numpy()
        
        metrics = {
            'accuracy': accuracy_score(labels, predictions),
            'precision': precision_score(labels, predictions, zero_division=0),
            'recall': recall_score(labels, predictions, zero_division=0),
            'f1': f1_score(labels, predictions, zero_division=0),
            'mcc': matthews_corrcoef(labels, predictions)
        }
        
        # AUC-ROC requires positive class probabilities
        if len(np.unique(labels)) > 1:
            metrics['auc_roc'] = roc_auc_score(labels, probabilities)
        else:
            metrics['auc_roc'] = 0.5
            
        return metrics
    
    def _log_training_step(
        self,
        loss: float,
        accuracy: float,
        grad_norm: float,
        outputs: Dict
    ):
        """Log training step metrics."""
        # WandB logging
        if self.config['logging']['use_wandb']:
            wandb.log({
                'train/loss': loss,
                'train/accuracy': accuracy,
                'train/grad_norm': grad_norm,
                'train/lr': self.optimizer.param_groups[0]['lr'],
                'train/direct_score': outputs['component_scores']['direct'].mean().item(),
                'train/motif_score': outputs['component_scores']['motif'].mean().item(),
                'train/allosteric_score': outputs['component_scores']['allosteric'].mean().item(),
                'step': self.global_step
            })
        
        # TensorBoard logging
        if self.tb_writer:
            self.tb_writer.add_scalar('train/loss', loss, self.global_step)
            self.tb_writer.add_scalar('train/accuracy', accuracy, self.global_step)
            self.tb_writer.add_scalar('train/grad_norm', grad_norm, self.global_step)
            self.tb_writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
    
    def _log_epoch_metrics(self, train_metrics: Dict, val_metrics: Dict):
        """Log epoch-level metrics."""
        print(f"\nEpoch {self.current_epoch} Summary:")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc_roc']:.4f}")
        
        # WandB logging
        if self.config['logging']['use_wandb']:
            log_dict = {
                'epoch': self.current_epoch,
                'train/epoch_loss': train_metrics['loss'],
                'train/epoch_accuracy': train_metrics['accuracy'],
            }
            for key, value in val_metrics.items():
                if key != 'num_samples':
                    log_dict[f'val/{key}'] = value
            wandb.log(log_dict)
        
        # TensorBoard logging
        if self.tb_writer:
            self.tb_writer.add_scalars('loss', {
                'train': train_metrics['loss'],
                'val': val_metrics['loss']
            }, self.current_epoch)
            
            self.tb_writer.add_scalars('accuracy', {
                'train': train_metrics['accuracy'],
                'val': val_metrics['accuracy']
            }, self.current_epoch)
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_metric': self.best_val_metric,
            'config': self.config
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save checkpoint
        checkpoint_dir = self.config['logging']['checkpoint_dir']
        
        if is_best:
            path = os.path.join(checkpoint_dir, 'best_model.pt')
        else:
            path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{self.current_epoch}.pt')
            
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_metric = checkpoint['best_val_metric']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def _cleanup(self):
        """Cleanup after training."""
        if self.tb_writer:
            self.tb_writer.close()
            
        if self.config['logging']['use_wandb']:
            wandb.finish()