import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass


@dataclass
class MetricsResult:
    """Container for evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    auc_pr: float
    mcc: float
    confusion_matrix: np.ndarray
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary (excluding confusion matrix)."""
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'auc_roc': self.auc_roc,
            'auc_pr': self.auc_pr,
            'mcc': self.mcc
        }
    
    def __str__(self) -> str:
        """String representation."""
        return (
            f"Accuracy: {self.accuracy:.4f}\n"
            f"Precision: {self.precision:.4f}\n"
            f"Recall: {self.recall:.4f}\n"
            f"F1-Score: {self.f1:.4f}\n"
            f"AUC-ROC: {self.auc_roc:.4f}\n"
            f"AUC-PR: {self.auc_pr:.4f}\n"
            f"MCC: {self.mcc:.4f}"
        )


class MetricsCalculator:
    """
    Calculates and tracks metrics for protein-protein interaction prediction.
    """
    
    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Classification threshold
        """
        self.threshold = threshold
        self.reset()
        
    def reset(self):
        """Reset accumulated predictions and labels."""
        self.all_predictions = []
        self.all_probabilities = []
        self.all_labels = []
        
    def update(
        self,
        predictions: torch.Tensor,
        probabilities: torch.Tensor,
        labels: torch.Tensor
    ):
        """
        Update with batch predictions.
        
        Args:
            predictions: Binary predictions [batch_size]
            probabilities: Prediction probabilities [batch_size]
            labels: True labels [batch_size]
        """
        self.all_predictions.extend(predictions.cpu().numpy())
        self.all_probabilities.extend(probabilities.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())
    
    def compute(self) -> MetricsResult:
        """Compute all metrics from accumulated data."""
        predictions = np.array(self.all_predictions)
        probabilities = np.array(self.all_probabilities)
        labels = np.array(self.all_labels)
        
        # Handle edge cases
        if len(np.unique(labels)) < 2:
            # Only one class present
            return MetricsResult(
                accuracy=accuracy_score(labels, predictions),
                precision=0.0,
                recall=0.0,
                f1=0.0,
                auc_roc=0.5,
                auc_pr=0.5,
                mcc=0.0,
                confusion_matrix=confusion_matrix(labels, predictions)
            )
        
        # Compute metrics
        metrics = MetricsResult(
            accuracy=accuracy_score(labels, predictions),
            precision=precision_score(labels, predictions, zero_division=0),
            recall=recall_score(labels, predictions, zero_division=0),
            f1=f1_score(labels, predictions, zero_division=0),
            auc_roc=roc_auc_score(labels, probabilities),
            auc_pr=average_precision_score(labels, probabilities),
            mcc=matthews_corrcoef(labels, predictions),
            confusion_matrix=confusion_matrix(labels, predictions)
        )
        
        return metrics
    
    def compute_at_threshold(self, threshold: float) -> MetricsResult:
        """Compute metrics at a specific threshold."""
        probabilities = np.array(self.all_probabilities)
        labels = np.array(self.all_labels)
        
        # Apply threshold
        predictions = (probabilities >= threshold).astype(int)
        
        # Compute metrics
        if len(np.unique(labels)) < 2:
            return MetricsResult(
                accuracy=accuracy_score(labels, predictions),
                precision=0.0,
                recall=0.0,
                f1=0.0,
                auc_roc=0.5,
                auc_pr=0.5,
                mcc=0.0,
                confusion_matrix=confusion_matrix(labels, predictions)
            )
        
        metrics = MetricsResult(
            accuracy=accuracy_score(labels, predictions),
            precision=precision_score(labels, predictions, zero_division=0),
            recall=recall_score(labels, predictions, zero_division=0),
            f1=f1_score(labels, predictions, zero_division=0),
            auc_roc=roc_auc_score(labels, probabilities),
            auc_pr=average_precision_score(labels, probabilities),
            mcc=matthews_corrcoef(labels, predictions),
            confusion_matrix=confusion_matrix(labels, predictions)
        )
        
        return metrics
    
    def find_optimal_threshold(
        self,
        metric: str = 'f1',
        thresholds: Optional[List[float]] = None
    ) -> Tuple[float, MetricsResult]:
        """
        Find optimal classification threshold.
        
        Args:
            metric: Metric to optimize ('f1', 'mcc', 'accuracy')
            thresholds: List of thresholds to try
            
        Returns:
            Tuple of (optimal_threshold, metrics_at_threshold)
        """
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 81)
            
        best_threshold = self.threshold
        best_value = -float('inf')
        best_metrics = None
        
        for threshold in thresholds:
            metrics = self.compute_at_threshold(threshold)
            
            if metric == 'f1':
                value = metrics.f1
            elif metric == 'mcc':
                value = metrics.mcc
            elif metric == 'accuracy':
                value = metrics.accuracy
            else:
                raise ValueError(f"Unknown metric: {metric}")
                
            if value > best_value:
                best_value = value
                best_threshold = threshold
                best_metrics = metrics
                
        return best_threshold, best_metrics
    
    def plot_roc_curve(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot ROC curve."""
        labels = np.array(self.all_labels)
        probabilities = np.array(self.all_probabilities)
        
        fpr, tpr, _ = roc_curve(labels, probabilities)
        auc = roc_auc_score(labels, probabilities)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic Curve')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_pr_curve(self, save_path: Optional[str] = None) -> plt.Figure:
        """Plot Precision-Recall curve."""
        labels = np.array(self.all_labels)
        probabilities = np.array(self.all_probabilities)
        
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        auc_pr = average_precision_score(labels, probabilities)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, linewidth=2, label=f'PR curve (AUC = {auc_pr:.3f})')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_confusion_matrix(
        self,
        metrics: MetricsResult,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot confusion matrix."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Normalize confusion matrix
        cm = metrics.confusion_matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(
            cm_normalized,
            annot=cm,
            fmt='d',
            cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'],
            ax=ax
        )
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_threshold_analysis(
        self,
        thresholds: Optional[List[float]] = None,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """Plot metrics vs threshold."""
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 81)
            
        metrics_dict = {
            'threshold': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'mcc': []
        }
        
        for threshold in thresholds:
            metrics = self.compute_at_threshold(threshold)
            metrics_dict['threshold'].append(threshold)
            metrics_dict['accuracy'].append(metrics.accuracy)
            metrics_dict['precision'].append(metrics.precision)
            metrics_dict['recall'].append(metrics.recall)
            metrics_dict['f1'].append(metrics.f1)
            metrics_dict['mcc'].append(metrics.mcc)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot accuracy, precision, recall, F1
        ax1.plot(metrics_dict['threshold'], metrics_dict['accuracy'], label='Accuracy', linewidth=2)
        ax1.plot(metrics_dict['threshold'], metrics_dict['precision'], label='Precision', linewidth=2)
        ax1.plot(metrics_dict['threshold'], metrics_dict['recall'], label='Recall', linewidth=2)
        ax1.plot(metrics_dict['threshold'], metrics_dict['f1'], label='F1-Score', linewidth=2)
        ax1.set_xlabel('Threshold')
        ax1.set_ylabel('Score')
        ax1.set_title('Classification Metrics vs Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot MCC
        ax2.plot(metrics_dict['threshold'], metrics_dict['mcc'], label='MCC', linewidth=2, color='red')
        ax2.set_xlabel('Threshold')
        ax2.set_ylabel('Matthews Correlation Coefficient')
        ax2.set_title('MCC vs Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    threshold: float = 0.5
) -> MetricsResult:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to run on
        threshold: Classification threshold
        
    Returns:
        MetricsResult object
    """
    model.eval()
    calculator = MetricsCalculator(threshold)
    
    with torch.no_grad():
        for batch in dataloader:
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
            predictions = (probabilities >= threshold).float()
            
            # Update calculator
            calculator.update(predictions, probabilities, labels)
    
    return calculator.compute()