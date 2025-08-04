import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from .encoder import ProteinPairEncoder
from .motif_discovery import MotifDiscoveryModule, MotifInteractionModule
from .complementarity import ComplementarityAnalyzer
from .gnn import AllostericGNN


class BioMotifPPI(nn.Module):
    """
    BioMotif-PPI: Complete model for protein-protein interaction prediction.
    Combines motif discovery, complementarity analysis, and allosteric GNN.
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config
        
        # Initialize components
        self.encoder = ProteinPairEncoder(config)
        self.motif_discovery = MotifDiscoveryModule(config)
        self.motif_interaction = MotifInteractionModule(config)
        self.complementarity = ComplementarityAnalyzer(config)
        self.allosteric_gnn = AllostericGNN(config)
        
        # Component MLPs
        pred_config = config['model']['prediction']
        
        # Direct MLP with projection layer
        hidden_dim = config['model']['bigru']['output_dim']
        combined_hidden_dim = hidden_dim * 2  # concatenated hidden states
        self.direct_projection = nn.Linear(combined_hidden_dim, pred_config['direct_mlp_dims'][0])
        self.direct_mlp = self._build_mlp(pred_config['direct_mlp_dims'])
        
        self.motif_mlp = self._build_mlp(pred_config['motif_mlp_dims'])
        self.allosteric_mlp = self._build_mlp(pred_config['allosteric_mlp_dims'])
        
        # Ensemble weights
        init_weights = pred_config['init_weights']
        self.ensemble_weights = nn.Parameter(torch.tensor(init_weights))
        
        # Gradient checkpointing settings
        self.use_checkpointing = config['training'].get('gradient_checkpointing', False)
        self.checkpoint_layers = config['training'].get('checkpoint_layers', [])
        
    def _build_mlp(self, dims: List[int]) -> nn.Module:
        """Build MLP from dimension list."""
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation after last layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
        return nn.Sequential(*layers)
    
    def forward(
        self,
        esm_a: torch.Tensor,
        esm_b: torch.Tensor,
        props_a: torch.Tensor,
        props_b: torch.Tensor,
        mask_a: Optional[torch.Tensor] = None,
        mask_b: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through BioMotif-PPI.
        
        Args:
            esm_a, esm_b: ESM-2 embeddings [batch, seq_len, 1280]
            props_a, props_b: Physicochemical properties [batch, seq_len, 12]
            mask_a, mask_b: Optional attention masks [batch, seq_len]
            
        Returns:
            Dictionary containing:
            - logits: Raw prediction scores [batch, 1]
            - probabilities: Interaction probabilities [batch, 1]
            - component_scores: Individual component scores
            - attention_info: Attention weights and motif usage
        """
        # Step 1: Encode protein sequences
        encoded = self.encoder(esm_a, esm_b, props_a, props_b, mask_a, mask_b)
        features_a = encoded['features_a']
        features_b = encoded['features_b']
        hidden_a = encoded['hidden_a']
        hidden_b = encoded['hidden_b']
        
        # Step 2: Motif discovery
        motif_a = self.motif_discovery(features_a, mask_a)
        motif_b = self.motif_discovery(features_b, mask_b)
        
        # Step 3: Direct interaction score (baseline)
        combined_hidden = torch.cat([hidden_a, hidden_b], dim=-1)
        # Use the pre-initialized projection layer
        combined_hidden_proj = self.direct_projection(combined_hidden)
        direct_score = self.direct_mlp(combined_hidden_proj)
        
        # Step 4: Motif-based interaction score
        motif_score = self.motif_interaction(
            motif_a['motif_usage'],
            motif_b['motif_usage'],
            hidden_a,
            hidden_b
        )
        
        # Step 5: Complementarity analysis
        if self.use_checkpointing and 0 in self.checkpoint_layers:
            comp_result = torch.utils.checkpoint.checkpoint(
                self.complementarity.chunked_analysis,
                features_a, features_b, props_a, props_b, mask_a, mask_b
            )
        else:
            comp_result = self.complementarity.chunked_analysis(
                features_a, features_b, props_a, props_b, mask_a, mask_b
            )
        
        # Step 6: Allosteric GNN analysis
        # Concatenate both proteins for joint graph analysis
        if mask_a is not None and mask_b is not None:
            # Combine features
            max_len = max(features_a.shape[1], features_b.shape[1])
            batch_size = features_a.shape[0]
            
            # Pad to same length
            if features_a.shape[1] < max_len:
                pad_len = max_len - features_a.shape[1]
                features_a = F.pad(features_a, (0, 0, 0, pad_len))
                props_a = F.pad(props_a, (0, 0, 0, pad_len))
                mask_a = F.pad(mask_a, (0, pad_len), value=False)
            
            if features_b.shape[1] < max_len:
                pad_len = max_len - features_b.shape[1]
                features_b = F.pad(features_b, (0, 0, 0, pad_len))
                props_b = F.pad(props_b, (0, 0, 0, pad_len))
                mask_b = F.pad(mask_b, (0, pad_len), value=False)
            
            # Stack proteins
            combined_features = torch.cat([features_a, features_b], dim=1)
            combined_props = torch.cat([props_a, props_b], dim=1)
            combined_mask = torch.cat([mask_a, mask_b], dim=1)
        else:
            combined_features = torch.cat([features_a, features_b], dim=1)
            combined_props = torch.cat([props_a, props_b], dim=1)
            combined_mask = None
        
        # Apply GNN
        if self.use_checkpointing and 1 in self.checkpoint_layers:
            gnn_result = torch.utils.checkpoint.checkpoint(
                self.allosteric_gnn,
                combined_features, combined_props, combined_mask
            )
        else:
            gnn_result = self.allosteric_gnn(
                combined_features, combined_props, combined_mask
            )
        
        # Step 7: Allosteric score
        allosteric_features = torch.cat([
            gnn_result['global_features'],
            gnn_result['avg_degree'].unsqueeze(-1),
            comp_result['complementarity_features']
        ], dim=-1)
        
        # Pad to expected dimension
        expected_dim = self.config['model']['prediction']['allosteric_mlp_dims'][0]
        if allosteric_features.shape[-1] < expected_dim:
            pad_size = expected_dim - allosteric_features.shape[-1]
            allosteric_features = F.pad(allosteric_features, (0, pad_size))
        elif allosteric_features.shape[-1] > expected_dim:
            allosteric_features = allosteric_features[:, :expected_dim]
            
        allosteric_score = self.allosteric_mlp(allosteric_features)
        
        # Step 8: Ensemble prediction
        # Normalize ensemble weights
        weights = F.softmax(self.ensemble_weights, dim=0)
        
        # Combine scores
        final_logits = (
            weights[0] * direct_score +
            weights[1] * motif_score +
            weights[2] * allosteric_score
        )
        
        # Compute probabilities
        probabilities = torch.sigmoid(final_logits)
        
        # Prepare output
        return {
            'logits': final_logits,
            'probabilities': probabilities,
            'component_scores': {
                'direct': direct_score,
                'motif': motif_score,
                'allosteric': allosteric_score
            },
            'ensemble_weights': weights,
            'attention_info': {
                'motif_usage_a': motif_a['motif_usage'],
                'motif_usage_b': motif_b['motif_usage']
            },
            'graph_stats': {
                'num_nodes': gnn_result['num_nodes'],
                'num_edges': gnn_result['num_edges'],
                'avg_degree': gnn_result['avg_degree']
            }
        }
    
    def get_motif_bank(self) -> torch.Tensor:
        """Get the learned motif bank."""
        return self.motif_discovery.get_motif_embeddings()
    
    def predict(
        self,
        esm_a: torch.Tensor,
        esm_b: torch.Tensor,
        props_a: torch.Tensor,
        props_b: torch.Tensor,
        mask_a: Optional[torch.Tensor] = None,
        mask_b: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """
        Make predictions with the model.
        
        Args:
            All inputs same as forward()
            threshold: Classification threshold
            
        Returns:
            Dictionary with predictions and scores
        """
        with torch.no_grad():
            output = self.forward(esm_a, esm_b, props_a, props_b, mask_a, mask_b)
            
        predictions = (output['probabilities'] > threshold).float()
        
        return {
            'predictions': predictions,
            'probabilities': output['probabilities'],
            'logits': output['logits'],
            'component_scores': output['component_scores']
        }


def create_model(config: Dict) -> BioMotifPPI:
    """Factory function to create BioMotif-PPI model."""
    return BioMotifPPI(config)