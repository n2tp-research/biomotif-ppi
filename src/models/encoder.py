import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class BiologicalFeatureEncoder(nn.Module):
    """
    Encodes protein sequences using ESM-2 embeddings and physicochemical properties.
    Combines features and processes through BiGRU layers.
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config
        
        # Extract dimensions from config
        self.esm_dim = config['esm']['embedding_dim']
        self.property_dim = 12  # Fixed number of physicochemical properties
        self.esm_proj_dim = config['model']['encoder']['esm_projection_dim']
        self.prop_proj_dim = config['model']['encoder']['property_projection_dim']
        self.combined_dim = config['model']['encoder']['combined_dim']
        
        # BiGRU parameters
        self.hidden_size = config['model']['bigru']['hidden_size']
        self.num_layers = config['model']['bigru']['num_layers']
        self.dropout = config['model']['bigru']['dropout']
        self.bidirectional = config['model']['bigru']['bidirectional']
        self.output_dim = config['model']['bigru']['output_dim']
        
        # Build layers
        self._build_layers()
        
    def _build_layers(self):
        """Build the encoder layers."""
        # ESM projection with LayerNorm
        self.esm_projection = nn.Sequential(
            nn.Linear(self.esm_dim, self.esm_proj_dim),
            nn.LayerNorm(self.esm_proj_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Property projection with BatchNorm
        self.property_projection = nn.Sequential(
            nn.Linear(self.property_dim, self.prop_proj_dim),
            nn.BatchNorm1d(self.prop_proj_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Combined projection
        actual_combined_dim = self.esm_proj_dim + self.prop_proj_dim
        if actual_combined_dim != self.combined_dim:
            # Add projection to match expected dimension
            self.combined_projection = nn.Linear(actual_combined_dim, self.combined_dim)
        else:
            self.combined_projection = nn.Identity()
        
        # BiGRU layers
        self.gru = nn.GRU(
            input_size=self.combined_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout if self.num_layers > 1 else 0,
            bidirectional=self.bidirectional
        )
        
        # Output projection
        gru_output_dim = self.hidden_size * 2 if self.bidirectional else self.hidden_size
        self.output_projection = nn.Sequential(
            nn.Linear(gru_output_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
    def forward(
        self,
        esm_embeddings: torch.Tensor,
        properties: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.
        
        Args:
            esm_embeddings: ESM-2 embeddings [batch, seq_len, esm_dim]
            properties: Physicochemical properties [batch, seq_len, 12]
            mask: Optional attention mask [batch, seq_len]
            
        Returns:
            Tuple of:
            - encoded_features: Encoded sequence features [batch, seq_len, output_dim]
            - final_hidden: Final hidden state [batch, output_dim]
        """
        batch_size, seq_len, _ = esm_embeddings.shape
        
        # Project ESM embeddings
        esm_proj = self.esm_projection(esm_embeddings)
        
        # Project properties (need to reshape for BatchNorm)
        props_flat = properties.reshape(-1, self.property_dim)
        props_proj = self.property_projection(props_flat)
        props_proj = props_proj.reshape(batch_size, seq_len, -1)
        
        # Combine features
        combined = torch.cat([esm_proj, props_proj], dim=-1)
        combined = self.combined_projection(combined)
        
        # Apply mask if provided
        if mask is not None:
            # Create length tensor for packing
            lengths = mask.sum(dim=1).cpu()
            
            # Pack sequences
            packed = nn.utils.rnn.pack_padded_sequence(
                combined,
                lengths,
                batch_first=True,
                enforce_sorted=False
            )
            
            # Process through GRU
            packed_output, hidden = self.gru(packed)
            
            # Unpack sequences
            output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=seq_len
            )
        else:
            # Process without packing
            output, hidden = self.gru(combined)
        
        # Project output
        encoded_features = self.output_projection(output)
        
        # Get final hidden state
        if self.bidirectional:
            # Concatenate forward and backward final states
            hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)
            # Take last layer's forward and backward states
            final_hidden = torch.cat([hidden[-1, 0], hidden[-1, 1]], dim=-1)
        else:
            # Take last layer's hidden state
            final_hidden = hidden[-1]
            
        # Project final hidden
        final_hidden = self.output_projection(final_hidden)
        
        return encoded_features, final_hidden


class ProteinPairEncoder(nn.Module):
    """
    Encodes a pair of proteins using the BiologicalFeatureEncoder.
    Handles both proteins independently and provides interaction features.
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config
        
        # Create shared encoder for both proteins
        self.encoder = BiologicalFeatureEncoder(config)
        
        # Optional separate encoders if needed
        self.share_encoder = config.get('model', {}).get('share_encoder', True)
        if not self.share_encoder:
            self.encoder_b = BiologicalFeatureEncoder(config)
        
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
        Encode protein pair.
        
        Args:
            esm_a, esm_b: ESM-2 embeddings for proteins A and B
            props_a, props_b: Physicochemical properties
            mask_a, mask_b: Optional attention masks
            
        Returns:
            Dictionary containing:
            - features_a: Encoded features for protein A [batch, seq_len_a, dim]
            - features_b: Encoded features for protein B [batch, seq_len_b, dim]
            - hidden_a: Final hidden state for protein A [batch, dim]
            - hidden_b: Final hidden state for protein B [batch, dim]
            - lengths_a: Sequence lengths for protein A
            - lengths_b: Sequence lengths for protein B
        """
        # Encode protein A
        features_a, hidden_a = self.encoder(esm_a, props_a, mask_a)
        
        # Encode protein B
        if self.share_encoder:
            features_b, hidden_b = self.encoder(esm_b, props_b, mask_b)
        else:
            features_b, hidden_b = self.encoder_b(esm_b, props_b, mask_b)
        
        # Calculate sequence lengths
        if mask_a is not None:
            lengths_a = mask_a.sum(dim=1)
        else:
            lengths_a = torch.full((features_a.shape[0],), features_a.shape[1], device=features_a.device)
            
        if mask_b is not None:
            lengths_b = mask_b.sum(dim=1)
        else:
            lengths_b = torch.full((features_b.shape[0],), features_b.shape[1], device=features_b.device)
        
        return {
            'features_a': features_a,
            'features_b': features_b,
            'hidden_a': hidden_a,
            'hidden_b': hidden_b,
            'lengths_a': lengths_a,
            'lengths_b': lengths_b
        }