import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import math

# Try to import flash_attn, fallback to standard attention if not available
try:
    from flash_attn import flash_attn_func, flash_attn_qkvpacked_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False
    print("Flash Attention not available, using standard attention")


class MotifDiscoveryModule(nn.Module):
    """
    Discovers interaction motifs using Flash Attention mechanism.
    Learns a bank of motifs and assigns protein positions to motifs.
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config
        
        # Extract parameters
        self.num_motifs = config['model']['motif_discovery']['num_motifs']
        self.motif_dim = config['model']['motif_discovery']['motif_dim']
        self.input_dim = config['model']['bigru']['output_dim']
        self.temperature_init = config['model']['motif_discovery']['temperature_init']
        self.temperature_range = config['model']['motif_discovery']['temperature_range']
        self.dropout = config['model']['motif_discovery']['dropout']
        self.use_flash = config['model']['motif_discovery']['use_flash_attention'] and FLASH_AVAILABLE
        
        # Initialize motif bank
        self._init_motif_bank()
        
        # Build layers
        self._build_layers()
        
    def _init_motif_bank(self):
        """Initialize the learnable motif bank using Xavier uniform."""
        # Calculate Xavier uniform bounds
        fan_in = self.num_motifs
        fan_out = self.motif_dim
        bound = math.sqrt(6.0 / (fan_in + fan_out))
        
        # Initialize motif bank
        self.motif_bank = nn.Parameter(
            torch.empty(self.num_motifs, self.motif_dim).uniform_(-bound, bound)
        )
        
    def _build_layers(self):
        """Build the module layers."""
        # Query projection for input features
        self.query_projection = nn.Linear(
            self.input_dim,
            self.motif_dim
        )
        
        # Temperature parameter (learnable)
        self.temperature = nn.Parameter(
            torch.tensor(self.temperature_init)
        )
        
        # Output projections
        self.motif_output = nn.Sequential(
            nn.Linear(self.motif_dim, self.input_dim),
            nn.LayerNorm(self.input_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Attention dropout
        self.attn_dropout = nn.Dropout(self.dropout)
        
    def _standard_attention(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Standard scaled dot-product attention (fallback when Flash not available).
        
        Args:
            queries: [batch, seq_len, dim]
            keys: [batch, num_motifs, dim]
            values: [batch, num_motifs, dim]
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, dim = queries.shape
        
        # Calculate attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1))  # [batch, seq_len, num_motifs]
        scores = scores / math.sqrt(dim)
        
        # Apply temperature scaling
        temperature = torch.clamp(self.temperature, self.temperature_range[0], self.temperature_range[1])
        scores = scores / temperature
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask for motifs dimension
            mask_expanded = mask.unsqueeze(-1)  # [batch, seq_len, 1]
            scores = scores.masked_fill(~mask_expanded, float('-inf'))
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, values)
        
        return output, attention_weights
    
    def _flash_attention(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Flash Attention implementation for efficient computation.
        
        Args:
            queries: [batch, seq_len, dim]
            keys: [batch, num_motifs, dim]
            values: [batch, num_motifs, dim]
            mask: Optional attention mask
            
        Returns:
            Attention output [batch, seq_len, dim]
        """
        batch_size, seq_len, dim = queries.shape
        
        # Ensure all tensors have the same dtype (use float16 for Flash Attention)
        dtype = torch.float16 if queries.device.type == 'cuda' else queries.dtype
        queries = queries.to(dtype)
        keys = keys.to(dtype)
        values = values.to(dtype)
        
        # Apply temperature scaling
        temperature = torch.clamp(self.temperature, self.temperature_range[0], self.temperature_range[1])
        scale = 1.0 / (math.sqrt(dim) * temperature)
        
        # Flash attention expects queries as [batch, seq_len, num_heads, head_dim]
        # We treat the entire dimension as a single head for motif discovery
        queries = queries.unsqueeze(2)  # [batch, seq_len, 1, dim]
        keys = keys.unsqueeze(2)  # [batch, num_motifs, 1, dim]
        values = values.unsqueeze(2)  # [batch, num_motifs, 1, dim]
        
        # Use flash attention
        output = flash_attn_func(
            queries,
            keys,
            values,
            dropout_p=self.dropout if self.training else 0.0,
            softmax_scale=scale,
            causal=False,
            window_size=(-1, -1),  # Full attention
            alibi_slopes=None,
            deterministic=False
        )
        
        # Remove head dimension
        output = output.squeeze(2)  # [batch, seq_len, dim]
        
        return output
    
    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Discover and assign motifs to sequence positions.
        
        Args:
            features: Encoded features [batch, seq_len, input_dim]
            mask: Optional attention mask [batch, seq_len]
            return_attention: Whether to return attention weights (only with standard attention)
            
        Returns:
            Dictionary containing:
            - motif_features: Features after motif assignment [batch, seq_len, input_dim]
            - attention_weights: Attention weights if requested [batch, seq_len, num_motifs]
            - motif_usage: Average attention per motif [batch, num_motifs]
        """
        batch_size, seq_len, _ = features.shape
        
        # Project features to query space
        queries = self.query_projection(features)  # [batch, seq_len, motif_dim]
        
        # Expand motif bank for batch and ensure same dtype as queries
        keys = self.motif_bank.unsqueeze(0).expand(batch_size, -1, -1).to(queries.dtype)  # [batch, num_motifs, motif_dim]
        values = keys  # Same as keys for motif discovery
        
        # Apply attention mechanism
        if self.use_flash and not return_attention:
            # Use Flash Attention (no attention weights returned)
            motif_assignments = self._flash_attention(queries, keys, values, mask)
            attention_weights = None
        else:
            # Use standard attention
            motif_assignments, attention_weights = self._standard_attention(queries, keys, values, mask)
        
        # Project back to input dimension
        motif_features = self.motif_output(motif_assignments)
        
        # Calculate motif usage statistics
        if attention_weights is not None:
            if mask is not None:
                # Mask out padded positions
                attention_weights = attention_weights * mask.unsqueeze(-1)
                valid_positions = mask.sum(dim=1, keepdim=True).clamp(min=1)
                motif_usage = attention_weights.sum(dim=1) / valid_positions
            else:
                motif_usage = attention_weights.mean(dim=1)
        else:
            # Approximate motif usage by recomputing attention scores
            with torch.no_grad():
                scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.motif_dim)
                if mask is not None:
                    scores = scores.masked_fill(~mask.unsqueeze(-1), float('-inf'))
                approx_weights = F.softmax(scores, dim=-1)
                motif_usage = approx_weights.mean(dim=1)
        
        # Residual connection
        output_features = features + motif_features
        
        results = {
            'motif_features': output_features,
            'motif_usage': motif_usage
        }
        
        if return_attention and attention_weights is not None:
            results['attention_weights'] = attention_weights
            
        return results
    
    def get_top_motifs(self, attention_weights: torch.Tensor, k: int = 5) -> torch.Tensor:
        """
        Get top-k motifs for each position.
        
        Args:
            attention_weights: Attention weights [batch, seq_len, num_motifs]
            k: Number of top motifs to return
            
        Returns:
            Indices of top motifs [batch, seq_len, k]
        """
        top_values, top_indices = torch.topk(attention_weights, k, dim=-1)
        return top_indices
    
    def get_motif_embeddings(self) -> torch.Tensor:
        """Return the current motif bank embeddings."""
        return self.motif_bank.detach()


class MotifInteractionModule(nn.Module):
    """
    Analyzes interactions between motifs discovered in two proteins.
    Computes pairwise motif compatibility scores.
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config
        
        # Parameters
        self.input_dim = config['model']['bigru']['output_dim']
        self.hidden_dim = 256
        
        # Interaction scoring network
        self.interaction_mlp = nn.Sequential(
            nn.Linear(self.input_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
    def forward(
        self,
        motif_usage_a: torch.Tensor,
        motif_usage_b: torch.Tensor,
        hidden_a: torch.Tensor,
        hidden_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute interaction score based on motif usage patterns.
        
        Args:
            motif_usage_a: Motif usage for protein A [batch, num_motifs]
            motif_usage_b: Motif usage for protein B [batch, num_motifs]
            hidden_a: Hidden representation for protein A [batch, hidden_dim]
            hidden_b: Hidden representation for protein B [batch, hidden_dim]
            
        Returns:
            Interaction scores [batch, 1]
        """
        # Combine hidden representations
        combined_hidden = torch.cat([hidden_a, hidden_b], dim=-1)
        
        # Compute interaction score
        interaction_score = self.interaction_mlp(combined_hidden)
        
        return interaction_score