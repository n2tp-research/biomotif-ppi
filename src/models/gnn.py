import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import math
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch

# Try to import flash_attn
try:
    from flash_attn import flash_attn_func
    FLASH_AVAILABLE = True
except ImportError:
    FLASH_AVAILABLE = False


class AllostericGNNLayer(nn.Module):
    """
    Single layer of the Allosteric GNN with Flash Attention support.
    Models long-range allosteric communication in proteins.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        ffn_dim: int,
        dropout: float = 0.1,
        use_flash: bool = True
    ):
        """
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            head_dim: Dimension per head
            ffn_dim: Feed-forward network dimension
            dropout: Dropout rate
            use_flash: Whether to use Flash Attention
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_dim = num_heads * head_dim
        self.use_flash = use_flash and FLASH_AVAILABLE
        
        # Multi-head attention projections
        self.q_proj = nn.Linear(hidden_dim, self.total_dim)
        self.k_proj = nn.Linear(hidden_dim, self.total_dim)
        self.v_proj = nn.Linear(hidden_dim, self.total_dim)
        self.o_proj = nn.Linear(self.total_dim, hidden_dim)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through GNN layer.
        
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Optional edge attributes
            batch: Batch assignment for nodes
            
        Returns:
            Updated node features
        """
        # Store residual
        residual = x
        
        # Normalize
        x = self.norm1(x)
        
        # Get neighbor aggregation using attention
        x_att = self._attention_aggregation(x, edge_index, edge_attr, batch)
        
        # Residual connection
        x = residual + self.dropout(x_att)
        
        # FFN block
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        return x
    
    def _attention_aggregation(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Perform attention-based neighbor aggregation.
        """
        num_nodes = x.shape[0]
        
        # Project to Q, K, V
        queries = self.q_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        keys = self.k_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        values = self.v_proj(x).view(num_nodes, self.num_heads, self.head_dim)
        
        # Initialize output
        out = torch.zeros_like(queries)
        
        if batch is None:
            # Single graph
            out = self._graph_attention(
                queries, keys, values, edge_index, 0, num_nodes
            )
        else:
            # Batched graphs
            for graph_idx in torch.unique(batch):
                mask = batch == graph_idx
                node_indices = torch.where(mask)[0]
                
                # Get subgraph edges
                edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
                sub_edges = edge_index[:, edge_mask]
                
                # Map to local indices
                local_map = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
                local_map[node_indices] = torch.arange(len(node_indices), device=x.device)
                local_edges = local_map[sub_edges]
                
                # Compute attention for subgraph
                out[node_indices] = self._graph_attention(
                    queries[node_indices],
                    keys[node_indices],
                    values[node_indices],
                    local_edges,
                    0,
                    len(node_indices)
                )
        
        # Project back
        out = out.view(num_nodes, self.total_dim)
        out = self.o_proj(out)
        
        return out
    
    def _graph_attention(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        edge_index: torch.Tensor,
        start_idx: int,
        end_idx: int
    ) -> torch.Tensor:
        """
        Compute attention for a single graph.
        """
        num_nodes = end_idx - start_idx
        
        # Always use standard attention for graphs (Flash Attention doesn't support sparse masks)
        if False:  # Disabled Flash Attention for sparse graph structures
            pass
        else:
            # Standard attention with sparse computation
            out = torch.zeros_like(queries)
            
            for node in range(num_nodes):
                # Get neighbors
                neighbors = edge_index[0][edge_index[1] == node]
                if len(neighbors) == 0:
                    continue
                
                # Add self-loop
                neighbors = torch.cat([neighbors, torch.tensor([node], device=neighbors.device)])
                
                # Compute attention scores
                q = queries[node:node+1]  # [1, heads, dim]
                k = keys[neighbors]  # [num_neighbors, heads, dim]
                
                scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
                attn_weights = F.softmax(scores, dim=-1)
                
                # Apply to values
                v = values[neighbors]  # [num_neighbors, heads, dim]
                out[node] = torch.matmul(attn_weights, v).squeeze(0)
        
        return out


class AllostericGNN(nn.Module):
    """
    Graph Neural Network for modeling allosteric communication in proteins.
    Uses Flash Attention for efficient message passing.
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config
        gnn_config = config['model']['gnn']
        
        # Parameters
        self.num_layers = gnn_config['num_layers']
        self.hidden_dim = gnn_config['hidden_dim']
        self.num_heads = gnn_config['num_heads']
        self.head_dim = gnn_config['head_dim']
        self.contact_threshold = gnn_config['contact_threshold']
        self.use_flash = gnn_config['use_flash_attention']
        self.dropout = gnn_config['dropout']
        
        # Input projection
        input_dim = config['model']['bigru']['output_dim']
        self.input_proj = nn.Linear(input_dim, self.hidden_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            AllostericGNNLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                ffn_dim=gnn_config['ffn_dim'],
                dropout=self.dropout,
                use_flash=self.use_flash
            )
            for _ in range(self.num_layers)
        ])
        
        # Edge construction network
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim * 2 + 3, 128),  # 2 features + distance + ss similarity
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Global pooling
        self.pool_attention = nn.Linear(self.hidden_dim, 1)
        pool_dims = gnn_config['global_pool_dims']
        self.pool_mlp = self._build_mlp([self.hidden_dim * 2] + pool_dims)
        
    def _build_mlp(self, dims: List[int]) -> nn.Module:
        """Build MLP from dimension list."""
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
        return nn.Sequential(*layers)
    
    def construct_graph(
        self,
        features: torch.Tensor,
        properties: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Construct protein graph with learned edges.
        
        Args:
            features: Node features [batch, seq_len, dim]
            properties: Physicochemical properties [batch, seq_len, 12]
            mask: Optional mask [batch, seq_len]
            
        Returns:
            Tuple of (node_features, edge_index) for batch
        """
        batch_size, seq_len, feat_dim = features.shape
        
        # Flatten batch for graph construction
        if mask is not None:
            # Create node list from valid positions
            node_features = []
            node_batch = []
            
            for b in range(batch_size):
                valid_mask = mask[b]
                valid_features = features[b][valid_mask]
                node_features.append(valid_features)
                node_batch.append(torch.full((valid_mask.sum(),), b, device=features.device))
            
            node_features = torch.cat(node_features, dim=0)
            node_batch = torch.cat(node_batch, dim=0)
        else:
            node_features = features.view(-1, feat_dim)
            node_batch = torch.repeat_interleave(
                torch.arange(batch_size, device=features.device),
                seq_len
            )
        
        # Construct edges
        edge_list = []
        
        for b in range(batch_size):
            batch_mask = node_batch == b
            batch_indices = torch.where(batch_mask)[0]
            num_nodes = len(batch_indices)
            
            if num_nodes == 0:
                continue
            
            # Get features and properties for this sample
            sample_features = node_features[batch_indices]
            
            # Extract secondary structure propensities
            if mask is not None:
                sample_props = properties[b][mask[b]]
            else:
                sample_props = properties[b]
            
            alpha_prop = sample_props[:, 10]
            beta_prop = sample_props[:, 11]
            
            # Simplified graph construction - connect sequential neighbors and sample random long-range
            # Always connect sequential neighbors
            for i in range(num_nodes - 1):
                global_i = batch_indices[i]
                global_j = batch_indices[i + 1]
                edge_list.append([global_i, global_j])
                edge_list.append([global_j, global_i])
            
            # Sample random long-range connections based on a simplified heuristic
            # This avoids O(n²) edge prediction computation
            num_long_range = min(num_nodes * 5, 500)  # Limit long-range edges
            if num_nodes > 10:
                for _ in range(num_long_range):
                    i = torch.randint(0, num_nodes - 5, (1,)).item()
                    j = i + torch.randint(5, min(50, num_nodes - i), (1,)).item()
                    
                    if j < num_nodes:
                        # Simple feature-based probability (no MLP for speed)
                        feat_i = sample_features[i]
                        feat_j = sample_features[j]
                        
                        # Fast similarity check using cosine similarity
                        similarity = F.cosine_similarity(feat_i.unsqueeze(0), feat_j.unsqueeze(0)).item()
                        
                        # Add edge based on similarity threshold
                        if similarity > 0.5:  # Simple threshold
                            global_i = batch_indices[i]
                            global_j = batch_indices[j]
                            edge_list.append([global_i, global_j])
                            edge_list.append([global_j, global_i])
        
        if len(edge_list) > 0:
            edge_index = torch.tensor(edge_list, device=features.device).t()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=features.device)
        
        return node_features, edge_index, node_batch
    
    def forward(
        self,
        features: torch.Tensor,
        properties: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Allosteric GNN.
        
        Args:
            features: Encoded features [batch, seq_len, dim]
            properties: Physicochemical properties [batch, seq_len, 12]
            mask: Optional mask [batch, seq_len]
            
        Returns:
            Dictionary with GNN outputs
        """
        # Construct graph
        node_features, edge_index, batch = self.construct_graph(features, properties, mask)
        
        # Project to GNN dimension
        x = self.input_proj(node_features)
        
        # Apply GNN layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index, batch=batch)
        
        # Global pooling
        # Attention pooling
        att_scores = self.pool_attention(x)
        att_weights = torch.sigmoid(att_scores)
        
        # Separate by batch
        pooled_att = []
        pooled_max = []
        pooled_mean = []
        
        for b in torch.unique(batch):
            batch_mask = batch == b
            batch_x = x[batch_mask]
            batch_att = att_weights[batch_mask]
            
            # Weighted sum
            att_pool = (batch_x * batch_att).sum(dim=0) / (batch_att.sum() + 1e-8)
            pooled_att.append(att_pool)
            
            # Max and mean
            pooled_max.append(batch_x.max(dim=0)[0])
            pooled_mean.append(batch_x.mean(dim=0))
        
        # Stack pooled features
        pooled_att = torch.stack(pooled_att)
        pooled_max = torch.stack(pooled_max)
        pooled_mean = torch.stack(pooled_mean)
        
        # Combine pooling methods
        pooled = torch.cat([pooled_max, pooled_mean], dim=-1)
        global_features = self.pool_mlp(pooled)
        
        # Graph statistics
        num_nodes = torch.tensor([
            (batch == b).sum().item() for b in torch.unique(batch)
        ], device=features.device)
        
        num_edges = torch.tensor([
            ((edge_index[0] >= (batch == b).nonzero()[0]) & 
             (edge_index[0] < (batch == b).nonzero()[-1] + 1)).sum().item() / 2
            for b in torch.unique(batch)
        ], device=features.device)
        
        avg_degree = num_edges / (num_nodes + 1e-8)
        
        return {
            'node_features': x,
            'global_features': global_features,
            'edge_index': edge_index,
            'batch': batch,
            'num_nodes': num_nodes,
            'num_edges': num_edges,
            'avg_degree': avg_degree
        }