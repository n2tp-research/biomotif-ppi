import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, List, Optional
import numpy as np
from scipy import stats
import heapq


class ComplementarityAnalyzer(nn.Module):
    """
    Analyzes physicochemical complementarity between protein pairs.
    Uses chunked computation for memory efficiency and biological rules.
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config
        
        # Extract parameters
        comp_config = config['model']['complementarity']
        self.chunk_size = comp_config['chunk_size']
        self.chunk_overlap = comp_config['chunk_overlap']
        self.top_k_pairs = comp_config['top_k_pairs']
        
        # Complementarity weights (learnable)
        self.comp_weights = nn.Parameter(torch.tensor([
            comp_config['weights']['hydrophobic'],
            comp_config['weights']['electrostatic'],
            comp_config['weights']['size'],
            comp_config['weights']['aromatic'],
            comp_config['weights']['hbond']
        ]))
        
        # Function parameters
        self.hydrophobic_threshold = comp_config['hydrophobic_threshold']
        self.electrostatic_decay = comp_config['electrostatic_decay']
        self.size_sigma = comp_config['size_sigma']
        self.aromatic_optimal_distance = comp_config['aromatic_optimal_distance']
        self.aromatic_distance_tolerance = comp_config['aromatic_distance_tolerance']
        
        # Build aggregation MLP
        mlp_dims = comp_config['mlp_dims']
        self.aggregation_mlp = self._build_mlp(mlp_dims)
        
    def _build_mlp(self, dims: List[int]) -> nn.Module:
        """Build MLP from dimension list."""
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation after last layer
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
        return nn.Sequential(*layers)
    
    def compute_hydrophobic_complementarity(
        self,
        hydro_a: torch.Tensor,
        hydro_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute hydrophobic complementarity.
        High score when both residues are hydrophobic.
        """
        # Check if both are hydrophobic (above threshold)
        both_hydrophobic = (hydro_a > self.hydrophobic_threshold) & (hydro_b > self.hydrophobic_threshold)
        
        # Compute score
        score = torch.sigmoid(2 * hydro_a * hydro_b)
        score = score * both_hydrophobic.float()
        
        return score
    
    def compute_electrostatic_complementarity(
        self,
        charge_a: torch.Tensor,
        charge_b: torch.Tensor,
        distance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute electrostatic complementarity.
        Positive for opposite charges, negative for same charges.
        """
        charge_product = charge_a * charge_b
        
        # Base score
        score = torch.zeros_like(charge_product)
        
        # Opposite charges attract
        opposite_mask = charge_product == -1
        score[opposite_mask] = 2.0
        
        # Same charges repel
        same_mask = charge_product == 1
        score[same_mask] = -1.0
        
        # Apply distance decay if provided
        if distance is not None:
            decay = torch.exp(-distance / self.electrostatic_decay)
            score = score * decay
            
        return score
    
    def compute_size_complementarity(
        self,
        size_a: torch.Tensor,
        size_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute size complementarity.
        Prefers similar sizes (shape complementarity).
        """
        size_diff = torch.abs(size_a - size_b)
        score = torch.exp(-size_diff**2 / (2 * self.size_sigma**2))
        return score
    
    def compute_aromatic_complementarity(
        self,
        arom_a: torch.Tensor,
        arom_b: torch.Tensor,
        seq_distance: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute aromatic complementarity.
        Pi-pi stacking has optimal distance ~7 residues.
        """
        # Both must be aromatic
        both_aromatic = arom_a * arom_b
        
        if seq_distance is not None:
            # Compute distance penalty
            distance_penalty = torch.maximum(
                torch.zeros_like(seq_distance),
                1 - torch.abs(seq_distance - self.aromatic_optimal_distance) / self.aromatic_distance_tolerance
            )
            score = both_aromatic * distance_penalty
        else:
            score = both_aromatic
            
        return score
    
    def compute_hbond_complementarity(
        self,
        donor_a: torch.Tensor,
        acceptor_a: torch.Tensor,
        donor_b: torch.Tensor,
        acceptor_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute hydrogen bonding complementarity.
        Donor-acceptor pairs score positively.
        """
        # A donor to B acceptor OR B donor to A acceptor
        score = donor_a * acceptor_b + donor_b * acceptor_a
        return score
    
    def compute_pairwise_complementarity(
        self,
        props_a: torch.Tensor,
        props_b: torch.Tensor,
        indices_a: torch.Tensor,
        indices_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute all complementarity scores for a chunk.
        
        Args:
            props_a: Properties for chunk A [chunk_size_a, 12]
            props_b: Properties for chunk B [chunk_size_b, 12]
            indices_a: Original indices for chunk A
            indices_b: Original indices for chunk B
            
        Returns:
            Complementarity scores [chunk_size_a, chunk_size_b]
        """
        # Extract individual properties
        # Properties order: hydrophobicity, charge, size, aromaticity, flexibility,
        # solvent_accessibility, h_bond_donor, h_bond_acceptor, ...
        
        hydro_a = props_a[:, 0].unsqueeze(1)
        hydro_b = props_b[:, 0].unsqueeze(0)
        
        charge_a = props_a[:, 1].unsqueeze(1)
        charge_b = props_b[:, 1].unsqueeze(0)
        
        size_a = props_a[:, 2].unsqueeze(1)
        size_b = props_b[:, 2].unsqueeze(0)
        
        arom_a = props_a[:, 3].unsqueeze(1)
        arom_b = props_b[:, 3].unsqueeze(0)
        
        donor_a = props_a[:, 6].unsqueeze(1)
        acceptor_a = props_a[:, 7].unsqueeze(1)
        donor_b = props_b[:, 6].unsqueeze(0)
        acceptor_b = props_b[:, 7].unsqueeze(0)
        
        # Compute individual complementarities
        hydrophobic_score = self.compute_hydrophobic_complementarity(hydro_a, hydro_b)
        electrostatic_score = self.compute_electrostatic_complementarity(charge_a, charge_b)
        size_score = self.compute_size_complementarity(size_a, size_b)
        aromatic_score = self.compute_aromatic_complementarity(arom_a, arom_b)
        hbond_score = self.compute_hbond_complementarity(donor_a, acceptor_a, donor_b, acceptor_b)
        
        # Stack scores
        scores = torch.stack([
            hydrophobic_score,
            electrostatic_score,
            size_score,
            aromatic_score,
            hbond_score
        ], dim=-1)
        
        # Apply learned weights
        weights = F.softmax(self.comp_weights, dim=0)
        weighted_scores = scores @ weights
        
        return weighted_scores
    
    def chunked_analysis(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        props_a: torch.Tensor,
        props_b: torch.Tensor,
        mask_a: Optional[torch.Tensor] = None,
        mask_b: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform chunked complementarity analysis.
        
        Args:
            features_a, features_b: Encoded features [batch, seq_len, dim]
            props_a, props_b: Properties [batch, seq_len, 12]
            mask_a, mask_b: Optional masks
            
        Returns:
            Dictionary with analysis results
        """
        batch_size = features_a.shape[0]
        results = []
        
        for batch_idx in range(batch_size):
            # Get single sample
            feat_a = features_a[batch_idx]
            feat_b = features_b[batch_idx]
            prop_a = props_a[batch_idx]
            prop_b = props_b[batch_idx]
            
            # Apply masks if provided
            if mask_a is not None:
                valid_a = mask_a[batch_idx]
                feat_a = feat_a[valid_a]
                prop_a = prop_a[valid_a]
            
            if mask_b is not None:
                valid_b = mask_b[batch_idx]
                feat_b = feat_b[valid_b]
                prop_b = prop_b[valid_b]
            
            len_a = feat_a.shape[0]
            len_b = feat_b.shape[0]
            
            # Priority queue for top-k pairs
            top_pairs = []
            
            # Process in chunks
            for i in range(0, len_a, self.chunk_size - self.chunk_overlap):
                for j in range(0, len_b, self.chunk_size - self.chunk_overlap):
                    # Get chunk boundaries
                    end_i = min(i + self.chunk_size, len_a)
                    end_j = min(j + self.chunk_size, len_b)
                    
                    # Extract chunks
                    chunk_prop_a = prop_a[i:end_i]
                    chunk_prop_b = prop_b[j:end_j]
                    
                    # Compute complementarity scores
                    chunk_scores = self.compute_pairwise_complementarity(
                        chunk_prop_a,
                        chunk_prop_b,
                        torch.arange(i, end_i, device=prop_a.device),
                        torch.arange(j, end_j, device=prop_b.device)
                    )
                    
                    # Find top scores in chunk
                    chunk_flat = chunk_scores.flatten()
                    top_values, top_indices = torch.topk(
                        chunk_flat,
                        min(self.top_k_pairs // 10, chunk_flat.shape[0])
                    )
                    
                    # Convert to original indices
                    for val, idx in zip(top_values, top_indices):
                        local_i = idx // chunk_scores.shape[1]
                        local_j = idx % chunk_scores.shape[1]
                        global_i = i + local_i
                        global_j = j + local_j
                        
                        # Add to priority queue (negative for min heap)
                        if len(top_pairs) < self.top_k_pairs:
                            heapq.heappush(top_pairs, (val.item(), global_i.item(), global_j.item()))
                        elif val.item() > top_pairs[0][0]:
                            heapq.heapreplace(top_pairs, (val.item(), global_i.item(), global_j.item()))
            
            # Extract features from top pairs
            sample_features = self._extract_pair_features(top_pairs, feat_a, feat_b, prop_a, prop_b)
            results.append(sample_features)
        
        # Stack batch results
        batch_features = torch.stack(results)
        
        # Process through aggregation MLP
        output = self.aggregation_mlp(batch_features)
        
        return {
            'complementarity_features': output,
            'batch_features': batch_features
        }
    
    def _extract_pair_features(
        self,
        top_pairs: List[Tuple[float, int, int]],
        features_a: torch.Tensor,
        features_b: torch.Tensor,
        props_a: torch.Tensor,
        props_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Extract statistical features from top complementary pairs.
        """
        if len(top_pairs) == 0:
            # Return zero features if no pairs found
            return torch.zeros(self.config['model']['complementarity']['mlp_dims'][0], 
                              device=features_a.device)
        
        # Sort pairs by score
        top_pairs = sorted(top_pairs, reverse=True)
        
        # Extract scores and positions
        scores = torch.tensor([p[0] for p in top_pairs], device=features_a.device)
        pos_a = torch.tensor([p[1] for p in top_pairs], device=features_a.device)
        pos_b = torch.tensor([p[2] for p in top_pairs], device=features_a.device)
        
        # Compute statistics
        features = []
        
        # Score statistics (5 features)
        features.extend([
            scores.max(),
            scores.mean(),
            scores.std(),
            scores[0] if len(scores) > 0 else 0,  # Best score
            scores[min(9, len(scores)-1)] if len(scores) > 0 else 0  # 10th best score
        ])
        
        # Distance distribution (10 features - histogram)
        distances = torch.abs(pos_a.float() - pos_b.float())
        hist = torch.histc(distances, bins=10, min=0, max=100)
        hist = hist / hist.sum()  # Normalize
        features.extend(hist.tolist())
        
        # Position statistics (6 features)
        features.extend([
            pos_a.float().mean(),
            pos_a.float().std(),
            pos_b.float().mean(),
            pos_b.float().std(),
            (pos_a.float() / features_a.shape[0]).mean(),  # Relative position A
            (pos_b.float() / features_b.shape[0]).mean()   # Relative position B
        ])
        
        # Complementarity type distribution (5 features)
        # Count predominant complementarity type for each top pair
        type_counts = torch.zeros(5, device=features_a.device)
        for score, i, j in top_pairs[:20]:  # Look at top 20
            # Recompute individual scores to find dominant type
            prop_vec_a = props_a[i:i+1]
            prop_vec_b = props_b[j:j+1]
            
            scores_individual = torch.stack([
                self.compute_hydrophobic_complementarity(prop_vec_a[:, 0:1], prop_vec_b[:, 0:1]),
                self.compute_electrostatic_complementarity(prop_vec_a[:, 1:2], prop_vec_b[:, 1:2]),
                self.compute_size_complementarity(prop_vec_a[:, 2:3], prop_vec_b[:, 2:3]),
                self.compute_aromatic_complementarity(prop_vec_a[:, 3:4], prop_vec_b[:, 3:4]),
                self.compute_hbond_complementarity(prop_vec_a[:, 6:7], prop_vec_a[:, 7:8], 
                                                  prop_vec_b[:, 6:7], prop_vec_b[:, 7:8])
            ]).squeeze()
            
            dominant_type = scores_individual.argmax()
            type_counts[dominant_type] += 1
        
        type_counts = type_counts / type_counts.sum()  # Normalize
        features.extend(type_counts.tolist())
        
        # Clustering coefficient (1 feature)
        # How clustered are the top pairs in sequence space
        if len(pos_a) > 1:
            cluster_score = 1.0 / (1.0 + distances.std())
        else:
            cluster_score = 0.0
        features.append(cluster_score.item() if isinstance(cluster_score, torch.Tensor) else cluster_score)
        
        # Interaction surface estimate (1 feature)
        # Approximate by counting unique positions involved
        unique_a = len(torch.unique(pos_a))
        unique_b = len(torch.unique(pos_b))
        surface_score = (unique_a + unique_b) / (features_a.shape[0] + features_b.shape[0])
        features.append(surface_score)
        
        # Mean feature similarity at top pairs (4 features)
        if len(top_pairs) > 0:
            feat_sim = []
            for _, i, j in top_pairs[:10]:
                sim = F.cosine_similarity(features_a[i:i+1], features_b[j:j+1])
                feat_sim.append(sim.item())
            feat_sim = torch.tensor(feat_sim)
            features.extend([
                feat_sim.mean().item(),
                feat_sim.std().item() if len(feat_sim) > 1 else 0,
                feat_sim.max().item(),
                feat_sim.min().item()
            ])
        else:
            features.extend([0, 0, 0, 0])
        
        # Pad to expected dimension if needed
        feature_tensor = torch.tensor(features, device=features_a.device)
        expected_dim = self.config['model']['complementarity']['mlp_dims'][0]
        
        if feature_tensor.shape[0] < expected_dim:
            padding = torch.zeros(expected_dim - feature_tensor.shape[0], device=features_a.device)
            feature_tensor = torch.cat([feature_tensor, padding])
        elif feature_tensor.shape[0] > expected_dim:
            feature_tensor = feature_tensor[:expected_dim]
            
        return feature_tensor