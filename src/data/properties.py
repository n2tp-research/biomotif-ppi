import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class AminoAcidProperties:
    """Container for amino acid physicochemical properties."""
    hydrophobicity: float
    charge: int
    size: float
    aromaticity: bool
    flexibility: float
    solvent_accessibility: float
    h_bond_donor: bool
    h_bond_acceptor: bool


class PhysicochemicalEncoder:
    """
    Encodes protein sequences into physicochemical property tensors.
    Implements multiple property scales as defined in the methodology.
    """
    
    # Kyte-Doolittle hydrophobicity scale (normalized to [-1, 1])
    KYTE_DOOLITTLE = {
        'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
        'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
        'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
        'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
    }
    
    # Amino acid charges at pH 7.0
    CHARGE = {
        'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
        'Q': 0, 'E': -1, 'G': 0, 'H': 0, 'I': 0,
        'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
        'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
    }
    
    # Normalized volumes (Zamyatnin scale)
    SIZE = {
        'A': 88.6, 'R': 173.4, 'N': 114.1, 'D': 111.1, 'C': 108.5,
        'Q': 143.8, 'E': 138.4, 'G': 60.1, 'H': 153.2, 'I': 166.7,
        'L': 166.7, 'K': 168.6, 'M': 162.9, 'F': 189.9, 'P': 112.7,
        'S': 89.0, 'T': 116.1, 'W': 227.8, 'Y': 193.6, 'V': 140.0
    }
    
    # Aromatic residues
    AROMATIC = {'F', 'W', 'Y'}
    
    # B-factor propensities (flexibility) from PDB statistics
    FLEXIBILITY = {
        'A': 0.357, 'R': 0.529, 'N': 0.463, 'D': 0.511, 'C': 0.346,
        'Q': 0.493, 'E': 0.497, 'G': 0.544, 'H': 0.323, 'I': 0.462,
        'L': 0.365, 'K': 0.466, 'M': 0.295, 'F': 0.314, 'P': 0.509,
        'S': 0.507, 'T': 0.444, 'W': 0.305, 'Y': 0.420, 'V': 0.386
    }
    
    # Janin solvent accessibility scale
    SOLVENT_ACCESSIBILITY = {
        'A': 0.74, 'R': 0.64, 'N': 0.63, 'D': 0.62, 'C': 0.91,
        'Q': 0.62, 'E': 0.62, 'G': 0.72, 'H': 0.78, 'I': 0.88,
        'L': 0.85, 'K': 0.52, 'M': 0.85, 'F': 0.88, 'P': 0.64,
        'S': 0.66, 'T': 0.70, 'W': 0.85, 'Y': 0.76, 'V': 0.86
    }
    
    # H-bond donors and acceptors
    H_BOND_DONORS = {'R', 'K', 'W', 'N', 'Q', 'H', 'S', 'T', 'Y', 'C'}
    H_BOND_ACCEPTORS = {'D', 'E', 'N', 'Q', 'H', 'S', 'T', 'Y', 'C', 'M'}
    
    # Chou-Fasman secondary structure propensities
    CHOU_FASMAN_ALPHA = {
        'A': 1.42, 'R': 0.98, 'N': 0.67, 'D': 1.01, 'C': 0.70,
        'Q': 1.11, 'E': 1.51, 'G': 0.57, 'H': 1.00, 'I': 1.08,
        'L': 1.21, 'K': 1.16, 'M': 1.45, 'F': 1.13, 'P': 0.57,
        'S': 0.77, 'T': 0.83, 'W': 1.08, 'Y': 0.69, 'V': 1.06
    }
    
    CHOU_FASMAN_BETA = {
        'A': 0.83, 'R': 0.93, 'N': 0.89, 'D': 0.54, 'C': 1.19,
        'Q': 1.10, 'E': 0.37, 'G': 0.75, 'H': 0.87, 'I': 1.60,
        'L': 1.30, 'K': 0.74, 'M': 1.05, 'F': 1.38, 'P': 0.55,
        'S': 0.75, 'T': 1.19, 'W': 1.37, 'Y': 1.47, 'V': 1.70
    }
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Normalize scales
        self._normalize_scales()
        
    def _normalize_scales(self):
        """Normalize property scales to appropriate ranges."""
        # Normalize hydrophobicity to [-1, 1]
        hydro_values = list(self.KYTE_DOOLITTLE.values())
        hydro_min, hydro_max = min(hydro_values), max(hydro_values)
        for aa in self.KYTE_DOOLITTLE:
            self.KYTE_DOOLITTLE[aa] = 2 * (self.KYTE_DOOLITTLE[aa] - hydro_min) / (hydro_max - hydro_min) - 1
        
        # Normalize size to [0, 1]
        size_values = list(self.SIZE.values())
        size_min, size_max = min(size_values), max(size_values)
        for aa in self.SIZE:
            self.SIZE[aa] = (self.SIZE[aa] - size_min) / (size_max - size_min)
            
        # Normalize flexibility to [0, 1]
        flex_values = list(self.FLEXIBILITY.values())
        flex_min, flex_max = min(flex_values), max(flex_values)
        for aa in self.FLEXIBILITY:
            self.FLEXIBILITY[aa] = (self.FLEXIBILITY[aa] - flex_min) / (flex_max - flex_min)
    
    def encode_sequence(self, sequence: str) -> torch.Tensor:
        """
        Encode a protein sequence into physicochemical property tensor.
        
        Args:
            sequence: Protein sequence string
            
        Returns:
            Tensor of shape (seq_len, 12) with properties
        """
        seq_len = len(sequence)
        properties = torch.zeros(seq_len, 12)
        
        for i, aa in enumerate(sequence):
            if aa not in self.KYTE_DOOLITTLE:
                # Handle unknown amino acids
                continue
                
            # Basic properties (6 features)
            properties[i, 0] = self.KYTE_DOOLITTLE[aa]
            properties[i, 1] = self.CHARGE[aa]
            properties[i, 2] = self.SIZE[aa]
            properties[i, 3] = 1.0 if aa in self.AROMATIC else 0.0
            properties[i, 4] = self.FLEXIBILITY[aa]
            properties[i, 5] = self.SOLVENT_ACCESSIBILITY[aa]
            
            # H-bonding (2 features)
            properties[i, 6] = 1.0 if aa in self.H_BOND_DONORS else 0.0
            properties[i, 7] = 1.0 if aa in self.H_BOND_ACCEPTORS else 0.0
            
            # Position-specific features (2 features)
            properties[i, 8] = i / seq_len  # Relative position
            properties[i, 9] = 1.0 - (i / seq_len)  # Distance from C-terminus
            
            # Placeholder for secondary structure (2 features)
            # Will be filled by secondary structure calculation
            properties[i, 10] = 0.0  # Alpha helix propensity
            properties[i, 11] = 0.0  # Beta sheet propensity
            
        return properties
    
    def calculate_secondary_structure_propensity(
        self,
        sequence: str,
        window_size: int = 17
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate secondary structure propensities using Chou-Fasman parameters.
        
        Args:
            sequence: Protein sequence
            window_size: Window size for averaging
            
        Returns:
            Tuple of (alpha_propensity, beta_propensity, coil_propensity) tensors
        """
        seq_len = len(sequence)
        alpha_prop = torch.zeros(seq_len)
        beta_prop = torch.zeros(seq_len)
        
        half_window = window_size // 2
        
        for i in range(seq_len):
            # Calculate window boundaries
            start = max(0, i - half_window)
            end = min(seq_len, i + half_window + 1)
            window_aa = sequence[start:end]
            
            # Calculate average propensities in window
            alpha_sum = 0
            beta_sum = 0
            count = 0
            
            for aa in window_aa:
                if aa in self.CHOU_FASMAN_ALPHA:
                    alpha_sum += self.CHOU_FASMAN_ALPHA[aa]
                    beta_sum += self.CHOU_FASMAN_BETA[aa]
                    count += 1
            
            if count > 0:
                alpha_prop[i] = alpha_sum / count
                beta_prop[i] = beta_sum / count
        
        # Normalize to probabilities
        total = alpha_prop + beta_prop + 1e-8
        alpha_prop = alpha_prop / total
        beta_prop = beta_prop / total
        coil_prop = 1 - alpha_prop - beta_prop
        
        return alpha_prop, beta_prop, coil_prop
    
    def encode_sequence_with_ss(self, sequence: str) -> torch.Tensor:
        """
        Encode sequence with secondary structure propensities included.
        
        Args:
            sequence: Protein sequence
            
        Returns:
            Tensor of shape (seq_len, 12) with all properties
        """
        # Get basic properties
        properties = self.encode_sequence(sequence)
        
        # Calculate secondary structure
        alpha_prop, beta_prop, _ = self.calculate_secondary_structure_propensity(
            sequence,
            window_size=self.config['properties']['secondary_structure']['window_size']
        )
        
        # Add to properties tensor
        properties[:, 10] = alpha_prop
        properties[:, 11] = beta_prop
        
        return properties
    
    def get_property_names(self) -> List[str]:
        """Return list of property names in order."""
        return [
            'hydrophobicity',
            'charge',
            'size',
            'aromaticity',
            'flexibility',
            'solvent_accessibility',
            'h_bond_donor',
            'h_bond_acceptor',
            'relative_position',
            'distance_from_c_term',
            'alpha_propensity',
            'beta_propensity'
        ]