import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Tuple, Optional, List
from datasets import load_dataset
import h5py
from tqdm import tqdm


class PPIDataset(Dataset):
    """
    Dataset class for protein-protein interaction prediction.
    Handles loading from Hugging Face and preprocessing.
    """
    
    def __init__(
        self,
        split: str,
        config: Dict,
        sequence_dict: Optional[Dict[str, str]] = None,
        embedding_cache: Optional[str] = None,
        transform=None
    ):
        """
        Args:
            split: One of 'Intra1' (train), 'Intra0' (val), 'Intra2' (test)
            config: Configuration dictionary
            sequence_dict: Pre-loaded protein sequences
            embedding_cache: Path to cached embeddings
            transform: Optional data transforms
        """
        self.split = split
        self.config = config
        self.transform = transform
        self.embedding_cache = embedding_cache
        
        # Load dataset from Hugging Face
        print(f"Loading {split} split from Hugging Face...")
        self.dataset = load_dataset(
            config['data']['dataset_name'],
            split=split,
            cache_dir=config['data']['cache_dir']
        )
        
        # Process interactions
        self.interactions = []
        self.protein_ids = set()
        
        for item in tqdm(self.dataset, desc=f"Processing {split} data"):
            protein_a = item['SeqA']
            protein_b = item['SeqB']
            label = item['labels']
            
            self.interactions.append({
                'protein_a': protein_a,
                'protein_b': protein_b,
                'label': label
            })
            
            self.protein_ids.add(protein_a)
            self.protein_ids.add(protein_b)
        
        print(f"Loaded {len(self.interactions)} interactions with {len(self.protein_ids)} unique proteins")
        
        # Load or prepare sequence dictionary
        if sequence_dict is None:
            self.sequence_dict = self._load_sequences()
        else:
            self.sequence_dict = sequence_dict
            
        # Validate sequences
        self._validate_sequences()
        
        # Initialize embedding cache if provided
        if self.embedding_cache:
            self.embedding_h5 = h5py.File(self.embedding_cache, 'r')
        else:
            self.embedding_h5 = None
    
    def _load_sequences(self) -> Dict[str, str]:
        """Load protein sequences from UniProt or cached file."""
        sequence_file = os.path.join(
            self.config['data']['cache_dir'],
            f"sequences_{self.config['data']['uniprot_release']}.pkl"
        )
        
        if os.path.exists(sequence_file):
            import pickle
            print(f"Loading sequences from cache: {sequence_file}")
            with open(sequence_file, 'rb') as f:
                return pickle.load(f)
        else:
            # In real implementation, would fetch from UniProt
            # For now, return empty dict
            print("Warning: No sequence file found. Sequences need to be loaded separately.")
            return {}
    
    def _validate_sequences(self):
        """Validate sequences according to criteria in config."""
        min_len = self.config['data']['min_seq_length']
        max_len = self.config['data']['max_seq_length']
        non_standard = set(self.config['data']['non_standard_aa'])
        max_gap_frac = self.config['data']['max_gap_fraction']
        max_unk_frac = self.config['data']['max_unknown_fraction']
        
        valid_interactions = []
        
        for interaction in tqdm(self.interactions, desc="Validating sequences"):
            protein_a = interaction['protein_a']
            protein_b = interaction['protein_b']
            
            # Check if sequences exist
            if protein_a not in self.sequence_dict or protein_b not in self.sequence_dict:
                continue
                
            seq_a = self.sequence_dict[protein_a]
            seq_b = self.sequence_dict[protein_b]
            
            # Length check
            if not (min_len <= len(seq_a) <= max_len and min_len <= len(seq_b) <= max_len):
                continue
            
            # Non-standard amino acid check
            if any(aa in non_standard for aa in seq_a) or any(aa in non_standard for aa in seq_b):
                continue
                
            # Gap and unknown residue check
            gap_frac_a = seq_a.count('-') / len(seq_a)
            gap_frac_b = seq_b.count('-') / len(seq_b)
            unk_frac_a = seq_a.count('X') / len(seq_a)
            unk_frac_b = seq_b.count('X') / len(seq_b)
            
            if (gap_frac_a > max_gap_frac or gap_frac_b > max_gap_frac or
                unk_frac_a > max_unk_frac or unk_frac_b > max_unk_frac):
                continue
                
            # Convert U to C
            seq_a = seq_a.replace('U', 'C')
            seq_b = seq_b.replace('U', 'C')
            
            # Update sequence dict with processed sequences
            self.sequence_dict[protein_a] = seq_a
            self.sequence_dict[protein_b] = seq_b
            
            valid_interactions.append(interaction)
        
        print(f"Retained {len(valid_interactions)}/{len(self.interactions)} interactions after validation")
        self.interactions = valid_interactions
    
    def __len__(self) -> int:
        return len(self.interactions)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary containing:
            - protein_a_id: Protein A identifier
            - protein_b_id: Protein B identifier
            - sequence_a: Protein A sequence
            - sequence_b: Protein B sequence
            - label: Interaction label (0 or 1)
            - embedding_a: ESM-2 embedding if cache available
            - embedding_b: ESM-2 embedding if cache available
        """
        interaction = self.interactions[idx]
        
        item = {
            'protein_a_id': interaction['protein_a'],
            'protein_b_id': interaction['protein_b'],
            'sequence_a': self.sequence_dict[interaction['protein_a']],
            'sequence_b': self.sequence_dict[interaction['protein_b']],
            'label': torch.tensor(interaction['label'], dtype=torch.float32)
        }
        
        # Load embeddings from cache if available
        if self.embedding_h5:
            if interaction['protein_a'] in self.embedding_h5:
                item['embedding_a'] = torch.from_numpy(
                    self.embedding_h5[interaction['protein_a']][:]
                )
            if interaction['protein_b'] in self.embedding_h5:
                item['embedding_b'] = torch.from_numpy(
                    self.embedding_h5[interaction['protein_b']][:]
                )
        
        if self.transform:
            item = self.transform(item)
            
        return item
    
    def get_protein_ids(self) -> List[str]:
        """Return list of all unique protein IDs in dataset."""
        return list(self.protein_ids)
    
    def close(self):
        """Close any open file handles."""
        if self.embedding_h5:
            self.embedding_h5.close()


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable length sequences.
    Pads sequences to maximum length in batch.
    """
    # Find max lengths in batch
    max_len_a = max(len(item['sequence_a']) for item in batch)
    max_len_b = max(len(item['sequence_b']) for item in batch)
    
    # Prepare batch tensors
    batch_size = len(batch)
    labels = torch.stack([item['label'] for item in batch])
    
    # Prepare output dict
    collated = {
        'protein_a_ids': [item['protein_a_id'] for item in batch],
        'protein_b_ids': [item['protein_b_id'] for item in batch],
        'sequences_a': [item['sequence_a'] for item in batch],
        'sequences_b': [item['sequence_b'] for item in batch],
        'labels': labels,
        'lengths_a': torch.tensor([len(item['sequence_a']) for item in batch]),
        'lengths_b': torch.tensor([len(item['sequence_b']) for item in batch])
    }
    
    # Add embeddings if available
    if 'embedding_a' in batch[0]:
        # Pad embeddings
        embedding_dim = batch[0]['embedding_a'].shape[-1]
        padded_embeddings_a = torch.zeros(batch_size, max_len_a, embedding_dim)
        padded_embeddings_b = torch.zeros(batch_size, max_len_b, embedding_dim)
        
        for i, item in enumerate(batch):
            len_a = len(item['sequence_a'])
            len_b = len(item['sequence_b'])
            padded_embeddings_a[i, :len_a] = item['embedding_a']
            padded_embeddings_b[i, :len_b] = item['embedding_b']
        
        collated['embeddings_a'] = padded_embeddings_a
        collated['embeddings_b'] = padded_embeddings_b
        
        # Create attention masks
        collated['mask_a'] = torch.arange(max_len_a)[None, :] < collated['lengths_a'][:, None]
        collated['mask_b'] = torch.arange(max_len_b)[None, :] < collated['lengths_b'][:, None]
    
    return collated