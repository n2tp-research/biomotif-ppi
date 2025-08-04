import os
import torch
import h5py
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from transformers import AutoModel, AutoTokenizer
import hashlib


class ESMEmbeddingGenerator:
    """
    Generates and caches ESM-2 embeddings for protein sequences.
    Implements efficient batching and HDF5-based caching.
    """
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        """
        Args:
            config: Configuration dictionary
            device: Device to run model on
        """
        self.config = config
        self.device = device
        
        # Load ESM-2 model and tokenizer
        print(f"Loading ESM-2 model: {config['esm']['model_name']}")
        self.tokenizer = AutoTokenizer.from_pretrained(config['esm']['model_name'])
        self.model = AutoModel.from_pretrained(config['esm']['model_name'])
        self.model.to(device)
        self.model.eval()
        
        # Model parameters
        self.batch_size = config['esm']['batch_size']
        self.max_length = config['esm']['max_length']
        self.embedding_dim = config['esm']['embedding_dim']
        self.truncation_strategy = config['esm']['truncation_strategy']
        
        # Cache settings
        self.cache_dir = config['data']['cache_dir']
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def _get_cache_path(self, dataset_name: str) -> str:
        """Generate cache file path based on dataset and model."""
        model_name = self.config['esm']['model_name'].replace('/', '_')
        cache_file = f"{dataset_name}_{model_name}_embeddings.h5"
        return os.path.join(self.cache_dir, cache_file)
    
    def _truncate_sequence(self, sequence: str) -> Tuple[str, bool]:
        """
        Truncate sequence according to strategy if needed.
        Returns truncated sequence and whether truncation occurred.
        """
        if len(sequence) <= self.max_length - 2:  # Account for special tokens
            return sequence, False
            
        if self.truncation_strategy == 'center':
            # Keep equal amounts from start and end
            keep_length = self.max_length - 2
            start_keep = keep_length // 2
            end_keep = keep_length - start_keep
            truncated = sequence[:start_keep] + sequence[-end_keep:]
        elif self.truncation_strategy == 'start':
            truncated = sequence[:self.max_length - 2]
        else:  # 'end'
            truncated = sequence[-(self.max_length - 2):]
            
        return truncated, True
    
    def _compute_sequence_hash(self, sequence: str) -> str:
        """Compute MD5 hash of sequence for integrity checking."""
        return hashlib.md5(sequence.encode()).hexdigest()
    
    def generate_embeddings(
        self,
        sequences: Dict[str, str],
        dataset_name: str,
        force_regenerate: bool = False
    ) -> str:
        """
        Generate embeddings for all sequences and save to HDF5 cache.
        
        Args:
            sequences: Dictionary mapping protein IDs to sequences
            dataset_name: Name for the cache file
            force_regenerate: Whether to regenerate even if cache exists
            
        Returns:
            Path to the generated cache file
        """
        cache_path = self._get_cache_path(dataset_name)
        
        # Determine whether to create new file or append
        if os.path.exists(cache_path):
            if force_regenerate:
                print(f"Force regenerating cache at {cache_path}")
                mode = 'w'  # Overwrite
            else:
                print(f"Appending to existing cache at {cache_path}")
                mode = 'a'  # Append
        else:
            print(f"Creating new embedding cache at {cache_path}")
            mode = 'w'  # Create new
            
        with h5py.File(cache_path, mode) as h5f:
            # Add metadata only if creating new file
            if mode == 'w':
                h5f.attrs['model_name'] = self.config['esm']['model_name']
                h5f.attrs['embedding_dim'] = self.embedding_dim
                h5f.attrs['max_length'] = self.max_length
                h5f.attrs['truncation_strategy'] = self.truncation_strategy
            
            # Process sequences in batches
            protein_ids = list(sequences.keys())
            num_batches = (len(protein_ids) + self.batch_size - 1) // self.batch_size
            
            with torch.no_grad():
                for batch_idx in tqdm(range(num_batches), desc="Generating embeddings"):
                    start_idx = batch_idx * self.batch_size
                    end_idx = min((batch_idx + 1) * self.batch_size, len(protein_ids))
                    batch_ids = protein_ids[start_idx:end_idx]
                    
                    # Prepare batch sequences
                    batch_sequences = []
                    truncation_masks = []
                    
                    for protein_id in batch_ids:
                        seq = sequences[protein_id]
                        truncated_seq, was_truncated = self._truncate_sequence(seq)
                        batch_sequences.append(truncated_seq)
                        truncation_masks.append(was_truncated)
                    
                    # Tokenize
                    inputs = self.tokenizer(
                        batch_sequences,
                        return_tensors='pt',
                        padding=True,
                        truncation=False,  # Already handled
                        max_length=self.max_length
                    ).to(self.device)
                    
                    # Generate embeddings
                    outputs = self.model(**inputs)
                    embeddings = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
                    
                    # Save to HDF5
                    for i, protein_id in enumerate(batch_ids):
                        # Skip if already exists (when appending)
                        if protein_id in h5f:
                            continue
                            
                        # Remove padding
                        seq_len = len(batch_sequences[i]) + 2  # Account for special tokens
                        embedding = embeddings[i, :seq_len].cpu().numpy()
                        
                        # Create dataset for this protein
                        grp = h5f.create_dataset(
                            protein_id,
                            data=embedding,
                            compression='lzf' if self.config['esm']['compression'] == 'lzf' else None
                        )
                        
                        # Add metadata
                        grp.attrs['sequence_length'] = len(sequences[protein_id])
                        grp.attrs['truncated'] = truncation_masks[i]
                        grp.attrs['sequence_hash'] = self._compute_sequence_hash(sequences[protein_id])
                    
                    # Clear GPU cache periodically
                    if batch_idx % 10 == 0:
                        torch.cuda.empty_cache()
        
        print(f"Successfully generated embeddings for {len(sequences)} proteins")
        return cache_path
    
    def load_embedding(self, protein_id: str, cache_path: str) -> Optional[torch.Tensor]:
        """Load single embedding from cache."""
        try:
            with h5py.File(cache_path, 'r') as h5f:
                if protein_id in h5f:
                    return torch.from_numpy(h5f[protein_id][:])
            return None
        except Exception as e:
            print(f"Error loading embedding for {protein_id}: {e}")
            return None
    
    def validate_cache(self, cache_path: str, sequences: Dict[str, str]) -> Dict[str, bool]:
        """
        Validate that cached embeddings match current sequences.
        Returns dict of protein_id -> is_valid.
        """
        validation_results = {}
        
        with h5py.File(cache_path, 'r') as h5f:
            for protein_id in sequences:
                if protein_id not in h5f:
                    validation_results[protein_id] = False
                    continue
                    
                # Check sequence hash
                stored_hash = h5f[protein_id].attrs.get('sequence_hash', '')
                current_hash = self._compute_sequence_hash(sequences[protein_id])
                validation_results[protein_id] = (stored_hash == current_hash)
        
        return validation_results
    
    def generate_single_embedding(self, sequence: str) -> torch.Tensor:
        """Generate embedding for a single sequence (no caching)."""
        truncated_seq, _ = self._truncate_sequence(sequence)
        
        with torch.no_grad():
            inputs = self.tokenizer(
                truncated_seq,
                return_tensors='pt',
                padding=False,
                truncation=False
            ).to(self.device)
            
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.squeeze(0)  # Remove batch dim
            
        return embedding