# Copyright (c) 2025, Jingze Shi.

"""
Unified Sparse Mask API for Flash Dynamic Mask Attention

This module provides Python classes and utilities for creating and managing
sparse attention masks with block-level skipping support.
"""

from typing import Optional, Union, Tuple, List
import torch
import numpy as np
from abc import ABC, abstractmethod

__all__ = [
    "SparseMask",
    "CausalMask", 
    "WindowMask",
    "CausalWindowMask",
    "BlockBitsetMask",
    "BCSRMask",
    "DynamicMask",
    "create_sparse_mask",
    "estimate_speedup",
    "calculate_memory_savings"
]


class SparseMask(ABC):
    """
    Abstract base class for unified sparse masks.
    
    This class defines the interface for all sparse mask implementations
    that can be used with Flash Dynamic Mask Attention kernels.
    """
    
    def __init__(self, 
                 seqlen_q: int, 
                 seqlen_k: int, 
                 block_size_m: int = 128, 
                 block_size_n: int = 128,
                 device: Optional[torch.device] = None):
        self.seqlen_q = seqlen_q
        self.seqlen_k = seqlen_k
        self.block_size_m = block_size_m
        self.block_size_n = block_size_n
        self.device = device or torch.device('cuda')
        self.num_query_blocks = (seqlen_q + block_size_m - 1) // block_size_m
        self.num_key_blocks = (seqlen_k + block_size_n - 1) // block_size_n
    
    @abstractmethod
    def get_mask_type(self) -> str:
        """Return the mask type identifier."""
        pass
    
    @abstractmethod
    def get_cuda_params(self) -> dict:
        """Return parameters needed by CUDA kernels."""
        pass
    
    @abstractmethod
    def estimate_memory_usage(self) -> int:
        """Estimate memory usage in bytes."""
        pass
    
    def to_dense(self) -> torch.Tensor:
        """Convert to dense attention mask for compatibility."""
        mask = torch.zeros(self.seqlen_q, self.seqlen_k, 
                          dtype=torch.float32, device=self.device)
        
        for q_block in range(self.num_query_blocks):
            for k_block in range(self.num_key_blocks):
                if self.is_block_active(q_block, k_block):
                    q_start = q_block * self.block_size_m
                    q_end = min(self.seqlen_q, (q_block + 1) * self.block_size_m)
                    k_start = k_block * self.block_size_n
                    k_end = min(self.seqlen_k, (k_block + 1) * self.block_size_n)
                    mask[q_start:q_end, k_start:k_end] = 1.0
        
        return mask
    
    @abstractmethod
    def is_block_active(self, query_block: int, key_block: int) -> bool:
        """Check if a block should be processed (not masked)."""
        pass
    
    def count_active_blocks(self) -> int:
        """Count total number of active blocks."""
        count = 0
        for q_block in range(self.num_query_blocks):
            for k_block in range(self.num_key_blocks):
                if self.is_block_active(q_block, k_block):
                    count += 1
        return count
    
    def get_sparsity_ratio(self) -> float:
        """Get the sparsity ratio (fraction of inactive blocks)."""
        total_blocks = self.num_query_blocks * self.num_key_blocks
        active_blocks = self.count_active_blocks()
        return 1.0 - (active_blocks / total_blocks)


class CausalMask(SparseMask):
    """
    Causal (lower triangular) mask for autoregressive attention.
    
    This is a parametric mask that requires no storage - the pattern
    is computed on-the-fly in the kernels.
    """
    
    def get_mask_type(self) -> str:
        return "PARAMETRIC_CAUSAL"
    
    def get_cuda_params(self) -> dict:
        return {
            "mask_type": 0,  # PARAMETRIC_CAUSAL
            "mask_data": None,
            "is_causal": True,
            "use_window": False,
            "window_size": 0,
            "doc_segment_id": -1
        }
    
    def estimate_memory_usage(self) -> int:
        return 0  # No storage required
    
    def is_block_active(self, query_block: int, key_block: int) -> bool:
        # Causal mask: key block must not extend beyond query block end
        query_end = (query_block + 1) * self.block_size_m - 1
        key_start = key_block * self.block_size_n
        return key_start <= query_end


class WindowMask(SparseMask):
    """
    Sliding window mask for local attention patterns.
    
    This is a parametric mask that computes the window pattern on-the-fly.
    """
    
    def __init__(self, window_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = window_size
    
    def get_mask_type(self) -> str:
        return "PARAMETRIC_WINDOW"
    
    def get_cuda_params(self) -> dict:
        return {
            "mask_type": 1,  # PARAMETRIC_WINDOW
            "mask_data": None,
            "is_causal": False,
            "use_window": True,
            "window_size": self.window_size,
            "doc_segment_id": -1
        }
    
    def estimate_memory_usage(self) -> int:
        return 0  # No storage required
    
    def is_block_active(self, query_block: int, key_block: int) -> bool:
        # Sliding window: check if blocks overlap with window
        query_center = query_block * self.block_size_m + self.block_size_m // 2
        key_start = key_block * self.block_size_n
        key_end = (key_block + 1) * self.block_size_n - 1
        
        window_start = max(0, query_center - self.window_size // 2)
        window_end = min(self.seqlen_k - 1, query_center + self.window_size // 2)
        
        return not (key_end < window_start or key_start > window_end)


class CausalWindowMask(SparseMask):
    """
    Hybrid causal + sliding window mask.
    
    Combines causal masking with a sliding window for efficient
    long-context attention.
    """
    
    def __init__(self, window_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.window_size = window_size
    
    def get_mask_type(self) -> str:
        return "PARAMETRIC_WINDOW"  # Use window type with causal flag
    
    def get_cuda_params(self) -> dict:
        return {
            "mask_type": 1,  # PARAMETRIC_WINDOW
            "mask_data": None,
            "is_causal": True,
            "use_window": True,
            "window_size": self.window_size,
            "doc_segment_id": -1
        }
    
    def estimate_memory_usage(self) -> int:
        return 0  # No storage required
    
    def is_block_active(self, query_block: int, key_block: int) -> bool:
        # First check causal constraint
        query_end = (query_block + 1) * self.block_size_m - 1
        key_start = key_block * self.block_size_n
        if key_start > query_end:
            return False
        
        # Then check window constraint
        query_center = query_block * self.block_size_m + self.block_size_m // 2
        key_end = (key_block + 1) * self.block_size_n - 1
        
        window_start = max(0, query_center - self.window_size // 2)
        window_end = min(self.seqlen_k - 1, query_center + self.window_size // 2)
        
        return not (key_end < window_start or key_start > window_end)


class BlockBitsetMask(SparseMask):
    """
    Block-level bitset mask for moderate sparsity patterns.
    
    Uses a compressed bitset representation where each bit indicates
    whether a (block_m x block_n) tile should be processed.
    """
    
    def __init__(self, bitset: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        expected_bits = self.num_query_blocks * self.num_key_blocks
        if bitset.numel() * 64 < expected_bits:
            raise ValueError(f"Bitset too small: need at least {expected_bits} bits, got {bitset.numel() * 64}")
        self.bitset = bitset.to(device=self.device, dtype=torch.uint64)
    
    def get_mask_type(self) -> str:
        return "BLOCK_BITSET"
    
    def get_cuda_params(self) -> dict:
        return {
            "mask_type": 2,  # BLOCK_BITSET
            "mask_data": self.bitset.data_ptr(),
            "num_query_blocks": self.num_query_blocks,
            "num_key_blocks": self.num_key_blocks,
            "bitset_size_words": self.bitset.numel()
        }
    
    def estimate_memory_usage(self) -> int:
        return self.bitset.numel() * 8  # 8 bytes per uint64
    
    def is_block_active(self, query_block: int, key_block: int) -> bool:
        bit_idx = query_block * self.num_key_blocks + key_block
        word_idx = bit_idx // 64
        bit_offset = bit_idx % 64
        
        if word_idx >= self.bitset.numel():
            return False
        
        word = self.bitset[word_idx].item()
        return bool((word >> bit_offset) & 1)
    
    @classmethod
    def from_dense_mask(cls, dense_mask: torch.Tensor, 
                       block_size_m: int = 128, 
                       block_size_n: int = 128, 
                       threshold: float = 0.0):
        """Create BlockBitsetMask from dense attention mask."""
        seqlen_q, seqlen_k = dense_mask.shape
        num_q_blocks = (seqlen_q + block_size_m - 1) // block_size_m
        num_k_blocks = (seqlen_k + block_size_n - 1) // block_size_n
        
        total_bits = num_q_blocks * num_k_blocks
        bitset_words = (total_bits + 63) // 64
        bitset = torch.zeros(bitset_words, dtype=torch.uint64, device=dense_mask.device)
        
        for q_block in range(num_q_blocks):
            for k_block in range(num_k_blocks):
                # Check if any element in the block is above threshold
                q_start = q_block * block_size_m
                q_end = min(seqlen_q, (q_block + 1) * block_size_m)
                k_start = k_block * block_size_n
                k_end = min(seqlen_k, (k_block + 1) * block_size_n)
                
                block_active = (dense_mask[q_start:q_end, k_start:k_end] > threshold).any().item()
                
                if block_active:
                    bit_idx = q_block * num_k_blocks + k_block
                    word_idx = bit_idx // 64
                    bit_offset = bit_idx % 64
                    bitset[word_idx] |= (1 << bit_offset)
        
        return cls(bitset, seqlen_q, seqlen_k, block_size_m, block_size_n, dense_mask.device)


class BCSRMask(SparseMask):
    """
    Block Compressed Sparse Row (BCSR) mask for irregular sparse patterns.
    
    Uses row pointers and column indices to represent sparse block patterns efficiently.
    """
    
    def __init__(self, row_ptr: torch.Tensor, col_idx: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.row_ptr = row_ptr.to(device=self.device, dtype=torch.int32)
        self.col_idx = col_idx.to(device=self.device, dtype=torch.int32)
        
        if self.row_ptr.numel() != self.num_query_blocks + 1:
            raise ValueError(f"row_ptr size mismatch: expected {self.num_query_blocks + 1}, got {self.row_ptr.numel()}")
    
    def get_mask_type(self) -> str:
        return "BCSR"
    
    def get_cuda_params(self) -> dict:
        return {
            "mask_type": 3,  # BCSR
            "mask_data": {
                "row_ptr": self.row_ptr.data_ptr(),
                "col_idx": self.col_idx.data_ptr(),
                "nnz_blocks": self.col_idx.numel()
            }
        }
    
    def estimate_memory_usage(self) -> int:
        return (self.row_ptr.numel() + self.col_idx.numel()) * 4  # 4 bytes per int32
    
    def is_block_active(self, query_block: int, key_block: int) -> bool:
        start = self.row_ptr[query_block].item()
        end = self.row_ptr[query_block + 1].item()
        
        for i in range(start, end):
            if self.col_idx[i].item() == key_block:
                return True
        return False
    
    @classmethod
    def from_dense_mask(cls, dense_mask: torch.Tensor, 
                       block_size_m: int = 128, 
                       block_size_n: int = 128, 
                       threshold: float = 0.0):
        """Create BCSRMask from dense attention mask."""
        seqlen_q, seqlen_k = dense_mask.shape
        num_q_blocks = (seqlen_q + block_size_m - 1) // block_size_m
        num_k_blocks = (seqlen_k + block_size_n - 1) // block_size_n
        
        row_ptr = torch.zeros(num_q_blocks + 1, dtype=torch.int32, device=dense_mask.device)
        col_indices = []
        
        for q_block in range(num_q_blocks):
            row_start = len(col_indices)
            
            for k_block in range(num_k_blocks):
                # Check if any element in the block is above threshold
                q_start = q_block * block_size_m
                q_end = min(seqlen_q, (q_block + 1) * block_size_m)
                k_start = k_block * block_size_n
                k_end = min(seqlen_k, (k_block + 1) * block_size_n)
                
                block_active = (dense_mask[q_start:q_end, k_start:k_end] > threshold).any().item()
                
                if block_active:
                    col_indices.append(k_block)
            
            row_ptr[q_block + 1] = len(col_indices)
        
        col_idx = torch.tensor(col_indices, dtype=torch.int32, device=dense_mask.device)
        return cls(row_ptr, col_idx, seqlen_q, seqlen_k, block_size_m, block_size_n, dense_mask.device)


class DynamicMask(BCSRMask):
    """
    Dynamic mask that can be updated at runtime.
    
    Uses BCSR format internally but allows for runtime updates
    of the sparse pattern based on attention scores or other criteria.
    """
    
    def get_mask_type(self) -> str:
        return "DYNAMIC"
    
    def get_cuda_params(self) -> dict:
        params = super().get_cuda_params()
        params["mask_type"] = 5  # DYNAMIC
        return params
    
    def update_from_scores(self, attention_scores: torch.Tensor, top_k: int):
        """Update the mask based on attention scores using top-k selection."""
        # Implementation for dynamic mask updates based on attention scores
        # This would be called during forward pass to adaptively prune attention
        pass


def create_sparse_mask(mask_type: str, **kwargs) -> SparseMask:
    """
    Factory function to create sparse masks.
    
    Args:
        mask_type: Type of mask ('causal', 'window', 'causal_window', 'bitset', 'bcsr', 'dynamic')
        **kwargs: Type-specific parameters
    
    Returns:
        SparseMask: Appropriate sparse mask implementation
    """
    if mask_type == "causal":
        return CausalMask(**kwargs)
    elif mask_type == "window":
        return WindowMask(**kwargs)
    elif mask_type == "causal_window":
        return CausalWindowMask(**kwargs)
    elif mask_type == "bitset":
        return BlockBitsetMask(**kwargs)
    elif mask_type == "bcsr":
        return BCSRMask(**kwargs)
    elif mask_type == "dynamic":
        return DynamicMask(**kwargs)
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")


def estimate_speedup(sparse_mask: SparseMask, skip_overhead_ratio: float = 0.01) -> float:
    """
    Estimate theoretical speedup from using sparse mask.
    
    Args:
        sparse_mask: Sparse mask to analyze
        skip_overhead_ratio: Ratio of time spent on skip logic vs computation
    
    Returns:
        float: Estimated speedup ratio
    """
    total_blocks = sparse_mask.num_query_blocks * sparse_mask.num_key_blocks
    active_blocks = sparse_mask.count_active_blocks()
    
    if active_blocks == 0:
        return 1.0
    
    active_fraction = active_blocks / total_blocks
    return 1.0 / (active_fraction + (1.0 - active_fraction) * skip_overhead_ratio)


def calculate_memory_savings(sparse_mask: SparseMask) -> float:
    """
    Calculate memory savings compared to dense mask.
    
    Args:
        sparse_mask: Sparse mask to analyze
    
    Returns:
        float: Memory savings ratio (0.0 to 1.0)
    """
    dense_memory = sparse_mask.seqlen_q * sparse_mask.seqlen_k * 4  # 4 bytes per float32
    compressed_memory = sparse_mask.estimate_memory_usage()
    return 1.0 - (compressed_memory / dense_memory) if dense_memory > 0 else 0.0