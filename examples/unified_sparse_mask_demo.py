#!/usr/bin/env python3
"""
Example demonstrating the Unified Sparse Mask Strategy with Block-Level Skipping

This example shows how to use different sparse mask types with Flash Dynamic Mask Attention
to achieve memory efficiency and computational speedup for long sequences.
"""

import torch
import numpy as np
import math
from typing import Optional

# Import the sparse mask API (when available)
try:
    from flash_dmattn import (
        CausalMask, WindowMask, CausalWindowMask, BlockBitsetMask, BCSRMask,
        create_sparse_mask, estimate_speedup, calculate_memory_savings,
        flash_dmattn_func_auto, get_available_backends
    )
    SPARSE_MASK_AVAILABLE = True
except ImportError:
    print("Warning: Sparse mask API not available. Install flash-dmattn with CUDA support.")
    SPARSE_MASK_AVAILABLE = False


def create_sample_inputs(batch_size: int = 1, 
                        seq_len: int = 4096, 
                        num_heads: int = 8, 
                        num_kv_heads: int = 8,
                        head_dim: int = 64,
                        device: str = 'cuda',
                        dtype: torch.dtype = torch.bfloat16):
    """Create sample Q, K, V tensors for testing."""
    
    query = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
    
    return query, key, value


def demonstrate_causal_mask(seq_len: int = 2048):
    """Demonstrate causal mask usage."""
    print(f"\n=== Causal Mask Demo (seq_len={seq_len}) ===")
    
    # Create causal mask
    causal_mask = CausalMask(seqlen_q=seq_len, seqlen_k=seq_len)
    
    print(f"Mask type: {causal_mask.get_mask_type()}")
    print(f"Memory usage: {causal_mask.estimate_memory_usage()} bytes")
    print(f"Sparsity ratio: {causal_mask.get_sparsity_ratio():.2%}")
    print(f"Active blocks: {causal_mask.count_active_blocks()}/{causal_mask.num_query_blocks * causal_mask.num_key_blocks}")
    
    # Estimate performance benefits
    speedup = estimate_speedup(causal_mask)
    memory_savings = calculate_memory_savings(causal_mask)
    print(f"Estimated speedup: {speedup:.2f}x")
    print(f"Memory savings: {memory_savings:.2%}")
    
    return causal_mask


def demonstrate_window_mask(seq_len: int = 4096, window_size: int = 512):
    """Demonstrate sliding window mask usage."""
    print(f"\n=== Window Mask Demo (seq_len={seq_len}, window={window_size}) ===")
    
    # Create window mask
    window_mask = WindowMask(window_size=window_size, seqlen_q=seq_len, seqlen_k=seq_len)
    
    print(f"Mask type: {window_mask.get_mask_type()}")
    print(f"Memory usage: {window_mask.estimate_memory_usage()} bytes")
    print(f"Sparsity ratio: {window_mask.get_sparsity_ratio():.2%}")
    print(f"Active blocks: {window_mask.count_active_blocks()}/{window_mask.num_query_blocks * window_mask.num_key_blocks}")
    
    # Estimate performance benefits
    speedup = estimate_speedup(window_mask)
    memory_savings = calculate_memory_savings(window_mask)
    print(f"Estimated speedup: {speedup:.2f}x")
    print(f"Memory savings: {memory_savings:.2%}")
    
    return window_mask


def demonstrate_hybrid_mask(seq_len: int = 8192, window_size: int = 1024):
    """Demonstrate hybrid causal + window mask usage."""
    print(f"\n=== Causal+Window Mask Demo (seq_len={seq_len}, window={window_size}) ===")
    
    # Create hybrid mask
    hybrid_mask = CausalWindowMask(window_size=window_size, seqlen_q=seq_len, seqlen_k=seq_len)
    
    print(f"Mask type: {hybrid_mask.get_mask_type()}")
    print(f"Memory usage: {hybrid_mask.estimate_memory_usage()} bytes")
    print(f"Sparsity ratio: {hybrid_mask.get_sparsity_ratio():.2%}")
    print(f"Active blocks: {hybrid_mask.count_active_blocks()}/{hybrid_mask.num_query_blocks * hybrid_mask.num_key_blocks}")
    
    # Estimate performance benefits
    speedup = estimate_speedup(hybrid_mask)
    memory_savings = calculate_memory_savings(hybrid_mask)
    print(f"Estimated speedup: {speedup:.2f}x")
    print(f"Memory savings: {memory_savings:.2%}")
    
    return hybrid_mask


def demonstrate_block_bitset_mask(seq_len: int = 4096):
    """Demonstrate block bitset mask usage."""
    print(f"\n=== Block Bitset Mask Demo (seq_len={seq_len}) ===")
    
    # Create a random sparse pattern
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dense_mask = torch.rand(seq_len, seq_len, device=device)
    
    # Create different sparsity patterns
    sparsity_levels = [0.5, 0.7, 0.9]
    
    for sparsity in sparsity_levels:
        # Apply sparsity threshold
        threshold = torch.quantile(dense_mask, sparsity)
        sparse_pattern = (dense_mask > threshold).float()
        
        # Convert to block bitset mask
        bitset_mask = BlockBitsetMask.from_dense_mask(sparse_pattern)
        
        print(f"\nSparsity level: {sparsity:.1%}")
        print(f"  Mask type: {bitset_mask.get_mask_type()}")
        print(f"  Memory usage: {bitset_mask.estimate_memory_usage()} bytes")
        print(f"  Actual sparsity: {bitset_mask.get_sparsity_ratio():.2%}")
        print(f"  Active blocks: {bitset_mask.count_active_blocks()}/{bitset_mask.num_query_blocks * bitset_mask.num_key_blocks}")
        
        # Estimate performance benefits
        speedup = estimate_speedup(bitset_mask)
        memory_savings = calculate_memory_savings(bitset_mask)
        print(f"  Estimated speedup: {speedup:.2f}x")
        print(f"  Memory savings: {memory_savings:.2%}")


def demonstrate_bcsr_mask(seq_len: int = 4096):
    """Demonstrate BCSR mask usage."""
    print(f"\n=== BCSR Mask Demo (seq_len={seq_len}) ===")
    
    # Create a block-diagonal sparse pattern
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dense_mask = torch.zeros(seq_len, seq_len, device=device)
    
    # Create block-diagonal pattern with some random blocks
    block_size = 128
    num_blocks = seq_len // block_size
    
    # Add diagonal blocks
    for i in range(num_blocks):
        start = i * block_size
        end = min((i + 1) * block_size, seq_len)
        dense_mask[start:end, start:end] = 1.0
    
    # Add some off-diagonal blocks randomly
    for _ in range(num_blocks // 4):
        i = torch.randint(0, num_blocks, (1,)).item()
        j = torch.randint(0, num_blocks, (1,)).item()
        i_start, i_end = i * block_size, min((i + 1) * block_size, seq_len)
        j_start, j_end = j * block_size, min((j + 1) * block_size, seq_len)
        dense_mask[i_start:i_end, j_start:j_end] = 1.0
    
    # Convert to BCSR mask
    bcsr_mask = BCSRMask.from_dense_mask(dense_mask)
    
    print(f"Mask type: {bcsr_mask.get_mask_type()}")
    print(f"Memory usage: {bcsr_mask.estimate_memory_usage()} bytes")
    print(f"Sparsity ratio: {bcsr_mask.get_sparsity_ratio():.2%}")
    print(f"Active blocks: {bcsr_mask.count_active_blocks()}/{bcsr_mask.num_query_blocks * bcsr_mask.num_key_blocks}")
    print(f"NNZ blocks: {bcsr_mask.col_idx.numel()}")
    
    # Estimate performance benefits
    speedup = estimate_speedup(bcsr_mask)
    memory_savings = calculate_memory_savings(bcsr_mask)
    print(f"Estimated speedup: {speedup:.2f}x")
    print(f"Memory savings: {memory_savings:.2%}")
    
    return bcsr_mask


def benchmark_attention_with_masks():
    """Benchmark attention computation with different sparse masks."""
    print(f"\n=== Attention Benchmarking ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available - skipping attention benchmarks")
        return
    
    # Setup
    batch_size, seq_len, num_heads, head_dim = 1, 4096, 8, 64
    device = torch.device('cuda')
    dtype = torch.bfloat16
    
    query, key, value = create_sample_inputs(
        batch_size, seq_len, num_heads, num_heads, head_dim, device, dtype
    )
    
    print(f"Tensor shapes: Q={query.shape}, K={key.shape}, V={value.shape}")
    
    # Test different mask types
    masks_to_test = [
        ("No Mask", None),
        ("Causal", CausalMask(seq_len, seq_len)),
        ("Window-512", WindowMask(512, seq_len, seq_len)),
        ("Causal+Window-1024", CausalWindowMask(1024, seq_len, seq_len)),
    ]
    
    if 'cuda' in get_available_backends():
        flash_attn_func = flash_dmattn_func_auto(backend='cuda')
        
        for mask_name, sparse_mask in masks_to_test:
            print(f"\nTesting {mask_name}:")
            
            # Prepare mask parameters
            attn_mask = None
            attn_bias = None
            sparse_mask_params = None
            
            if sparse_mask is not None:
                if hasattr(sparse_mask, 'get_cuda_params'):
                    sparse_mask_params = sparse_mask.get_cuda_params()
                    print(f"  Sparsity: {sparse_mask.get_sparsity_ratio():.1%}")
                    print(f"  Expected speedup: {estimate_speedup(sparse_mask):.2f}x")
            
            # Note: The actual CUDA kernel integration would happen here
            # For now, we just demonstrate the API
            print(f"  Mask type: {sparse_mask.get_mask_type() if sparse_mask else 'Dense'}")
            print(f"  Memory usage: {sparse_mask.estimate_memory_usage() if sparse_mask else 'Full'} bytes")
    else:
        print("CUDA backend not available - skipping kernel tests")


def main():
    """Main demonstration function."""
    print("Flash Dynamic Mask Attention - Unified Sparse Mask Strategy Demo")
    print("=" * 70)
    
    if not SPARSE_MASK_AVAILABLE:
        print("Sparse mask API not available. Please install flash-dmattn with CUDA support.")
        return
    
    print(f"Available backends: {get_available_backends()}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Demonstrate different mask types
    demonstrate_causal_mask(2048)
    demonstrate_window_mask(4096, 512)
    demonstrate_hybrid_mask(8192, 1024)
    demonstrate_block_bitset_mask(4096)
    demonstrate_bcsr_mask(4096)
    
    # Benchmark with actual attention computation
    benchmark_attention_with_masks()
    
    print("\n" + "=" * 70)
    print("Demo completed! The unified sparse mask strategy enables:")
    print("• Memory-efficient representation of sparse attention patterns")
    print("• Block-level computation skipping for significant speedups")
    print("• Support for parametric, bitset, and BCSR mask formats")
    print("• Seamless integration with Flash Attention kernels")
    print("• Automatic fallback to dense computation when needed")


if __name__ == "__main__":
    main()