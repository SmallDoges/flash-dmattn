#!/usr/bin/env python3
"""
Example demonstrating variable-length batch inference with broadcastable key-based masks and bias.

This example shows how to use the new total_k-based mask/bias layout for efficient decoding
with variable-length sequences and large KV caches.
"""

import torch
from flash_dmattn import flash_dmattn_varlen_func


def create_varlen_sequences(batch_size, max_seqlen_q, max_seqlen_k):
    """Create variable-length sequences for demonstration."""
    import random
    
    # Generate random sequence lengths
    seqlens_q = [random.randint(1, max_seqlen_q) for _ in range(batch_size)]
    seqlens_k = [random.randint(max_seqlen_k // 2, max_seqlen_k) for _ in range(batch_size)]
    
    # Create cumulative sequence length tensors
    cu_seqlens_q = torch.tensor([0] + list(torch.tensor(seqlens_q).cumsum(0).tolist()), 
                                 dtype=torch.int32, device='cuda')
    cu_seqlens_k = torch.tensor([0] + list(torch.tensor(seqlens_k).cumsum(0).tolist()), 
                                 dtype=torch.int32, device='cuda')
    
    total_q = cu_seqlens_q[-1].item()
    total_k = cu_seqlens_k[-1].item()
    
    return cu_seqlens_q, cu_seqlens_k, total_q, total_k, max(seqlens_q), max(seqlens_k)


def example_k_based_mask_bias():
    """Example using key-based broadcastable mask and bias (NEW FEATURE)."""
    print("=" * 80)
    print("Example: Key-based Broadcastable Mask and Bias")
    print("=" * 80)
    
    # Configuration
    batch_size = 4
    max_seqlen_q = 8  # Short queries (typical for decoding)
    max_seqlen_k = 1024  # Large KV cache
    num_heads = 32
    num_heads_k = 8  # Grouped-query attention
    head_dim = 128
    
    print(f"Batch size: {batch_size}")
    print(f"Max query length: {max_seqlen_q}")
    print(f"Max key length: {max_seqlen_k}")
    print(f"Num query heads: {num_heads}")
    print(f"Num KV heads: {num_heads_k}")
    print(f"Head dimension: {head_dim}")
    print()
    
    # Create variable-length sequences
    cu_seqlens_q, cu_seqlens_k, total_q, total_k, actual_max_q, actual_max_k = \
        create_varlen_sequences(batch_size, max_seqlen_q, max_seqlen_k)
    
    print(f"Total query tokens: {total_q}")
    print(f"Total key tokens: {total_k}")
    print(f"Actual max query length: {actual_max_q}")
    print(f"Actual max key length: {actual_max_k}")
    print()
    
    # Create query, key, value tensors
    q = torch.randn(total_q, num_heads, head_dim, dtype=torch.float16, device='cuda')
    k = torch.randn(total_k, num_heads_k, head_dim, dtype=torch.float16, device='cuda')
    v = torch.randn(total_k, num_heads_k, head_dim, dtype=torch.float16, device='cuda')
    
    # KEY FEATURE: Key-based broadcastable mask and bias
    # Shape: (total_k, num_heads_k) - broadcasts across ALL query positions
    # This saves memory compared to (total_q, num_heads_k, max_seqlen_k)
    attn_mask = torch.randint(0, 2, (total_k, num_heads_k), dtype=torch.bool, device='cuda')
    attn_bias = torch.randn(total_k, num_heads_k, dtype=torch.float16, device='cuda')
    
    print(f"Query shape: {q.shape}")
    print(f"Key shape: {k.shape}")
    print(f"Value shape: {v.shape}")
    print(f"Mask shape (key-based): {attn_mask.shape}")
    print(f"Bias shape (key-based): {attn_bias.shape}")
    print()
    
    # Memory comparison
    q_based_elements = total_q * num_heads_k * actual_max_k
    k_based_elements = total_k * num_heads_k
    memory_saving = (1 - k_based_elements / q_based_elements) * 100
    
    print("Memory comparison:")
    print(f"  Query-based layout would need: {q_based_elements:,} elements")
    print(f"  Key-based layout needs: {k_based_elements:,} elements")
    print(f"  Memory savings: {memory_saving:.1f}%")
    print()
    
    # Call flash attention with automatic layout detection
    print("Running flash attention with key-based mask/bias...")
    output = flash_dmattn_varlen_func(
        query=q,
        key=k,
        value=v,
        attn_mask=attn_mask,  # Automatically detected as key-based
        attn_bias=attn_bias,  # Automatically detected as key-based
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=actual_max_q,
        max_seqlen_k=actual_max_k,
        softmax_scale=1.0 / (head_dim ** 0.5),
    )
    
    print(f"Output shape: {output.shape}")
    print("Success! ✓")
    print()


def example_q_based_mask_bias():
    """Example using traditional query-based mask and bias (for comparison)."""
    print("=" * 80)
    print("Example: Traditional Query-based Mask and Bias")
    print("=" * 80)
    
    # Configuration
    batch_size = 4
    max_seqlen_q = 8
    max_seqlen_k = 1024
    num_heads = 32
    num_heads_k = 8
    head_dim = 128
    
    # Create variable-length sequences
    cu_seqlens_q, cu_seqlens_k, total_q, total_k, actual_max_q, actual_max_k = \
        create_varlen_sequences(batch_size, max_seqlen_q, max_seqlen_k)
    
    print(f"Total query tokens: {total_q}")
    print(f"Total key tokens: {total_k}")
    print()
    
    # Create query, key, value tensors
    q = torch.randn(total_q, num_heads, head_dim, dtype=torch.float16, device='cuda')
    k = torch.randn(total_k, num_heads_k, head_dim, dtype=torch.float16, device='cuda')
    v = torch.randn(total_k, num_heads_k, head_dim, dtype=torch.float16, device='cuda')
    
    # Traditional query-based mask and bias
    # Shape: (total_q, num_heads_k, max_seqlen_k)
    attn_mask = torch.randint(0, 2, (total_q, num_heads_k, actual_max_k), 
                               dtype=torch.bool, device='cuda')
    attn_bias = torch.randn(total_q, num_heads_k, actual_max_k, 
                            dtype=torch.float16, device='cuda')
    
    print(f"Query shape: {q.shape}")
    print(f"Key shape: {k.shape}")
    print(f"Value shape: {v.shape}")
    print(f"Mask shape (query-based): {attn_mask.shape}")
    print(f"Bias shape (query-based): {attn_bias.shape}")
    print()
    
    # Call flash attention
    print("Running flash attention with query-based mask/bias...")
    output = flash_dmattn_varlen_func(
        query=q,
        key=k,
        value=v,
        attn_mask=attn_mask,
        attn_bias=attn_bias,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=actual_max_q,
        max_seqlen_k=actual_max_k,
        softmax_scale=1.0 / (head_dim ** 0.5),
    )
    
    print(f"Output shape: {output.shape}")
    print("Success! ✓")
    print()


def example_mixed_layouts():
    """Example using key-based mask with query-based bias (mixed layouts)."""
    print("=" * 80)
    print("Example: Mixed Layouts - Key-based Mask + Query-based Bias")
    print("=" * 80)
    
    # Configuration
    batch_size = 2
    max_seqlen_q = 4
    max_seqlen_k = 512
    num_heads = 16
    num_heads_k = 4
    head_dim = 64
    
    # Create variable-length sequences
    cu_seqlens_q, cu_seqlens_k, total_q, total_k, actual_max_q, actual_max_k = \
        create_varlen_sequences(batch_size, max_seqlen_q, max_seqlen_k)
    
    print(f"Total query tokens: {total_q}")
    print(f"Total key tokens: {total_k}")
    print()
    
    # Create query, key, value tensors
    q = torch.randn(total_q, num_heads, head_dim, dtype=torch.float16, device='cuda')
    k = torch.randn(total_k, num_heads_k, head_dim, dtype=torch.float16, device='cuda')
    v = torch.randn(total_k, num_heads_k, head_dim, dtype=torch.float16, device='cuda')
    
    # Key-based mask (broadcast across queries)
    attn_mask = torch.randint(0, 2, (total_k, num_heads_k), dtype=torch.bool, device='cuda')
    
    # Query-based bias (per-query values)
    attn_bias = torch.randn(total_q, num_heads_k, actual_max_k, dtype=torch.float16, device='cuda')
    
    print(f"Mask shape (key-based): {attn_mask.shape}")
    print(f"Bias shape (query-based): {attn_bias.shape}")
    print()
    
    # Call flash attention - each tensor uses its own layout
    print("Running flash attention with mixed layouts...")
    output = flash_dmattn_varlen_func(
        query=q,
        key=k,
        value=v,
        attn_mask=attn_mask,  # Key-based
        attn_bias=attn_bias,  # Query-based
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=actual_max_q,
        max_seqlen_k=actual_max_k,
        softmax_scale=1.0 / (head_dim ** 0.5),
    )
    
    print(f"Output shape: {output.shape}")
    print("Success! ✓")
    print()


def main():
    """Run all examples."""
    if not torch.cuda.is_available():
        print("CUDA is not available. These examples require a CUDA-capable GPU.")
        return
    
    print("Flash Dynamic Mask Attention - Variable Length Broadcastable Examples")
    print()
    
    # Example 1: Key-based mask and bias (NEW FEATURE)
    example_k_based_mask_bias()
    
    # Example 2: Traditional query-based (for comparison)
    example_q_based_mask_bias()
    
    # Example 3: Mixed layouts
    example_mixed_layouts()
    
    print("=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
