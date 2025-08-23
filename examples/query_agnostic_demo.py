#!/usr/bin/env python3
"""
Demonstration of Query-Agnostic Masking Behavior in Flash Dynamic Mask Attention

This script demonstrates how the current implementation applies the same mask
to all queries, showing both the benefits and limitations of this approach.
"""

import torch
import torch.nn.functional as F

def calculate_zoh_states(value_states, dt_proj, A):
    """Calculate ZOH states from value vectors only (query-agnostic)."""
    batch_size, _, key_len, _ = value_states.shape
    
    # Compute importance scores from Value vectors only
    dt_result = torch.matmul(
        value_states.transpose(-2, -3).reshape(batch_size, key_len, -1), 
        dt_proj.T
    )
    
    dt_states = torch.exp(F.softplus(dt_result) * A)
    return dt_states.transpose(-1, -2)

def prepare_dynamic_mask(query_states, zoh_states, keep_window_size=4):
    """Prepare dynamic mask - demonstrates query-agnostic behavior."""
    dtype = query_states.dtype
    device = query_states.device
    
    # Broadcast same ZOH scores to all queries
    attn_bias = zoh_states[:, :, None, :].expand(-1, -1, query_states.shape[2], -1)
    
    # TopK selection: same keys for all queries
    if attn_bias.shape[-1] > keep_window_size:
        topk_indices = torch.topk(attn_bias, keep_window_size, dim=-1, 
                                 largest=True, sorted=False).indices
        active_mask = torch.zeros_like(attn_bias, dtype=dtype, device=device)
        active_mask = active_mask.scatter(-1, topk_indices, 1.0)
    else:
        active_mask = torch.ones_like(attn_bias, dtype=dtype, device=device)
    
    return attn_bias, active_mask, topk_indices

def main():
    print("=" * 70)
    print("Flash Dynamic Mask Attention: Query-Agnostic Behavior Demonstration")
    print("=" * 70)
    
    # Setup simple example
    batch_size, num_heads, seq_len, head_dim = 1, 2, 8, 4
    keep_window_size = 4
    device = 'cpu'
    
    # Create example data with clear patterns
    torch.manual_seed(42)
    
    # Values with clear importance pattern: positions 1, 3, 5, 7 are "important"
    value_states = torch.zeros(batch_size, num_heads, seq_len, head_dim)
    value_states[:, :, [1, 3, 5, 7], :] = 1.0  # Important positions
    value_states[:, :, [0, 2, 4, 6], :] = 0.1  # Less important positions
    
    # Queries with different "intentions"
    query_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
    query_states[:, :, 0, :] = torch.tensor([1.0, 0.0, 0.0, 0.0])  # Query 0: looking for pattern A
    query_states[:, :, 1, :] = torch.tensor([0.0, 1.0, 0.0, 0.0])  # Query 1: looking for pattern B
    query_states[:, :, 2, :] = torch.tensor([0.0, 0.0, 1.0, 0.0])  # Query 2: looking for pattern C
    
    # Learned parameters (simplified)
    dt_proj = torch.ones(num_heads, num_heads * head_dim) * 0.5
    A = torch.ones(num_heads)
    
    print(f"Sequence length: {seq_len}")
    print(f"Keep window size: {keep_window_size}")
    print(f"Important value positions: [1, 3, 5, 7]")
    print(f"Less important positions: [0, 2, 4, 6]")
    print()
    
    # Calculate ZOH states (value-based importance)
    zoh_states = calculate_zoh_states(value_states, dt_proj, A)
    print("ZOH States (Value-based importance scores):")
    print(f"Shape: {zoh_states.shape}")  # [batch, heads, key_len]
    print("Head 0:", zoh_states[0, 0].round(decimals=3).tolist())
    print("Head 1:", zoh_states[0, 1].round(decimals=3).tolist())
    print()
    
    # Generate dynamic mask
    attn_bias, active_mask, topk_indices = prepare_dynamic_mask(
        query_states, zoh_states, keep_window_size
    )
    
    print("TopK Selected Keys (same for ALL queries):")
    print("Head 0:", topk_indices[0, 0, 0].tolist())  # Same for all queries
    print("Head 1:", topk_indices[0, 1, 0].tolist())  # Same for all queries
    print()
    
    print("Active Mask Verification (1.0 = attend, 0.0 = ignore):")
    for head in range(num_heads):
        print(f"\nHead {head}:")
        print("Query positions -> Key positions:")
        for query in range(min(3, seq_len)):  # Show first 3 queries
            mask_row = active_mask[0, head, query].tolist()
            attended_keys = [i for i, val in enumerate(mask_row) if val == 1.0]
            print(f"  Query {query}: attends to keys {attended_keys}")
    
    print("\n" + "=" * 50)
    print("KEY OBSERVATIONS:")
    print("=" * 50)
    print("1. ZOH states computed from Values only (no Query information)")
    print("2. Same TopK keys selected for ALL queries")
    print("3. Query intentions (patterns A, B, C) are ignored in key selection")
    print("4. Computational efficiency: O(N) mask generation vs O(N²) for query-aware")
    print("5. Trade-off: efficiency vs. query-specific precision")
    
    print("\n" + "=" * 50)
    print("IMPLICATIONS FOR ASSOCIATIVE RECALL:")
    print("=" * 50)
    print("✅ Works well when:")
    print("   - Important information is globally relevant")
    print("   - Document has clear hierarchical structure")
    print("   - Similar information needs across queries")
    
    print("\n❌ Limitations for:")
    print("   - Fine-grained query-specific retrieval")
    print("   - Tasks requiring different keys per query")
    print("   - Precise associative recall ('What did Alice say about X?')")
    
    print("\n💡 Potential improvements:")
    print("   - Larger keep_window_size for more coverage")
    print("   - Query-conditioned importance scoring")
    print("   - Multi-stage selection (global + query-specific)")

if __name__ == "__main__":
    main()