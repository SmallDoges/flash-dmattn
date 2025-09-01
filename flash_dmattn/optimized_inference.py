# Copyright (c) 2025, Jingze Shi.

import torch
from typing import Optional

from .kv_cache_optimizer import LinearKVCache, linear_kv_cache_attention


def calculate_zoh_states(value_states, dt_proj, A):
    """Calculate ZOH states for dynamic mask attention."""
    # This is a placeholder - in the real implementation, this would be more complex
    # For now, just return random importance scores
    batch_size, num_heads, seq_len, head_dim = value_states.shape
    return torch.randn(batch_size, num_heads, seq_len, device=value_states.device, dtype=value_states.dtype)


# Optimized inference function using linear KV cache
def dynamic_mask_attention_cuda_optimized(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    dt_proj: torch.Tensor,
    A: torch.Tensor,
    scaling: float,
    cache_position: torch.Tensor,
    kv_cache: Optional[LinearKVCache] = None,
    keep_window_size=2048,
    is_causal=True,
    inference_mode=True,
):
    """
    Optimized CUDA implementation of dynamic mask attention for inference.
    
    This version uses linear KV cache optimization to maintain only
    keep_window_size tokens instead of growing cache indefinitely.
    
    Args:
        query_states: [batch_size, num_heads, query_len, head_dim]
        key_states: [batch_size, num_kv_heads, key_len, head_dim]  
        value_states: [batch_size, num_kv_heads, key_len, head_dim]
        dt_proj: [num_kv_heads, num_kv_heads * head_dim]
        A: [num_kv_heads]
        scaling: Attention scaling factor
        cache_position: Cache position for causal masking
        kv_cache: Existing LinearKVCache or None
        keep_window_size: Number of tokens to keep in attention window
        is_causal: Whether to apply causal masking
        inference_mode: Whether to use inference optimizations
    
    Returns:
        (attn_outputs, updated_cache): Attention outputs and updated cache
    """
    # Calculate zoh_states for the new token(s)
    zoh_states = calculate_zoh_states(value_states, dt_proj, A)
    
    # For inference, we typically process one token at a time
    # Extract the new token's attention bias
    new_bias = zoh_states[:, :, None, -1:]  # [batch, num_kv_heads, 1, 1]
    
    # Use the optimized linear KV cache attention
    attn_outputs, updated_cache = linear_kv_cache_attention(
        query_states,
        key_states,
        value_states,
        new_bias,
        cache=kv_cache,
        keep_window_size=keep_window_size,
        sequence_position=cache_position.item() if cache_position is not None else 0,
        inference_mode=inference_mode,
    )
    
    return attn_outputs, updated_cache