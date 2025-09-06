# Copyright (c) 2025, Jingze Shi.

import torch
from typing import Optional, Tuple, Union


class LinearKVCache:
    """
    Optimized KV cache for inference that maintains only keep_window_size tokens.
    
    During inference, since attention scores are static, evicted tokens will never
    be selected again. This allows us to maintain a fixed-size cache instead of
    growing indefinitely.
    """
    
    def __init__(
        self,
        keep_window_size: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
    ):
        """
        Initialize the linear KV cache.
        
        Args:
            keep_window_size: Maximum number of tokens to keep in cache
            num_heads: Number of attention heads
            head_dim: Dimension of each head
            dtype: Data type for cache tensors
            device: Device to store cache tensors
        """
        self.keep_window_size = keep_window_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        
        # Cache tensors [1, num_heads, keep_window_size, head_dim]
        self.key_cache = torch.zeros(
            1, num_heads, keep_window_size, head_dim,
            dtype=dtype, device=device
        )
        self.value_cache = torch.zeros(
            1, num_heads, keep_window_size, head_dim,
            dtype=dtype, device=device
        )
        
        # Track which cache positions are valid and their original sequence positions
        self.cache_valid = torch.zeros(keep_window_size, dtype=torch.bool, device=device)
        self.cache_positions = torch.full((keep_window_size,), -1, dtype=torch.long, device=device)
        self.current_length = 0
        self.next_position = 0  # Circular buffer position
        
        # Track importance scores for each cached token
        self.importance_scores = torch.full(
            (keep_window_size,), float('-inf'), dtype=dtype, device=device
        )
    
    def update(
        self,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
        new_scores: torch.Tensor,
        sequence_position: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key-value pairs and their importance scores.
        
        Args:
            new_keys: New key tensor [1, num_heads, 1, head_dim]
            new_values: New value tensor [1, num_heads, 1, head_dim]
            new_scores: Importance scores for the new token [1, num_heads]
            sequence_position: Original sequence position of the new token
            
        Returns:
            Tuple of (cached_keys, cached_values) for attention computation
        """
        # Average importance across heads for simplicity
        avg_score = new_scores.mean().item()
        
        if self.current_length < self.keep_window_size:
            # Cache not full, just add the new token
            pos = self.current_length
            self.key_cache[:, :, pos:pos+1, :] = new_keys
            self.value_cache[:, :, pos:pos+1, :] = new_values
            self.cache_valid[pos] = True
            self.cache_positions[pos] = sequence_position
            self.importance_scores[pos] = avg_score
            self.current_length += 1
        else:
            # Cache is full, need to decide whether to evict
            min_score_idx = torch.argmin(self.importance_scores)
            min_score = self.importance_scores[min_score_idx].item()
            
            if avg_score > min_score:
                # New token is more important, evict the least important
                pos = min_score_idx.item()
                self.key_cache[:, :, pos:pos+1, :] = new_keys
                self.value_cache[:, :, pos:pos+1, :] = new_values
                self.cache_positions[pos] = sequence_position
                self.importance_scores[pos] = avg_score
            # If new token is less important, it's discarded (cache unchanged)
        
        # Return the currently cached keys and values
        valid_positions = self.cache_valid[:self.current_length]
        return (
            self.key_cache[:, :, :self.current_length, :],
            self.value_cache[:, :, :self.current_length, :]
        )
    
    def get_cache_info(self) -> dict:
        """Get information about the current cache state."""
        return {
            'current_length': self.current_length,
            'cached_positions': self.cache_positions[:self.current_length].tolist(),
            'importance_scores': self.importance_scores[:self.current_length].tolist(),
            'capacity_utilization': self.current_length / self.keep_window_size,
        }
    
    def reset(self):
        """Reset the cache to empty state."""
        self.cache_valid.fill_(False)
        self.cache_positions.fill_(-1)
        self.importance_scores.fill_(float('-inf'))
        self.current_length = 0
        self.next_position = 0


def linear_kv_cache_attention(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_bias: torch.Tensor,
    cache: Optional[LinearKVCache] = None,
    keep_window_size: int = 2048,
    sequence_position: int = 0,
    inference_mode: bool = True,
) -> Tuple[torch.Tensor, Optional[LinearKVCache]]:
    """
    Perform attention with linear KV cache optimization for inference.
    
    Args:
        query_states: Query tensor [batch_size, num_heads, 1, head_dim] for inference
        key_states: Key tensor [batch_size, num_heads, seq_len, head_dim]
        value_states: Value tensor [batch_size, num_heads, seq_len, head_dim]
        attention_bias: Attention bias/scores [batch_size, num_heads, 1, seq_len]
        cache: Existing linear KV cache or None
        keep_window_size: Number of tokens to keep in cache
        sequence_position: Current sequence position
        inference_mode: Whether to use inference optimizations
        
    Returns:
        Tuple of (attention_output, updated_cache)
    """
    if not inference_mode or query_states.shape[-2] != 1:
        # Training mode or multi-token queries, use standard attention
        # Apply standard dynamic masking
        if attention_bias.shape[-1] > keep_window_size:
            topk_values, topk_indices = torch.topk(
                attention_bias, keep_window_size, dim=-1, largest=True, sorted=False
            )
            # Create attention mask
            attention_mask = torch.zeros_like(attention_bias)
            attention_mask.scatter_(-1, topk_indices, 1.0)
            
            # Apply mask to select relevant K, V
            expanded_mask = attention_mask.unsqueeze(-1)  # [batch, heads, 1, seq_len, 1]
            masked_keys = key_states * expanded_mask
            masked_values = value_states * expanded_mask
            
            # Compute attention normally
            scores = torch.matmul(query_states, masked_keys.transpose(-2, -1))
            scores = scores + attention_bias.masked_fill(attention_mask == 0, float('-inf'))
            attention_weights = torch.softmax(scores, dim=-1)
            attention_output = torch.matmul(attention_weights, masked_values)
        else:
            # Standard attention for short sequences
            scores = torch.matmul(query_states, key_states.transpose(-2, -1))
            scores = scores + attention_bias
            attention_weights = torch.softmax(scores, dim=-1)
            attention_output = torch.matmul(attention_weights, value_states)
        
        return attention_output, None
    
    # Inference mode with single query token
    batch_size, num_heads, _, head_dim = query_states.shape
    
    # Initialize cache if needed
    if cache is None:
        cache = LinearKVCache(
            keep_window_size=keep_window_size,
            num_heads=num_heads,
            head_dim=head_dim,
            dtype=key_states.dtype,
            device=key_states.device,
        )
    
    # Extract the new key-value pair (last token in sequence)
    new_key = key_states[:, :, -1:, :]  # [batch, heads, 1, head_dim]
    new_value = value_states[:, :, -1:, :]  # [batch, heads, 1, head_dim]
    new_score = attention_bias[:, :, :, -1:]  # [batch, heads, 1, 1]
    
    # Update cache with new token
    cached_keys, cached_values = cache.update(
        new_key, new_value, new_score.squeeze(-1), sequence_position
    )
    
    # Compute attention with cached K, V
    scores = torch.matmul(query_states, cached_keys.transpose(-2, -1))
    
    # Create appropriate bias for cached tokens
    valid_length = cache.current_length
    if valid_length > 0:
        # Get importance scores for cached tokens
        cached_bias = cache.importance_scores[:valid_length].unsqueeze(0).unsqueeze(0).unsqueeze(0).to(scores.dtype)
        scores = scores + cached_bias
    
    attention_weights = torch.softmax(scores, dim=-1)
    attention_output = torch.matmul(attention_weights, cached_values)
    
    return attention_output, cache