#!/usr/bin/env python3
"""
Test script for linear KV cache optimization during inference.
"""

import torch
import torch.nn.functional as F
import time
import sys
import os

# Add the flash_dmattn module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flash_dmattn.kv_cache_optimizer import LinearKVCache, linear_kv_cache_attention


def standard_attention_with_topk(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_bias: torch.Tensor,
    keep_window_size: int = 2048,
) -> torch.Tensor:
    """Standard attention with TopK masking (current implementation)."""
    if attention_bias.shape[-1] > keep_window_size:
        topk_values, topk_indices = torch.topk(
            attention_bias, keep_window_size, dim=-1, largest=True, sorted=False
        )
        attention_mask = torch.zeros_like(attention_bias)
        attention_mask.scatter_(-1, topk_indices, 1.0)
        
        # Apply mask
        masked_bias = attention_bias.masked_fill(attention_mask == 0, float('-inf'))
    else:
        masked_bias = attention_bias
    
    # Compute attention
    scores = torch.matmul(query_states, key_states.transpose(-2, -1))
    scores = scores + masked_bias
    attention_weights = torch.softmax(scores, dim=-1)
    attention_output = torch.matmul(attention_weights, value_states)
    
    return attention_output


def create_test_tensors(batch_size, num_heads, seq_len, head_dim, device):
    """Create test tensors for attention computation."""
    # Create query (single token for inference)
    query = torch.randn(batch_size, num_heads, 1, head_dim, device=device, dtype=torch.bfloat16)
    
    # Create full key and value sequences
    keys = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    values = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    
    # Create attention bias (importance scores)
    # Simulate realistic importance scores with some high-importance tokens
    bias = torch.randn(batch_size, num_heads, 1, seq_len, device=device, dtype=torch.bfloat16)
    
    # Make some tokens clearly more important
    important_positions = torch.randperm(seq_len)[:seq_len//4]  # 25% of tokens are important
    bias[:, :, :, important_positions] += 2.0  # Boost important tokens
    
    return query, keys, values, bias


def test_correctness():
    """Test that the linear KV cache produces similar results to standard attention."""
    print("Testing correctness...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size, num_heads, seq_len, head_dim = 1, 8, 4096, 64
    keep_window_size = 512
    
    # Create test data
    query, keys, values, bias = create_test_tensors(batch_size, num_heads, seq_len, head_dim, device)
    
    # Standard attention with TopK
    standard_output = standard_attention_with_topk(query, keys, values, bias, keep_window_size)
    
    # Optimized attention with linear KV cache
    # Simulate inference by processing tokens sequentially
    cache = None
    for i in range(seq_len):
        # Current query
        current_query = query
        
        # Keys and values up to current position
        current_keys = keys[:, :, :i+1, :]
        current_values = values[:, :, :i+1, :]
        current_bias = bias[:, :, :, :i+1]
        
        optimized_output, cache = linear_kv_cache_attention(
            current_query, current_keys, current_values, current_bias,
            cache=cache, keep_window_size=keep_window_size, 
            sequence_position=i, inference_mode=True
        )
    
    # Compare outputs (they won't be identical due to different token selection strategies)
    cosine_sim = F.cosine_similarity(
        standard_output.flatten(), optimized_output.flatten(), dim=0
    ).item()
    
    print(f"Cosine similarity between standard and optimized: {cosine_sim:.4f}")
    print(f"Standard output norm: {standard_output.norm().item():.4f}")
    print(f"Optimized output norm: {optimized_output.norm().item():.4f}")
    
    if cache is not None:
        cache_info = cache.get_cache_info()
        print(f"Final cache state: {cache_info}")
    
    print("Correctness test completed.\n")


def test_memory_efficiency():
    """Test memory efficiency of the linear KV cache."""
    print("Testing memory efficiency...")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test.\n")
        return
    
    device = torch.device('cuda')
    batch_size, num_heads, head_dim = 1, 32, 128
    keep_window_size = 2048
    
    def measure_memory():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        return torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()
    
    # Test with different sequence lengths
    seq_lengths = [4096, 8192, 16384, 32768]
    
    for seq_len in seq_lengths:
        print(f"\nSequence length: {seq_len}")
        
        # Standard approach - maintain full KV cache
        query, keys, values, bias = create_test_tensors(batch_size, num_heads, seq_len, head_dim, device)
        
        start_mem, _ = measure_memory()
        standard_output = standard_attention_with_topk(query, keys, values, bias, keep_window_size)
        torch.cuda.synchronize()
        end_mem, peak_mem = measure_memory()
        
        standard_memory = peak_mem - start_mem
        print(f"Standard memory usage: {standard_memory / 1e6:.2f} MB")
        
        # Optimized approach - linear KV cache
        del query, keys, values, bias, standard_output
        torch.cuda.empty_cache()
        
        query, keys, values, bias = create_test_tensors(batch_size, num_heads, seq_len, head_dim, device)
        
        start_mem, _ = measure_memory()
        cache = None
        for i in range(min(seq_len, 1000)):  # Simulate first 1000 tokens of inference
            current_query = query
            current_keys = keys[:, :, i:i+1, :]
            current_values = values[:, :, i:i+1, :]
            current_bias = bias[:, :, :, i:i+1]
            
            optimized_output, cache = linear_kv_cache_attention(
                current_query, current_keys, current_values, current_bias,
                cache=cache, keep_window_size=keep_window_size,
                sequence_position=i, inference_mode=True
            )
        torch.cuda.synchronize()
        end_mem, peak_mem = measure_memory()
        
        optimized_memory = peak_mem - start_mem
        print(f"Optimized memory usage: {optimized_memory / 1e6:.2f} MB")
        print(f"Memory reduction: {(1 - optimized_memory / standard_memory) * 100:.1f}%")
        
        del query, keys, values, bias, optimized_output, cache
    
    print("\nMemory efficiency test completed.\n")


def test_performance():
    """Test performance of the linear KV cache during inference simulation."""
    print("Testing performance...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size, num_heads, head_dim = 1, 32, 128
    keep_window_size = 2048
    num_inference_steps = 1000
    
    # Create base tensors
    query = torch.randn(batch_size, num_heads, 1, head_dim, device=device, dtype=torch.bfloat16)
    
    def simulate_inference_standard():
        """Simulate standard inference (growing KV cache)."""
        total_time = 0
        for step in range(num_inference_steps):
            seq_len = step + 1
            keys = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
            values = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
            bias = torch.randn(batch_size, num_heads, 1, seq_len, device=device, dtype=torch.bfloat16)
            
            start_time = time.time()
            output = standard_attention_with_topk(query, keys, values, bias, keep_window_size)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            
            total_time += (end_time - start_time)
        return total_time
    
    def simulate_inference_optimized():
        """Simulate optimized inference (linear KV cache)."""
        total_time = 0
        cache = None
        for step in range(num_inference_steps):
            # New token
            new_key = torch.randn(batch_size, num_heads, 1, head_dim, device=device, dtype=torch.bfloat16)
            new_value = torch.randn(batch_size, num_heads, 1, head_dim, device=device, dtype=torch.bfloat16)
            new_bias = torch.randn(batch_size, num_heads, 1, 1, device=device, dtype=torch.bfloat16)
            
            start_time = time.time()
            output, cache = linear_kv_cache_attention(
                query, new_key, new_value, new_bias,
                cache=cache, keep_window_size=keep_window_size,
                sequence_position=step, inference_mode=True
            )
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            
            total_time += (end_time - start_time)
        return total_time
    
    # Warmup
    print("Warming up...")
    for _ in range(10):
        simulate_inference_standard()
        simulate_inference_optimized()
    
    # Benchmark
    print("Running benchmarks...")
    standard_time = simulate_inference_standard()
    optimized_time = simulate_inference_optimized()
    
    print(f"Standard inference time: {standard_time:.4f}s")
    print(f"Optimized inference time: {optimized_time:.4f}s")
    print(f"Speedup: {standard_time / optimized_time:.2f}x")
    
    print("Performance test completed.\n")


def main():
    """Run all tests."""
    print("Linear KV Cache Optimization Tests")
    print("=" * 50)
    
    test_correctness()
    test_memory_efficiency()
    test_performance()
    
    print("All tests completed!")


if __name__ == "__main__":
    main()