#!/usr/bin/env python3
"""
Performance Benchmark for Dynamic Mask Attention

This script measures and compares the performance of multiple Dynamic Mask Attention 
implementations against SDPA baseline across various configurations.

Implementations tested:
- PyTorch SDPA - Baseline
- Dynamic Mask Attention CUDA - Custom CUDA kernel implementation  
- Dynamic Mask Attention Triton - Triton kernel implementation
- Dynamic Mask Attention Flex - Flex Attention implementation

Benchmark includes:
- Multiple sequence lengths and batch sizes
- Head count and dimension variations
- Throughput and latency measurements
- Memory usage analysis
- Speedup comparisons across all implementations
"""

import argparse
import gc
import os
import sys
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    import colorama

    colorama.just_fix_windows_console()
except ImportError:  # pragma: no cover - optional dependency
    colorama = None



# Import the compiled CUDA extension
try:
    from flash_dmattn.flash_dmattn_interface import flash_dmattn_func
    print("✅ Successfully imported flash_dmattn interface")
except ImportError as e:
    print(f"❌ Failed to import flash_dmattn interface: {e}")
    print("Please make sure the package is properly installed with: pip install .")
    # Don't exit here, just warn
    flash_dmattn_func = None

# Import the Triton implementation
try:
    from flash_dmattn.flash_dmattn_triton import triton_dmattn_func
    print("✅ Successfully imported flash_dmattn_triton")
except ImportError as e:
    print(f"❌ Failed to import flash_dmattn_triton: {e}")
    print("Please make sure the Triton implementation is available.")
    # Don't exit here, just warn
    triton_dmattn_func = None

# Import the Flex Attention implementation
try:
    from flash_dmattn.flash_dmattn_flex import flex_dmattn_func
    print("✅ Successfully imported flash_dmattn_flex")
except ImportError as e:
    print(f"❌ Failed to import flash_dmattn_flex: {e}")
    print("Please make sure the Flex Attention implementation is available.")
    # Don't exit here, just warn
    flex_dmattn_func = None


ANSI_RESET = "\033[0m"
ANSI_CODES = {
    "green": "32",
    "yellow": "33",
    "red": "31",
    "blue": "34",
    "magenta": "35",
    "cyan": "36",
}


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    stream = getattr(sys, "stdout", None)
    return bool(stream and stream.isatty())


ENABLE_COLOR = _supports_color()


def style_text(text: str, color: Optional[str] = None, bold: bool = False) -> str:
    """Apply optional ANSI styling if the current environment supports color."""

    if not ENABLE_COLOR or (color is None and not bold):
        return text

    codes = []
    if bold:
        codes.append("1")
    if color and color in ANSI_CODES:
        codes.append(ANSI_CODES[color])

    if not codes:
        return text

    return f"\033[{';'.join(codes)}m{text}{ANSI_RESET}"


def format_metric_cell(status: str, value: str, success_color: str = "cyan") -> str:
    """Color-code metric cells based on status."""

    if status == "success":
        return style_text(value, success_color)

    if status in {"OOM", "Not Available"}:
        return style_text(value, "red")

    if status in {"N/A"}:
        return style_text(value, "yellow")

    return style_text(value, "yellow")


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). 
    Transform from (batch, num_key_value_heads, seqlen, head_dim) 
    to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def prepare_dynamic_mask(
    hidden_states: torch.Tensor,
    zoh_states: torch.Tensor,
    keep_window_size: int = 2048,
    cache_position: torch.Tensor = None,
):
    """
    Calculate dynamic attention mask to mask tokens for sparse attention.

    Combine `zoh_states` with `attention_mask` to generate the final `attn_mask`.

    Args:
        hidden_states: Input hidden states to determine dtype minimum value
        zoh_states: zoh_states of shape (batch_size, num_kv_heads, key_sequence_length)
        keep_window_size: Window size of tokens not dynamically masked
        cache_position: Optional cache position for causal masking
    
    Returns:
        tuple: (attn_bias, attn_mask)
    """
    dtype = hidden_states.dtype
    min_dtype = torch.finfo(dtype).min
    attn_bias = zoh_states[:, :, None, :].expand(
        -1, -1, hidden_states.shape[2], -1
    ).to(dtype)     # [batch_size, num_kv_heads, query_len, key_len]
    
    if cache_position is not None:
        attn_bias = attn_bias.masked_fill(
            torch.arange(attn_bias.shape[-1], device=attn_bias.device) > cache_position.reshape(-1, 1),
            min_dtype
        )

    if attn_bias.shape[-1] > keep_window_size:
        topk_values, topk_indices = torch.topk(
            attn_bias, keep_window_size, dim=-1, largest=True, sorted=False
        )
        valid_topk = topk_values != min_dtype
        attn_mask = torch.zeros_like(attn_bias, dtype=torch.bool, device=attn_bias.device)
        attn_mask = attn_mask.scatter(-1, topk_indices, valid_topk)
        attn_bias = attn_bias.masked_fill(~attn_mask, min_dtype)
    else:
        attn_mask = torch.ones_like(attn_bias, dtype=torch.bool, device=attn_bias.device)
    return attn_bias, attn_mask


def calculate_zoh_states(value_states, dt_proj, A):
    """
    Calculate zoh states for dynamic mask attention.
    
    Args:
        value_states: [batch_size, num_kv_heads, key_len, head_dim]
        dt_proj: [num_kv_heads, num_kv_heads * head_dim]
        A: [num_kv_heads]
        causal_mask: Optional causal mask
    
    Returns:
        zoh_states: [batch_size, num_kv_heads, key_len]
    """
    batch_size, _, key_len, _ = value_states.shape
    
    # Transpose and reshape value_states, then matrix multiply with dt_proj.T
    dt_result = torch.matmul(
        value_states.transpose(-2, -3).reshape(batch_size, key_len, -1), 
        dt_proj.T
    )
    
    # Apply softplus activation and coefficient A
    dt_states = torch.exp(F.softplus(dt_result) * A)
    zoh_states = dt_states.transpose(-1, -2)  # [batch_size, num_kv_heads, key_len]

    return zoh_states


def scaled_dot_product_attention_cuda(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    scaling: float,
    causal_mask: torch.Tensor,
    is_causal=True,
):
    """
    CUDA implementation of SDPA baseline.
    
    Args:
        query_states: [batch_size, num_heads, query_len, head_dim]
        key_states: [batch_size, num_kv_heads, key_len, head_dim]
        value_states: [batch_size, num_kv_heads, key_len, head_dim]
        scaling: Attention scaling factor
        causal_mask: Causal attention mask
        is_causal: Whether to apply causal masking
    
    Returns:
        attn_outputs or "OOM" if out of memory
    """
    _, _, query_len, _ = query_states.shape
    _, _, key_len, _ = key_states.shape
    if query_len > 32768 and key_len > 32768:
        return "OOM"

    query_states = query_states.contiguous()
    key_states = key_states.contiguous()
    value_states = value_states.contiguous()

    try:
        # Only measure the core attention computation
        torch.cuda.synchronize()
        start_time = time.time()

        attn_outputs = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            scale=scaling,
            # is_causal=is_causal if query_len == key_len else False,
            enable_gqa=True
        )
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        attn_outputs = attn_outputs.transpose(1, 2).contiguous()  # Transpose to [batch, query_len, num_heads, head_dim]
        return attn_outputs, (end_time - start_time) * 1000  # Return output and time in ms
    except torch.cuda.OutOfMemoryError:
        return "OOM", 0


def dynamic_mask_attention_cuda(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    dt_proj: torch.Tensor,
    A: torch.Tensor,
    scaling: float,
    cache_position: torch.Tensor,
    keep_window_size=2048,
    is_causal=True,
    return_softmax=False
):
    """
    CUDA implementation of dynamic mask attention.
    
    Args:
        query_states: [batch_size, num_heads, query_len, head_dim]
        key_states: [batch_size, num_kv_heads, key_len, head_dim]
        value_states: [batch_size, num_kv_heads, key_len, head_dim]
        dt_proj: [num_kv_heads, num_kv_heads * head_dim]
        A: [num_kv_heads]
        scaling: Attention scaling factor
        cache_position: Cache position for causal masking
        keep_window_size: Number of tokens to keep in attention window
        is_causal: Whether to apply causal masking
        return_softmax: Whether to return softmax weights
    
    Returns:
        attn_outputs: [batch_size, query_len, num_heads, head_dim]
    """
    if flash_dmattn_func is None:
        return "Not Available", 0

    # Calculate zoh_states
    zoh_states = calculate_zoh_states(value_states, dt_proj, A)

    # Use prepare_dynamic_mask to get the processed attention mask  
    attn_bias, attn_mask = prepare_dynamic_mask(
        query_states,
        zoh_states,
        keep_window_size,
        cache_position if is_causal else None
    )  # [batch_size, num_kv_heads, query_len, key_len]
    
    # Ensure correct data types and memory layout for CUDA function
    # CUDA function expects: q, k, v in [batch, seqlen, num_heads, head_dim] format
    query_states = query_states.transpose(1, 2)     # [batch, query_len, num_heads, head_dim]
    key_states = key_states.transpose(1, 2)         # [batch, key_len, num_kv_heads, head_dim]
    value_states = value_states.transpose(1, 2)     # [batch, key_len, num_kv_heads, head_dim]

    try:
        torch.cuda.synchronize()
        start_time = time.time()

        # Call the new flash_dmattn_func interface
        attn_outputs = flash_dmattn_func(
            query_states,               # [batch, query_len, num_heads, head_dim]
            key_states,                 # [batch, key_len, num_kv_heads, head_dim]
            value_states,               # [batch, key_len, num_kv_heads, head_dim]
            attn_mask=attn_mask,        # [batch, num_kv_heads, query_len, key_len]
            attn_bias=attn_bias,        # [batch, num_kv_heads, query_len, key_len]
            is_causal=is_causal,
            scale=scaling,
            softcap=0.0,
            deterministic=False,
            return_attn_probs=return_softmax
        )
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        return attn_outputs, (end_time - start_time) * 1000  # Return output and time in ms
    except torch.cuda.OutOfMemoryError:
        return "OOM", 0


def dynamic_mask_attention_triton(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    dt_proj: torch.Tensor,
    A: torch.Tensor,
    scaling: float,
    cache_position: torch.Tensor,
    keep_window_size=2048,
    is_causal=True,
):
    """
    Triton implementation of dynamic mask attention.
    
    Args:
        query_states: [batch_size, num_heads, query_len, head_dim]
        key_states: [batch_size, num_kv_heads, key_len, head_dim]
        value_states: [batch_size, num_kv_heads, key_len, head_dim]
        dt_proj: [num_kv_heads, num_kv_heads * head_dim]
        A: [num_kv_heads]
        scaling: Attention scaling factor
        cache_position: Cache position for causal masking
        keep_window_size: Number of tokens to keep in attention window
        is_causal: Whether to apply causal masking
    
    Returns:
        attn_outputs: [batch_size, query_len, num_heads, head_dim]
    """
    if triton_dmattn_func is None:
        return "Not Available", 0
    
    _, num_heads, _, _ = query_states.shape
    _, num_kv_heads, _, _ = key_states.shape
    num_queries_per_kv = num_heads // num_kv_heads

    try:
        # Calculate zoh_states
        zoh_states = calculate_zoh_states(value_states, dt_proj, A)

        # Use prepare_dynamic_mask to get the processed attention mask  
        attn_bias, attn_mask = prepare_dynamic_mask(
            query_states,
            zoh_states,
            keep_window_size,
            cache_position if is_causal else None
        )  # [batch_size, num_kv_heads, query_len, key_len]
        
        # Repeat KV for multi-head attention (GQA support)
        key_states = repeat_kv(key_states, num_queries_per_kv)
        value_states = repeat_kv(value_states, num_queries_per_kv)
        attn_mask = repeat_kv(attn_mask, num_queries_per_kv)
        attn_bias = repeat_kv(attn_bias, num_queries_per_kv)
        
        # Triton function expects: q, k, v in [batch, seqlen, num_heads, head_dim] format
        query_states = query_states.transpose(1, 2).contiguous()    # [batch, query_len, num_heads, head_dim]
        key_states = key_states.transpose(1, 2).contiguous()        # [batch, key_len, num_heads, head_dim]
        value_states = value_states.transpose(1, 2).contiguous()    # [batch, key_len, num_heads, head_dim]
        attn_mask = attn_mask.contiguous()                          # [batch, num_heads, seqlen_q, seqlen_k]
        attn_bias = attn_bias.contiguous()                          # [batch, num_heads, seqlen_q, seqlen_k]
        
        # Only measure the core Triton kernel computation
        torch.cuda.synchronize()
        start_time = time.time()
        
        # Call the Triton implementation
        attn_outputs = triton_dmattn_func(
            query_states,               # q: [batch, seqlen_q, num_heads, head_dim]
            key_states,                 # k: [batch, seqlen_k, num_heads, head_dim]
            value_states,               # v: [batch, seqlen_k, num_heads, head_dim]
            attn_mask=attn_mask,        # mask: [batch, num_heads, seqlen_q, seqlen_k]
            attn_bias=attn_bias,        # bias: [batch, num_heads, seqlen_q, seqlen_k]
            is_causal=is_causal,        # causal masking
            scale=scaling               # scaling factor
        )
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        return attn_outputs, (end_time - start_time) * 1000  # Return output and time in ms
    except torch.cuda.OutOfMemoryError:
        return "OOM", 0


def dynamic_mask_attention_flex(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    dt_proj: torch.Tensor,
    A: torch.Tensor,
    scaling: float,
    cache_position: torch.Tensor,
    keep_window_size=2048,
    is_causal=True,
):
    """
    Flex Attention implementation of dynamic mask attention.
    
    Args:
        query_states: [batch_size, num_heads, query_len, head_dim]
        key_states: [batch_size, num_kv_heads, key_len, head_dim]
        value_states: [batch_size, num_kv_heads, key_len, head_dim]
        dt_proj: [num_kv_heads, num_kv_heads * head_dim]
        A: [num_kv_heads]
        scaling: Attention scaling factor
        cache_position: Cache position for causal masking
        keep_window_size: Number of tokens to keep in attention window
        is_causal: Whether to apply causal masking
    
    Returns:
        attn_outputs: [batch_size, query_len, num_heads, head_dim]
    """
    if flex_dmattn_func is None:
        return "Not Available", 0
    
    _, num_heads, _, _ = query_states.shape
    _, num_kv_heads, _, _ = key_states.shape
    num_queries_per_kv = num_heads // num_kv_heads

    try:
        # Calculate zoh_states
        zoh_states = calculate_zoh_states(value_states, dt_proj, A)

        # Use prepare_dynamic_mask to get the processed attention mask  
        attn_bias, attn_mask = prepare_dynamic_mask(
            query_states,
            zoh_states,
            keep_window_size,
            cache_position if is_causal else None
        )  # [batch_size, num_kv_heads, query_len, key_len]
        
        # Repeat KV for multi-head attention (GQA support)
        key_states = repeat_kv(key_states, num_queries_per_kv)
        value_states = repeat_kv(value_states, num_queries_per_kv)
        attn_mask = repeat_kv(attn_mask, num_queries_per_kv)
        attn_bias = repeat_kv(attn_bias, num_queries_per_kv)
        
        # Flex attention expects: q, k, v in [batch, num_heads, seqlen, head_dim] format
        # But attention_mask and attention_bias in [batch, num_heads, query_len, key_len] format
        
        # Only measure the core Flex Attention computation
        torch.cuda.synchronize()
        start_time = time.time()
        
        # Call the Flex Attention implementation
        attn_outputs = flex_dmattn_func(
            query_states.transpose(1, 2),               # q: [batch, query_len, num_heads, head_dim]
            key_states.transpose(1, 2),                 # k: [batch, key_len, num_heads, head_dim]
            value_states.transpose(1, 2),               # v: [batch, key_len, num_heads, head_dim]
            attn_mask=attn_mask,                        # attn_mask: [batch, num_heads, query_len, key_len]
            attn_bias=attn_bias,                        # attn_bias: [batch, num_heads, query_len, key_len]
            is_causal=is_causal,                        # is_causal: whether to apply causal masking
            scale=scaling                               # scaling factor
        )
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        return attn_outputs, (end_time - start_time) * 1000  # Return output and time in ms  
    except torch.cuda.OutOfMemoryError:
        return "OOM", 0


def measure_memory_usage():
    """
    Measure current GPU memory usage.
    
    Returns:
        tuple: (allocated_mb, reserved_mb)
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        reserved = torch.cuda.memory_reserved() / 1024**2   # MB
        return allocated, reserved
    return 0, 0


def compute_sdpa_tflops(
    batch_size: int,
    num_heads: int,
    query_len: int,
    key_len: int,
    head_dim: int,
    elapsed_ms: float,
) -> float:
    """Estimate effective TFLOPs for the SDPA baseline given execution time.

    Uses the conventional attention flop model consisting of two matrix
    multiplications (QK^T and softmax-V application)."""

    if elapsed_ms is None or elapsed_ms <= 0:
        return 0.0

    flop_count = 4 * batch_size * num_heads * query_len * key_len * head_dim
    return flop_count / (elapsed_ms / 1000.0) / 1e12


def compute_dmattn_tflops(
    batch_size: int,
    num_heads: int,
    num_kv_heads: int,
    query_len: int,
    key_len: int,
    head_dim: int,
    keep_window_size: int,
    elapsed_ms: float,
) -> float:
    """Estimate effective TFLOPs for dynamic mask attention kernels.

    Calculates compute density using the configured keep window as the
    effective key length processed per query."""

    if elapsed_ms is None or elapsed_ms <= 0:
        return 0.0

    effective_keys = min(key_len, keep_window_size)
    # Dynamic mask kernels ultimately operate over repeated heads after GQA expansion.
    flop_count = 4 * batch_size * num_heads * query_len * effective_keys * head_dim
    return flop_count / (elapsed_ms / 1000.0) / 1e12


def benchmark_attention_performance(config, test_type='all', num_runs=5, warmup_runs=2):
    """
    Benchmark attention performance for a given configuration.
    
    Args:
        config: Tuple of (batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim, keep_window_size, is_causal)
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
    
    Returns:
        dict: Performance metrics
    """
    batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim, keep_window_size, is_causal = config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create random input data
    query_states = torch.randn(
        batch_size, num_heads, query_len, head_dim, 
        device=device, dtype=torch.bfloat16
    )
    key_states = torch.randn(
        batch_size, num_kv_heads, key_len, head_dim, 
        device=device, dtype=torch.bfloat16
    )
    value_states = torch.randn(
        batch_size, num_kv_heads, key_len, head_dim, 
        device=device, dtype=torch.bfloat16
    )
    dt_proj = torch.randn(
        num_kv_heads, num_kv_heads * head_dim, 
        device=device, dtype=torch.bfloat16
    )
    A = torch.randn(num_kv_heads, device=device, dtype=torch.bfloat16)
    
    # Create custom causal mask with cache position
    cache_position = torch.arange(key_len - query_len, key_len, device=device)
    min_type = torch.finfo(value_states.dtype).min
    causal_mask = torch.full(
        (query_len, key_len), fill_value=min_type, 
        device=device, dtype=value_states.dtype
    )
    causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask *= torch.arange(key_len, device=device) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    
    # Set scaling factor from config
    scaling = head_dim ** -0.5
    
    results = {
        'config': config,
        'sdpa_forward_times': [],
        'sdpa_forward_tflops': [],
        'fdma_cuda_forward_times': [],
        'fdma_cuda_forward_tflops': [],
        'fdma_triton_forward_times': [],
        'fdma_triton_forward_tflops': [],
        'fdma_flex_forward_times': [],
        'fdma_flex_forward_tflops': [],
        'sdpa_forward_memory': 0,
        'fdma_cuda_forward_memory': 0,
        'fdma_triton_forward_memory': 0,
        'fdma_flex_forward_memory': 0,
        'sdpa_forward_status': 'success',
        'fdma_cuda_forward_status': 'success',
        'fdma_triton_forward_status': 'success',
        'fdma_flex_forward_status': 'success'
    }
    
    # Determine which implementations to run
    run_flash = test_type in ['all', 'sdpa', 'sdpa-vs-cuda', 'sdpa-vs-triton', 'sdpa-vs-flex']
    run_cuda = test_type in ['all', 'cuda', 'sdpa-vs-cuda']
    run_triton = test_type in ['all', 'triton', 'sdpa-vs-triton']
    run_flex = test_type in ['all', 'flex', 'sdpa-vs-flex']
    
    # Benchmark SDPA
    if run_flash:
        gc.collect()
        torch.cuda.empty_cache()
        
        # Warmup runs
        for _ in range(warmup_runs):
            result = scaled_dot_product_attention_cuda(
                query_states, key_states, value_states,
                scaling, causal_mask, is_causal
            )
            if result[0] == "OOM":
                results['sdpa_forward_status'] = 'OOM'
                break
            torch.cuda.synchronize()

        if results['sdpa_forward_status'] == 'success':
            # Measure memory before benchmark
            mem_before = measure_memory_usage()
            
            # Actual benchmark runs
            for _ in range(num_runs):
                result = scaled_dot_product_attention_cuda(
                    query_states, key_states, value_states,
                    scaling, causal_mask, is_causal
                )
                
                if result[0] == "OOM":
                    results['sdpa_forward_status'] = 'OOM'
                    break
                
                # Use the timing from the function instead of measuring here
                elapsed_ms = result[1]
                results['sdpa_forward_times'].append(elapsed_ms)  # ms
                results['sdpa_forward_tflops'].append(
                    compute_sdpa_tflops(
                        batch_size,
                        num_heads,
                        query_len,
                        key_len,
                        head_dim,
                        elapsed_ms,
                    )
                )
            
            # Measure memory after
            mem_after = measure_memory_usage()
            results['sdpa_forward_memory'] = mem_after[0] - mem_before[0]
    else:
        results['sdpa_forward_status'] = 'N/A'
    
    # Benchmark Dynamic Mask Attention
    if run_cuda:
        gc.collect()
        torch.cuda.empty_cache()
        
        # Warmup runs
        for _ in range(warmup_runs):
            result = dynamic_mask_attention_cuda(
                query_states, key_states, value_states,
                dt_proj, A, scaling, cache_position,
                keep_window_size, is_causal
            )
            if result[0] == "OOM":
                results['fdma_cuda_forward_status'] = 'OOM'
                break
            torch.cuda.synchronize()
        
        if results['fdma_cuda_forward_status'] == 'success':
            # Measure memory before benchmark
            mem_before = measure_memory_usage()
            
            # Actual benchmark runs
            for _ in range(num_runs):
                result = dynamic_mask_attention_cuda(
                    query_states, key_states, value_states,
                    dt_proj, A, scaling, cache_position,
                    keep_window_size, is_causal
                )
                
                if result[0] == "OOM":
                    results['fdma_cuda_forward_status'] = 'OOM'
                    break
                
                # Use the timing from the function instead of measuring here
                elapsed_ms = result[1]
                results['fdma_cuda_forward_times'].append(elapsed_ms)  # ms
                results['fdma_cuda_forward_tflops'].append(
                    compute_dmattn_tflops(
                        batch_size,
                        num_heads,
                        num_kv_heads,
                        query_len,
                        key_len,
                        head_dim,
                        keep_window_size,
                        elapsed_ms,
                    )
                )
            
            # Measure memory after
            mem_after = measure_memory_usage()
            results['fdma_cuda_forward_memory'] = mem_after[0] - mem_before[0]
    else:
        results['fdma_cuda_forward_status'] = 'N/A'
    
    # Benchmark Dynamic Mask Attention (Triton)
    if run_triton:
        gc.collect()
        torch.cuda.empty_cache()
        
        # Warmup runs
        for _ in range(warmup_runs):
            result = dynamic_mask_attention_triton(
                query_states, key_states, value_states,
                dt_proj, A, scaling, cache_position,
                keep_window_size, is_causal
            )
            if result[0] in ["OOM", "Not Available"]:
                results['fdma_triton_forward_status'] = result[0]
                break
            torch.cuda.synchronize()

        if results['fdma_triton_forward_status'] == 'success':
            # Measure memory before benchmark
            mem_before = measure_memory_usage()
            
            # Actual benchmark runs
            for _ in range(num_runs):
                result = dynamic_mask_attention_triton(
                    query_states, key_states, value_states,
                    dt_proj, A, scaling, cache_position,
                    keep_window_size, is_causal
                )
                
                if result[0] in ["OOM", "Not Available"]:
                    results['fdma_triton_forward_status'] = result[0]
                    break
                
                # Use the timing from the function instead of measuring here
                elapsed_ms = result[1]
                results['fdma_triton_forward_times'].append(elapsed_ms)  # ms
                results['fdma_triton_forward_tflops'].append(
                    compute_dmattn_tflops(
                        batch_size,
                        num_heads,
                        num_kv_heads,
                        query_len,
                        key_len,
                        head_dim,
                        keep_window_size,
                        elapsed_ms,
                    )
                )
            
            # Measure memory after
            mem_after = measure_memory_usage()
            results['fdma_triton_forward_memory'] = mem_after[0] - mem_before[0]
    else:
        results['fdma_triton_forward_status'] = 'N/A'
    
    # Benchmark Dynamic Mask Attention (Flex)
    if run_flex:
        gc.collect()
        torch.cuda.empty_cache()
        
        # Warmup runs
        for _ in range(warmup_runs):
            result = dynamic_mask_attention_flex(
                query_states, key_states, value_states,
                dt_proj, A, scaling, cache_position,
                keep_window_size, is_causal
            )
            if result[0] in ["OOM", "Not Available"]:
                results['fdma_flex_forward_status'] = result[0]
                break
            torch.cuda.synchronize()

        if results['fdma_flex_forward_status'] == 'success':
            # Measure memory before benchmark
            mem_before = measure_memory_usage()
            
            # Actual benchmark runs
            for _ in range(num_runs):
                result = dynamic_mask_attention_flex(
                    query_states, key_states, value_states,
                    dt_proj, A, scaling, cache_position,
                    keep_window_size, is_causal
                )
                
                if result[0] in ["OOM", "Not Available"]:
                    results['fdma_flex_forward_status'] = result[0]
                    break
                
                # Use the timing from the function instead of measuring here
                elapsed_ms = result[1]
                results['fdma_flex_forward_times'].append(elapsed_ms)  # ms
                results['fdma_flex_forward_tflops'].append(
                    compute_dmattn_tflops(
                        batch_size,
                        num_heads,
                        num_kv_heads,
                        query_len,
                        key_len,
                        head_dim,
                        keep_window_size,
                        elapsed_ms,
                    )
                )

            # Measure memory after
            mem_after = measure_memory_usage()
            results['fdma_flex_forward_memory'] = mem_after[0] - mem_before[0]
    else:
        results['fdma_flex_forward_status'] = 'N/A'

    return results


def run_performance_benchmark(test_type='all', num_runs=3, warmup_runs=2):
    """Run comprehensive performance benchmark across different configurations."""
    print("\n" + "🏆" + "=" * 76 + "🏆")
    
    # Update title based on test type
    if test_type == 'all':
        title = "⚡ Performance Benchmark: SDPA vs CUDA vs Triton vs Flex ⚡"
    elif test_type == 'sdpa-vs-cuda':
        title = "⚡ Performance Benchmark: SDPA Attention vs CUDA ⚡"
    elif test_type == 'sdpa-vs-triton':
        title = "⚡ Performance Benchmark: SDPA Attention vs Triton ⚡"
    elif test_type == 'sdpa-vs-flex':
        title = "⚡ Performance Benchmark: SDPA Attention vs Flex ⚡"
    elif test_type == 'sdpa':
        title = "⚡ Performance Benchmark: SDPA Attention Only ⚡"
    elif test_type == 'cuda':
        title = "⚡ Performance Benchmark: CUDA Implementations ⚡"
    elif test_type == 'triton':
        title = "⚡ Performance Benchmark: Triton Implementation ⚡"
    elif test_type == 'flex':
        title = "⚡ Performance Benchmark: Flex Implementation ⚡"
    else:
        title = "⚡ Performance Benchmark ⚡"
    
    print(title)
    print("🏆" + "=" * 76 + "🏆")
    
    # Test configurations: (batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim, keep_window_size, is_causal)
    configs = [
        # Vary sequence length
        (1, 2, 1, 256, 256, 64, 1024, True),
        (1, 2, 1, 512, 512, 64, 1024, True),
        (1, 2, 1, 1024, 1024, 64, 1024, True),
        (1, 2, 1, 2048, 2048, 64, 1024, True),
        (1, 2, 1, 4096, 4096, 64, 1024, True),
        (1, 2, 1, 8192, 8192, 64, 1024, True),
        (1, 2, 1, 16384, 16384, 64, 1024, True),
        (1, 2, 1, 32768, 32768, 64, 1024, True),

        # Inference
        (1, 2, 1, 1, 256, 64, 1024, True),
        (1, 2, 1, 1, 512, 64, 1024, True),
        (1, 2, 1, 1, 1024, 64, 1024, True),
        (1, 2, 1, 1, 2048, 64, 1024, True),
        (1, 2, 1, 1, 4096, 64, 1024, True),
        (1, 2, 1, 1, 8192, 64, 1024, True),
        (1, 2, 1, 1, 16384, 64, 1024, True),
        (1, 2, 1, 1, 32768, 64, 1024, True),
        (1, 2, 1, 1, 65536, 64, 1024, True),
        (1, 2, 1, 1, 131072, 64, 1024, True),
        (1, 2, 1, 1, 262144, 64, 1024, True),
        (1, 2, 1, 1, 524288, 64, 1024, True),
        
        # Vary batch size
        (1, 2, 1, 4096, 4096, 64, 1024, True),
        (2, 2, 1, 4096, 4096, 64, 1024, True),
        (4, 2, 1, 4096, 4096, 64, 1024, True),
        (8, 2, 1, 4096, 4096, 64, 1024, True),
        
        # Vary head count
        (1, 1, 1, 4096, 4096, 64, 1024, True),
        (1, 2, 1, 4096, 4096, 64, 1024, True),
        (1, 4, 1, 4096, 4096, 64, 1024, True),
        (1, 8, 2, 4096, 4096, 64, 1024, True),
        
        # # Vary head dimension
        # (1, 2, 1, 4096, 4096, 32, 1024, True),
        # (1, 2, 1, 4096, 4096, 64, 1024, True),
        # (1, 2, 1, 4096, 4096, 96, 1024, True),
        # (1, 2, 1, 4096, 4096, 128, 1024, True),
        # (1, 2, 1, 4096, 4096, 192, 1024, True),
        # (1, 2, 1, 4096, 4096, 256, 1024, True),
        
        # Vary keep_window_size
        (1, 2, 1, 32768, 32768, 64, 32, True),
        (1, 2, 1, 32768, 32768, 64, 64, True),
        (1, 2, 1, 32768, 32768, 64, 128, True),
        (1, 2, 1, 32768, 32768, 64, 256, True),
        (1, 2, 1, 32768, 32768, 64, 512, True),
        (1, 2, 1, 32768, 32768, 64, 1024, True),
        (1, 2, 1, 32768, 32768, 64, 2048, True),
        (1, 2, 1, 32768, 32768, 64, 4096, True),
        (1, 2, 1, 32768, 32768, 64, 8192, True),
        (1, 2, 1, 32768, 32768, 64, 16384, True),
        (1, 2, 1, 32768, 32768, 64, 32768, True),
    ]
    
    print(f"\n📊 Benchmark Results (averaged over {num_runs} runs):")
    table_rows = []
    all_results = []

    for config in configs:
        batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim, keep_window_size, is_causal = config
        
        results = benchmark_attention_performance(config, test_type, num_runs, warmup_runs)
        all_results.append(results)
        
        # Calculate averages for all implementations
        implementations = {
            'sdpa': (
                'sdpa_forward',
                results['sdpa_forward_status'],
                results['sdpa_forward_times'],
                results['sdpa_forward_tflops'],
            ),
            'cuda': (
                'fdma_cuda_forward',
                results['fdma_cuda_forward_status'],
                results['fdma_cuda_forward_times'],
                results['fdma_cuda_forward_tflops'],
            ),
            'triton': (
                'fdma_triton_forward',
                results['fdma_triton_forward_status'],
                results['fdma_triton_forward_times'],
                results['fdma_triton_forward_tflops'],
            ),
            'flex': (
                'fdma_flex_forward',
                results['fdma_flex_forward_status'],
                results['fdma_flex_forward_times'],
                results['fdma_flex_forward_tflops'],
            ),
        }
        
        metrics = {}
        time_avgs = {}

        for impl_key, (_, status, times, tflops) in implementations.items():
            metric = {
                "status": status,
                "avg_time": None,
                "avg_tflops": None,
                "display": status if status != 'success' else "N/A",
            }

            if status == 'success' and times:
                avg_time = sum(times) / len(times)
                avg_tflops = sum(tflops) / len(tflops) if tflops else 0.0
                metric["avg_time"] = avg_time
                metric["avg_tflops"] = avg_tflops
                metric["display"] = f"{avg_time:.2f} ms / {avg_tflops:.2f}"

            metrics[impl_key] = metric
            time_avgs[impl_key] = (
                metric["avg_time"] if metric["avg_time"] is not None else float('inf')
            )

        flash_avg = time_avgs.get('sdpa', float('inf'))

        speedup_values = {}
        for impl_key in ['cuda', 'triton', 'flex']:
            impl_avg = time_avgs.get(impl_key, float('inf'))
            if flash_avg != float('inf') and impl_avg != float('inf') and impl_avg > 0:
                speedup_values[impl_key] = flash_avg / impl_avg
            else:
                speedup_values[impl_key] = None

        best_impl = "N/A"
        best_speedup_value = None
        for impl_key, speedup_val in speedup_values.items():
            if speedup_val is not None and (
                best_speedup_value is None or speedup_val > best_speedup_value
            ):
                best_speedup_value = speedup_val
                best_impl = impl_key.upper()

        if best_speedup_value is not None:
            speedup_summary = f"{best_impl}:{best_speedup_value:.2f}x"
        else:
            speedup_summary = "N/A"

        config_label = (
            f"B{batch_size} Hq{num_heads} Hkv{num_kv_heads} "
            f"Q{query_len} K{key_len} D{head_dim} W{keep_window_size} "
            f"{'C' if is_causal else 'N'}"
        )

        status_cells_display = []
        status_cells_plain = []
        for impl_key in ['sdpa', 'cuda', 'triton', 'flex']:
            status = metrics[impl_key]['status']
            if status == 'success':
                icon = "✅"
                color = "green"
            elif status in {"OOM", "Not Available"}:
                icon = "❌"
                color = "red"
            else:
                icon = "⚠️"
                color = "yellow"
            status_cells_display.append(style_text(icon, color))
            status_cells_plain.append(icon)
        status_display = " ".join(status_cells_display) + " "
        status_plain = " ".join(status_cells_plain) + " "

        config_display = style_text(config_label, "magenta")
        config_plain = config_label

        sdpa_plain = metrics['sdpa']['display']
        cuda_plain = metrics['cuda']['display']
        triton_plain = metrics['triton']['display']
        flex_plain = metrics['flex']['display']

        sdpa_display = format_metric_cell(
            metrics['sdpa']['status'], sdpa_plain, success_color="blue"
        )
        cuda_display = format_metric_cell(
            metrics['cuda']['status'], cuda_plain, success_color="green"
        )
        triton_display = format_metric_cell(
            metrics['triton']['status'], triton_plain, success_color="magenta"
        )
        flex_display = format_metric_cell(
            metrics['flex']['status'], flex_plain, success_color="cyan"
        )

        if speedup_summary != "N/A":
            speedup_color_map = {'CUDA': 'green', 'TRITON': 'magenta', 'FLEX': 'cyan'}
            speedup_display = style_text(speedup_summary, speedup_color_map.get(best_impl, 'blue'), bold=True)
        else:
            speedup_display = style_text(speedup_summary, 'yellow')
        speedup_plain = speedup_summary

        table_rows.append([
            (status_display, status_plain),
            (config_display, config_plain),
            (sdpa_display, sdpa_plain),
            (cuda_display, cuda_plain),
            (triton_display, triton_plain),
            (flex_display, flex_plain),
            (speedup_display, speedup_plain),
        ])

    if table_rows:
        header = [
            "Status",
            "Configuration",
            "SDPA (ms / TFLOPs)",
            "CUDA (ms / TFLOPs)",
            "Triton (ms / TFLOPs)",
            "Flex (ms / TFLOPs)",
            "Speedup",
        ]

        num_cols = len(header)
        all_plain_rows = [header] + [[cell[1] for cell in row] for row in table_rows]
        col_widths = [
            max(len(row[i]) for row in all_plain_rows)
            for i in range(num_cols)
        ]

        header_cells = [
            header[i].ljust(col_widths[i])
            for i in range(num_cols)
        ]
        print("| " + " | ".join(header_cells) + " |")

        separator_cells = ["-" * (width + 2) for width in col_widths]
        print("|" + "|".join(separator_cells) + "|")

        for row in table_rows:
            padded_cells = []
            for idx, (display, plain) in enumerate(row):
                padding = col_widths[idx] - len(plain)
                if padding < 0:
                    padding = 0
                padded_cells.append(display + " " * padding)
            print("| " + " | ".join(padded_cells) + " |")

    # Summary statistics
    implementation_speedups = {
        'cuda': [],
        'triton': [],
        'flex': []
    }
    implementation_tflops = {
        'sdpa': [],
        'cuda': [],
        'triton': [],
        'flex': []
    }

    for results in all_results:
        if results['sdpa_forward_status'] != 'success' or not results['sdpa_forward_times']:
            continue

        flash_avg = sum(results['sdpa_forward_times']) / len(results['sdpa_forward_times'])
        if results['sdpa_forward_tflops']:
            implementation_tflops['sdpa'].append(
                sum(results['sdpa_forward_tflops']) / len(results['sdpa_forward_tflops'])
            )

        for impl_key in implementation_speedups.keys():
            if impl_key == 'cuda':
                status_key = 'fdma_cuda_forward_status'
                times_key = 'fdma_cuda_forward_times'
                tflops_key = 'fdma_cuda_forward_tflops'
            else:
                status_key = f'fdma_{impl_key}_forward_status'
                times_key = f'fdma_{impl_key}_forward_times'
                tflops_key = f'fdma_{impl_key}_forward_tflops'

            if (
                status_key in results
                and results[status_key] == 'success'
                and times_key in results
                and results[times_key]
            ):
                impl_avg = sum(results[times_key]) / len(results[times_key])
                if impl_avg > 0:
                    implementation_speedups[impl_key].append(flash_avg / impl_avg)

            if (
                status_key in results
                and results[status_key] == 'success'
                and tflops_key in results
                and results[tflops_key]
            ):
                implementation_tflops[impl_key].append(
                    sum(results[tflops_key]) / len(results[tflops_key])
                )

    print(f"\n🏆 Summary:")

    for impl_key, speedups in implementation_speedups.items():
        impl_name = impl_key.replace('_', '-').upper()
        impl_label = style_text(impl_name, bold=True)
        if speedups:
            avg_speedup = np.mean(speedups)
            max_speedup = np.max(speedups)
            min_speedup = np.min(speedups)

            if avg_speedup > 2.0:
                icon = "🔥"
                color = "green"
            elif avg_speedup > 1.5:
                icon = "🚀"
                color = "cyan"
            elif avg_speedup > 1.0:
                icon = "📈"
                color = "orange"
            else:
                icon = "😐"
                color = "yellow"

            speedup_text = style_text(
                f"Avg: {avg_speedup:.2f}x (Best: {max_speedup:.2f}x, Worst: {min_speedup:.2f}x)",
                color,
            )
            print(f"  {icon} {impl_label} vs SDPA - {speedup_text}")
        else:
            message = style_text("No successful runs", "red")
            print(f"  ❌ {impl_label} vs SDPA - {message}")

    print("\n📊 Average TFLOPs by implementation:")
    for impl_key, values in implementation_tflops.items():
        label = "SDPA" if impl_key == 'sdpa' else impl_key.upper()
        label_text = style_text(label, bold=True)
        if values:
            avg_tflops = np.mean(values)
            max_tflops = np.max(values)
            min_tflops = np.min(values)
            tflops_text = style_text(
                f"Avg: {avg_tflops:.2f} TFLOPs (Best: {max_tflops:.2f}, Worst: {min_tflops:.2f})",
                "blue",
            )
            print(f"  💡 {label_text} {tflops_text}")
        else:
            message = style_text("No TFLOPs measurements", "yellow")
            print(f"  ⚠️ {label_text} {message}")


def main():
    """
    Run performance benchmarks for Dynamic Mask Attention.
    
    This script measures and compares performance including:
    - Latency measurements across different configurations
    - Memory usage analysis
    - Throughput comparisons
    - Scalability analysis
    """
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Performance benchmark for Dynamic Mask Attention'
    )
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--runs', type=int, default=3, help='Number of benchmark runs')
    parser.add_argument('--warmup', type=int, default=2, help='Number of warmup runs')
    parser.add_argument('--test-type', type=str, default='all', 
                        choices=['all', 'sdpa', 'cuda', 'triton', 'flex', 'sdpa-vs-cuda', 'sdpa-vs-triton', 'sdpa-vs-flex'],
                        help='Type of benchmark to run (default: all)')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Print test environment information
    print(f"🐍 PyTorch version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_icon = "🔥" if device.type == "cuda" else "💻"
    print(f"{device_icon} Device: {device}")
    
    if torch.cuda.is_available():
        print(f"🎮 CUDA device: {torch.cuda.get_device_name()}")
        print(f"💾 Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print(f"🎲 Random seed: {args.seed}")
    print(f"📊 Test type: {args.test_type}")
    print(f"🔄 Runs: {args.runs}, Warmup: {args.warmup}")
    
    # Run performance benchmark
    run_performance_benchmark(args.test_type, args.runs, args.warmup)


if __name__ == "__main__":
    main()