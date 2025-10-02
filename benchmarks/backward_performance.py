#!/usr/bin/env python3
"""
Backward Performance Benchmark for Dynamic Mask Attention

This script measures and compares the backward pass performance of multiple Dynamic Mask Attention 
implementations against SDPA baseline across various configurations.

Implementations tested:
- PyTorch SDPA - Baseline (backward pass)
- Dynamic Mask Attention CUDA - Custom CUDA kernel implementation (backward pass)
- Dynamic Mask Attention Triton - Triton kernel implementation (backward pass)
- Dynamic Mask Attention Flex - Flex Attention implementation (backward pass)

Benchmark includes:
- Multiple sequence lengths and batch sizes
- Head count and dimension variations
- Backward pass throughput and latency measurements
- Memory usage analysis during backward pass
- Speedup comparisons across all implementations for gradient computation
"""

import torch
import torch.nn.functional as F
import argparse
import time
import gc
import sys
import os
from typing import Optional
import numpy as np

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


def compute_sdpa_backward_tflops(
    batch_size: int,
    num_heads: int,
    query_len: int,
    key_len: int,
    head_dim: int,
    elapsed_ms: float,
) -> float:
    """Estimate effective TFLOPs for SDPA backward given execution time.

    Utilizes the same dense attention flop model as the forward benchmark, which
    is a reasonable lower bound for backward pass compute density."""

    if elapsed_ms is None or elapsed_ms <= 0:
        return 0.0

    # Backward attention FLOPs are roughly ~2x forward matmuls; approximate as 8 * B * H * Q * K * D
    flop_count = 8 * batch_size * num_heads * query_len * key_len * head_dim
    return flop_count / (elapsed_ms / 1000.0) / 1e12


def compute_dmattn_backward_tflops(
    batch_size: int,
    num_heads: int,
    num_kv_heads: int,
    query_len: int,
    key_len: int,
    head_dim: int,
    keep_window_size: int,
    elapsed_ms: float,
) -> float:
    """Estimate effective TFLOPs for dynamic mask attention backward kernels."""

    if elapsed_ms is None or elapsed_ms <= 0:
        return 0.0

    effective_keys = min(key_len, keep_window_size)
    # Apply same ~2x multiplier for dynamic mask (sparse) backward
    flop_count = 8 * batch_size * num_heads * query_len * effective_keys * head_dim
    return flop_count / (elapsed_ms / 1000.0) / 1e12


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


def scaled_dot_product_attention_backward(
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
    SDPA baseline backward pass implementation.
    
    Args:
        query_states: [batch_size, num_heads, query_len, head_dim]
        key_states: [batch_size, num_kv_heads, key_len, head_dim]
        value_states: [batch_size, num_kv_heads, key_len, head_dim]
        scaling: Attention scaling factor
        causal_mask: Causal attention mask
        is_causal: Whether to apply causal masking
    
    Returns:
        tuple: (output_tensor, timing_ms) or ("OOM", 0) if out of memory
    """
    _, num_heads, query_len, _ = query_states.shape
    _, num_kv_heads, key_len, _ = key_states.shape
    num_queries_per_kv = num_heads // num_kv_heads
    
    # Calculate zoh_states
    zoh_states = calculate_zoh_states(value_states, dt_proj, A)

    # Use prepare_dynamic_mask to get the processed attention mask
    attn_bias, attn_mask = prepare_dynamic_mask(
        query_states,
        zoh_states,
        keep_window_size,
        cache_position if is_causal else None
    )   # [batch_size, num_kv_heads, query_len, key_len]

    query_states = query_states.contiguous()
    key_states = key_states.contiguous()
    value_states = value_states.contiguous()
    attn_bias = repeat_kv(attn_bias, num_queries_per_kv).contiguous()

    try:
        # Forward pass - SDPA expects q, k, v in [batch, num_heads, seq_len, head_dim] format
        attn_outputs = F.scaled_dot_product_attention(
            query_states,                   # [batch, num_heads, query_len, head_dim]
            key_states,                     # [batch, num_kv_heads, key_len, head_dim]
            value_states,                   # [batch, num_kv_heads, key_len, head_dim]
            attn_mask=attn_bias,            # [batch, num_kv_heads, query_len, key_len]
            scale=scaling,
            # is_causal=is_causal if query_len == key_len else False,
            enable_gqa=True
        )

        # Transpose to match expected output format
        attn_outputs = attn_outputs.transpose(1, 2).contiguous()    # [batch, query_len, num_heads, head_dim]
        
        torch.cuda.synchronize()
        start_time = time.time()

        # Backward pass
        attn_outputs.sum().backward()

        torch.cuda.synchronize()
        end_time = time.time()
        
        return attn_outputs, (end_time - start_time) * 1000  # Convert to milliseconds
        
    except torch.cuda.OutOfMemoryError:
        return "OOM", 0


def dynamic_mask_attention_backward_cuda(
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
    CUDA implementation of dynamic mask attention backward pass.
    
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
        tuple: (output_tensor, timing_ms) or ("OOM", 0) or ("Not Available", 0)
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
    )   # [batch_size, num_kv_heads, query_len, key_len]
    
    # Ensure correct data types and memory layout for CUDA function
    # CUDA function expects: q, k, v in [batch, seqlen, num_heads, head_dim] format
    query_states = query_states.transpose(1, 2).contiguous()        # [batch, query_len, num_heads, head_dim]
    key_states = key_states.transpose(1, 2).contiguous()            # [batch, key_len, num_kv_heads, head_dim]
    value_states = value_states.transpose(1, 2).contiguous()        # [batch, key_len, num_kv_heads, head_dim]

    try:
        # Call the flash_dmattn_func interface
        attn_outputs = flash_dmattn_func(
            query=query_states,             # q: [batch, query_len, num_heads, head_dim]
            key=key_states,                 # k: [batch, key_len, num_kv_heads, head_dim]
            value=value_states,             # v: [batch, key_len, num_kv_heads, head_dim]
            attn_mask=attn_mask,            # mask: [batch, num_kv_heads, query_len, key_len]
            attn_bias=attn_bias,            # bias: [batch, num_kv_heads, query_len, key_len]
            is_causal=is_causal,            # causal masking
            scale=scaling,                  # scaling factor
            softcap=0.0,
            deterministic=False,
            return_attn_probs=False
        )

        torch.cuda.synchronize()
        start_time = time.time()
        
        # Backward pass
        attn_outputs.sum().backward()
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        return attn_outputs, (end_time - start_time) * 1000  # Convert to milliseconds
        
    except torch.cuda.OutOfMemoryError:
        return "OOM", 0


def dynamic_mask_attention_backward_triton(
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
    Triton implementation of dynamic mask attention backward pass.
    
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
        tuple: (output_tensor, timing_ms) or ("OOM", 0) or ("Not Available", 0)
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
        query_states = query_states.transpose(1, 2).contiguous()        # [batch, query_len, num_heads, head_dim]  
        key_states = key_states.transpose(1, 2).contiguous()            # [batch, key_len, num_heads, head_dim]  
        value_states = value_states.transpose(1, 2).contiguous()        # [batch, key_len, num_heads, head_dim]  
        attn_mask = attn_mask.contiguous()                              # [batch, num_heads, seqlen_q, seqlen_k]
        attn_bias = attn_bias.contiguous()                              # [batch, num_heads, seqlen_q, seqlen_k]
        
        # Call the Triton implementation
        attn_outputs = triton_dmattn_func(
            query=query_states,                                         # q: [batch, seqlen_q, num_heads, head_dim]
            key=key_states,                                             # k: [batch, seqlen_k, num_heads, head_dim]
            value=value_states,                                         # v: [batch, seqlen_k, num_heads, head_dim]
            attn_mask=attn_mask,                                        # mask: [batch, num_heads, seqlen_q, seqlen_k]
            attn_bias=attn_bias,                                        # bias: [batch, num_heads, seqlen_q, seqlen_k]
            is_causal=is_causal,                                        # causal masking
            scale=scaling                                               # scaling factor
        )

        torch.cuda.synchronize()
        start_time = time.time()

        # Backward pass
        attn_outputs.sum().backward()
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        return attn_outputs, (end_time - start_time) * 1000  # Convert to milliseconds
        
    except torch.cuda.OutOfMemoryError:
        return "OOM", 0


def dynamic_mask_attention_backward_flex(
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
    Flex Attention implementation of dynamic mask attention backward pass.
    
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
        tuple: (output_tensor, timing_ms) or ("OOM", 0) or ("Not Available", 0)
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
        start_time = time.time()

        # Backward pass
        attn_outputs.sum().backward()
        
        torch.cuda.synchronize()
        end_time = time.time()
        
        return attn_outputs, (end_time - start_time) * 1000  # Convert to milliseconds
        
    except torch.cuda.OutOfMemoryError:
        return "OOM", 0


def measure_memory_usage():
    """
    Measure current GPU memory usage.
    
    Returns:
        tuple: (allocated_mb, reserved_mb)
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
        reserved = torch.cuda.memory_reserved() / 1024 / 1024    # MB
        return allocated, reserved
    return 0, 0


def benchmark_backward_attention_performance(config, test_type='all', num_runs=5, warmup_runs=2):
    """
    Benchmark backward attention performance for a given configuration.
    
    Args:
        config: Tuple of (batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim, keep_window_size, is_causal)
        test_type: Type of test to run ('all', 'sdpa', 'cuda', 'triton', 'flex', etc.)
        num_runs: Number of benchmark runs
        warmup_runs: Number of warmup runs
    
    Returns:
        dict: Performance metrics
    """
    batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim, keep_window_size, is_causal = config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create random input data (requires_grad=True for backward pass)
    query_states = torch.randn(
        batch_size, num_heads, query_len, head_dim, 
        device=device, dtype=torch.bfloat16, requires_grad=True
    )
    key_states = torch.randn(
        batch_size, num_kv_heads, key_len, head_dim, 
        device=device, dtype=torch.bfloat16, requires_grad=True
    )
    value_states = torch.randn(
        batch_size, num_kv_heads, key_len, head_dim, 
        device=device, dtype=torch.bfloat16, requires_grad=True
    )
    dt_proj = torch.randn(
        num_kv_heads, num_kv_heads * head_dim, 
        device=device, dtype=torch.bfloat16, requires_grad=True
    )
    A = torch.randn(num_kv_heads, device=device, dtype=torch.bfloat16, requires_grad=True)
    
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
        'sdpa_backward_times': [],
        'sdpa_backward_tflops': [],
        'fdma_cuda_backward_times': [],
        'fdma_cuda_backward_tflops': [],
        'fdma_triton_backward_times': [],
        'fdma_triton_backward_tflops': [],
        'fdma_flex_backward_times': [],
        'fdma_flex_backward_tflops': [],
        'sdpa_backward_memory': 0,
        'fdma_cuda_backward_memory': 0,
        'fdma_triton_backward_memory': 0,
        'fdma_flex_backward_memory': 0,
        'sdpa_backward_status': 'success',
        'fdma_cuda_backward_status': 'success',
        'fdma_triton_backward_status': 'success',
        'fdma_flex_backward_status': 'success'
    }
    
    # Determine which implementations to run
    run_sdpa = test_type in ['all', 'sdpa', 'sdpa-vs-cuda', 'sdpa-vs-triton', 'sdpa-vs-flex']
    run_cuda = test_type in ['all', 'cuda', 'sdpa-vs-cuda']
    run_triton = test_type in ['all', 'triton', 'sdpa-vs-triton']
    run_flex = test_type in ['all', 'flex', 'sdpa-vs-flex']
    
    # Benchmark SDPA Backward
    if run_sdpa:
        gc.collect()
        torch.cuda.empty_cache()
        
        # Warmup runs
        for _ in range(warmup_runs):
            # Clone inputs for each run
            q_clone = query_states.clone().detach().requires_grad_(True)
            k_clone = key_states.clone().detach().requires_grad_(True)
            v_clone = value_states.clone().detach().requires_grad_(True)
            dt_clone = dt_proj.clone().detach().requires_grad_(True)
            a_clone = A.clone().detach().requires_grad_(True)
            
            result = scaled_dot_product_attention_backward(
                q_clone, k_clone, v_clone, dt_clone, a_clone,
                scaling, cache_position, keep_window_size, is_causal
            )
            if result[0] == "OOM":
                results['sdpa_backward_status'] = 'OOM'
                break
            torch.cuda.synchronize()
        
        if results['sdpa_backward_status'] == 'success':
            # Measure memory before benchmark
            mem_before = measure_memory_usage()
            
            # Actual benchmark runs
            for _ in range(num_runs):
                # Clone inputs for each run
                q_clone = query_states.clone().detach().requires_grad_(True)
                k_clone = key_states.clone().detach().requires_grad_(True)
                v_clone = value_states.clone().detach().requires_grad_(True)
                dt_clone = dt_proj.clone().detach().requires_grad_(True)
                a_clone = A.clone().detach().requires_grad_(True)
                
                result = scaled_dot_product_attention_backward(
                    q_clone, k_clone, v_clone, dt_clone, a_clone,
                    scaling, cache_position, keep_window_size, is_causal
                )
                
                if result[0] == "OOM":
                    results['sdpa_backward_status'] = 'OOM'
                    break
                
                # Use the timing from the function instead of measuring here
                elapsed_ms = result[1]
                results['sdpa_backward_times'].append(elapsed_ms)  # ms
                results['sdpa_backward_tflops'].append(
                    compute_sdpa_backward_tflops(
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
            results['sdpa_backward_memory'] = mem_after[0] - mem_before[0]
    else:
        results['sdpa_backward_status'] = 'N/A'
    
    # Benchmark Dynamic Mask Attention CUDA Backward
    if run_cuda:
        gc.collect()
        torch.cuda.empty_cache()
        
        # Warmup runs
        for _ in range(warmup_runs):
            # Clone inputs for each run
            q_clone = query_states.clone().detach().requires_grad_(True)
            k_clone = key_states.clone().detach().requires_grad_(True)
            v_clone = value_states.clone().detach().requires_grad_(True)
            dt_clone = dt_proj.clone().detach().requires_grad_(True)
            a_clone = A.clone().detach().requires_grad_(True)
            
            result = dynamic_mask_attention_backward_cuda(
                q_clone, k_clone, v_clone, dt_clone, a_clone,
                scaling, cache_position, keep_window_size, is_causal
            )
            if result[0] in ["OOM", "Not Available"]:
                results['fdma_cuda_backward_status'] = result[0]
                break
            torch.cuda.synchronize()
        
        if results['fdma_cuda_backward_status'] == 'success':
            # Measure memory before benchmark
            mem_before = measure_memory_usage()
            
            # Actual benchmark runs
            for _ in range(num_runs):
                # Clone inputs for each run
                q_clone = query_states.clone().detach().requires_grad_(True)
                k_clone = key_states.clone().detach().requires_grad_(True)
                v_clone = value_states.clone().detach().requires_grad_(True)
                dt_clone = dt_proj.clone().detach().requires_grad_(True)
                a_clone = A.clone().detach().requires_grad_(True)
                
                result = dynamic_mask_attention_backward_cuda(
                    q_clone, k_clone, v_clone, dt_clone, a_clone,
                    scaling, cache_position, keep_window_size, is_causal
                )
                
                if result[0] in ["OOM", "Not Available"]:
                    results['fdma_cuda_backward_status'] = result[0]
                    break
                
                # Use the timing from the function instead of measuring here
                elapsed_ms = result[1]
                results['fdma_cuda_backward_times'].append(elapsed_ms)  # ms
                results['fdma_cuda_backward_tflops'].append(
                    compute_dmattn_backward_tflops(
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
            results['fdma_cuda_backward_memory'] = mem_after[0] - mem_before[0]
    else:
        results['fdma_cuda_backward_status'] = 'N/A'
    
    # Benchmark Dynamic Mask Attention Triton Backward
    if run_triton:
        gc.collect()
        torch.cuda.empty_cache()
        
        # Warmup runs
        for _ in range(warmup_runs):
            # Clone inputs for each run
            q_clone = query_states.clone().detach().requires_grad_(True)
            k_clone = key_states.clone().detach().requires_grad_(True)
            v_clone = value_states.clone().detach().requires_grad_(True)
            dt_clone = dt_proj.clone().detach().requires_grad_(True)
            a_clone = A.clone().detach().requires_grad_(True)
            
            result = dynamic_mask_attention_backward_triton(
                q_clone, k_clone, v_clone, dt_clone, a_clone,
                scaling, cache_position, keep_window_size, is_causal
            )
            if result[0] in ["OOM", "Not Available"]:
                results['fdma_triton_backward_status'] = result[0]
                break
            torch.cuda.synchronize()

        if results['fdma_triton_backward_status'] == 'success':
            # Measure memory before benchmark
            mem_before = measure_memory_usage()
            
            # Actual benchmark runs
            for _ in range(num_runs):
                # Clone inputs for each run
                q_clone = query_states.clone().detach().requires_grad_(True)
                k_clone = key_states.clone().detach().requires_grad_(True)
                v_clone = value_states.clone().detach().requires_grad_(True)
                dt_clone = dt_proj.clone().detach().requires_grad_(True)
                a_clone = A.clone().detach().requires_grad_(True)
                
                result = dynamic_mask_attention_backward_triton(
                    q_clone, k_clone, v_clone, dt_clone, a_clone,
                    scaling, cache_position, keep_window_size, is_causal
                )
                
                if result[0] in ["OOM", "Not Available"]:
                    results['fdma_triton_backward_status'] = result[0]
                    break
                
                # Use the timing from the function instead of measuring here
                elapsed_ms = result[1]
                results['fdma_triton_backward_times'].append(elapsed_ms)  # ms
                results['fdma_triton_backward_tflops'].append(
                    compute_dmattn_backward_tflops(
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
            results['fdma_triton_backward_memory'] = mem_after[0] - mem_before[0]
    else:
        results['fdma_triton_backward_status'] = 'N/A'
    
    # Benchmark Dynamic Mask Attention Flex Backward
    if run_flex:
        gc.collect()
        torch.cuda.empty_cache()
        
        # Warmup runs
        for _ in range(warmup_runs):
            # Clone inputs for each run
            q_clone = query_states.clone().detach().requires_grad_(True)
            k_clone = key_states.clone().detach().requires_grad_(True)
            v_clone = value_states.clone().detach().requires_grad_(True)
            dt_clone = dt_proj.clone().detach().requires_grad_(True)
            a_clone = A.clone().detach().requires_grad_(True)
            
            result = dynamic_mask_attention_backward_flex(
                q_clone, k_clone, v_clone, dt_clone, a_clone,
                scaling, cache_position, keep_window_size, is_causal
            )
            if result[0] in ["OOM", "Not Available"]:
                results['fdma_flex_backward_status'] = result[0]
                break
            torch.cuda.synchronize()
        
        if results['fdma_flex_backward_status'] == 'success':
            # Measure memory before benchmark
            mem_before = measure_memory_usage()
            
            # Actual benchmark runs
            for _ in range(num_runs):
                # Clone inputs for each run
                q_clone = query_states.clone().detach().requires_grad_(True)
                k_clone = key_states.clone().detach().requires_grad_(True)
                v_clone = value_states.clone().detach().requires_grad_(True)
                dt_clone = dt_proj.clone().detach().requires_grad_(True)
                a_clone = A.clone().detach().requires_grad_(True)
                
                result = dynamic_mask_attention_backward_flex(
                    q_clone, k_clone, v_clone, dt_clone, a_clone,
                    scaling, cache_position, keep_window_size, is_causal
                )
                
                if result[0] in ["OOM", "Not Available"]:
                    results['fdma_flex_backward_status'] = result[0]
                    break
                
                # Use the timing from the function instead of measuring here
                elapsed_ms = result[1]
                results['fdma_flex_backward_times'].append(elapsed_ms)  # ms
                results['fdma_flex_backward_tflops'].append(
                    compute_dmattn_backward_tflops(
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
            results['fdma_flex_backward_memory'] = mem_after[0] - mem_before[0]
    else:
        results['fdma_flex_backward_status'] = 'N/A'
    
    return results


def run_backward_performance_benchmark(test_type='all', num_runs=3, warmup_runs=2):
    """Run comprehensive backward pass performance benchmark across different configurations."""
    print("\n" + "🏆" + "=" * 76 + "🏆")
    
    # Update title based on test type
    if test_type == 'all':
        title = "🔥 Backward Pass Performance Benchmark: All Implementations 🔥"
    elif test_type == 'sdpa-vs-cuda':
        title = "🚀 Backward Pass Performance: SDPA vs CUDA 🚀"
    elif test_type == 'sdpa-vs-triton':
        title = "🌟 Backward Pass Performance: SDPA vs Triton 🌟"
    elif test_type == 'sdpa-vs-flex':
        title = "✨ Backward Pass Performance: SDPA vs Flex ✨"
    elif test_type == 'sdpa':
        title = "📊 Backward Pass Performance: SDPA Only 📊"
    elif test_type == 'cuda':
        title = "🚀 Backward Pass Performance: CUDA Only 🚀"
    elif test_type == 'triton':
        title = "🌟 Backward Pass Performance: Triton Only 🌟"
    elif test_type == 'flex':
        title = "✨ Backward Pass Performance: Flex Only ✨"
    else:
        title = "🔥 Backward Pass Performance Benchmark 🔥"
    
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
        
        # Vary head dimension
        (1, 2, 1, 16384, 16384, 32, 1024, True),
        (1, 2, 1, 16384, 16384, 64, 1024, True),
        (1, 2, 1, 16384, 16384, 96, 1024, True),
        (1, 2, 1, 16384, 16384, 128, 1024, True),

        # Vary keep_window_size
        (1, 2, 1, 16384, 16384, 64, 32, True),
        (1, 2, 1, 16384, 16384, 64, 64, True),
        (1, 2, 1, 16384, 16384, 64, 128, True),
        (1, 2, 1, 16384, 16384, 64, 256, True),
        (1, 2, 1, 16384, 16384, 64, 512, True),
        (1, 2, 1, 16384, 16384, 64, 1024, True),
        (1, 2, 1, 16384, 16384, 64, 2048, True),
        (1, 2, 1, 16384, 16384, 64, 4096, True),
        (1, 2, 1, 16384, 16384, 64, 8192, True),
        (1, 2, 1, 16384, 16384, 64, 16384, True),
    ]
    
    print(f"\n📊 Backward Pass Benchmark Results (averaged over {num_runs} runs):")

    table_rows = []
    all_results = []

    for config in configs:
        try:
            results = benchmark_backward_attention_performance(config, test_type, num_runs, warmup_runs)
            all_results.append(results)

            batch_size, num_heads, num_kv_heads, query_len, key_len, head_dim, keep_window_size, is_causal = config
            config_label = (
                f"B{batch_size} Hq{num_heads} Hkv{num_kv_heads} "
                f"Q{query_len} K{key_len} D{head_dim} W{keep_window_size} "
                f"{'C' if is_causal else 'N'}"
            )

            implementations = {
                'sdpa': (
                    'sdpa_backward',
                    results['sdpa_backward_status'],
                    results['sdpa_backward_times'],
                    results['sdpa_backward_tflops'],
                ),
                'cuda': (
                    'fdma_cuda_backward',
                    results['fdma_cuda_backward_status'],
                    results['fdma_cuda_backward_times'],
                    results['fdma_cuda_backward_tflops'],
                ),
                'triton': (
                    'fdma_triton_backward',
                    results['fdma_triton_backward_status'],
                    results['fdma_triton_backward_times'],
                    results['fdma_triton_backward_tflops'],
                ),
                'flex': (
                    'fdma_flex_backward',
                    results['fdma_flex_backward_status'],
                    results['fdma_flex_backward_times'],
                    results['fdma_flex_backward_tflops'],
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

            sdpa_avg = time_avgs.get('sdpa', float('inf'))

            speedup_values = {}
            for impl_key in ['cuda', 'triton', 'flex']:
                impl_avg = time_avgs.get(impl_key, float('inf'))
                if sdpa_avg != float('inf') and impl_avg != float('inf') and impl_avg > 0:
                    speedup_values[impl_key] = sdpa_avg / impl_avg
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

            status_display = " ".join(status_cells_display)
            status_plain = " ".join(status_cells_plain)

            config_display = style_text(config_label, "magenta")

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
                speedup_display = style_text(
                    speedup_summary, speedup_color_map.get(best_impl, 'blue'), bold=True
                )
            else:
                speedup_display = style_text(speedup_summary, 'yellow')

            table_rows.append([
                (status_display, status_plain),
                (config_display, config_label),
                (sdpa_display, sdpa_plain),
                (cuda_display, cuda_plain),
                (triton_display, triton_plain),
                (flex_display, flex_plain),
                (speedup_display, speedup_summary),
            ])

        except Exception as e:
            print(f"❌ Error in config {config}: {e}")
            continue

    if table_rows:
        header = [
            "Status",
            "Configuration",
            "SDPA Backward (ms / TFLOPs)",
            "CUDA Backward (ms / TFLOPs)",
            "Triton Backward (ms / TFLOPs)",
            "Flex Backward (ms / TFLOPs)",
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
    else:
        print(style_text("No benchmark data available.", "yellow"))
    
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
        if results['sdpa_backward_times'] and results['fdma_cuda_backward_times']:
            sdpa_time = sum(results['sdpa_backward_times'])/len(results['sdpa_backward_times'])
            cuda_time = sum(results['fdma_cuda_backward_times'])/len(results['fdma_cuda_backward_times'])
            implementation_speedups['cuda'].append(sdpa_time/cuda_time)

        if results['sdpa_backward_times'] and results['fdma_triton_backward_times']:
            sdpa_time = sum(results['sdpa_backward_times'])/len(results['sdpa_backward_times'])
            triton_time = sum(results['fdma_triton_backward_times'])/len(results['fdma_triton_backward_times'])
            implementation_speedups['triton'].append(sdpa_time/triton_time)

        if results['sdpa_backward_times'] and results['fdma_flex_backward_times']:
            sdpa_time = sum(results['sdpa_backward_times'])/len(results['sdpa_backward_times'])
            flex_time = sum(results['fdma_flex_backward_times'])/len(results['fdma_flex_backward_times'])
            implementation_speedups['flex'].append(sdpa_time/flex_time)

        if results['sdpa_backward_status'] == 'success' and results['sdpa_backward_tflops']:
            implementation_tflops['sdpa'].append(
                sum(results['sdpa_backward_tflops']) / len(results['sdpa_backward_tflops'])
            )

        if results['fdma_cuda_backward_status'] == 'success' and results['fdma_cuda_backward_tflops']:
            implementation_tflops['cuda'].append(
                sum(results['fdma_cuda_backward_tflops']) / len(results['fdma_cuda_backward_tflops'])
            )

        if results['fdma_triton_backward_status'] == 'success' and results['fdma_triton_backward_tflops']:
            implementation_tflops['triton'].append(
                sum(results['fdma_triton_backward_tflops']) / len(results['fdma_triton_backward_tflops'])
            )

        if results['fdma_flex_backward_status'] == 'success' and results['fdma_flex_backward_tflops']:
            implementation_tflops['flex'].append(
                sum(results['fdma_flex_backward_tflops']) / len(results['fdma_flex_backward_tflops'])
            )
    
    print(f"\n🏆 Backward Pass Summary:")

    for impl_key, speedups in implementation_speedups.items():
        impl_name = impl_key.replace('_', '-').upper()
        impl_label = style_text(impl_name, bold=True)
        if speedups:
            avg_speedup = sum(speedups) / len(speedups)
            max_speedup = max(speedups)
            min_speedup = min(speedups)

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
            message = style_text("No valid measurements", "red")
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
    Run backward pass performance benchmarks for Dynamic Mask Attention.
    
    This script measures and compares backward pass performance including:
    - Gradient computation latency measurements across different configurations
    - Memory usage analysis during backward pass
    - Throughput comparisons for gradient computation
    - Scalability analysis for large sequences
    """
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Backward pass performance benchmark for Dynamic Mask Attention'
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
        print(f"💾 CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print(f"🎲 Random seed: {args.seed}")
    print(f"📊 Test type: {args.test_type}")
    print(f"🔄 Runs: {args.runs}, Warmup: {args.warmup}")
    
    # Run backward pass performance benchmark
    run_backward_performance_benchmark(args.test_type, args.runs, args.warmup)


if __name__ == "__main__":
    main()
