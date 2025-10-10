# Variable Length Batch Inference with Broadcastable Key-based Masks/Bias

## Overview

This feature enables efficient batch inference with variable-length sequences using key-side broadcastable masks and bias tensors. Instead of materializing per-query masks/bias of shape `(total_q, num_heads, max_seqlen_k)`, you can now provide compact key-side tensors of shape `(total_k, num_heads_variant)` that broadcast across query positions.

## Motivation

In autoregressive decoding with dynamic sparsity:
- Queries are typically short (1-8 tokens per batch element)
- Keys/values can be thousands of tokens from the KV cache
- Precomputed key-side gating scores are naturally shaped `(total_k, num_heads)`
- Reshaping to per-query layout wastes O(total_q * num_heads) memory
- Streaming workloads cannot backfill materialized copies

## Supported Layouts

### Traditional Query-based Layout (existing)
```python
# Mask: (total_q, {1|num_heads_k|num_heads}, max_seqlen_k)
# Bias: (total_q, {1|num_heads_k|num_heads}, max_seqlen_k)
```
Each query position has its own mask/bias slice. This is the default when the first dimension equals `total_q`.

### New Key-based Broadcastable Layout
```python
# Mask: (total_k, {1|num_heads_k|num_heads})
# Bias: (total_k, {1|num_heads_k|num_heads})
```
A single mask/bias value per key position, broadcast across all query positions. Automatically detected when the first dimension equals `total_k`.

## Usage Example

```python
import torch
from flash_dmattn import flash_dmattn_varlen_func

batch_size = 4
max_seqlen_q = 2  # Typical for decoding
max_seqlen_k = 1024  # Large KV cache
num_heads = 32
num_heads_k = 8  # GQA
head_dim = 128

# Create variable length sequences
cu_seqlens_q = torch.tensor([0, 1, 3, 4, 6], dtype=torch.int32, device='cuda')  # total_q = 6
cu_seqlens_k = torch.tensor([0, 256, 512, 768, 1024], dtype=torch.int32, device='cuda')  # total_k = 1024

# Query, key, value tensors
q = torch.randn(6, num_heads, head_dim, dtype=torch.float16, device='cuda')
k = torch.randn(1024, num_heads_k, head_dim, dtype=torch.float16, device='cuda')
v = torch.randn(1024, num_heads_k, head_dim, dtype=torch.float16, device='cuda')

# Key-based broadcastable mask and bias (NEW!)
# Shape: (total_k, num_heads_variant) - broadcasts across query positions
attn_mask = torch.randint(0, 2, (1024, num_heads_k), dtype=torch.bool, device='cuda')
attn_bias = torch.randn(1024, num_heads_k, dtype=torch.float16, device='cuda')

# Call varlen function - layout detection is automatic
output = flash_dmattn_varlen_func(
    query=q,
    key=k,
    value=v,
    attn_mask=attn_mask,  # Automatically detected as k-based
    attn_bias=attn_bias,  # Automatically detected as k-based
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
)
```

## Layout Detection

The implementation automatically detects which layout is being used:

```python
if mask.dim() == 2 and mask.size(0) == total_k:
    # Key-based layout: (total_k, num_heads_variant)
    # Broadcast across query positions
elif mask.dim() == 3 and mask.size(0) == total_q:
    # Query-based layout: (total_q, num_heads_variant, max_seqlen_k)
    # Per-query mask slices
```

The same logic applies independently to both mask and bias tensors.

## Head Dimension Broadcasting

Both layouts support flexible head dimensions:
- `1`: Single mask/bias for all heads (broadcast across all heads)
- `num_heads_k`: One per KV head (broadcast across query head groups in GQA)
- `num_heads`: One per query head (no broadcasting)

## Performance Benefits

### Memory Savings
For typical decoding scenarios:
- Query-based: `total_q × num_heads × max_seqlen_k` elements
- Key-based: `total_k × num_heads` elements
- **Savings**: ~`(total_q × max_seqlen_k) / num_heads` reduction

Example: With `total_q=8`, `max_seqlen_k=2048`, `num_heads=32`:
- Query-based: 524,288 elements
- Key-based: 65,536 elements
- **87.5% memory reduction**

### Computational Efficiency
- No host-side tensor reshaping or copying
- Direct key-side indexing in CUDA kernels
- Maintains streaming-friendly data layout
- Zero materialization overhead

## Implementation Details

### Kernel Changes
The CUDA kernels handle broadcasting by:
1. Using stride `_0{}` for the query dimension in key-based tensors
2. Adjusting offset calculations to skip query-position indexing
3. Reading from the same key-position value for all queries

```cpp
// Key-based layout
Tensor gMask = make_tensor(
    ptr,
    Shape<Int<kBlockM>, Int<kBlockN>>{},
    make_stride(_0{}, _1{})  // Zero stride = broadcast across M
);

// Query-based layout
Tensor gMask = make_tensor(
    ptr,
    Shape<Int<kBlockM>, Int<kBlockN>>{},
    make_stride(mask_row_stride, _1{})  // Normal 2D indexing
);
```

### API Parameters

The C++ API signature is:
```cpp
std::vector<at::Tensor> mha_varlen_fwd(
    at::Tensor &q,                          // total_q x num_heads x head_size
    const at::Tensor &k,                    // total_k x num_heads_k x head_size
    const at::Tensor &v,                    // total_k x num_heads_k x head_size
    std::optional<at::Tensor> &mask_,       // (total_q, h, k) or (total_k, h)
    std::optional<at::Tensor> &bias_,       // (total_q, h, k) or (total_k, h)
    std::optional<at::Tensor> &out_,        // total_q x num_heads x head_size
    const at::Tensor &cu_seqlens_q,         // batch_size + 1
    const at::Tensor &cu_seqlens_k,         // batch_size + 1
    ...
);
```

Layout detection happens automatically based on tensor shapes.

## Use Cases

### Autoregressive Decoding with Dynamic Sparsity
```python
# Precompute key-side attention scores from dependency graph
key_scores = compute_dependency_scores(kv_cache)  # (total_k, num_heads)
key_mask = key_scores > threshold

# Use directly without reshaping
output = flash_dmattn_varlen_func(..., attn_mask=key_mask)
```

### Batch Decode with Shared Key Filtering
```python
# Apply same key filtering to all queries in batch
key_importance = model.compute_key_importance(keys)  # (total_k, 1)
key_mask = key_importance > threshold

# Broadcast to all heads
output = flash_dmattn_varlen_func(..., attn_mask=key_mask)
```

### MaskMod Pipelines
```python
# Dependency-aware masking from MaskMod
from torch.nn.attention.flex_attention import create_mask

# Generate key-side mask efficiently
key_mask = create_mask_mod_k_based(...)  # (total_k, num_heads)

# Direct usage without conversion
output = flash_dmattn_varlen_func(..., attn_mask=key_mask)
```

## Limitations

- Only supported in `mha_varlen_fwd` (variable length forward pass)
- Backward pass (gradient computation) uses query-based layout
- Paged KV cache support is experimental
- Both mask and bias can independently use either layout

## Compatibility

- GPU: Requires Ampere (SM80) or newer
- PyTorch: Compatible with existing Flash Attention interfaces
- Mixed Layouts: Mask and bias can use different layouts in the same call
- GQA/MQA: Full support for grouped-query and multi-query attention

## Related Work

This feature aligns with:
- Sparse attention patterns in modern LLMs
- Efficient KV cache management
- Streaming inference workloads
- MaskMod and FlexAttention paradigms
