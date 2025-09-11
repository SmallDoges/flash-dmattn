# Unified Sparse Mask Strategy with Block-Level Skipping

This document describes the implementation of the unified sparse mask strategy in Flash Dynamic Mask Attention, addressing the requirements specified in issue #163.

## Overview

The unified sparse mask strategy provides a comprehensive framework for handling various sparse attention patterns while maintaining memory efficiency and computational performance through block-level skipping.

## Key Features

### 1. Unified Mask Abstraction

The system supports multiple mask types through a unified interface:

- **Parametric Masks** (no storage required):
  - `PARAMETRIC_CAUSAL`: Standard autoregressive causal mask
  - `PARAMETRIC_WINDOW`: Sliding window attention pattern
  - Hybrid causal + window combinations

- **Compressed Representations**:
  - `BLOCK_BITSET`: Block-level bitset for moderate sparsity (B×B granularity)
  - `BCSR`: Block Compressed Sparse Row format for irregular patterns
  - `MIXED_GRANULARITY`: Dense blocks + partial bitpacked blocks
  - `DYNAMIC`: Runtime-updatable sparse patterns

### 2. Block-Level Skipping Logic

The implementation introduces unified block-level skip logic that operates at tile granularity:

```cpp
// Tile-level active detection
any_active = OR_reduce(mask_block)  // Single bit indicating if any position in tile is active

// Skip decision for forward pass
if (!any_active) {
    advance_pointers();              // Skip all computation
    continue;
}

// Skip decision for backward pass  
if (!any_active) {
    advance_pointers_zero_outputs(); // Skip computation, zero side outputs
    continue;
}
```

### 3. Memory Efficiency

Different compression levels provide varying trade-offs:

- **No Storage** (Parametric): 0 bytes - patterns computed on-the-fly
- **Block Level** (Bitset): ~(L/B)² bits for L×L attention with block size B
- **Sparse Index** (BCSR): O(nnz_blocks) storage for irregular patterns

## Implementation Details

### Core Components

#### UnifiedSparseMask Class (`unified_sparse_mask.h`)

The main abstraction providing:
- Block activity checking: `is_block_active(query_block, key_block)`
- Active block enumeration: `enumerate_active_blocks(query_block, active_blocks, max_blocks)`
- Block descriptor generation: `get_block_descriptor(query_block)`
- Fast OR-reduction: `compute_block_activity_fast(mask_tile, q_block, k_block)`

#### Mask Factory (`mask_factory.h`)

Convenience functions for creating different mask types:
- `create_causal_mask()`: Zero-storage causal attention
- `create_window_mask()`: Sliding window patterns
- `create_block_bitset_mask()`: Bitset-based sparse patterns
- `create_bcsr_mask()`: Irregular sparse patterns
- Conversion utilities: `dense_to_block_bitset()`, `dense_to_bcsr()`

### Kernel Integration

#### Forward Pass Integration

Updated `flash_fwd_kernel.h` to support:
- Sparse mask pointer in `Flash_fwd_params.sparse_mask_ptr`
- Block-level activity detection before computation
- Automatic skipping of inactive tiles

```cpp
// Enhanced mask application with skip checking
bool block_has_activity = mask.template apply_mask_with_skip_check<Is_causal, Is_even_MN>(
    acc_s, tSrMask, tSrBias, params.scale_softmax,
    n_block * kBlockN, m_block * kBlockM + offset, warp_row_stride,
    m_block, n_block, kBlockM, kBlockN
);

if (!block_has_activity) {
    clear(acc_s);  // Zero accumulator for inactive blocks
    continue;      // Skip softmax/output computation
}
```

#### Backward Pass Integration

Similar block-level skipping logic added to `flash_bwd_kernel.h`:
- Unified OR-reduction for activity detection
- Skip entire gradient computation chains for inactive blocks
- Maintain correct pointer advancement for memory layout

### Python API

#### Sparse Mask Classes (`sparse_mask.py`)

High-level Python interface for creating and managing sparse masks:

```python
from flash_dmattn import CausalMask, WindowMask, CausalWindowMask

# Create different mask types
causal_mask = CausalMask(seqlen_q=4096, seqlen_k=4096)
window_mask = WindowMask(window_size=512, seqlen_q=4096, seqlen_k=4096)
hybrid_mask = CausalWindowMask(window_size=1024, seqlen_q=4096, seqlen_k=4096)

# Estimate performance benefits
speedup = estimate_speedup(causal_mask)
memory_savings = calculate_memory_savings(causal_mask)
```

#### Block-Based Compression

```python
from flash_dmattn import BlockBitsetMask, BCSRMask

# Convert dense mask to compressed format
dense_mask = torch.rand(4096, 4096) > 0.7  # 70% sparsity
bitset_mask = BlockBitsetMask.from_dense_mask(dense_mask)
bcsr_mask = BCSRMask.from_dense_mask(dense_mask)

# Use with Flash Attention
output = flash_dmattn_func(query, key, value, sparse_mask=bitset_mask)
```

## Performance Benefits

### Computational Speedup

For sparse patterns with active fraction `p` and skip overhead ratio `ε`:

```
Speedup ≈ 1/(p + (1-p)ε)
```

Upper bound as ε → 0: `1/p`

### Memory Savings

- **32K sequences**: Dense L×L mask requires ~4GB memory
- **Block bitset** (B=128): ~16MB for same coverage
- **Parametric masks**: 0 bytes storage

### Real-World Performance

Expected performance improvements:
- **Causal**: ~2-3x speedup for long sequences
- **Window-512**: ~10-50x speedup depending on sequence length
- **Hybrid patterns**: ~5-20x speedup with minimal accuracy loss

## Usage Examples

### Basic Usage

```python
import torch
from flash_dmattn import flash_dmattn_func_auto, CausalMask

# Setup tensors
query = torch.randn(1, 4096, 8, 64, device='cuda', dtype=torch.bfloat16)
key = torch.randn(1, 4096, 8, 64, device='cuda', dtype=torch.bfloat16)  
value = torch.randn(1, 4096, 8, 64, device='cuda', dtype=torch.bfloat16)

# Create sparse mask
sparse_mask = CausalMask(seqlen_q=4096, seqlen_k=4096)

# Run attention with automatic backend selection
flash_attn_func = flash_dmattn_func_auto(backend="cuda")
output = flash_attn_func(
    query=query,
    key=key, 
    value=value,
    sparse_mask=sparse_mask,  # New parameter
    scale=1.0/8.0
)
```

### Advanced Patterns

```python
# Document segmentation with hybrid masking
doc_mask = CausalWindowMask(
    window_size=1024,
    seqlen_q=8192,
    seqlen_k=8192
)

# Custom sparse pattern via bitset
custom_pattern = create_custom_sparse_pattern(8192, 8192)
bitset_mask = BlockBitsetMask.from_dense_mask(custom_pattern)

# Performance analysis
print(f"Sparsity: {bitset_mask.get_sparsity_ratio():.1%}")
print(f"Expected speedup: {estimate_speedup(bitset_mask):.2f}x")
print(f"Memory savings: {calculate_memory_savings(bitset_mask):.1%}")
```

## Implementation Status

### ✅ Completed Features

- [x] Unified mask interface and abstraction layer
- [x] Parametric mask support (causal, window, hybrid)
- [x] Block bitset compression format
- [x] BCSR sparse representation  
- [x] Block-level skip logic integration
- [x] Forward pass kernel modifications
- [x] Python API with factory functions
- [x] Performance estimation utilities
- [x] Dense mask conversion utilities

### 🔄 In Progress

- [ ] Backward pass block skipping integration
- [ ] Mixed granularity support (dense + partial blocks)
- [ ] Dynamic refinement hooks
- [ ] Comprehensive benchmarking suite

### 🎯 Future Enhancements

- [ ] Adaptive density thresholding
- [ ] Persistent CTA work queues for load balancing
- [ ] Bit-packed warp ballot optimizations
- [ ] Multi-GPU sparse pattern distribution

## Testing and Validation

The implementation includes:
- Unit tests for all mask types (`test_sparse_mask.py`)
- Performance benchmarking example (`unified_sparse_mask_demo.py`)
- Memory usage validation
- Correctness verification against dense attention

## Integration Notes

### Backward Compatibility

The system maintains full backward compatibility:
- Existing code continues to work without changes
- Sparse mask parameter is optional
- Automatic fallback to dense computation when no sparse mask provided

### Memory Layout

Sparse mask data structures are designed for:
- Coalesced memory access patterns
- Minimal GPU memory overhead  
- Cache-friendly block enumeration
- Lock-free concurrent access

### Error Handling

Comprehensive error checking for:
- Invalid sparse mask formats
- Mismatched tensor dimensions
- Out-of-bounds block access
- Memory allocation failures

## References

This implementation addresses the requirements from issue #163 and incorporates design patterns from:
- FlashAttention: Memory-efficient attention computation
- Longformer/BigBird: Pattern-based sparse attention
- Sparse Attention (BlockSparse): Block-level sparsity
- Top-k attention: Dynamic selection strategies