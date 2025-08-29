# Compute Bubble Reduction Usage Guide

This guide explains how to use and benchmark the compute bubble reduction optimizations implemented in the backward kernel.

## Quick Start

The optimizations are **automatically enabled** when using flash-dmattn - no code changes required!

```python
import torch
from flash_dmattn import flash_dmattn_func

# Create sparse attention data
q = torch.randn(2, 512, 8, 64, dtype=torch.float16, device="cuda")
k = torch.randn(2, 512, 8, 64, dtype=torch.float16, device="cuda") 
v = torch.randn(2, 512, 8, 64, dtype=torch.float16, device="cuda")

# Create sparse mask (70% sparse = 30% density)
mask = torch.rand(2, 8, 512, 512, device="cuda") > 0.7

# Run attention with automatic bubble reduction optimizations
output = flash_dmattn_func(q, k, v, mask=mask)
loss = output.sum()
loss.backward()  # Optimizations automatically apply here
```

## Performance Testing

### Basic Performance Test

```bash
# Run the test suite
python test_bubble_reduction.py

# Run performance benchmark
python benchmark_bubble_reduction.py --seq-len 512 --batch-size 4
```

### Advanced Benchmarking

```python
# Benchmark specific sparsity patterns
python benchmark_bubble_reduction.py \
    --batch-size 2 \
    --num-heads 8 \
    --seq-len 1024 \
    --head-dim 64 \
    --output results.json
```

### Expected Performance Gains

| Sparsity Level | Density | Pattern Type | Expected Speedup |
|----------------|---------|--------------|------------------|
| 90%            | 10%     | Any          | 2-4x             |
| 70%            | 30%     | Block-sparse | 1.5-3x           |
| 50%            | 50%     | Random       | 1.2-2x           |
| 30%            | 70%     | Structured   | 1.1-1.5x         |
| 10%            | 90%     | Dense        | 1.0x (adaptive)  |

## Optimization Details

### When Optimizations Apply

1. **High Benefit Scenarios**:
   - Sparsity ≥ 70% (density ≤ 30%)
   - Block-sparse patterns with large masked regions
   - Structured attention with many inactive tiles

2. **Medium Benefit Scenarios**:
   - Sparsity 40-70% (density 30-60%)
   - Random sparsity patterns
   - Mixed dense/sparse regions

3. **Adaptive Fallback**:
   - Density > 85% → Skip logic automatically disabled
   - Prevents optimization overhead in dense scenarios

### Verification

To verify optimizations are working:

```python
import torch
from flash_dmattn import flash_dmattn_func

# Test with very sparse mask (should see significant benefit)
sparse_mask = torch.zeros(1, 1, 128, 128, device="cuda")
sparse_mask[:, :, :16, :16] = 1.0  # Only 1/64 of attention active

q = torch.randn(1, 128, 1, 64, dtype=torch.float16, device="cuda")
k = torch.randn(1, 128, 1, 64, dtype=torch.float16, device="cuda")
v = torch.randn(1, 128, 1, 64, dtype=torch.float16, device="cuda")

# Time with sparse mask
import time
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    output = flash_dmattn_func(q, k, v, mask=sparse_mask)
    output.sum().backward()
torch.cuda.synchronize()
sparse_time = time.time() - start

# Compare with dense mask  
dense_mask = torch.ones(1, 1, 128, 128, device="cuda")
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    output = flash_dmattn_func(q, k, v, mask=dense_mask)
    output.sum().backward()
torch.cuda.synchronize()
dense_time = time.time() - start

speedup = dense_time / sparse_time
print(f"Speedup: {speedup:.2f}x")
```

## Troubleshooting

### Performance Not as Expected?

1. **Check sparsity level**:
   ```python
   density = float(torch.sum(mask)) / mask.numel()
   print(f"Mask density: {density:.2f}")
   # Should be < 0.85 for optimizations to apply
   ```

2. **Verify sparse pattern**:
   ```python
   # Count fully masked tiles (most beneficial)
   block_size = 64  # Typical block size
   masked_blocks = 0
   total_blocks = 0
   for i in range(0, mask.shape[-2], block_size):
       for j in range(0, mask.shape[-1], block_size):
           block = mask[:, :, i:i+block_size, j:j+block_size]
           if torch.sum(block) == 0:
               masked_blocks += 1
           total_blocks += 1
   
   print(f"Fully masked blocks: {masked_blocks}/{total_blocks} ({100*masked_blocks/total_blocks:.1f}%)")
   ```

3. **Profile memory bandwidth**:
   ```bash
   # Use nvidia-smi or nsight-compute to verify reduced memory traffic
   nvidia-smi dmon -s u -d 1
   ```

### Common Issues

- **No speedup with random sparsity**: Random patterns have fewer fully masked tiles
- **Overhead with dense attention**: Adaptive mode should disable optimizations automatically  
- **Memory errors**: Optimizations don't change memory requirements
- **Numerical differences**: Should be within floating-point precision

## Integration with Existing Code

### Drop-in Replacement

```python
# Before: Standard attention
output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

# After: Flash attention with automatic optimizations  
output = flash_dmattn_func(q, k, v, mask=mask)
```

### Hugging Face Integration

```python
from transformers import AutoModel
from flash_dmattn.integrations import replace_attention_with_flash_dmattn

# Replace attention implementation
model = AutoModel.from_pretrained("bert-base-uncased")
model = replace_attention_with_flash_dmattn(model)

# Sparse attention patterns automatically optimized
```

### Custom Training Loops

```python
for batch in dataloader:
    q, k, v, mask = batch
    
    # Forward pass with optimizations
    output = flash_dmattn_func(q, k, v, mask=mask)
    loss = criterion(output, target)
    
    # Backward pass with bubble reduction
    loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()
```

## Monitoring Performance

### Basic Timing

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    output = flash_dmattn_func(q, k, v, mask=sparse_mask)
    output.sum().backward()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Advanced Profiling

```bash
# Use nsight-compute for detailed kernel analysis
ncu --set full --target-processes application python your_script.py

# Look for:
# - Reduced memory transactions for masked tiles
# - Higher instruction throughput 
# - Fewer stalled cycles
```

## Best Practices

1. **Design sparse patterns to maximize block-level sparsity**
2. **Use structured patterns when possible (better than random)**
3. **Monitor density - adjust sparsity thresholds if needed**
4. **Profile end-to-end performance, not just attention**
5. **Consider attention pattern evolution during training**

## Support

For issues or questions:
- Check the test suite: `python test_bubble_reduction.py`
- Run benchmarks: `python benchmark_bubble_reduction.py`
- Review documentation: `docs/compute_bubble_reduction.md`
- File issues on GitHub with performance profiles