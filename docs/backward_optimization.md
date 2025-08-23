# Backward Launch Template Optimizations

This document describes the architecture-specific optimizations implemented for the backward pass in Flash Dynamic Mask Attention.

## Overview

The backward launch template optimization provides adaptive kernel selection based on:
- GPU architecture (SM 8.0, 8.6, 8.9, 9.0)
- Problem dimensions (sequence length, batch size, head dimension)
- Available shared memory
- Performance characteristics of different configurations

## Architecture-Specific Features

### SM 9.0 (H100/H200)
- **Large Block Optimization**: Uses 128x128 blocks for optimal memory bandwidth
- **Multi-level Shared Memory**: Leverages advanced memory hierarchy
- **Long Sequence Support**: Optimized for sequences ≥8K tokens
- **High Memory Bandwidth**: Target >85% peak bandwidth utilization

### SM 8.9 (Ada Lovelace/H200)
- **Variable Sequence Optimization**: Adaptive block sizes based on sequence length
- **Medium-Large Block Support**: 64x128 to 128x64 blocks depending on workload
- **Memory-Aware Selection**: Adjusts configuration based on available shared memory
- **Register Optimization**: Uses V-in-registers for memory-constrained scenarios

### SM 8.6 (A100)
- **Memory-Optimized Configurations**: Balances performance with memory constraints
- **Double Buffering Control**: Adaptive enable/disable based on memory availability
- **Standard Block Sizes**: 64x128 blocks for most scenarios

### SM 8.0 and below
- **Legacy Fallback**: Compatible configurations for older architectures
- **Conservative Memory Usage**: Single buffering, smaller block sizes
- **Reduced Feature Set**: Focus on correctness over peak performance

## Configuration Selection Logic

### Head Dimension 32
- **H100**: Large blocks (64x128) with V-in-registers for optimal register usage
- **Ada**: Sequence-aware optimization - larger blocks for long sequences (≥4K)
- **A100**: Standard configuration with memory-aware buffering

### Head Dimension 64
- **H100**: Very large blocks (128x128) when memory allows, optimal bandwidth
- **Ada**: Long sequence detection (≥8K) triggers large block optimization
- **A100**: Standard 128x128 or fallback to 64x128 with V-in-registers

### Head Dimension 128
- **H100**: Adaptive 128x128 vs 64x128 based on memory availability
- **Ada**: Sequence-aware block selection (128x64 for long, 64x128 for standard)
- **A100**: Traditional 64x128 vs 64x64 with register optimization

### Head Dimension 256
- **H100**: Memory tier-aware selection (176KB → 144KB → <144KB tiers)
- **Ada**: Long sequence detection with optimized memory patterns
- **A100**: Progressive degradation: double buffering → single buffering → V-in-registers

## Performance Profiling

### Enabling Profiling

Set the environment variable to enable optimization choice logging:

```bash
export FLASH_DMATTN_PROFILE_BACKWARD=1
```

### Profiling Output

When enabled, the system logs optimization choices in the format:
```
FLASH_DMATTN_PROFILE: HeadDim=128, Arch=SM9.0, SeqQ=8192, SeqK=8192, Batch=2, Choice=SM90_LargeBlock_128x128
```

### Optimization Choice Codes

- `SM90_LargeBlock_128x128`: H100 large block optimization
- `SM90_MediumBlock_64x128`: H100 medium block optimization
- `SM89_LongSeq_128x64`: Ada long sequence optimization
- `SM89_StandardSeq_64x128`: Ada standard sequence optimization
- `SM89_LowMem_64x64_VinRegs`: Ada memory-constrained optimization
- `SM86_Standard_64x128`: A100 standard optimization
- `SM86_LowMem_64x64_VinRegs`: A100 memory-constrained optimization

## Performance Expectations

### Target Improvements

- **15-25% reduction** in backward pass latency for long sequences
- **>85% memory bandwidth** utilization on H100/H200
- **Zero register spilling** for common configurations
- **>80% occupancy** maintained across problem sizes

### Sequence Length Optimizations

- **≥8K tokens**: Long sequence optimizations (large blocks, bandwidth focus)
- **≥4K tokens**: Medium sequence optimizations (balanced approach)
- **<4K tokens**: Standard optimizations (occupancy focus)

### Batch Size Optimizations

- **≤4 batch size**: Smaller block M dimension for improved occupancy
- **>4 batch size**: Standard block sizes for throughput

## Compatibility

### Backward Compatibility
- All existing kernel launches continue to work
- Graceful fallback for unsupported architectures
- No changes to Python API

### Architecture Support
- **Required**: SM 8.0+ for Flash Attention features
- **Optimized**: SM 8.6+ for advanced optimizations
- **Latest**: SM 8.9+ and 9.0 for cutting-edge features

### Memory Requirements

Different shared memory tiers:
- **High (176+ KB)**: Full optimization set (H100)
- **Medium (144+ KB)**: Standard optimizations (A100)
- **Low (<144 KB)**: Memory-constrained optimizations (older cards)

## Debugging

### Common Issues

1. **Insufficient Shared Memory**: System automatically selects memory-constrained variants
2. **Unsupported Architecture**: Falls back to legacy optimizations
3. **Very Long Sequences**: May require memory optimization or chunking

### Performance Analysis

Use profiling output to understand optimization choices:
```bash
export FLASH_DMATTN_PROFILE_BACKWARD=1
python your_training_script.py 2>&1 | grep FLASH_DMATTN_PROFILE
```

### Memory Analysis

Check shared memory availability:
```python
import torch
props = torch.cuda.get_device_properties(0)
print(f"Max shared memory: {props.max_shared_memory_per_block_optin / 1024:.0f} KB")
```

## Future Enhancements

### Planned Improvements

1. **Runtime Auto-tuning**: Benchmark and cache optimal configurations
2. **Heuristic Models**: Mathematical models for configuration prediction
3. **Advanced Memory Patterns**: Multi-level shared memory utilization
4. **Occupancy Optimization**: Dynamic warp scheduling improvements

### Integration Points

- **PyTorch 2.0+ support**: Full compatibility with latest PyTorch versions
- **CUDA 12.x features**: Asynchronous execution pattern utilization
- **Multi-GPU scaling**: Distributed training optimizations
- **Mixed precision**: Enhanced bfloat16/float16 gradient handling

## Examples

### Basic Usage

```python
import torch
from flash_dmattn import flash_dmattn_func

# Enable profiling (optional)
import os
os.environ['FLASH_DMATTN_PROFILE_BACKWARD'] = '1'

# Your model will automatically use optimized backward kernels
q = torch.randn(2, 16, 8192, 128, device='cuda', requires_grad=True)
k = torch.randn(2, 16, 8192, 128, device='cuda', requires_grad=True)  
v = torch.randn(2, 16, 8192, 128, device='cuda', requires_grad=True)

out = flash_dmattn_func(q, k, v, is_causal=True)
loss = out.sum()
loss.backward()  # This will use the optimized backward kernels
```

### Performance Monitoring

```python
# Monitor optimization choices
import subprocess
import os

os.environ['FLASH_DMATTN_PROFILE_BACKWARD'] = '1'

# Run your training
# Check optimization log
result = subprocess.run(['python', 'train.py'], capture_output=True, text=True)
profile_lines = [line for line in result.stderr.split('\n') if 'FLASH_DMATTN_PROFILE' in line]
for line in profile_lines:
    print(line)
```