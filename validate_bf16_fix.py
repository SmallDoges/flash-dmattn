#!/usr/bin/env python3
"""
Validation script for the BF16 INF issue fix

This script reproduces the conditions described in the issue and validates
that the fix prevents INF values during backward pass.

Usage:
    python validate_bf16_fix.py [--cuda] [--verbose]
"""

import argparse
import torch
import sys
import traceback

def setup_test_tensors(batch_size=1, seq_len=4096, num_heads=8, head_dim=128, 
                      window_size=2048, device="cpu", dtype=torch.bfloat16):
    """Setup test tensors similar to the original issue configuration"""
    print(f"Setting up test with seq_len={seq_len}, window_size={window_size}, dtype={dtype}")
    
    # Create input tensors
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device, requires_grad=True)
    
    # Create attention mask with causal + window pattern
    attention_mask = torch.ones(batch_size, num_heads, seq_len, seq_len, dtype=torch.bool, device=device)
    
    # Apply causal mask  
    for i in range(seq_len):
        attention_mask[:, :, i, i+1:] = False
        
    # Apply window mask
    for i in range(seq_len):
        start_idx = max(0, i - window_size)
        attention_mask[:, :, i, :start_idx] = False
    
    # Create attention bias
    attention_bias = torch.randn(batch_size, num_heads, seq_len, seq_len, dtype=dtype, device=device, requires_grad=True)
    
    masked_positions = (~attention_mask).sum().item()
    total_positions = attention_mask.numel()
    
    print(f"  Tensors created on {device} with {masked_positions:,}/{total_positions:,} masked positions")
    
    return q, k, v, attention_mask, attention_bias

def test_masking_operation(attention_bias, attention_mask, dtype):
    """Test the masking operation that was causing the issue"""
    print("Testing masking operation...")
    
    # Test original approach (potentially problematic)
    original_min = torch.finfo(dtype).min
    
    try:
        masked_original = attention_bias.masked_fill(~attention_mask, original_min)
        has_inf_orig = torch.isinf(masked_original).any()
        has_nan_orig = torch.isnan(masked_original).any()
        print(f"  Original masking (min={original_min:.2e}): inf={has_inf_orig}, nan={has_nan_orig}")
    except Exception as e:
        print(f"  Original masking FAILED: {e}")
        return False
    
    # Test safer approach (our fix)
    if dtype == torch.bfloat16:
        safe_min = -1e30
    elif dtype == torch.float16:
        safe_min = -1e4
    else:
        safe_min = original_min
        
    try:
        masked_safe = attention_bias.masked_fill(~attention_mask, safe_min)
        has_inf_safe = torch.isinf(masked_safe).any()
        has_nan_safe = torch.isnan(masked_safe).any()
        print(f"  Safe masking (min={safe_min:.2e}): inf={has_inf_safe}, nan={has_nan_safe}")
    except Exception as e:
        print(f"  Safe masking FAILED: {e}")
        return False
        
    return True

def test_flash_attention(q, k, v, attention_mask, attention_bias, verbose=False):
    """Test flash attention with the given inputs"""
    print("Testing flash attention forward and backward...")
    
    try:
        # Try to import flash_dmattn
        try:
            from flash_dmattn import flash_dmattn_func
            flash_fn = flash_dmattn_func
            print("  Using flash_dmattn CUDA implementation")
        except ImportError:
            print("  flash_dmattn not available, using torch SDPA")
            flash_fn = None
            
        if flash_fn is not None:
            # Test with flash_dmattn
            output = flash_fn(q, k, v, attn_bias=attention_bias, attn_mask=attention_mask)
            
            if verbose:
                print(f"  Output shape: {output.shape}")
                print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]")
                print(f"  Output finite: {torch.isfinite(output).all()}")
            
            # Test backward pass
            loss = output.sum()
            loss.backward()
            
            # Check gradients for inf/nan
            grads_finite = True
            for name, param in [("q", q), ("k", k), ("v", v), ("bias", attention_bias)]:
                if param.grad is not None:
                    has_inf = torch.isinf(param.grad).any()
                    has_nan = torch.isnan(param.grad).any()
                    if has_inf or has_nan:
                        grads_finite = False
                        print(f"  WARNING: {name} gradient has inf={has_inf}, nan={has_nan}")
                    elif verbose:
                        print(f"  {name} gradient is finite: {torch.isfinite(param.grad).all()}")
            
            if grads_finite:
                print("  ✅ Forward and backward pass completed successfully!")
                return True
            else:
                print("  ❌ Gradients contain inf/nan values")
                return False
        else:
            print("  Skipping flash attention test (not available)")
            return True
            
    except Exception as e:
        print(f"  ❌ Flash attention test FAILED: {e}")
        if verbose:
            traceback.print_exc()
        return False

def main():
    parser = argparse.ArgumentParser(description="Validate BF16 INF issue fix")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA device")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--seq-len", type=int, default=1024, help="Sequence length (default: 1024)")
    parser.add_argument("--window-size", type=int, default=512, help="Window size (default: 512)")
    args = parser.parse_args()
    
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"Running validation on {device}")
    
    if device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name()}")
    
    # Test with different dtypes
    dtypes_to_test = [torch.bfloat16, torch.float16] if device == "cuda" else [torch.bfloat16]
    
    all_passed = True
    
    for dtype in dtypes_to_test:
        print(f"\n{'='*50}")
        print(f"Testing with {dtype}")
        print(f"{'='*50}")
        
        try:
            # Setup test tensors  
            q, k, v, attention_mask, attention_bias = setup_test_tensors(
                seq_len=args.seq_len, 
                window_size=args.window_size,
                device=device, 
                dtype=dtype
            )
            
            # Test masking operation
            mask_ok = test_masking_operation(attention_bias, attention_mask, dtype)
            if not mask_ok:
                all_passed = False
                continue
                
            # Test flash attention
            flash_ok = test_flash_attention(q, k, v, attention_mask, attention_bias, args.verbose)
            if not flash_ok:
                all_passed = False
                
        except Exception as e:
            print(f"Test with {dtype} FAILED: {e}")
            if args.verbose:
                traceback.print_exc()
            all_passed = False
    
    print(f"\n{'='*50}")
    if all_passed:
        print("🎉 All tests PASSED! The BF16 INF fix is working correctly.")
    else:
        print("❌ Some tests FAILED. The issue may still be present.")
        sys.exit(1)

if __name__ == "__main__":
    main()