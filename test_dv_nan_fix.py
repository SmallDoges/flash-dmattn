#!/usr/bin/env python3
"""
Test script to validate the NaN/Inf fix in dV backward pass.
This script specifically tests the failing configuration mentioned in the issue.
"""
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import sys
import traceback

def test_dv_nan_fix():
    """Test the specific configuration that was failing with NaN/Inf in dV gradients."""
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return True
    
    try:
        # Import the triton implementation
        from flash_dmattn.flash_dmattn_triton import triton_dmattn_func
        print("✅ Successfully imported flash_dmattn_triton")
    except ImportError as e:
        print(f"❌ Failed to import flash_dmattn_triton: {e}")
        return False

    # Test configuration from the issue
    torch.manual_seed(42)
    device = "cuda"
    B, H, HKV = 1, 1, 1
    Q_LEN = 256
    K_LEN = 256
    D = 64
    is_causal = True

    print(f"Testing configuration: B={B}, H={H}, HKV={HKV}, Q_LEN={Q_LEN}, K_LEN={K_LEN}, D={D}, is_causal={is_causal}")

    # Create input tensors
    q = torch.randn(B, Q_LEN, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, K_LEN, HKV, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, K_LEN, HKV, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    attn_mask = None
    attn_bias = None

    # Test multiple runs to ensure stability
    for run in range(5):
        print(f"\nRun {run + 1}/5:")
        
        # Clear gradients
        if q.grad is not None:
            q.grad.zero_()
        if k.grad is not None:
            k.grad.zero_()
        if v.grad is not None:
            v.grad.zero_()
        
        # Forward and backward pass
        out = triton_dmattn_func(q, k, v, attn_mask, attn_bias, is_causal=is_causal, scale=None)
        loss = out.sum()
        loss.backward()

        # Check for NaN/Inf in gradients
        has_nan_dv = torch.isnan(v.grad).any().item()
        has_inf_dv = torch.isinf(v.grad).any().item()
        has_nan_dk = torch.isnan(k.grad).any().item()
        has_inf_dk = torch.isinf(k.grad).any().item()
        has_nan_dq = torch.isnan(q.grad).any().item()
        has_inf_dq = torch.isinf(q.grad).any().item()
        
        print(f"  dV - NaN: {has_nan_dv}, Inf: {has_inf_dv}")
        print(f"  dK - NaN: {has_nan_dk}, Inf: {has_inf_dk}")
        print(f"  dQ - NaN: {has_nan_dq}, Inf: {has_inf_dq}")
        
        # Check gradient ranges
        if v.grad is not None:
            dv_min = torch.min(v.grad).item()
            dv_max = torch.max(v.grad).item()
            print(f"  dV range: [{dv_min:.6f}, {dv_max:.6f}]")
        
        if k.grad is not None:
            dk_min = torch.min(k.grad).item()
            dk_max = torch.max(k.grad).item()
            print(f"  dK range: [{dk_min:.6f}, {dk_max:.6f}]")
        
        if q.grad is not None:
            dq_min = torch.min(q.grad).item()
            dq_max = torch.max(q.grad).item()
            print(f"  dQ range: [{dq_min:.6f}, {dq_max:.6f}]")
        
        # Fail if any gradient contains NaN/Inf
        if has_nan_dv or has_inf_dv or has_nan_dk or has_inf_dk or has_nan_dq or has_inf_dq:
            print(f"❌ Run {run + 1} FAILED: Found NaN/Inf in gradients")
            return False
        else:
            print(f"✅ Run {run + 1} PASSED: All gradients are finite")
    
    print("\n🎉 All test runs passed! NaN/Inf issue appears to be fixed.")
    return True


def test_additional_configurations():
    """Test additional configurations to ensure the fix is robust."""
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping additional tests")
        return True
    
    try:
        from flash_dmattn.flash_dmattn_triton import triton_dmattn_func
    except ImportError as e:
        print(f"❌ Failed to import flash_dmattn_triton: {e}")
        return False

    # Additional test configurations
    test_configs = [
        # (B, H, HKV, Q_LEN, K_LEN, D, is_causal)
        (1, 1, 1, 128, 128, 64, True),
        (1, 1, 1, 256, 256, 32, True),
        (1, 2, 1, 128, 128, 64, True),
        (2, 1, 1, 128, 128, 64, True),
        (1, 1, 1, 256, 256, 64, False),
    ]
    
    device = "cuda"
    all_passed = True
    
    for i, (B, H, HKV, Q_LEN, K_LEN, D, is_causal) in enumerate(test_configs):
        print(f"\nAdditional Test {i+1}: B={B}, H={H}, HKV={HKV}, Q_LEN={Q_LEN}, K_LEN={K_LEN}, D={D}, is_causal={is_causal}")
        
        torch.manual_seed(42 + i)  # Different seed for each config
        
        q = torch.randn(B, Q_LEN, H, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(B, K_LEN, HKV, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(B, K_LEN, HKV, D, device=device, dtype=torch.bfloat16, requires_grad=True)
        
        out = triton_dmattn_func(q, k, v, None, None, is_causal=is_causal, scale=None)
        loss = out.sum()
        loss.backward()
        
        # Check for NaN/Inf
        has_nan = any([
            torch.isnan(q.grad).any().item() if q.grad is not None else False,
            torch.isnan(k.grad).any().item() if k.grad is not None else False,
            torch.isnan(v.grad).any().item() if v.grad is not None else False,
        ])
        has_inf = any([
            torch.isinf(q.grad).any().item() if q.grad is not None else False,
            torch.isinf(k.grad).any().item() if k.grad is not None else False,
            torch.isinf(v.grad).any().item() if v.grad is not None else False,
        ])
        
        if has_nan or has_inf:
            print(f"❌ Additional Test {i+1} FAILED: Found NaN/Inf in gradients")
            all_passed = False
        else:
            print(f"✅ Additional Test {i+1} PASSED")
    
    return all_passed


if __name__ == "__main__":
    print("🧪 Testing NaN/Inf fix in dV backward pass")
    print("=" * 50)
    
    try:
        # Test the specific failing configuration
        main_test_passed = test_dv_nan_fix()
        
        # Test additional configurations
        additional_tests_passed = test_additional_configurations()
        
        # Overall result
        if main_test_passed and additional_tests_passed:
            print("\n🎉 ALL TESTS PASSED! The NaN/Inf issue in dV gradients appears to be resolved.")
            sys.exit(0)
        else:
            print("\n😞 SOME TESTS FAILED! The fix may need further refinement.")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n💥 Test execution failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)