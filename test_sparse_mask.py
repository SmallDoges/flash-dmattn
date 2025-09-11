#!/usr/bin/env python3
"""
Basic tests for the Unified Sparse Mask functionality

This test suite validates the core functionality of different sparse mask types
without requiring CUDA kernels to be built.
"""

import sys
import os

# Add the parent directory to Python path to import flash_dmattn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
except ImportError:
    print("PyTorch not available - skipping tests")
    sys.exit(0)

try:
    from flash_dmattn.sparse_mask import (
        CausalMask, WindowMask, CausalWindowMask, 
        BlockBitsetMask, BCSRMask, create_sparse_mask,
        estimate_speedup, calculate_memory_savings
    )
    SPARSE_MASK_AVAILABLE = True
except ImportError as e:
    print(f"Sparse mask API not available: {e}")
    SPARSE_MASK_AVAILABLE = False


def test_causal_mask():
    """Test causal mask functionality."""
    print("Testing CausalMask...")
    
    mask = CausalMask(seqlen_q=256, seqlen_k=256, block_size_m=64, block_size_n=64)
    
    # Test basic properties
    assert mask.get_mask_type() == "PARAMETRIC_CAUSAL"
    assert mask.estimate_memory_usage() == 0  # No storage required
    assert mask.num_query_blocks == 4
    assert mask.num_key_blocks == 4
    
    # Test block activity (causal pattern)
    assert mask.is_block_active(0, 0) == True   # Diagonal block
    assert mask.is_block_active(1, 0) == True   # Lower triangular
    assert mask.is_block_active(0, 1) == False  # Upper triangular
    assert mask.is_block_active(3, 3) == True   # Last diagonal
    
    # Test sparsity
    active_blocks = mask.count_active_blocks()
    total_blocks = mask.num_query_blocks * mask.num_key_blocks
    expected_active = 10  # For 4x4 causal: blocks (0,0), (1,0), (1,1), (2,0), (2,1), (2,2), (3,0), (3,1), (3,2), (3,3)
    assert active_blocks == expected_active, f"Expected {expected_active} active blocks, got {active_blocks}"
    
    print("✓ CausalMask tests passed")


def test_window_mask():
    """Test sliding window mask functionality."""
    print("Testing WindowMask...")
    
    mask = WindowMask(window_size=128, seqlen_q=256, seqlen_k=256, block_size_m=64, block_size_n=64)
    
    # Test basic properties
    assert mask.get_mask_type() == "PARAMETRIC_WINDOW"
    assert mask.estimate_memory_usage() == 0  # No storage required
    assert mask.window_size == 128
    
    # Test CUDA parameters
    params = mask.get_cuda_params()
    assert params["mask_type"] == 1
    assert params["use_window"] == True
    assert params["window_size"] == 128
    
    print("✓ WindowMask tests passed")


def test_causal_window_mask():
    """Test hybrid causal + window mask functionality."""
    print("Testing CausalWindowMask...")
    
    mask = CausalWindowMask(window_size=128, seqlen_q=256, seqlen_k=256, block_size_m=64, block_size_n=64)
    
    # Test basic properties
    assert mask.get_mask_type() == "PARAMETRIC_WINDOW"
    assert mask.estimate_memory_usage() == 0
    
    # Test CUDA parameters (hybrid: causal + window)
    params = mask.get_cuda_params()
    assert params["is_causal"] == True
    assert params["use_window"] == True
    assert params["window_size"] == 128
    
    print("✓ CausalWindowMask tests passed")


def test_block_bitset_mask():
    """Test block bitset mask functionality."""
    print("Testing BlockBitsetMask...")
    
    # Create a simple test pattern
    device = torch.device('cpu')  # Use CPU for testing
    seqlen_q, seqlen_k = 128, 128
    block_size_m, block_size_n = 32, 32
    
    # Create dense mask (diagonal pattern)
    dense_mask = torch.eye(seqlen_q, seqlen_k, device=device)
    
    # Convert to bitset mask
    mask = BlockBitsetMask.from_dense_mask(dense_mask, block_size_m, block_size_n)
    
    # Test basic properties
    assert mask.get_mask_type() == "BLOCK_BITSET"
    assert mask.seqlen_q == seqlen_q
    assert mask.seqlen_k == seqlen_k
    assert mask.num_query_blocks == 4  # 128/32
    assert mask.num_key_blocks == 4
    
    # Test diagonal blocks are active
    assert mask.is_block_active(0, 0) == True
    assert mask.is_block_active(1, 1) == True
    assert mask.is_block_active(2, 2) == True
    assert mask.is_block_active(3, 3) == True
    
    # Test off-diagonal blocks are inactive
    assert mask.is_block_active(0, 1) == False
    assert mask.is_block_active(1, 0) == False
    
    # Test memory usage estimation
    assert mask.estimate_memory_usage() > 0
    
    print("✓ BlockBitsetMask tests passed")


def test_bcsr_mask():
    """Test BCSR mask functionality."""
    print("Testing BCSRMask...")
    
    # Create a simple test pattern
    device = torch.device('cpu')
    seqlen_q, seqlen_k = 128, 128
    block_size_m, block_size_n = 32, 32
    
    # Create dense mask (block diagonal pattern)
    dense_mask = torch.zeros(seqlen_q, seqlen_k, device=device)
    # Add diagonal blocks
    for i in range(0, seqlen_q, block_size_m):
        end_i = min(i + block_size_m, seqlen_q)
        for j in range(0, seqlen_k, block_size_n):
            end_j = min(j + block_size_n, seqlen_k)
            if i == j:  # Diagonal blocks
                dense_mask[i:end_i, j:end_j] = 1.0
    
    # Convert to BCSR mask
    mask = BCSRMask.from_dense_mask(dense_mask, block_size_m, block_size_n)
    
    # Test basic properties
    assert mask.get_mask_type() == "BCSR"
    assert mask.seqlen_q == seqlen_q
    assert mask.seqlen_k == seqlen_k
    assert mask.num_query_blocks == 4
    assert mask.num_key_blocks == 4
    
    # Test diagonal blocks are active
    assert mask.is_block_active(0, 0) == True
    assert mask.is_block_active(1, 1) == True
    assert mask.is_block_active(2, 2) == True
    assert mask.is_block_active(3, 3) == True
    
    # Test off-diagonal blocks are inactive
    assert mask.is_block_active(0, 1) == False
    assert mask.is_block_active(1, 0) == False
    
    # Test row pointer structure
    assert mask.row_ptr.numel() == mask.num_query_blocks + 1
    assert mask.col_idx.numel() == 4  # 4 diagonal blocks
    
    print("✓ BCSRMask tests passed")


def test_mask_factory():
    """Test mask factory function."""
    print("Testing mask factory...")
    
    # Test creating different mask types
    causal = create_sparse_mask("causal", seqlen_q=128, seqlen_k=128)
    assert isinstance(causal, CausalMask)
    
    window = create_sparse_mask("window", window_size=64, seqlen_q=128, seqlen_k=128)
    assert isinstance(window, WindowMask)
    
    hybrid = create_sparse_mask("causal_window", window_size=64, seqlen_q=128, seqlen_k=128)
    assert isinstance(hybrid, CausalWindowMask)
    
    print("✓ Mask factory tests passed")


def test_performance_estimation():
    """Test performance estimation functions."""
    print("Testing performance estimation...")
    
    # Test with causal mask
    mask = CausalMask(seqlen_q=256, seqlen_k=256)
    
    speedup = estimate_speedup(mask)
    assert speedup > 1.0, f"Speedup should be > 1.0, got {speedup}"
    
    memory_savings = calculate_memory_savings(mask)
    assert 0.0 <= memory_savings <= 1.0, f"Memory savings should be in [0,1], got {memory_savings}"
    
    # Parametric masks should have maximum memory savings
    assert memory_savings > 0.99, f"Parametric mask should have ~100% memory savings, got {memory_savings:.2%}"
    
    print("✓ Performance estimation tests passed")


def test_dense_conversion():
    """Test conversion to dense mask format."""
    print("Testing dense mask conversion...")
    
    # Test causal mask conversion
    mask = CausalMask(seqlen_q=64, seqlen_k=64, block_size_m=16, block_size_n=16)
    dense = mask.to_dense()
    
    assert dense.shape == (64, 64)
    assert dense.dtype == torch.float32
    
    # Check causal pattern in dense mask
    for i in range(64):
        for j in range(64):
            if j <= i:
                assert dense[i, j] == 1.0, f"Causal mask should be 1 at ({i},{j})"
            else:
                assert dense[i, j] == 0.0, f"Causal mask should be 0 at ({i},{j})"
    
    print("✓ Dense conversion tests passed")


def run_all_tests():
    """Run all available tests."""
    if not SPARSE_MASK_AVAILABLE:
        print("Sparse mask API not available - skipping tests")
        return False
    
    try:
        test_causal_mask()
        test_window_mask() 
        test_causal_window_mask()
        test_block_bitset_mask()
        test_bcsr_mask()
        test_mask_factory()
        test_performance_estimation()
        test_dense_conversion()
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Running Unified Sparse Mask Tests")
    print("=" * 40)
    
    success = run_all_tests()
    sys.exit(0 if success else 1)