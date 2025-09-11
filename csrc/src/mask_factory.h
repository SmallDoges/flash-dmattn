/******************************************************************************
 * Copyright (c) 2025, Jingze Shi and Tri Dao.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
#include "unified_sparse_mask.h"

namespace FLASH_NAMESPACE {

////////////////////////////////////////////////////////////////////////////////////////////////////
// Mask Factory Functions
////////////////////////////////////////////////////////////////////////////////////////////////////

// Create a causal mask (parametric - no storage)
__forceinline__ __device__ UnifiedSparseMask create_causal_mask(
    int32_t max_seqlen_q,
    int32_t max_seqlen_k,
    int32_t block_size_m = 128,
    int32_t block_size_n = 128
) {
    ParametricMaskParams params;
    params.is_causal = true;
    params.use_window = false;
    params.window_size = 0;
    params.doc_segment_id = -1;
    
    return UnifiedSparseMask(
        MaskType::PARAMETRIC_CAUSAL,
        nullptr, // No storage needed
        params,
        max_seqlen_q,
        max_seqlen_k,
        block_size_m,
        block_size_n
    );
}

// Create a sliding window mask (parametric - no storage)
__forceinline__ __device__ UnifiedSparseMask create_window_mask(
    int32_t window_size,
    int32_t max_seqlen_q,
    int32_t max_seqlen_k,
    int32_t block_size_m = 128,
    int32_t block_size_n = 128
) {
    ParametricMaskParams params;
    params.is_causal = false;
    params.use_window = true;
    params.window_size = window_size;
    params.doc_segment_id = -1;
    
    return UnifiedSparseMask(
        MaskType::PARAMETRIC_WINDOW,
        nullptr, // No storage needed
        params,
        max_seqlen_q,
        max_seqlen_k,
        block_size_m,
        block_size_n
    );
}

// Create a hybrid causal + window mask
__forceinline__ __device__ UnifiedSparseMask create_causal_window_mask(
    int32_t window_size,
    int32_t max_seqlen_q,
    int32_t max_seqlen_k,
    int32_t block_size_m = 128,
    int32_t block_size_n = 128
) {
    ParametricMaskParams params;
    params.is_causal = true;
    params.use_window = true;
    params.window_size = window_size;
    params.doc_segment_id = -1;
    
    return UnifiedSparseMask(
        MaskType::PARAMETRIC_WINDOW, // Use window type with causal flag
        nullptr, // No storage needed
        params,
        max_seqlen_q,
        max_seqlen_k,
        block_size_m,
        block_size_n
    );
}

// Create a block bitset mask
__forceinline__ __device__ UnifiedSparseMask create_block_bitset_mask(
    uint64_t* bitset,
    uint32_t num_query_blocks,
    uint32_t num_key_blocks,
    int32_t max_seqlen_q,
    int32_t max_seqlen_k,
    int32_t block_size_m = 128,
    int32_t block_size_n = 128
) {
    static BlockBitsetData data;
    data.bitset = bitset;
    data.num_query_blocks = num_query_blocks;
    data.num_key_blocks = num_key_blocks;
    data.bitset_size_words = ((num_query_blocks * num_key_blocks) + 63) / 64;
    
    ParametricMaskParams params{}; // Not used for bitset masks
    
    return UnifiedSparseMask(
        MaskType::BLOCK_BITSET,
        &data,
        params,
        max_seqlen_q,
        max_seqlen_k,
        block_size_m,
        block_size_n
    );
}

// Create a BCSR (Block Compressed Sparse Row) mask
__forceinline__ __device__ UnifiedSparseMask create_bcsr_mask(
    uint32_t* row_ptr,
    uint32_t* col_idx,
    uint32_t nnz_blocks,
    uint32_t num_query_blocks,
    int32_t max_seqlen_q,
    int32_t max_seqlen_k,
    int32_t block_size_m = 128,
    int32_t block_size_n = 128,
    uint8_t* partial_masks = nullptr
) {
    static BCSRMaskData data;
    data.row_ptr = row_ptr;
    data.col_idx = col_idx;
    data.partial_masks = partial_masks;
    data.nnz_blocks = nnz_blocks;
    
    ParametricMaskParams params{}; // Not used for BCSR masks
    
    return UnifiedSparseMask(
        MaskType::BCSR,
        &data,
        params,
        max_seqlen_q,
        max_seqlen_k,
        block_size_m,
        block_size_n
    );
}

// Create a dynamic mask (uses BCSR format with runtime updates)
__forceinline__ __device__ UnifiedSparseMask create_dynamic_mask(
    uint32_t* row_ptr,
    uint32_t* col_idx,
    uint32_t nnz_blocks,
    uint32_t num_query_blocks,
    int32_t max_seqlen_q,
    int32_t max_seqlen_k,
    int32_t block_size_m = 128,
    int32_t block_size_n = 128
) {
    static BCSRMaskData data;
    data.row_ptr = row_ptr;
    data.col_idx = col_idx;
    data.partial_masks = nullptr;
    data.nnz_blocks = nnz_blocks;
    
    ParametricMaskParams params{}; // Not used for dynamic masks
    
    return UnifiedSparseMask(
        MaskType::DYNAMIC,
        &data,
        params,
        max_seqlen_q,
        max_seqlen_k,
        block_size_m,
        block_size_n
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Mask Conversion Utilities
////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert dense mask to block bitset representation
__forceinline__ __device__ void dense_to_block_bitset(
    const float* dense_mask,
    int32_t seqlen_q,
    int32_t seqlen_k,
    int32_t block_size_m,
    int32_t block_size_n,
    uint64_t* bitset_out
) {
    int32_t num_q_blocks = (seqlen_q + block_size_m - 1) / block_size_m;
    int32_t num_k_blocks = (seqlen_k + block_size_n - 1) / block_size_n;
    
    // Initialize bitset to zero
    int32_t bitset_words = ((num_q_blocks * num_k_blocks) + 63) / 64;
    for (int32_t w = 0; w < bitset_words; ++w) {
        bitset_out[w] = 0ULL;
    }
    
    for (int32_t q_block = 0; q_block < num_q_blocks; ++q_block) {
        for (int32_t k_block = 0; k_block < num_k_blocks; ++k_block) {
            bool block_active = false;
            
            // Check if any element in the block is active
            int32_t q_start = q_block * block_size_m;
            int32_t q_end = min(seqlen_q, (q_block + 1) * block_size_m);
            int32_t k_start = k_block * block_size_n;
            int32_t k_end = min(seqlen_k, (k_block + 1) * block_size_n);
            
            for (int32_t q = q_start; q < q_end && !block_active; ++q) {
                for (int32_t k = k_start; k < k_end && !block_active; ++k) {
                    if (dense_mask[q * seqlen_k + k] != 0.0f) {
                        block_active = true;
                    }
                }
            }
            
            if (block_active) {
                uint32_t bit_idx = q_block * num_k_blocks + k_block;
                uint32_t word_idx = bit_idx / 64;
                uint32_t bit_offset = bit_idx % 64;
                bitset_out[word_idx] |= (1ULL << bit_offset);
            }
        }
    }
}

// Convert dense mask to BCSR representation
__forceinline__ __device__ uint32_t dense_to_bcsr(
    const float* dense_mask,
    int32_t seqlen_q,
    int32_t seqlen_k,
    int32_t block_size_m,
    int32_t block_size_n,
    uint32_t* row_ptr_out,
    uint32_t* col_idx_out,
    uint32_t max_nnz
) {
    int32_t num_q_blocks = (seqlen_q + block_size_m - 1) / block_size_m;
    int32_t num_k_blocks = (seqlen_k + block_size_n - 1) / block_size_n;
    
    uint32_t nnz_count = 0;
    row_ptr_out[0] = 0;
    
    for (int32_t q_block = 0; q_block < num_q_blocks; ++q_block) {
        uint32_t row_start = nnz_count;
        
        for (int32_t k_block = 0; k_block < num_k_blocks; ++k_block) {
            bool block_active = false;
            
            // Check if any element in the block is active
            int32_t q_start = q_block * block_size_m;
            int32_t q_end = min(seqlen_q, (q_block + 1) * block_size_m);
            int32_t k_start = k_block * block_size_n;
            int32_t k_end = min(seqlen_k, (k_block + 1) * block_size_n);
            
            for (int32_t q = q_start; q < q_end && !block_active; ++q) {
                for (int32_t k = k_start; k < k_end && !block_active; ++k) {
                    if (dense_mask[q * seqlen_k + k] != 0.0f) {
                        block_active = true;
                    }
                }
            }
            
            if (block_active && nnz_count < max_nnz) {
                col_idx_out[nnz_count] = k_block;
                nnz_count++;
            }
        }
        
        row_ptr_out[q_block + 1] = nnz_count;
    }
    
    return nnz_count;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Performance Estimation Utilities
////////////////////////////////////////////////////////////////////////////////////////////////////

// Estimate speedup from sparse mask usage
__forceinline__ __device__ float estimate_sparse_speedup(
    const UnifiedSparseMask& mask,
    int32_t num_query_blocks,
    int32_t num_key_blocks,
    float skip_overhead_ratio = 0.01f
) {
    uint32_t total_blocks = num_query_blocks * num_key_blocks;
    uint32_t active_blocks = 0;
    
    for (int32_t q = 0; q < num_query_blocks; ++q) {
        for (int32_t k = 0; k < num_key_blocks; ++k) {
            if (mask.is_block_active(q, k)) {
                active_blocks++;
            }
        }
    }
    
    if (active_blocks == 0) return 1.0f; // Avoid division by zero
    
    float active_fraction = float(active_blocks) / float(total_blocks);
    return 1.0f / (active_fraction + (1.0f - active_fraction) * skip_overhead_ratio);
}

// Calculate memory savings from compressed representation
__forceinline__ __device__ float calculate_memory_savings(
    MaskType mask_type,
    int32_t seqlen_q,
    int32_t seqlen_k,
    int32_t block_size_m,
    int32_t block_size_n,
    uint32_t nnz_blocks = 0
) {
    float dense_memory = float(seqlen_q) * float(seqlen_k) * sizeof(float);
    float compressed_memory = 0.0f;
    
    switch (mask_type) {
        case MaskType::PARAMETRIC_CAUSAL:
        case MaskType::PARAMETRIC_WINDOW:
            compressed_memory = 0.0f; // No storage
            break;
        case MaskType::BLOCK_BITSET: {
            int32_t num_blocks = ((seqlen_q + block_size_m - 1) / block_size_m) * 
                                ((seqlen_k + block_size_n - 1) / block_size_n);
            compressed_memory = float((num_blocks + 63) / 64) * sizeof(uint64_t);
            break;
        }
        case MaskType::BCSR:
        case MaskType::DYNAMIC: {
            int32_t num_q_blocks = (seqlen_q + block_size_m - 1) / block_size_m;
            compressed_memory = float(num_q_blocks + 1) * sizeof(uint32_t) + // row_ptr
                               float(nnz_blocks) * sizeof(uint32_t);          // col_idx
            break;
        }
        default:
            compressed_memory = dense_memory; // Dense fallback
    }
    
    return 1.0f - (compressed_memory / dense_memory);
}

} // namespace FLASH_NAMESPACE