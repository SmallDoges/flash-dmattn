/******************************************************************************
 * Copyright (c) 2025, Jingze Shi and Tri Dao.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/array.h>

namespace FLASH_NAMESPACE {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Mask Type Enumerations
////////////////////////////////////////////////////////////////////////////////////////////////////

enum class MaskType {
    PARAMETRIC_CAUSAL = 0,
    PARAMETRIC_WINDOW = 1,
    BLOCK_BITSET = 2,
    BCSR = 3,
    MIXED_GRANULARITY = 4,
    DYNAMIC = 5,
    DENSE_FALLBACK = 6
};

enum class MaskCompressionLevel {
    NO_STORAGE = 0,     // Parametric masks (causal, window)
    BLOCK_LEVEL = 1,    // Block bitset (B×B granularity)
    MIXED = 2,          // Dense blocks + partial bitpacked
    SPARSE_INDEX = 3    // BCSR format
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Block Descriptor Structures
////////////////////////////////////////////////////////////////////////////////////////////////////

struct BlockDescriptor {
    // Lightweight descriptor for per-query block
    uint32_t active_block_count;    // Number of active key blocks for this query block
    uint32_t descriptor_offset;     // Offset into descriptor data (bitset/indices)
    uint8_t mask_type;             // MaskType enum value
    uint8_t compression_level;     // MaskCompressionLevel enum value
    uint16_t partial_mask_bits;    // For mixed granularity: partial block mask
};

struct ParametricMaskParams {
    int32_t window_size;           // For sliding window masks
    int32_t doc_segment_id;        // For document segmentation
    bool is_causal;                // Causal mask flag
    bool use_window;               // Window mask flag
};

struct BCSRMaskData {
    uint32_t* row_ptr;             // Row pointers (query blocks)
    uint32_t* col_idx;             // Column indices (key blocks)
    uint8_t* partial_masks;        // Optional: partial block masks (bitpacked)
    uint32_t nnz_blocks;           // Number of non-zero blocks
};

struct BlockBitsetData {
    uint64_t* bitset;              // Bitset for block-level sparsity
    uint32_t num_query_blocks;     // Number of query blocks
    uint32_t num_key_blocks;       // Number of key blocks
    uint32_t bitset_size_words;    // Size of bitset in 64-bit words
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Unified Sparse Mask Interface
////////////////////////////////////////////////////////////////////////////////////////////////////

class UnifiedSparseMask {
public:
    __forceinline__ __device__ UnifiedSparseMask(
        MaskType type,
        void* mask_data,
        const ParametricMaskParams& params,
        int32_t max_seqlen_q,
        int32_t max_seqlen_k,
        int32_t block_size_m = 128,
        int32_t block_size_n = 128
    ) : mask_type_(type)
      , mask_data_(mask_data)
      , params_(params)
      , max_seqlen_q_(max_seqlen_q)
      , max_seqlen_k_(max_seqlen_k)
      , block_size_m_(block_size_m)
      , block_size_n_(block_size_n)
      , num_query_blocks_((max_seqlen_q + block_size_m - 1) / block_size_m)
      , num_key_blocks_((max_seqlen_k + block_size_n - 1) / block_size_n) {
    }

    // Core API: Check if a block is active (should be processed)
    __forceinline__ __device__ bool is_block_active(
        int32_t query_block_idx,
        int32_t key_block_idx
    ) const {
        switch (mask_type_) {
            case MaskType::PARAMETRIC_CAUSAL:
                return is_causal_block_active(query_block_idx, key_block_idx);
            case MaskType::PARAMETRIC_WINDOW:
                return is_window_block_active(query_block_idx, key_block_idx);
            case MaskType::BLOCK_BITSET:
                return is_bitset_block_active(query_block_idx, key_block_idx);
            case MaskType::BCSR:
                return is_bcsr_block_active(query_block_idx, key_block_idx);
            case MaskType::MIXED_GRANULARITY:
                return is_mixed_block_active(query_block_idx, key_block_idx);
            case MaskType::DYNAMIC:
                return is_dynamic_block_active(query_block_idx, key_block_idx);
            default:
                return true; // Dense fallback
        }
    }

    // Enumerate active key blocks for a given query block
    __forceinline__ __device__ uint32_t enumerate_active_blocks(
        int32_t query_block_idx,
        uint32_t* active_key_blocks,
        uint32_t max_blocks
    ) const {
        uint32_t count = 0;
        
        switch (mask_type_) {
            case MaskType::PARAMETRIC_CAUSAL:
            case MaskType::PARAMETRIC_WINDOW:
                count = enumerate_parametric_blocks(query_block_idx, active_key_blocks, max_blocks);
                break;
            case MaskType::BLOCK_BITSET:
                count = enumerate_bitset_blocks(query_block_idx, active_key_blocks, max_blocks);
                break;
            case MaskType::BCSR:
                count = enumerate_bcsr_blocks(query_block_idx, active_key_blocks, max_blocks);
                break;
            case MaskType::MIXED_GRANULARITY:
                count = enumerate_mixed_blocks(query_block_idx, active_key_blocks, max_blocks);
                break;
            case MaskType::DYNAMIC:
                count = enumerate_dynamic_blocks(query_block_idx, active_key_blocks, max_blocks);
                break;
            default:
                // Dense fallback: all blocks are active
                for (uint32_t i = 0; i < min(max_blocks, (uint32_t)num_key_blocks_); ++i) {
                    active_key_blocks[i] = i;
                }
                count = min(max_blocks, (uint32_t)num_key_blocks_);
        }
        
        return count;
    }

    // Get mask type (public accessor)
    __forceinline__ __device__ MaskType get_mask_type() const {
        return mask_type_;
    }

    // Get block descriptor for lightweight kernel access
    __forceinline__ __device__ BlockDescriptor get_block_descriptor(
        int32_t query_block_idx
    ) const {
        BlockDescriptor desc;
        desc.mask_type = static_cast<uint8_t>(mask_type_);
        
        switch (mask_type_) {
            case MaskType::PARAMETRIC_CAUSAL:
            case MaskType::PARAMETRIC_WINDOW:
                desc.compression_level = static_cast<uint8_t>(MaskCompressionLevel::NO_STORAGE);
                desc.active_block_count = count_parametric_active_blocks(query_block_idx);
                desc.descriptor_offset = 0; // No storage needed
                desc.partial_mask_bits = 0;
                break;
            case MaskType::BLOCK_BITSET:
                desc.compression_level = static_cast<uint8_t>(MaskCompressionLevel::BLOCK_LEVEL);
                desc.active_block_count = count_bitset_active_blocks(query_block_idx);
                desc.descriptor_offset = query_block_idx * ((num_key_blocks_ + 63) / 64);
                desc.partial_mask_bits = 0;
                break;
            case MaskType::BCSR:
                desc.compression_level = static_cast<uint8_t>(MaskCompressionLevel::SPARSE_INDEX);
                desc.active_block_count = count_bcsr_active_blocks(query_block_idx);
                desc.descriptor_offset = get_bcsr_row_offset(query_block_idx);
                desc.partial_mask_bits = 0;
                break;
            default:
                desc.compression_level = static_cast<uint8_t>(MaskCompressionLevel::BLOCK_LEVEL);
                desc.active_block_count = num_key_blocks_;
                desc.descriptor_offset = 0;
                desc.partial_mask_bits = 0;
        }
        
        return desc;
    }

    // Fast path: OR-reduction over entire block to determine activity
    template<typename TensorMask>
    __forceinline__ __device__ bool compute_block_activity_fast(
        TensorMask& mask_tile,
        int32_t query_block_idx,
        int32_t key_block_idx
    ) const {
        if (mask_type_ == MaskType::PARAMETRIC_CAUSAL || 
            mask_type_ == MaskType::PARAMETRIC_WINDOW) {
            // Parametric masks: no need to load, compute directly
            return is_block_active(query_block_idx, key_block_idx);
        }
        
        // For non-parametric masks, perform OR reduction on loaded tile
        bool any_active = false;
        #pragma unroll
        for (int i = 0; i < size<0>(mask_tile); ++i) {
            #pragma unroll
            for (int j = 0; j < size<1>(mask_tile); ++j) {
                if (mask_tile(i, j) != 0.0f) {
                    any_active = true;
                    break;
                }
            }
            if (any_active) break;
        }
        return any_active;
    }

private:
    MaskType mask_type_;
    void* mask_data_;
    ParametricMaskParams params_;
    int32_t max_seqlen_q_;
    int32_t max_seqlen_k_;
    int32_t block_size_m_;
    int32_t block_size_n_;
    int32_t num_query_blocks_;
    int32_t num_key_blocks_;

    // Parametric mask implementations
    __forceinline__ __device__ bool is_causal_block_active(
        int32_t query_block_idx, int32_t key_block_idx) const {
        if (!params_.is_causal) return true;
        
        // Causal mask: key block must not extend beyond query block end
        int32_t query_end = (query_block_idx + 1) * block_size_m_ - 1;
        int32_t key_start = key_block_idx * block_size_n_;
        
        return key_start <= query_end;
    }

    __forceinline__ __device__ bool is_window_block_active(
        int32_t query_block_idx, int32_t key_block_idx) const {
        if (!params_.use_window) return true;
        
        // Sliding window: check if blocks overlap with window
        int32_t query_center = query_block_idx * block_size_m_ + block_size_m_ / 2;
        int32_t key_start = key_block_idx * block_size_n_;
        int32_t key_end = (key_block_idx + 1) * block_size_n_ - 1;
        
        int32_t window_start = max(0, query_center - params_.window_size / 2);
        int32_t window_end = min(max_seqlen_k_ - 1, query_center + params_.window_size / 2);
        
        return !(key_end < window_start || key_start > window_end);
    }

    __forceinline__ __device__ bool is_bitset_block_active(
        int32_t query_block_idx, int32_t key_block_idx) const {
        auto* bitset_data = static_cast<BlockBitsetData*>(mask_data_);
        if (!bitset_data || !bitset_data->bitset) return false;
        
        uint32_t bit_idx = query_block_idx * num_key_blocks_ + key_block_idx;
        uint32_t word_idx = bit_idx / 64;
        uint32_t bit_offset = bit_idx % 64;
        
        if (word_idx >= bitset_data->bitset_size_words) return false;
        
        return (bitset_data->bitset[word_idx] >> bit_offset) & 1ULL;
    }

    __forceinline__ __device__ bool is_bcsr_block_active(
        int32_t query_block_idx, int32_t key_block_idx) const {
        auto* bcsr_data = static_cast<BCSRMaskData*>(mask_data_);
        if (!bcsr_data || !bcsr_data->row_ptr || !bcsr_data->col_idx) return false;
        
        uint32_t start = bcsr_data->row_ptr[query_block_idx];
        uint32_t end = bcsr_data->row_ptr[query_block_idx + 1];
        
        for (uint32_t i = start; i < end; ++i) {
            if (bcsr_data->col_idx[i] == static_cast<uint32_t>(key_block_idx)) {
                return true;
            }
        }
        return false;
    }

    __forceinline__ __device__ bool is_mixed_block_active(
        int32_t query_block_idx, int32_t key_block_idx) const {
        // For mixed granularity: first check block-level, then partial masks
        return is_bitset_block_active(query_block_idx, key_block_idx);
    }

    __forceinline__ __device__ bool is_dynamic_block_active(
        int32_t query_block_idx, int32_t key_block_idx) const {
        // Dynamic masks use BCSR-like storage with runtime updates
        return is_bcsr_block_active(query_block_idx, key_block_idx);
    }

    // Block enumeration implementations
    __forceinline__ __device__ uint32_t enumerate_parametric_blocks(
        int32_t query_block_idx, uint32_t* active_blocks, uint32_t max_blocks) const {
        uint32_t count = 0;
        
        for (int32_t k = 0; k < num_key_blocks_ && count < max_blocks; ++k) {
            if (is_block_active(query_block_idx, k)) {
                active_blocks[count++] = k;
            }
        }
        return count;
    }

    __forceinline__ __device__ uint32_t enumerate_bitset_blocks(
        int32_t query_block_idx, uint32_t* active_blocks, uint32_t max_blocks) const {
        auto* bitset_data = static_cast<BlockBitsetData*>(mask_data_);
        if (!bitset_data || !bitset_data->bitset) return 0;
        
        uint32_t count = 0;
        uint32_t base_bit = query_block_idx * num_key_blocks_;
        
        for (int32_t k = 0; k < num_key_blocks_ && count < max_blocks; ++k) {
            uint32_t bit_idx = base_bit + k;
            uint32_t word_idx = bit_idx / 64;
            uint32_t bit_offset = bit_idx % 64;
            
            if (word_idx < bitset_data->bitset_size_words &&
                ((bitset_data->bitset[word_idx] >> bit_offset) & 1ULL)) {
                active_blocks[count++] = k;
            }
        }
        return count;
    }

    __forceinline__ __device__ uint32_t enumerate_bcsr_blocks(
        int32_t query_block_idx, uint32_t* active_blocks, uint32_t max_blocks) const {
        auto* bcsr_data = static_cast<BCSRMaskData*>(mask_data_);
        if (!bcsr_data || !bcsr_data->row_ptr || !bcsr_data->col_idx) return 0;
        
        uint32_t start = bcsr_data->row_ptr[query_block_idx];
        uint32_t end = bcsr_data->row_ptr[query_block_idx + 1];
        uint32_t count = min(end - start, max_blocks);
        
        for (uint32_t i = 0; i < count; ++i) {
            active_blocks[i] = bcsr_data->col_idx[start + i];
        }
        return count;
    }

    __forceinline__ __device__ uint32_t enumerate_mixed_blocks(
        int32_t query_block_idx, uint32_t* active_blocks, uint32_t max_blocks) const {
        return enumerate_bitset_blocks(query_block_idx, active_blocks, max_blocks);
    }

    __forceinline__ __device__ uint32_t enumerate_dynamic_blocks(
        int32_t query_block_idx, uint32_t* active_blocks, uint32_t max_blocks) const {
        return enumerate_bcsr_blocks(query_block_idx, active_blocks, max_blocks);
    }

    // Block counting implementations
    __forceinline__ __device__ uint32_t count_parametric_active_blocks(int32_t query_block_idx) const {
        uint32_t count = 0;
        for (int32_t k = 0; k < num_key_blocks_; ++k) {
            if (is_block_active(query_block_idx, k)) {
                count++;
            }
        }
        return count;
    }

    __forceinline__ __device__ uint32_t count_bitset_active_blocks(int32_t query_block_idx) const {
        auto* bitset_data = static_cast<BlockBitsetData*>(mask_data_);
        if (!bitset_data || !bitset_data->bitset) return 0;
        
        uint32_t count = 0;
        uint32_t base_bit = query_block_idx * num_key_blocks_;
        
        for (int32_t k = 0; k < num_key_blocks_; ++k) {
            uint32_t bit_idx = base_bit + k;
            uint32_t word_idx = bit_idx / 64;
            uint32_t bit_offset = bit_idx % 64;
            
            if (word_idx < bitset_data->bitset_size_words &&
                ((bitset_data->bitset[word_idx] >> bit_offset) & 1ULL)) {
                count++;
            }
        }
        return count;
    }

    __forceinline__ __device__ uint32_t count_bcsr_active_blocks(int32_t query_block_idx) const {
        auto* bcsr_data = static_cast<BCSRMaskData*>(mask_data_);
        if (!bcsr_data || !bcsr_data->row_ptr) return 0;
        
        return bcsr_data->row_ptr[query_block_idx + 1] - bcsr_data->row_ptr[query_block_idx];
    }

    __forceinline__ __device__ uint32_t get_bcsr_row_offset(int32_t query_block_idx) const {
        auto* bcsr_data = static_cast<BCSRMaskData*>(mask_data_);
        if (!bcsr_data || !bcsr_data->row_ptr) return 0;
        
        return bcsr_data->row_ptr[query_block_idx];
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Convenience Functions for Block-Level Skip Logic
////////////////////////////////////////////////////////////////////////////////////////////////////

// Unified OR-reduction for block activity detection
template<typename TensorMask>
__forceinline__ __device__ bool compute_mask_block_activity(
    TensorMask& mask_tile,
    const UnifiedSparseMask& sparse_mask,
    int32_t query_block_idx,
    int32_t key_block_idx
) {
    return sparse_mask.compute_block_activity_fast(mask_tile, query_block_idx, key_block_idx);
}

// Warp-level ballot for efficient OR reduction across warps
__forceinline__ __device__ bool warp_ballot_mask_activity(bool thread_active) {
    #if __CUDA_ARCH__ >= 300
        return __any_sync(0xFFFFFFFF, thread_active);
    #else
        return __any(thread_active);
    #endif
}

} // namespace FLASH_NAMESPACE