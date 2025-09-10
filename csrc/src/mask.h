/******************************************************************************
 * Copyright (c) 2025, Jingze Shi and Yifan Wu and Bingheng Wu and Tri Dao.
 ******************************************************************************/

#pragma once
#include "namespace_config.h"

#include <cute/tensor.hpp>

namespace FLASH_NAMESPACE {

using namespace cute;

template <bool Causal_mask=false, typename TensorType, typename MaskType, typename BiasType, typename Params>
__forceinline__ __device__ void apply_mask(
    TensorType &tensor,
    MaskType &mask,
    BiasType &bias,
    const float scale_softmax,
    const int col_idx_offset_,
    const int max_seqlen_k,
    const int row_idx_offset,
    const int max_seqlen_q,
    const int warp_row_stride,
    const Params &params
) {
    // tensor has shape (nrow=(2, MMA_M), ncol=(2, MMA_N))
    static_assert(TensorType::rank == 2, "Only support 2D Tensor");
    static_assert(MaskType::rank == 2, "Only support 2D Mask");
    static_assert(BiasType::rank == 2, "Only support 2D Bias");
    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    #pragma unroll
    for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
        const int row_idx_base = row_idx_offset + mi * warp_row_stride;
        #pragma unroll
        for (int i = 0; i < size<0, 0>(tensor); ++i) {
            const int row_idx = row_idx_base + i * 8;
            const int col_idx_limit = Causal_mask ? std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q) : max_seqlen_k;
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                const int col_idx_base = col_idx_offset + nj * 8;
                #pragma unroll
                for (int j = 0; j < size<1, 0>(tensor); ++j) {
                    const int col_idx = col_idx_base + j;
                    // Without the "make_coord" we get wrong results
                    auto coord = make_coord(make_coord(i, mi), make_coord(j, nj));
                    
                    // Conditional mask and bias application
                    bool is_masked = false;
                    float bias_val = 0.0f;
                    
                    // Apply causal mask or boundary check
                    if (col_idx >= col_idx_limit) {
                        is_masked = true;
                    }
                    
                    // Apply mask if enabled
                    if (params.use_mask && mask(coord) == 0.0f) {
                        is_masked = true;
                    }
                    
                    // Add bias if enabled
                    if (params.use_bias) {
                        bias_val = bias(coord);
                    }
                    
                    // Apply scaling and bias or set to -INFINITY if masked
                    tensor(coord) = is_masked 
                        ? -INFINITY
                        : tensor(coord) * scale_softmax + bias_val;
                }
            }
        }
    }
}

template <bool Is_causal>
struct Mask {
    const int max_seqlen_k, max_seqlen_q;

    __forceinline__ __device__ Mask(
        const int max_seqlen_k,
        const int max_seqlen_q
    )  // Constructor
        : max_seqlen_k(max_seqlen_k)
        , max_seqlen_q(max_seqlen_q) {
    };

    template <bool Causal_mask=false, bool Is_even_MN=true, typename TensorType, typename MaskType, typename BiasType, typename Params>
    __forceinline__ __device__ void apply_mask(
        TensorType &tensor_,                        // acc_s (attention scores, MMA=4, MMA_M, MMA_N)
        MaskType &tSrMask,                          // Attention Mask (MMA=4, MMA_M, MMA_N)
        BiasType &tSrBias,                          // Attention Bias (MMA=4, MMA_M, MMA_N)
        const float scale_softmax,                  // Scale for softmax
        const int col_idx_offset_,                  // Column index offset
        const int row_idx_offset,                   // Row index offset
        const int warp_row_stride,                  // Warp row stride
        const Params &params                        // Parameters containing use_mask and use_bias flags
    ) {
        static_assert(TensorType::rank == 3, "tensor_ must be 3D Tensor");
        static_assert(MaskType::rank == 3, "Mask must be 3D Tensor");
        static_assert(BiasType::rank == 3, "Bias must be 3D Tensor");
        static_assert(decltype(size<0>(tensor_))::value == 4, "First dimension must be 4");

        // const bool Need_masking = Causal_mask || !Is_even_MN || (keep_window_size < max_seqlen_k);

        // Reshape tensors from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
        Tensor tensor = make_tensor(tensor_.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tensor_.layout()));
        Tensor mask = make_tensor(tSrMask.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tSrMask.layout()));
        Tensor bias = make_tensor(tSrBias.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(tSrBias.layout()));

        const int lane_id = threadIdx.x % 32;
        const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
        #pragma unroll
        for (int mi = 0; mi < size<0, 1>(tensor); ++mi) {
            const int row_idx_base = row_idx_offset + mi * warp_row_stride;
            #pragma unroll
            for (int i = 0; i < size<0, 0>(tensor); ++i) {
                const int row_idx = row_idx_base + i * 8;
                const int col_idx_limit = Causal_mask ? std::min(max_seqlen_k, row_idx + 1 + max_seqlen_k - max_seqlen_q) : max_seqlen_k;
                #pragma unroll
                for (int nj = 0; nj < size<1, 1>(tensor); ++nj) {
                    const int col_idx_base = col_idx_offset + nj * 8;
                    #pragma unroll
                    for (int j = 0; j < size<1, 0>(tensor); ++j) {
                        const int col_idx = col_idx_base + j;
                        auto coord = make_coord(make_coord(i, mi), make_coord(j, nj));
                        
                        // Conditional mask and bias application
                        bool is_masked = false;
                        float bias_val = 0.0f;
                        
                        // Apply causal mask or boundary check
                        if (col_idx >= col_idx_limit) {
                            is_masked = true;
                        }
                        
                        // Apply mask if enabled
                        if (params.use_mask && mask(coord) == 0.0f) {
                            is_masked = true;
                        }
                        
                        // Add bias if enabled
                        if (params.use_bias) {
                            bias_val = bias(coord);
                        }
                        
                        // Apply scaling and bias or set to -INFINITY if masked
                        tensor(coord) = is_masked 
                            ? -INFINITY
                            : tensor(coord) * scale_softmax + bias_val;
                    }
                }
            }
        }
        
    }
};

} // namespace FLASH_NAMESPACE
