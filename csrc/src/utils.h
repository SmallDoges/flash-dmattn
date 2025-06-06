/******************************************************************************
 * Copyright (c) 2025, Jingze Shi and Yiran Peng and Tri Dao.
 ******************************************************************************/

#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

#include "namespace_config.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace FLASH_NAMESPACE {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
__forceinline__ __device__ uint32_t relu2(const uint32_t x);

template<>
__forceinline__ __device__ uint32_t relu2<cutlass::half_t>(const uint32_t x) {
    uint32_t res;
    const uint32_t zero = 0u;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("max.f16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
#else
    asm volatile( \
        "{\n" \
        "\t .reg .f16x2 sela;\n" \
        "\t set.gtu.u32.f16x2 sela, %1, %2;\n" \
        "\t and.b32 %0, sela, %1;\n" 
        "}\n" : "=r"(res) : "r"(x), "r"(zero));
#endif
    return res;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template<>
__forceinline__ __device__ uint32_t relu2<cutlass::bfloat16_t>(const uint32_t x) {
    uint32_t res;
    const uint32_t zero = 0u;
    asm volatile("max.bf16x2 %0, %1, %2;\n" : "=r"(res) : "r"(x), "r"(zero));
    return res;
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

template<typename T>
__forceinline__ __device__ uint32_t convert_relu2(const float2 x);

template<>
__forceinline__ __device__ uint32_t convert_relu2<cutlass::half_t>(const float2 x) {
    uint32_t res;
    const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
    const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
    asm volatile("cvt.rn.relu.f16x2.f32 %0, %1, %2;\n" : "=r"(res) : "r"(b), "r"(a));
    return res;
}

template<>
__forceinline__ __device__ uint32_t convert_relu2<cutlass::bfloat16_t>(const float2 x) {
    uint32_t res;
    const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
    const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
    asm volatile("cvt.rn.relu.bf16x2.f32 %0, %1, %2;\n" : "=r"(res) : "r"(b), "r"(a));
    return res;
}

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct MaxOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
// This is slightly faster
__device__ __forceinline__ float operator()(float const &x, float const &y) { return max(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct SumOp {
__device__ __forceinline__ T operator()(T const & x, T const & y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<int THREADS>
struct Allreduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template<typename T, typename Operator>
    static __device__ __forceinline__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Allreduce<2> {
template<typename T, typename Operator> 
static __device__ __forceinline__ T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
}
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool A_in_regs=false, bool B_in_regs=false,
          typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3, typename Tensor4,
          typename TiledMma, typename TiledCopyA, typename TiledCopyB,
          typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm(
    Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA, Tensor4 const& tCsB,
    TiledMma tiled_mma, TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
    ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B
) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                        // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                        // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                       // MMA_K
    Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));             // M
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));             // N
    if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{})); }
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            if (!B_in_regs) { cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1)); }
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNWarps, bool A_in_regs=false, bool B_in_regs=false,
          typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3, typename Tensor4, typename Tensor5,
          typename TiledMma, typename TiledCopyA, typename TiledCopyB,
          typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void sparse_gemm(
    Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsA, Tensor4 const& tCsB, Tensor5 const &active_mask,
    TiledMma tiled_mma, TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
    ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B
) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                        // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                        // MMA_N
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(active_mask));                // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(active_mask));                // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                       // MMA_K
    auto tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));             // M
    auto tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));             // N
    bool mma_active[kNWarps] = {};  // MMA
    // Considering the characteristics of MMA and the chain of thoughts,
    // when there is any activated element in the query row or key column, 
    // we will mark the MMA block as activated.
    #pragma unroll
    for (int mma = 0; mma < size<0>(active_mask); ++mma) {
        mma_active[mma] = false;
        #pragma unroll
        for (int m = 0; m < size<1>(active_mask); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(active_mask); ++n) {
                if (active_mask(mma, m, n)) {
                    mma_active[mma] = true;
                    goto mma_active_found;
                }
            }
        }
        mma_active_found:;
    }
    if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{})); }
    if (!B_in_regs) {
        #pragma unroll
        for (int mma = 0; mma < size<0>(active_mask); ++mma) {
            if (mma_active[mma]) {
                cute::copy(smem_tiled_copy_B, tCsB(mma, _, _0{}), tCrB_copy_view(mma, _, _0{}));
            } else {
                cute::clear(tCrB_copy_view(mma, _, _0{}));
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            if (!A_in_regs) { cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1)); }
            if (!B_in_regs) {
                #pragma unroll
                for (int mma = 0; mma < size<0>(active_mask); ++mma) {
                    if (mma_active[mma]) {
                        cute::copy(smem_tiled_copy_B, tCsB(mma, _, i + 1), tCrB_copy_view(mma, _, i + 1));
                    } else {
                        cute::clear(tCrB_copy_view(mma, _, i + 1));
                    }
                    
                }
            }
        }
        // We must create a view to match `TiledMma` layout.
        #pragma unroll
        for (int mma = 0; mma < size<0>(active_mask); ++mma) {  // MMA
            if (mma_active[mma]) {
                cute::gemm(
                    tiled_mma,
                    tCrA(mma, _, i),                            // (MMA_M, MMA_K)
                    tCrB(mma, _, i),                            // (MMA_N, MMA_K)
                    acc(mma, _, _)                              // (MMA_M, MMA_N)
                );
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
          typename TiledMma, typename TiledCopy,
          typename ThrCopy>
__forceinline__ __device__ void gemm_rs(
    Tensor0 &acc, Tensor1 &tCrA, Tensor2 &tCrB, Tensor3 const& tCsB,
    TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
    ThrCopy smem_thr_copy_B
) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                        // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                        // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                       // MMA_K
    Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));             // N
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
        }
        cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int kNWarps,
          typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3, typename Tensor4,
          typename TiledMma, typename TiledCopy,
          typename ThrCopy>
__forceinline__ __device__ void sparse_gemm_rs(
    Tensor0 &acc,
    Tensor1 &tCrA,
    Tensor2 &tCrB,
    Tensor3 const& tCsB,
    Tensor4 const &active_mask,
    TiledMma tiled_mma,
    TiledCopy smem_tiled_copy_B,
    ThrCopy smem_thr_copy_B
) {
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));                        // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));                        // MMA_N
    CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(active_mask));                // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(active_mask));                // MMA_N
    CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));                       // MMA_K
    // Retile B for thread-wise copy from shared memory to registers
    auto tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));             // N
    // Check if any row or column in the MMA block is active.
    bool mma_active[kNWarps] = {};  // MMA
    #pragma unroll
    for (int mma = 0; mma < size<0>(active_mask); ++mma) {
        mma_active[mma] = false;
        #pragma unroll
        for (int m = 0; m < size<1>(active_mask); ++m) {
            #pragma unroll
            for (int n = 0; n < size<2>(active_mask); ++n) {
                if (active_mask(mma, m, n)) {
                    mma_active[mma] = true;
                    goto mma_active_found;
                }
            }
        }
        mma_active_found:;
    }
    #pragma unroll
    for (int mma = 0; mma < size<0>(active_mask); ++mma) {
        if (mma_active[mma]) {
            cute::copy(smem_tiled_copy_B, tCsB(mma, _, _0{}), tCrB_copy_view(mma, _, _0{}));
        } else {
            cute::clear(tCrB_copy_view(mma, _, _0{}));
        }
    }
    #pragma unroll
    for (int i = 0; i < size<2>(tCrA); ++i) {
        if (i < size<2>(tCrA) - 1) {
            #pragma unroll
            for (int mma = 0; mma < size<0>(active_mask); ++mma) {
                if (mma_active[mma]) {
                    cute::copy(smem_tiled_copy_B, tCsB(mma, _, i + 1), tCrB_copy_view(mma, _, i + 1));
                } else {
                    cute::clear(tCrB_copy_view(mma, _, i + 1));
                }
            }
        }
        #pragma unroll
        for (int mma = 0; mma < size<0>(active_mask); ++mma) {
            if (mma_active[mma]) {
                cute::gemm(tiled_mma, tCrA(mma, _, i), tCrB(mma, _, i), acc(mma, _, _));
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
    return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
// if using m16n8k16, or to (4, MMA_M, MMA_N) if using m16n8k8.
template<typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    if constexpr (mma_shape_K == 8) {
        return acc_layout;
    } else {
        auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
        return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
    }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
template<typename Layout>
__forceinline__ __device__ auto convert_layout_acc_dropout(Layout acc_layout) {
    using X = Underscore;
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
    return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    constexpr int numel = decltype(size(tensor))::value;
    cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
    // HACK: this requires tensor to be "contiguous"
    auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel> *>(tensor.data()));
    return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert 1D global ZOH tensor to 3D MMA layout tensor for apply_mask
template <typename Tensor, typename LayoutMMA>
__forceinline__ __device__ auto convert_global_zoh_to_mma_zoh(
    Tensor const &gZOH,                                       // ZOH tensor (actual_seqlen_k)
    LayoutMMA const &mma_layout,                              // Target MMA layout (4, MMA_M, MMA_N)
    const int col_idx_offset_,                                // Column index offset
    const int row_idx_offset,                                 // Row index offset
    const int warp_row_stride                                 // Warp row stride
) {
    using Element = typename Tensor::value_type;
    // Create 3D tensor with MMA layout
    constexpr int mma_size = decltype(size(mma_layout))::value;
    Element mma_data[mma_size];
    auto mma_fragment = make_tensor(make_rmem_ptr<Element>(mma_data), mma_layout);

    // Initialize the MMA fragment to -INFINITY
    #pragma unroll
    for (int i = 0; i < mma_size; ++i) {
        mma_data[i] = static_cast<Element>(-INFINITY);
    }

    // Convert layout to rowcol format for easier indexing
    auto fragment_rowcol = make_tensor(mma_fragment.data(), convert_layout_acc_rowcol(mma_layout));

    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    const int actual_seqlen_k = size<0>(gZOH);

    #pragma unroll
    for (int mi = 0; mi < size<0, 1>(fragment_rowcol); ++mi) {
        #pragma unroll
        for (int i = 0; i < size<0, 0>(fragment_rowcol); ++i) {
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(fragment_rowcol); ++nj) {
                const int col_idx_base = col_idx_offset + nj * 8;
                #pragma unroll
                for (int j = 0; j < size<1, 0>(fragment_rowcol); ++j) {
                    const int col_idx = col_idx_base + j;
                    auto coord = make_coord(make_coord(i, mi), make_coord(j, nj));
                    if (col_idx < actual_seqlen_k) {
                        fragment_rowcol(coord) = gZOH(col_idx);
                    }
                }
            }
        }
    }
    
    return mma_fragment;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert 1D global active mask to 3D MMA layout mask for apply_mask
template <typename Tensor, typename Layout>
__forceinline__ __device__ auto convert_global_mask_to_mma_mask(
    Tensor const &gActiveMask,                    // Active mask tensor (actual_seqlen_k)
    Layout const &mma_layout,                     // Target MMA layout (4, MMA_M, MMA_N)
    const int col_idx_offset_,                    // Column index offset
    const int row_idx_offset,                     // Row index offset
    const int warp_row_stride                     // Warp row stride
) {
    using Element = bool;
    // Create 3D tensor with MMA layout for mask
    constexpr int mma_size = decltype(size(mma_layout))::value;
    Element mma_data[mma_size];
    auto mma_fragment = make_tensor(make_rmem_ptr<Element>(mma_data), mma_layout);

    // Initialize the MMA fragment to false
    #pragma unroll
    for (int i = 0; i < mma_size; ++i) {
        mma_data[i] = static_cast<Element>(false);
    }

    // Convert layout to rowcol format for easier indexing
    auto fragment_rowcol = make_tensor(mma_fragment.data(), convert_layout_acc_rowcol(mma_layout));

    const int lane_id = threadIdx.x % 32;
    const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
    const int actual_seqlen_k = size<0>(gActiveMask);

    #pragma unroll
    for (int mi = 0; mi < size<0, 1>(fragment_rowcol); ++mi) {
        #pragma unroll
        for (int i = 0; i < size<0, 0>(fragment_rowcol); ++i) {  
            #pragma unroll
            for (int nj = 0; nj < size<1, 1>(fragment_rowcol); ++nj) {
                const int col_idx_base = col_idx_offset + nj * 8;
                #pragma unroll
                for (int j = 0; j < size<1, 0>(fragment_rowcol); ++j) {
                    const int col_idx = col_idx_base + j;
                    auto coord = make_coord(make_coord(i, mi), make_coord(j, nj));
                    if (col_idx < actual_seqlen_k) {
                        fragment_rowcol(coord) = gActiveMask(col_idx);
                    }
                }
            }
        }
    }
    
    return mma_fragment;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout>
__forceinline__ __device__ void relu_(Tensor<Engine, Layout> &tensor) {
    constexpr int numel = decltype(size(tensor))::value;
    static_assert(numel % 2 == 0);
    using value_t = typename Engine::value_type;
    // HACK: this requires tensor to be "contiguous"
    Tensor tensor_uint32 = recast<uint32_t>(tensor);
    #pragma unroll
    for (int i = 0; i < size(tensor_uint32); ++i) {
        tensor_uint32(i) = relu2<value_t>(tensor_uint32(i));
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// On SM80 and above, we can fuse fp32 -> fp16/bf16 conversion and relu into 1 instruction
template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type_relu(Tensor<Engine, Layout> const &tensor) {
    using From_type = typename Engine::value_type;
    static_assert(std::is_same_v<To_type, cutlass::half_t> || std::is_same_v<To_type, cutlass::bfloat16_t>);
    static_assert(std::is_same_v<float, From_type>);
    constexpr int numel = decltype(size(tensor))::value;
    static_assert(numel % 2 == 0);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    // HACK: this requires tensor to be "contiguous"
    Tensor tensor_float2 = recast<float2>(tensor);
    Tensor out_uint32 = make_tensor<uint32_t>(tensor_float2.layout());
    #pragma unroll
    for (int i = 0; i < size(out_uint32); ++i) {
        out_uint32(i) = convert_relu2<To_type>(tensor_float2(i));
    }
    Tensor out = make_tensor(make_rmem_ptr<To_type>(out_uint32.data()), tensor.layout());
#else
    Tensor out = FLASH_NAMESPACE::convert_type<To_type>(tensor);
    FLASH_NAMESPACE::relu_(out);
#endif
    return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Blocks until all but N previous cp.async.commit_group operations have committed.
// This differs from cute::cp_async_wait in that when N = 0 we don't call cp.async.wait_all
// (which is equivalent to commit_group then wait_group 0).
// Instead we just call cp.async.wait_group 0, which is slightly faster.
// https://github.com/NVIDIA/cutlass/blob/master/include/cute/arch/copy_sm80.hpp#L113
template <int N>
CUTE_HOST_DEVICE
void cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN=true, bool Is_even_K=true, bool Clear_OOB_MN=false, bool Clear_OOB_K=true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            Tensor<Engine3, Layout3> const &predicate_K, const int max_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // There's no case where !Clear_OOB_K && Clear_OOB_MN
    static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
                } else if (Clear_OOB_K) {
                    cute::clear(D(_, m, k));
                }
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }
    }
    // TD [2023-04-13]: Strange that the code below can cause race condition.
    // I think it's because the copies are under an if statement.
    // if (Is_even_K) {
    //     #pragma unroll
    //     for (int m = 0; m < size<1>(S); ++m) {
    //         if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
    //             copy(tiled_copy, S(_, m, _), D(_, m, _));
    //         } else if (Clear_OOB_MN) {
    //             clear(D(_, m, _));
    //         }
    //     }
    // } else {  // It's slightly faster in this case if iterate over K first
    //     #pragma unroll
    //     for (int k = 0; k < size<2>(S); ++k) {
    //         if (predicate_K(k)) {
    //             #pragma unroll
    //             for (int m = 0; m < size<1>(S); ++m) {
    //                 if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
    //                     copy(tiled_copy, S(_, m, k), D(_, m, k));
    //                 } else if (Clear_OOB_MN) {
    //                     clear(D(_, m, k));
    //                 }
    //             }
    //         } else if (Clear_OOB_K) {  // There's no case where !Clear_OOB_K && Clear_OOB_MN
    //             if (Clear_OOB_MN || Is_even_MN) {
    //                 clear(D(_, _, k));
    //             } else {
    //                 #pragma unroll
    //                 for (int m = 0; m < size<1>(S); ++m) {
    //                     if (!(Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN)) {
    //                         clear(D(_, m, k));
    //                     }
    //                 }
    //             }
    //         }
    //     }
    // }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K=true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy_w_min_idx(Tensor<Engine0, Layout0> const &S,
                                      Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                                      Tensor<Engine3, Layout3> const &predicate_K,
                                      const int max_MN=0, const int min_MN=0) {
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));                     // MMA
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));                     // MMA_M
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));                     // MMA_K
    // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("blockIdx.y = %d, max_MN = %d, min_MN = %d\n", blockIdx.y, max_MN, min_MN); }
    #pragma unroll
    for (int m = 0; m < size<1>(S); ++m) {
        // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("blockIdx.y = %d, m = %d\n", blockIdx.y, get<0>(identity_MN(0, m, 0))); }
        if (get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN) {
            // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("Inner loop, blockIdx.y = %d, m = %d\n", blockIdx.y, get<0>(identity_MN(0, m, 0))); }
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    cute::copy(S(_, m, k), D(_, m, k));
                }
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_softcap(Tensor<Engine, Layout> &tensor, const float softcap){
    #pragma unroll
    for (int i = 0; i < size(tensor); ++i) {
        tensor(i) = cutlass::fast_tanh(tensor(i) * softcap);
    }
}

template <typename Engine0, typename Layout0, typename Engine1, typename Layout1>
__forceinline__ __device__ void calculate_dtanh(Tensor<Engine0, Layout0> &src_tensor, Tensor<Engine1, Layout1> &dst_tensor, const float softcap){
    #pragma unroll
    for (int i = 0; i < size(src_tensor); ++i) {
        dst_tensor(i) = (1.f - (src_tensor(i) * src_tensor(i))) * softcap;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace FLASH_NAMESPACE
