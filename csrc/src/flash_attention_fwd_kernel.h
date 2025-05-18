/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
#include <ATen/cuda/detail/UnpackRaw.cuh> // For at::cuda::philox::unpack

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"

namespace FLASH_NAMESPACE {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ElementAccum, typename Params, int kBlockM, bool Is_even_MN>
__forceinline__ __device__ auto get_lse_tile(const Params &params, const int bidb, const int bidh, const int m_block, const BlockInfo</*Varlen=*/!Is_even_MN> &binfo) {
        // When params.unpadded_lse is false, LSE is written as (b, h, seqlen_q) - this is non-variable seqlen path.
        // Otherwise, when params.seqlenq_ngroups_swapped is true, it is written as (h, seqlen_q, b) to account for seqlen_q <-> h swapping trick.
        // Otherwise, it's written as (h, b, seqlen_q).
        const bool varlen_q = params.unpadded_lse && !params.seqlenq_ngroups_swapped;
        auto lse_offset = varlen_q ? binfo.q_offset(params.seqlen_q, 1, bidb) : 0;
        auto gmem_ptr_lse = make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + lse_offset);

        auto lse_shape = varlen_q ? make_shape(1, params.h, params.total_q) : make_shape(params.b, params.h, params.seqlen_q);
        auto lse_stride = params.seqlenq_ngroups_swapped ? make_stride(1, params.seqlen_q * params.b, params.b) : (
            params.unpadded_lse ? make_stride(params.h * params.total_q, params.total_q, 1) :  make_stride(params.h * params.seqlen_q, params.seqlen_q, 1)
            );

        auto lse_layout = make_layout(lse_shape, lse_stride);
        Tensor mLSE = make_tensor(gmem_ptr_lse, lse_layout);
        auto mLSE_slice = varlen_q ? mLSE(0, bidh, _) : mLSE(bidb, bidh, _);
        return local_tile(mLSE_slice, Shape<Int<kBlockM>>{}, make_coord(m_block));
}

template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Return_softmax, typename Params>
inline __device__ void compute_attn_1rowblock(const Params &params, const int bidb, const int bidh, const int m_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    // Check if there are any queries to process in the block
    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    // 计算实际要处理的N块范围
    const int n_block_min = 0;
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal) {
        n_block_max = std::min(n_block_max,
                            cute::ceil_div((m_block + 1) * kBlockM, kBlockN));
    }

    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(
            make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
            make_shape(binfo.actual_seqlen_q, params.h, params.d), make_stride(params.o_row_stride, params.o_head_stride, _1{})
        );
        Tensor gO = local_tile(
            mO(_, bidh, _),
            Shape<Int<kBlockM>, Int<kHeadDim>>{},
            make_coord(m_block, 0)
        );  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));  // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) {
                tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
            }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }

    // 全局内存张量配置
    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                          + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));
                           
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                          + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));
                            
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));
                           
    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                          + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
                            
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));
    
    Tensor mZeroHold = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.zero_hold_ptr)
                                               + bidb * params.zero_hold_batch_stride),
                                 make_shape(params.h, binfo.actual_seqlen_q, binfo.actual_seqlen_k), // Assuming h is num_kv_heads for zero_hold
                                 make_stride(params.zero_hold_head_stride, params.zero_hold_query_stride, _1{}));
    Tensor gZeroHold = local_tile(mZeroHold(bidh / params.h_h_k_ratio, _, _), // Use bidh / params.h_h_k_ratio if zero_hold is per kv_head
                              Shape<Int<kBlockM>, Int<kBlockN>>{},
                              make_coord(m_block, 0)); // m_block for query row, n_block for key column
    
    Tensor mCausalMask = params.causal_mask_ptr != nullptr
                       ? make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.causal_mask_ptr)
                                                  + bidb * params.causal_mask_batch_stride),
                                    make_shape(1, binfo.actual_seqlen_q, binfo.actual_seqlen_k),
                                    make_stride(params.causal_mask_head_stride, params.causal_mask_query_len_stride, _1{}))
                       : Tensor();  // Empty tensor if no causal mask is provided
    Tensor gCausalMask = params.causal_mask_ptr != nullptr
                       ? local_tile(mCausalMask(0, _, _),
                                    Shape<Int<kBlockM>, Int<kBlockN>>{},
                                    make_coord(_, 0))
                       : Tensor();  // Empty tensor if no causal mask is provided
                        


    // 共享内存配置
    // QKV的共享内存布局
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    // Dynamic mask的共享内存布局
    Tensor sZeroHold = make_tensor(sV.data().get() + size(sV), typename Kernel_traits::SmemLayoutZeroHold{});
    Tensor sCausalMask = params.causal_mask_ptr != nullptr
                       ? make_tensor(sZeroHold.data().get() + size(sZeroHold),
                            typename Kernel_traits::SmemLayoutZeroHold{})
                       : Tensor();
    Tensor sDynamicMaskValues = make_tensor(
        (params.causal_mask_ptr != nullptr ? 
            sCausalMask.data().get() + size(sCausalMask) : 
            sZeroHold.data().get() + size(sZeroHold)),
        typename Kernel_traits::SmemLayoutDynamicMaskValues{}
    );
    Tensor sDynamicMaskSortKeys = make_tensor(
        sDynamicMaskValues.data().get() + size(sDynamicMaskValues),
        typename Kernel_traits::SmemLayoutDynamicMaskSortKeys{}
    );
    Tensor sDynamicMaskSortIndices = make_tensor(
        sDynamicMaskSortKeys.data().get() + size(sDynamicMaskSortKeys),
        typename Kernel_traits::SmemLayoutDynamicMaskSortIndices{}
    );
    Tensor sNonZeroIndices = make_tensor(
        sDynamicMaskSortIndices.data().get() + size(sDynamicMaskSortIndices),
        typename Kernel_traits::SmemLayoutNonZeroIndices{}
    );
    Tensor sPredicate = make_tensor(
        sNonZeroIndices.data().get() + size(sNonZeroIndices),
        typename Kernel_traits::SmemLayoutZeroHold{}
    );
    

    // 设置全局内存到共享内存的拷贝
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopyZeroHold gmem_tiled_copy_ZeroHold;
    auto gmem_thr_copy_ZeroHold = gmem_tiled_copy_ZeroHold.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopyZeroHold gmem_tiled_copy_CausalMask;
    auto gmem_thr_copy_CausalMask = gmem_tiled_copy_CausalMask.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor tZeroHoldgZeroHold = gmem_thr_copy_ZeroHold.partition_S(gZeroHold);
    Tensor tZeroHoldsZeroHold = gmem_thr_copy_ZeroHold.partition_D(sZeroHold);
    Tensor tCausalMaskgCausalMask = params.causal_mask_ptr != nullptr
                                  ? gmem_thr_copy_CausalMask.partition_S(gCausalMask)
                                  : Tensor();
    Tensor tCausalMasksCausalMask = params.causal_mask_ptr != nullptr
                                  ? gmem_thr_copy_CausalMask.partition_D(sCausalMask)
                                  : Tensor();

    // 设置矩阵乘法操作
    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);
    
    // 设置从共享内存到寄存器的拷贝
    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    auto smem_tiled_copy_ZeroHold = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_ZeroHold = smem_tiled_copy_ZeroHold.get_thread_slice(tidx);
    Tensor tSsZeroHold = smem_thr_copy_ZeroHold.partition_S(sZeroHold);

    auto smem_tiled_copy_CausalMask = params.causal_mask_ptr != nullptr 
                                    ? make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma)
                                    : decltype(make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma)){};
    auto smem_thr_copy_CausalMask = params.causal_mask_ptr != nullptr
                                  ? smem_tiled_copy_CausalMask.get_thread_slice(tidx)
                                  : decltype(smem_tiled_copy_CausalMask.get_thread_slice(tidx)){};
    Tensor tSsCausalMask = params.causal_mask_ptr != nullptr
                         ? smem_thr_copy_CausalMask.partition_S(sCausalMask)
                         : Tensor();

    // 设置谓词
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    Tensor cZeroHold = make_identity_tensor(make_shape(size<0>(sZeroHold), size<1>(sZeroHold)));
    Tensor cCausalMask = params.causal_mask_ptr != nullptr
                       ? make_identity_tensor(make_shape(size<0>(sCausalMask), size<1>(sCausalMask)))
                       : Tensor();
    
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);
    Tensor tZeroHoldcZeroHold = gmem_thr_copy_ZeroHold.partition_S(cZeroHold);
    Tensor tCausalMaskcCausalMask = params.causal_mask_ptr != nullptr
                                  ? gmem_thr_copy_CausalMask.partition_S(cCausalMask)
                                  : Tensor();
    
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));
    Tensor tZeroHoldpZeroHold = make_tensor<bool>(make_shape(size<2>(tZeroHoldsZeroHold)));
    Tensor tCausalMaskpCausalMask = params.causal_mask_ptr != nullptr
                                  ? make_tensor<bool>(make_shape(size<2>(tCausalMasksCausalMask)))
                                  : Tensor();

    // 设置K维度的谓词
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) {
            tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d;
        }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) {
            tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d;
        }
    }

    // 初始化动态掩码处理器
    DynamicMask<Is_causal> dynamic_mask(params.keep_window_size);
    
    // 加载Q到共享内存
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(
        gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
        binfo.actual_seqlen_q - m_block * kBlockM
    );
    
    if (Kernel_traits::Is_Q_in_regs) {
        cute::cp_async_fence();
    }

    // 如果共享Q和K的内存，需要等待并同步
    if (Kernel_traits::Share_Q_K_smem) {
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    // 反向迭代N块
    int n_block = n_block_max - 1;
    
    // 加载第一个K块到共享内存
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(
        gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
        binfo.actual_seqlen_k - n_block * kBlockN
    );
    cute::cp_async_fence();

    // 加载第一个ZeroHold块到共享内存
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(
        gmem_tiled_copy_ZeroHold, tZeroHoldgZeroHold(_, _, _, n_block), tZeroHoldsZeroHold, tZeroHoldcZeroHold, tZeroHoldpZeroHold,
        binfo.actual_seqlen_k - n_block * kBlockN
    );
    cute::cp_async_fence();

    // 加载第一个CausalMask块到共享内存(如果有)
    if (params.causal_mask_ptr != nullptr) {
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(
            gmem_tiled_copy_CausalMask, tCausalMaskgCausalMask(_, _, _, n_block), tCausalMasksCausalMask, tCausalMaskcCausalMask, tCausalMaskpCausalMask,
            binfo.actual_seqlen_k - n_block * kBlockN
        );
        cute::cp_async_fence();
    }
    
    // 将Q从共享内存加载到寄存器（如果需要）
    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        FLASH_NAMESPACE::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }
    
    // 初始化输出累加器
    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});
    clear(acc_o);
    
    // 创建softmax计算器
    FLASH_NAMESPACE::Softmax<2 * size<1>(acc_o)> softmax;
    
    // 处理需要掩码的块（通常是最后几个块）
    constexpr int n_masking_steps = (!Is_causal)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        // 等待K数据
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        
        // 加载V块到共享内存
        FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(
            gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV,
            binfo.actual_seqlen_k - n_block * kBlockN
        );
        cute::cp_async_fence();
        
        // 计算块中实际键的数量
        const int block_key_len = min(kBlockN, binfo.actual_seqlen_k - n_block * kBlockN);
        
        // 为当前块内的每个查询行处理动态掩码
        const int queries_in_block = min(kBlockM, binfo.actual_seqlen_q - m_block * kBlockM);
        for (int m_idx = 0; m_idx < queries_in_block; ++m_idx) {
            // 获取当前查询的全局索引
            const int query_idx = m_block * kBlockM + m_idx;

            // 获取当前查询行的动态掩码内存
            Tensor mask_values = sDynamicMaskValues(m_idx, _);
            Tensor sort_keys = sDynamicMaskSortKeys(m_idx, _);
            Tensor sort_indices = sDynamicMaskSortIndices(m_idx, _);
            Tensor nonzero_indices = sNonZeroIndices(m_idx, _);
            Tensor predicate_k = sPredicate(m_idx, _);

            // 获取当前查询行的zero_hold和causal_mask
            const Element* zero_hold_row = &sZeroHold[m_idx][0];
            const Element* causal_mask_row = params.causal_mask_ptr != nullptr ? 
                &sCausalMask[m_idx][0] : nullptr;
            
            // 使用DynamicMask结构体来应用掩码
            dynamic_mask.apply_mask_1rowblock(
                mask_values,
                zero_hold_row,
                causal_mask_row,
                block_key_len,
                mask_values.data().get(),
                sort_keys.data().get(),
                reinterpret_cast<int*>(sort_indices.data().get()),
            );
            __syncthreads();
            
            // 初始化键的活性状态谓词
            if (tidx == 0) {
                // 只需一个线程来初始化整个谓词数组
                #pragma unroll
                for (int k_idx = 0; k_idx < kBlockN; ++k_idx) {
                    predicate_k(k_idx) = false;
                }
            }
            __syncthreads();

            // 找出非零位置
            int nonzero_count = 0;
            // 每个线程负责处理部分键位置
            for (int k_idx = tidx; k_idx < block_key_len; k_idx += blockDim.x) {
                if (mask_values(k_idx) != 0.0f) {
                    // 使用原子操作安全地增加计数并获取索引位置
                    int idx = atomicAdd(&nonzero_count, 1);
                    if (idx < Kernel_traits::kMaxKeysPerBlock) {
                        nonzero_indices(idx) = k_idx;
                        // 标记该键为活跃状态
                        predicate_k(k_idx) = true;
                    }
                }
            }
            __syncthreads();

            // 如果没有非零键，跳过当前查询行
            if (nonzero_count == 0) {
                continue;
            }

            // 处理多查询头情况 (MQA/GQA)
            const int num_queries_per_kv = params.h_h_k_ratio;
            
            // 对于每个查询组内的查询头
            for (int q_group_idx = 0; q_group_idx < num_queries_per_kv; q_group_idx++) {
                // 创建累加器用于注意力分数
                Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
                clear(acc_s);
                
                // 执行稀疏矩阵乘法
                FLASH_NAMESPACE::sparse_gemm_rs(
                    acc_s(_, m_idx, _),    // 当前查询行的累加器
                    tSrQ(_, m_idx, _),     // 当前查询
                    tSrK,                  // 键值
                    tSsK,                  // 共享内存中的键值
                    tiled_mma,
                    smem_tiled_copy_K,
                    smem_thr_copy_K,
                    predicate_k            // 活跃键的谓词
                );
                
                // 应用掩码添加（zero_hold状态既是掩码也是要添加到注意力分数的值）
                for (int s_idx = 0; s_idx < size(acc_s); ++s_idx) {
                    const int k_idx = get<2>(thr_mma.get_slice_idx(s_idx, acc_s));
                    if (k_idx < block_key_len && predicate_k(k_idx)) {
                        acc_s(s_idx) += static_cast<ElementAccum>(mask_values[k_idx]);
                    }
                }
                
                // 执行softmax并更新输出累加器
                if (q_group_idx == 0 && n_block == n_block_max - 1) {
                    softmax.template softmax_rescale_o<true, true>(
                        acc_s, acc_o, params.scale_softmax_log2);
                } else {
                    softmax.template softmax_rescale_o<false, true>(
                        acc_s, acc_o, params.scale_softmax_log2);
                }
                
                // 将浮点分数转换为Element类型进行输出计算
                Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
                Tensor tOrP = make_tensor(
                    rP.data(), 
                    FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout())
                );
                
                // 计算该查询头的输出
                FLASH_NAMESPACE::sparse_gemm_rs(
                    acc_o,              // 输出累加器
                    tOrP,               // 注意力权重
                    tOrVt,              // 值向量
                    tOsVt,              // 共享内存中的值向量
                    tiled_mma,
                    smem_tiled_copy_V,
                    smem_thr_copy_V,
                    predicate_k         // 应用相同的谓词来进行稀疏V矩阵乘法
                );
            }
            __syncthreads();
        }
        
        // 等待V数据
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        
        // 准备加载下一个K块（如果有）
        if (n_block > n_block_min) {
            FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(
                gmem_tiled_copy_QKV, tKgK(_, _, _, n_block-1), tKsK, tKVcKV, tKVpKV,
                binfo.actual_seqlen_k - (n_block-1) * kBlockN
            );
            cute::cp_async_fence();
            
            // 加载下一个ZeroHold块到共享内存
            FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(
                gmem_tiled_copy_ZeroHold, tZeroHoldgZeroHold(_, _, _, n_block-1), tZeroHoldsZeroHold, tZeroHoldcZeroHold, tZeroHoldpZeroHold,
                binfo.actual_seqlen_k - (n_block-1) * kBlockN
            );
            cute::cp_async_fence();
            
            // 加载下一个CausalMask块到共享内存(如果有)
            if (params.causal_mask_ptr != nullptr) {
                FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(
                    gmem_tiled_copy_CausalMask, tCausalMaskgCausalMask(_, _, _, n_block-1), tCausalMasksCausalMask, tCausalMaskcCausalMask, tCausalMaskpCausalMask,
                    binfo.actual_seqlen_k - (n_block-1) * kBlockN
                );
                cute::cp_async_fence();
            }
        }
        
        // 提前退出检查
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            break;
        }
    }

    // 处理不需要掩码的块
    for (; n_block >= n_block_min; --n_block) {
        // 等待K数据
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        
        // 加载V块到共享内存
        FLASH_NAMESPACE::copy<true, Is_even_K>(
            gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV
        );
        cute::cp_async_fence();
        
        // 计算块中实际键的数量
        const int block_key_len = min(kBlockN, binfo.actual_seqlen_k - n_block * kBlockN);
        const int queries_in_block = min(kBlockM, binfo.actual_seqlen_q - m_block * kBlockM);
        
        // 为当前块内的每个查询行处理动态掩码
        for (int m_idx = 0; m_idx < queries_in_block; ++m_idx) {
            // 获取当前查询的零状态行
            Tensor mask_values = sDynamicMaskValues(m_idx, _);
            Tensor sort_keys = sDynamicMaskSortKeys(m_idx, _);
            Tensor sort_indices = sDynamicMaskSortIndices(m_idx, _);
            Tensor nonzero_indices = sNonZeroIndices(m_idx, _);
            Tensor predicate_k = sPredicate(m_idx, _);
            
            // 获取当前查询行的zero_hold
            const Element* zero_hold_row = &sZeroHold[m_idx][0];
            
            // 使用DynamicMask结构体来应用掩码，没有因果掩码
            dynamic_mask.apply_mask_1rowblock(
                mask_values,
                zero_hold_row,
                nullptr,  // 无因果掩码
                block_key_len,
                mask_values.data().get(),
                sort_keys.data().get(),
                reinterpret_cast<int*>(sort_indices.data().get())
            );
            __syncthreads();
            
            // 初始化键的活性状态谓词
            if (tidx == 0) {
                // 只需一个线程来初始化整个谓词数组
                #pragma unroll
                for (int k_idx = 0; k_idx < kBlockN; ++k_idx) {
                    predicate_k(k_idx) = false;
                }
            }
            __syncthreads();

            // 找出非零位置
            int nonzero_count = 0;
            // 每个线程负责处理部分键位置
            for (int k_idx = tidx; k_idx < block_key_len; k_idx += blockDim.x) {
                if (mask_values(k_idx) != 0.0f) {
                    // 使用原子操作安全地增加计数并获取索引位置
                    int idx = atomicAdd(&nonzero_count, 1);
                    if (idx < Kernel_traits::kMaxKeysPerBlock) {
                        nonzero_indices(idx) = k_idx;
                        // 标记该键为活跃状态
                        predicate_k(k_idx) = true;
                    }
                }
            }
            __syncthreads();

            // 如果没有非零键，跳过当前查询行
            if (nonzero_count == 0) {
                continue;
            }
            
            // 处理多查询头情况 (MQA/GQA)
            const int num_queries_per_kv = params.h_h_k_ratio;
            
            // 对于每个查询组内的查询头
            for (int q_group_idx = 0; q_group_idx < num_queries_per_kv; q_group_idx++) {
                // 创建累加器用于注意力分数
                Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
                clear(acc_s);
                
                // 执行稀疏矩阵乘法
                FLASH_NAMESPACE::sparse_gemm_rs(
                    acc_s(_, m_idx, _),    // 当前查询行的累加器
                    tSrQ(_, m_idx, _),     // 当前查询
                    tSrK,                  // 键值
                    tSsK,                  // 共享内存中的键值
                    tiled_mma,
                    smem_tiled_copy_K,
                    smem_thr_copy_K,
                    predicate_k            // 活跃键的谓词
                );
                
                // 应用掩码添加
                for (int s_idx = 0; s_idx < size(acc_s); ++s_idx) {
                    const int k_idx = get<2>(thr_mma.get_slice_idx(s_idx, acc_s));
                    if (k_idx < block_key_len && predicate_k(k_idx)) {
                        acc_s(s_idx) += static_cast<ElementAccum>(mask_values[k_idx]);
                    }
                }
                
                // 执行softmax并更新输出累加器
                softmax.template softmax_rescale_o<false, false>(
                    acc_s, acc_o, params.scale_softmax_log2);
                
                // 将浮点分数转换为Element类型进行输出计算
                Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
                Tensor tOrP = make_tensor(
                    rP.data(), 
                    FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout())
                );
                
                // 计算该查询头的输出
                FLASH_NAMESPACE::sparse_gemm_rs(
                    acc_o,              // 输出累加器
                    tOrP,               // 注意力权重
                    tOrVt,              // 值向量
                    tOsVt,              // 共享内存中的值向量
                    tiled_mma,
                    smem_tiled_copy_V,
                    smem_thr_copy_V,
                    predicate_k         // 应用相同的谓词来进行稀疏V矩阵乘法
                );
            }
            __syncthreads();
        }
        
        // 等待V数据
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        
        if (n_block > n_block_min) {
            // 准备加载下一个K块（如果有）
            FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(
                gmem_tiled_copy_QKV, tKgK(_, _, _, n_block-1), tKsK, tKVcKV, tKVpKV,
                binfo.actual_seqlen_k - (n_block-1) * kBlockN
            );
            cute::cp_async_fence();
            
            // 加载下一个ZeroHold块到共享内存
            FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(
                gmem_tiled_copy_ZeroHold, tZeroHoldgZeroHold(_, _, _, n_block-1), tZeroHoldsZeroHold, tZeroHoldcZeroHold, tZeroHoldpZeroHold,
                binfo.actual_seqlen_k - (n_block-1) * kBlockN
            );
            cute::cp_async_fence();
            
            // 加载下一个CausalMask块到共享内存(如果有)
            if (params.causal_mask_ptr != nullptr) {
                FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K>(
                    gmem_tiled_copy_CausalMask, tCausalMaskgCausalMask(_, _, _, n_block-1), tCausalMasksCausalMask, tCausalMaskcCausalMask, tCausalMaskpCausalMask,
                    binfo.actual_seqlen_k - (n_block-1) * kBlockN
                );
                cute::cp_async_fence();
            }
        }
    }
    
    // 后处理和输出归一化
    Tensor lse = softmax.template normalize_softmax_lse<false>(
        acc_o, params.scale_softmax, 1.0f
    );
    
    // 转换acc_o到Element类型
    Tensor rO = FLASH_NAMESPACE::convert_type<Element>(acc_o);
    
    // 准备共享内存用于输出
    Tensor sO = make_tensor(
        sQ.data(), 
        typename Kernel_traits::SmemLayoutO{}
    );
    
    // 设置从累加器到共享内存的拷贝
    auto smem_tiled_copy_O = make_tiled_copy_C(
        typename Kernel_traits::SmemCopyAtomO{}, 
        tiled_mma
    );
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);
    
    // 确保共享内存区域可以安全使用
    if (Kernel_traits::Share_Q_K_smem) {
        __syncthreads();
    }
    
    // 拷贝输出到共享内存
    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);
    
    // 设置全局内存输出张量
    Tensor mO = make_tensor(
        make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
            + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
        make_shape(binfo.actual_seqlen_q, params.h, params.d),
        make_stride(params.o_row_stride, params.o_head_stride, _1{})
    );
    
    Tensor gO = local_tile(
        mO(_, bidh, _), 
        Shape<Int<kBlockM>, Int<kHeadDim>>{},
        make_coord(m_block, 0)
    );
    
    Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(
        params, bidb, bidh, m_block, binfo
    );
    
    // 设置从共享内存到全局内存的拷贝
    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    
    __syncthreads();
    
    // 从共享内存拷贝到寄存器，准备写入全局内存
    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);
    
    // 设置输出的谓词
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) {
            tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d;
        }
    }
    
    // 写入输出到全局内存
    FLASH_NAMESPACE::copy<Is_even_MN, Is_even_K, false, false>(
        gmem_tiled_copy_O, 
        tOrO, 
        tOgO, 
        tOcO, 
        tOpO, 
        binfo.actual_seqlen_q - m_block * kBlockM
    );
    
    // 写入LSE值到全局内存
    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor taccOcO = thr_mma.partition_C(caccO);
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    
    // 将张量转换为(2,2)形式，然后只获取行索引
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));
    
    // 只有第一个线程写入LSE值
    if (get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            if (m_block * kBlockM + get<0>(taccOcO_row(mi)) < binfo.actual_seqlen_q) {
                gLSE(get<0>(taccOcO_row(mi))) = lse(mi);
            }
        }
    }
}

template<typename Kernel_traits, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Return_softmax, typename Params>
inline __device__ void compute_attn(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    // 调用主要的计算函数
    compute_attn_1rowblock<Kernel_traits, Is_causal, Is_even_MN, Is_even_K, Return_softmax>(params, bidb, bidh, m_block);
}

}  // namespace FLASH_NAMESPACE