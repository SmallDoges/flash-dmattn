#define FLASH_ATTENTION_ENABLE_BF16
/******************************************************************************
 * Copyright (c) 2025, Jingze Shi and Tri Dao.
 ******************************************************************************/

#pragma once
#include "namespace_config.h"
#include <c10/cuda/CUDAException.h>  // For C10_CUDA_CHECK and C10_CUDA_KERNEL_LAUNCH_CHECK

#include "static_switch.h"
#include "hardware_info.h"
#include "flash.h"
#include "flash_fwd_kernel.h"

namespace FLASH_NAMESPACE {

// Determine if the architecture supports FLASH and define a macro to handle parameter modifiers
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_FLASH
#define KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define KERNEL_PARAM_MODIFIER
#endif

// Define a macro for unsupported architecture handling to centralize the error message
#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FlashDynamicMaskAttention requires building with sm version sm80-sm90, but was built for < 8.0!");

// Use a macro to clean up kernel definitions
#define DEFINE_FLASH_FORWARD_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_fwd_params params)

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_kernel, bool Is_dropout, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax) {
    #if defined(ARCH_SUPPORTS_FLASH)
        FLASH_NAMESPACE::compute_attn<Kernel_traits, Is_dropout, Is_causal, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_kernel, bool Is_causal, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split) {
    #if defined(ARCH_SUPPORTS_FLASH)
        FLASH_NAMESPACE::compute_attn_splitkv<Kernel_traits, Is_causal, Is_even_MN, Is_even_K, Is_softcap, Split>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_combine_kernel, int kBlockM, int Log_max_splits, bool Is_even_K) {
    static_assert(Log_max_splits >= 1);
    FLASH_NAMESPACE::combine_attn_seqk_parallel<Kernel_traits, kBlockM, Log_max_splits, Is_even_K>(params);
}

template<typename Kernel_traits, bool Is_dropout, bool Is_causal>
void run_flash_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    const size_t smem_size = Kernel_traits::kSmemSize;
    // printf("smem_size = %d (includes mask memory)\n", int(smem_size));

    // Work-around for gcc 7. It doesn't like nested BOOL_SWITCH.
    // https://github.com/kokkos/kokkos-kernels/issues/349
    // https://github.com/HazyResearch/flash-attention/issues/21

    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    const bool return_softmax = params.p_ptr != nullptr;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            BOOL_SWITCH(return_softmax, ReturnSoftmaxConst, [&] {
                SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                    // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                    // If return_softmax, set IsEvenMNConst to false to reduce number of templates
                    // If head dim > 128, set IsEvenMNConst to false to reduce number of templates
                    auto kernel = &flash_fwd_kernel<Kernel_traits, Is_dropout && !Is_softcap, Is_causal, IsEvenMNConst && IsEvenKConst && !ReturnSoftmaxConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst && !ReturnSoftmaxConst, Is_softcap, ReturnSoftmaxConst && Is_dropout && !Is_softcap>;
                    // auto kernel = &flash_fwd_kernel<Kernel_traits, false, Is_causal, false, false, true, true, false>;
                    // printf("run_flash_fwd: IsEvenMNConst = %d, IsEvenKConst = %d, Is_causal = %d, ReturnSoftmaxConst = %d, Is_dropout = %d\n", int(IsEvenMNConst), int(IsEvenKConst), int(Is_causal), int(ReturnSoftmaxConst), int(Is_dropout));
                    // auto kernel = &flash_fwd_kernel<Kernel_traits, false, Is_causal, false, true, true, false>;
                    if (smem_size >= 48 * 1024) {
                        C10_CUDA_CHECK(cudaFuncSetAttribute(
                            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                    }
                    // int ctas_per_sm;
                    // cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    //     &ctas_per_sm, kernel, Kernel_traits::kNThreads, smem_size);
                    // printf("run_flash_fwd: smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);
                    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
            });
        });
    });
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_splitkv_fwd(Flash_fwd_params &params, cudaStream_t stream) {
    static_assert(!Kernel_traits::Is_Q_in_regs, "SplitKV implementation does not support Is_Q_in_regs");
    static_assert(!Kernel_traits::Share_Q_K_smem, "SplitKV implementation does not support Share_Q_K_smem");
    constexpr size_t smem_size = Kernel_traits::kSmemSize;
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.b, params.num_splits > 1 ? params.b * params.h : params.h);
    const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr && params.seqlen_k % Kernel_traits::kBlockN == 0 && params.seqlen_q % Kernel_traits::kBlockM == 0;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            BOOL_SWITCH(params.num_splits > 1, Split, [&] {
                SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
                    // If not IsEvenKConst, we also set IsEvenMNConst to false to reduce number of templates.
                    // If Is_local, set Is_causal to false
                    auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Is_causal, IsEvenMNConst && IsEvenKConst && Kernel_traits::kHeadDim <= 128, IsEvenKConst, Is_softcap, Split>;
                    // auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Is_causal, false, true, Split>;
                    // auto kernel = &flash_fwd_splitkv_kernel<Kernel_traits, Is_causal, false, IsEvenKConst>;
                    // printf("run_flash_splitkv_fwd: Split = %d, Is_causal = %d, IsEvenMNConst = %d, IsEvenKConst = %d, Is_softcap = %d\n", int(Split), int(Is_causal), int(IsEvenMNConst), int(IsEvenKConst), int(Is_softcap));
                    if (smem_size >= 48 * 1024) {
                        C10_CUDA_CHECK(cudaFuncSetAttribute(
                            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
                    }
                    // int ctas_per_sm;
                    // cudaError status_ = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                    //     &ctas_per_sm, kernel, Kernel_traits::kNThreads, smem_size);
                    // printf("run_flash_splitkv_fwd: smem_size = %d, CTAs per SM = %d\n", int(smem_size), ctas_per_sm);
                    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
                    C10_CUDA_KERNEL_LAUNCH_CHECK();
                });
            });
        });
    });
    if (params.num_splits > 1) {
        // We want kBlockM to be as small as possible for more parallelism.
        // With 128 threads we can load 512 elements at a time, so if headdim is divisible by 128, kBlockM = 4.
        // If headdim is divisible by 64, then we set kBlockM = 8, etc.
        constexpr static int kBlockM = Kernel_traits::kHeadDim % 128 == 0 ? 4 : (Kernel_traits::kHeadDim % 64 == 0 ? 8 : 16);
        dim3 grid_combine((params.b * params.h * params.seqlen_q + kBlockM - 1) / kBlockM);
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            if (params.num_splits <= 2) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 1, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 4) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 2, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 8) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 3, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 16) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 4, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 32) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 5, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 64) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 6, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 128) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 7, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    }
}

template<typename T, int Headdim, bool Is_causal>
void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int kBlockM = 64;  // Fixed for all head dimensions
    constexpr static int kBlockN = Headdim <= 64 ? 64 : (Headdim < 128 ? 64 : 32);
    run_flash_splitkv_fwd<Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, 4, false, false, T>, Is_causal>(params, stream);
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim32(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 32;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            // Using 8 warps is 18% slower for seqlen=2k, 2 warps is 5% slower
            // Using block size (64 x 128) is 27% slower for seqlen=2k
            // Using block size (128 x 64) is 85% slower for seqlen=2k, because of register spilling
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim96(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 96;
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor > 0;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // For sm86 or sm89, 64 x 64 is the fastest for causal (because it's square),
        if (is_sm8x) {
            if constexpr(!Is_causal) {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
        // These two are always slower
        // run_flash_fwd<Flash_fwd_kernel_traits<96, 128, 128, 4, true, T>>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<96, 64, 128, 4, true, T>>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    auto [cc_major, cc_minor] = get_compute_capability(get_current_device());
    bool is_sm8x = cc_major == 8 && cc_minor > 0;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            // For sm86 or sm89, 64 x 32 (40 KB smem) is the fastest for causal and non-causal since we get 2 CTAs per SM.
            // Use block configuration (kBlockM = 64, kBlockN = 64) for better memory alignment
            if (is_sm8x) {
                if constexpr(!Is_causal) {
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                } else {
                    run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
                }
            } else {
                run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            }
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 128, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // Using 8 warps (128 x 128 and 256 x 64) is 28% slower for seqlen=2k
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // 1st ones are good for H100, A100
            // 2nd one is good for A6000 bc we get slightly better occupancy
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, true, false, T>, Is_dropout, Is_causal>(params, stream);
            // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 4, true, true, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim192(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 192;
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        if constexpr(!Is_dropout) {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 32, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 32, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 32, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, T>>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 128, 4, false, T>>(params, stream);
        // run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 128, 128, 8, false, T>>(params, stream);
    });
}

template<typename T, bool Is_causal>
void run_mha_fwd_hdim256(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_sm, max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_sm, cudaDevAttrMaxSharedMemoryPerMultiprocessor, device);
    status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
      C10_CUDA_CHECK(status_);
    }
    // printf("max_smem_per_sm = %d, max_smem_per_block = %d\n", max_smem_per_sm, max_smem_per_block);
    DROPOUT_SWITCH(params.p_dropout < 1.f, Is_dropout, [&] {
        // For A100, we want to run with 64 x 64 (112KB smem).
        // For H100 we want to run with 64 x 32 (72KB smem) since then we can get 2 CTAs per SM.
        if (max_smem_per_block >= 2 * Headdim * (128 + 2 * 64) && max_smem_per_sm < 4 * Headdim * (64 + 2 * 64)) {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 64, 8, false, false, T>, Is_dropout, Is_causal>(params, stream);
        } else {
            run_flash_fwd<Flash_fwd_kernel_traits<Headdim, 64, 32, 4, false, false, T>, Is_dropout, Is_causal>(params, stream);
        }
    });
}

} // namespace FLASH_NAMESPACE
