/******************************************************************************
 * Copyright (c) 2025, Jingze Shi and Tri Dao.
 ******************************************************************************/

#pragma once

#include <tuple>
#include <string>
#include <cstdlib>

#if !defined(__CUDACC_RTC__)
#include "cuda_runtime.h"
#endif

#define CHECK_CUDA(call)                                                        \
  do {                                                                          \
    cudaError_t status_ = call;                                                 \
    if (status_ != cudaSuccess) {                                               \
      fprintf(                                                                  \
        stderr,                                                                 \
        "CUDA error (%s:%d): %s\n", __FILE__, __LINE__,                         \
        cudaGetErrorString(status_)                                             \
      );                                                                        \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)


inline int get_current_device() {
    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    return device;
}

inline std::tuple<int, int> get_compute_capability(int device) {
    int capability_major, capability_minor;
    CHECK_CUDA(cudaDeviceGetAttribute(&capability_major, cudaDevAttrComputeCapabilityMajor, device));
    CHECK_CUDA(cudaDeviceGetAttribute(&capability_minor, cudaDevAttrComputeCapabilityMinor, device));
    return {capability_major, capability_minor};
}

inline int get_num_sm(int device) {
    int multiprocessor_count;
    CHECK_CUDA(cudaDeviceGetAttribute(&multiprocessor_count, cudaDevAttrMultiProcessorCount, device));
    return multiprocessor_count;
}

// Check if device supports specific architecture features
inline bool supports_sm89_features(int device) {
    auto [major, minor] = get_compute_capability(device);
    return (major == 8 && minor >= 9) || major >= 9;
}

inline bool supports_sm90_features(int device) {
    auto [major, minor] = get_compute_capability(device);
    return major >= 9;
}

// Get optimal configurations based on GPU architecture and problem size
struct ArchOptimizationConfig {
    bool use_async_copy;
    bool use_multi_level_smem;
    int preferred_block_m;
    int preferred_block_n;
    int max_smem_usage_kb;
    bool enable_double_buffering;
    bool enable_profiling;
};

inline ArchOptimizationConfig get_arch_optimization_config(int device, int seqlen_q, int seqlen_k, int batch_size) {
    auto [major, minor] = get_compute_capability(device);
    
    ArchOptimizationConfig config;
    config.use_async_copy = major >= 8;
    config.use_multi_level_smem = supports_sm90_features(device);
    config.enable_double_buffering = true;
    config.enable_profiling = false;  // Can be enabled via environment variable
    
    // Check for performance profiling environment variable
    const char* enable_profiling_env = std::getenv("FLASH_DMATTN_PROFILE_BACKWARD");
    if (enable_profiling_env && std::string(enable_profiling_env) == "1") {
        config.enable_profiling = true;
    }
    
    // Get max shared memory per block
    int max_smem_per_block;
    CHECK_CUDA(cudaDeviceGetAttribute(&max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
    config.max_smem_usage_kb = max_smem_per_block / 1024;
    
    // Architecture-specific optimizations
    if (major == 9) {  // SM 9.0 (H100)
        config.preferred_block_m = 128;
        config.preferred_block_n = 128;
        // Use larger block sizes for long sequences to improve memory bandwidth utilization
        if (seqlen_q >= 8192) {
            config.preferred_block_m = 128;
            config.preferred_block_n = 128;
        }
    } else if (major == 8 && minor >= 9) {  // SM 8.9 (H200/Ada)
        config.preferred_block_m = 64;
        config.preferred_block_n = 128;
        // Optimize for variable sequence lengths
        if (seqlen_q >= 4096) {
            config.preferred_block_m = 128;
            config.preferred_block_n = 64;
        }
    } else if (major == 8 && minor >= 6) {  // SM 8.6 (A100)
        config.preferred_block_m = 64;
        config.preferred_block_n = 128;
        // Disable double buffering for memory constrained scenarios
        if (config.max_smem_usage_kb < 144) {
            config.enable_double_buffering = false;
        }
    } else {  // SM 8.0 and below
        config.preferred_block_m = 64;
        config.preferred_block_n = 64;
        config.enable_double_buffering = false;
    }
    
    // Adjust for small batch sizes to improve occupancy
    if (batch_size <= 4) {
        config.preferred_block_m = std::min(config.preferred_block_m, 64);
    }
    
    return config;
}

// Performance monitoring hook for backward pass optimization
inline void log_backward_optimization_choice(const char* headdim_str, int device, 
                                            int seqlen_q, int seqlen_k, int batch_size,
                                            const char* optimization_choice) {
    const char* enable_profiling_env = std::getenv("FLASH_DMATTN_PROFILE_BACKWARD");
    if (enable_profiling_env && std::string(enable_profiling_env) == "1") {
        auto [major, minor] = get_compute_capability(device);
        printf("FLASH_DMATTN_PROFILE: HeadDim=%s, Arch=SM%d.%d, SeqQ=%d, SeqK=%d, Batch=%d, Choice=%s\n",
               headdim_str, major, minor, seqlen_q, seqlen_k, batch_size, optimization_choice);
    }
}
