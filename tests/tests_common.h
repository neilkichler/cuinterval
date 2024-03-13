#pragma once

#ifndef CUINTERVAL_TESTS_COMMON_H
#define CUINTERVAL_TESTS_COMMON_H

#include <cstdio>
#include <cstddef>
#include <span>

#include <cuda_runtime.h>

static constexpr std::size_t n_streams = 4;

using cuda_streams = std::span<cudaStream_t, n_streams>;

struct cuda_buffer {
    char *host;
    char *device;
};

using cuda_buffers = std::span<cuda_buffer, n_streams>;

#define CUDA_CHECK(x)                                                                \
    do {                                                                             \
        cudaError_t err = x;                                                         \
        if (err != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, \
                    __FILE__, __LINE__, cudaGetErrorString(err),                     \
                    cudaGetErrorName(err), err);                                     \
            abort();                                                                 \
        }                                                                            \
    } while (0)


#endif // CUINTERVAL_TESTS_COMMON_H
