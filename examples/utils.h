#pragma once

#ifndef CUDA_INTERVAL_EXAMPLES_UTILS_H
#define CUDA_INTERVAL_EXAMPLES_UTILS_H

#include <cstdio>

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


#endif // CUDA_INTERVAL_EXAMPLES_UTILS_H

