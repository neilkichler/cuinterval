#pragma once

#ifndef CUINTERVAL_TESTS_COMMON_CUH
#define CUINTERVAL_TESTS_COMMON_CUH

#include <cstddef>
#include <span>

static constexpr std::size_t n_streams = 4;

using cuda_streams = std::span<cudaStream_t, n_streams>;

struct cuda_buffers {
    char *host;
    char *device;
};

#endif // CUINTERVAL_TESTS_COMMON_CUH
