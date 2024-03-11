#pragma once

#ifndef CUINTERVAL_TESTS_COMMON_H
#define CUINTERVAL_TESTS_COMMON_H

#include <cstddef>
#include <span>

#include <cuda_runtime.h>

static constexpr std::size_t n_streams = 4;

using cuda_streams = std::span<cudaStream_t, n_streams>;

struct cuda_buffers {
    char *host;
    char *device;
};

#endif // CUINTERVAL_TESTS_COMMON_H
