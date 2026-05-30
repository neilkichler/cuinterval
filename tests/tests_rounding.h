#pragma once
#ifndef CUDA_INTERVAL_TESTS_ROUNDING_H
#define CUDA_INTERVAL_TESTS_ROUNDING_H

#include <cuda_runtime.h>

void tests_rounding(cudaStream_t stream, cudaEvent_t event);

#endif // CUDA_INTERVAL_TESTS_ROUNDING_H
