#pragma once
#ifndef CUDA_INTERVAL_TESTS_NESTED_H
#define CUDA_INTERVAL_TESTS_NESTED_H

#include <cuda_runtime.h>

void tests_nested(cudaStream_t stream, cudaEvent_t event);

#endif // CUDA_INTERVAL_TESTS_NESTED_H
