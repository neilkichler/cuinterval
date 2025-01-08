#pragma once
#ifndef CUDA_INTERVAL_TESTS_POWF_H
#define CUDA_INTERVAL_TESTS_POWF_H

#include <cuda_runtime.h>

void tests_powf(cudaStream_t stream, cudaEvent_t event);

#endif // CUDA_INTERVAL_TESTS_POWF_H
