#pragma once
#ifndef CUDA_INTERVAL_TESTS_LOOP_H
#define CUDA_INTERVAL_TESTS_LOOP_H

#include <cuda_runtime.h>

void tests_pi_approximation(cudaStream_t stream);
void tests_horner(cudaStream_t stream);

#endif // CUDA_INTERVAL_TESTS_LOOP_H
