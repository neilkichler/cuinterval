#ifndef CUDA_INTERVAL_TESTS_OPERATOR_OVERLOADING_H
#define CUDA_INTERVAL_TESTS_OPERATOR_OVERLOADING_H

#include "tests_common.h"

#include <cuda_runtime.h>

void tests_operator_overloading(cuda_buffer buffer, cudaStream_t stream, cudaEvent_t event);

#endif // CUDA_INTERVAL_TESTS_OPERATOR_OVERLOADING_H
