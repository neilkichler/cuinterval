#pragma once

#include "tests_common.h"

void tests_bisect(cuda_buffer buffer, cuda_streams streams);

void tests_mince(cuda_buffer buffer, cudaStream_t stream);

void tests_bisection(cuda_buffer buffer, cudaStream_t stream);
