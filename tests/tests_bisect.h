#pragma once

#include "tests_common.h"

void tests_bisect(cuda_buffer buffer, cuda_streams streams, cuda_events events);

void tests_mince(cuda_buffer buffer, cudaStream_t stream, cudaEvent_t  events);

void tests_bisection(cuda_buffer buffer, cudaStream_t stream, cudaEvent_t events);
