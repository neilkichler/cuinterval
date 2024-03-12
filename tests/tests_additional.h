#pragma once

#include "tests_bisect.h"
#include "tests_common.h"
#include "tests_loop.h"

#include <omp.h>

template<typename T>
void tests_additional(cuda_buffers buffers, cuda_streams streams)
{
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            #pragma omp task
            tests_bisect<T>(buffers, streams);
            #pragma omp task
            tests_bisection(buffers, streams[1]);
            #pragma omp task
            tests_pi_approximation(streams[2]);
            #pragma omp task
            tests_horner(streams[3]);
        }
    }
}
