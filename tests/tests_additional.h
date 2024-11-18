#pragma once

#include "tests_bisect.h"
#include "tests_common.h"
#include "tests_loop.h"
#include "tests_operator_overloading.h"

#include <omp.h>

template<typename T>
void tests_additional(cuda_buffers buffers, cuda_streams streams, cuda_events events)
{
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            #pragma omp task depend(inout:buffers[0].host,buffers[0].device)
            tests_bisect(buffers[0], streams, events);
            #pragma omp task depend(inout:buffers[1].host,buffers[1].device)
            tests_bisection(buffers[1], streams[1], events[1]);
            #pragma omp task
            tests_pi_approximation(streams[2], events[2]);
            #pragma omp task
            tests_horner(streams[3], events[3]);
            #pragma omp task depend(inout:buffers[0].host,buffers[0].device)
            tests_mince(buffers[0], streams[0], events[0]);
            #pragma omp task depend(inout:buffers[1].host,buffers[1].device)
            tests_operator_overloading(buffers[1], streams[1], events[1]);
        }
    }
}
