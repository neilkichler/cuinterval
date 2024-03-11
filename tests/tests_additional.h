#include "tests_bisect.h"
#include "tests_common.h"
// #include "tests_loop.cu"

#include <omp.h>

template<typename T>
void tests_additional(cuda_buffers buffers, cuda_streams streams)
{
    #pragma omp parallel
    {
        #pragma omp single
        {
            #pragma omp task
            tests_bisect<T>(buffers, streams);
            #pragma omp task
            tests_bisection(buffers, streams[1]);
            // tests_bisection<T>(buffers, streams[1]);
            // #pragma omp task
            // tests_pi_approximation<T>(streams[2]);
            // #pragma omp task
            // tests_horner<T>(streams[3]);
        }
    }
}
