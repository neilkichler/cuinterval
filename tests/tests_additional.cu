#include "tests_bisect.cu"
#include "tests_loop.cu"
#include "tests_common.cuh"

template<typename T>
void tests_additional(cuda_buffers buffers, cuda_streams streams)
{
    tests_bisect<T>(buffers, streams);
    tests_bisection<T>(buffers, streams[1]);
    tests_pi_approximation<T>(streams[2]);
    tests_horner<T>(streams[3]);
}
