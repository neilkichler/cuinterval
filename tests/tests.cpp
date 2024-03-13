#include "tests.h"
#include "generated/tests_generated.h"
#include "tests_additional.h"
#include "tests_common.h"

#include <cuda_runtime.h>
#include <omp.h>

#include <array>
#include <cstddef>
#include <cstdio>

void tests_generated(cuda_buffers buffers, cuda_streams streams);

int main(int argc, char *argv[])
{
    std::size_t n_bytes = 128 * 1024 * 2 * sizeof(double);

    CUDA_CHECK(cudaSetDevice(0));

    #pragma omp parallel // we could use: [[omp::directive(parallel)]]
    {
        printf("hello from omp thread %i\n", omp_get_thread_num());
    }

    std::array<cuda_buffer, n_streams> buffers {};
    for (auto &buffer : buffers) {
        CUDA_CHECK(cudaMallocHost(&buffer.host, n_bytes));
        CUDA_CHECK(cudaMalloc(&buffer.device, n_bytes));
    }

    std::array<cudaStream_t, n_streams> streams {};
    for (auto &stream : streams)
        CUDA_CHECK(cudaStreamCreate(&stream));

    tests_additional<double>(buffers, streams);
    tests_generated(buffers, streams);

    for (auto &stream : streams)
        CUDA_CHECK(cudaStreamDestroy(stream));

    for (auto &buffer : buffers) {
        CUDA_CHECK(cudaFree(buffer.device));
        CUDA_CHECK(cudaFreeHost(buffer.host));
    }

    return 0;
}
