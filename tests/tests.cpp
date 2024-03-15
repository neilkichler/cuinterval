#include "generated/tests_generated.h"
#include "tests_additional.h"
#include "tests_common.h"

#include <cuda_runtime.h>
#include <omp.h>

#include <array>
#include <cstddef>
#include <cstdio>

int main(int argc, char *argv[])
{
    CUDA_CHECK(cudaSetDevice(0));

    std::size_t n_bytes = 128 * 1024 * 2 * sizeof(double);
    std::array<cuda_buffer, n_streams> buffers {};

    char *host_backing_buffer;
    char *device_backing_buffer;
    CUDA_CHECK(cudaMallocHost(&host_backing_buffer, buffers.size() * n_bytes));
    CUDA_CHECK(cudaMalloc(&device_backing_buffer, buffers.size() * n_bytes));

    std::size_t offset = 0;
    for (auto &buffer : buffers) {
        buffer.host   = host_backing_buffer + offset;
        buffer.device = device_backing_buffer + offset;
        offset += n_bytes;
    }

    std::array<cudaStream_t, n_streams> streams {};
    for (auto &stream : streams)
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    tests_additional<double>(buffers, streams);
    tests_generated(buffers, streams);

    for (auto &stream : streams)
        CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(device_backing_buffer));
    CUDA_CHECK(cudaFreeHost(host_backing_buffer));

    return 0;
}
