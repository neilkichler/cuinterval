#include "generated/tests_generated.cu"
#include "tests_additional.cu"
#include "tests_common.cuh"

#include <array>
#include <cstddef>

int main(int argc, char *argv[])
{
    cuda_buffers buffers;
    std::size_t n_bytes = 128 * 1024 * 2 * sizeof(double);

    CUDA_CHECK(cudaSetDevice(0));

    CUDA_CHECK(cudaMallocHost(&buffers.host, n_bytes));
    CUDA_CHECK(cudaMalloc(&buffers.device, n_bytes));

    std::array<cudaStream_t, n_streams> streams {};
    for (auto &stream : streams)
        CUDA_CHECK(cudaStreamCreate(&stream));

    tests_additional<double>(buffers, streams);
    tests_generated<double>(buffers, streams);

    for (auto &stream : streams)
        CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaFree(buffers.device));
    CUDA_CHECK(cudaFreeHost(buffers.host));
    return 0;
}
