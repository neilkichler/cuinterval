#include "generated/tests_generated.cu"
#include "tests_additional.cu"

#include <cstddef>

int main()
{
    char *buffer;
    std::size_t n_bytes = 1024 * 1024 * 2 * sizeof(double);

    CUDA_CHECK(cudaMalloc(&buffer, n_bytes));

    tests_additional<double>(buffer);
    tests_generated<double>(buffer);

    CUDA_CHECK(cudaFree(buffer));
    return 0;
}
