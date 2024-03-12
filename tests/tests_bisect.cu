
#include <cuinterval/arithmetic/interval.h>
#include <cuinterval/examples/bisection.cuh>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "tests_common.h"
#include "tests_ops.cuh"

void test_bisect_call(cudaStream_t stream, int n,
                      interval<double> *x, double *y, split<double> *res)
{
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;

    test_bisect<<<numBlocks, blockSize, 0, stream>>>(n, x, y, res);
}

void test_bisection_call(cudaStream_t stream, interval<double> x, double tolerance,
                         interval<double> *roots, std::size_t *max_roots)
{

    constexpr std::size_t max_depth = 512;
    bisection<double, max_depth><<<1, 1, 0, stream>>>(x, tolerance, roots, max_roots);
}


// template<typename I>
thrust::host_vector<interval<double>> test_bisection_kernel(cudaStream_t stream, cuda_buffers buffers, interval<double> x, double tolerance) {
    using T = double;
    using I = interval<T>;
    constexpr std::size_t max_depth = 512;
    std::size_t max_roots           = 16;

    std::size_t *d_max_roots = (std::size_t *)buffers.device;
    CUDA_CHECK(cudaMemcpy(d_max_roots, &max_roots, sizeof(*d_max_roots), cudaMemcpyHostToDevice));
    thrust::device_vector<I> roots(max_roots);

    I *d_roots = thrust::raw_pointer_cast(roots.data());
    bisection<T, max_depth><<<1, 1, 0, stream>>>(x, tolerance, d_roots, d_max_roots);

    CUDA_CHECK(cudaMemcpy(&max_roots, d_max_roots, sizeof(*d_max_roots), cudaMemcpyDeviceToHost));

    roots.resize(max_roots);
    thrust::host_vector<I> h_roots = roots;
    return h_roots;
}
