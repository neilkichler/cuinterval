
#include <cuinterval/interval.h>
#include <cuinterval/examples/bisection.cuh>

#include <thrust/async/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "tests_common.h"
#include "tests_ops.cuh"

void test_bisect_call(cudaStream_t stream, int n,
                      interval<double> *x, double *y, split<double> *res)
{
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;

    test_bisect<<<numBlocks, blockSize, 0, stream>>>(n, x, y, res);
}

template<typename I>
__host__ __device__ I f(I x)
{
    return pow(x, 3) - pow(x, 2) - 17.0 * x - 15.0;
};

typedef interval<double> (*fn_t)(interval<double>);
__device__ fn_t d_f = f<interval<double>>;

thrust::host_vector<interval<double>> test_bisection_kernel(cudaStream_t stream, cuda_buffer buffer, interval<double> x, double tolerance)
{
    using T                         = double;
    using I                         = interval<T>;
    constexpr std::size_t max_depth = 512;
    std::size_t max_roots           = 16;

    std::size_t *d_max_roots = (std::size_t *)buffer.device;
    CUDA_CHECK(cudaMemcpyAsync(d_max_roots, &max_roots, sizeof(*d_max_roots), cudaMemcpyHostToDevice, stream));
    thrust::device_vector<I> roots(max_roots);

    I *d_roots = thrust::raw_pointer_cast(roots.data());

    fn_t h_f;
    cudaMemcpyFromSymbol(&h_f, d_f, sizeof(h_f));
    bisection<T, max_depth><<<1, 1, 0, stream>>>(h_f, x, tolerance, d_roots, d_max_roots);

    CUDA_CHECK(cudaMemcpyAsync(&max_roots, d_max_roots, sizeof(*d_max_roots), cudaMemcpyDeviceToHost, stream));
    thrust::host_vector<I> h_roots(max_roots);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    thrust::device_event e = thrust::async::copy(roots.begin(), roots.begin() + max_roots, h_roots.begin());

    return h_roots;
}

void tests_mince_call(int numBlocks, int blockSize, cudaStream_t stream,
                      int n, interval<double> *d_xs, int *d_offsets, interval<double> *d_res)
{
    test_mince<<<numBlocks, blockSize, 0, stream>>>(n, d_xs, d_offsets, d_res);
}
