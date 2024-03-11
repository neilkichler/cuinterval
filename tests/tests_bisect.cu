
#include <cuinterval/arithmetic/interval.h>
#include <cuinterval/examples/bisection.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "test_ops.cuh"
#include "tests.h"
#include "tests_common.h"

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

template<typename I>
__device__ I f(I x)
{
    return exp(I { -3.0, -3.0 } * x) - sin(x) * sin(x) * sin(x);
};

#include <omp.h>

void tests_bisection(cuda_buffers buffers, cudaStream_t stream)
{
    printf("Bisection: Inside OpenMP thread %i\n", omp_get_thread_num());

    using namespace boost::ut;
    using T = double;
    using I = interval<T>;

    I x = { -10.0, 10.0 };

    T ref_roots[4] = {
        0.588532743981861,
        3.09636393241065,
        6.28504927338259,
        9.42469725473852
    };

    constexpr double tolerance = 1e-7;
    constexpr std::size_t max_depth = 512;
    std::size_t max_roots      = 16;

    std::size_t *d_max_roots = (std::size_t *)buffers.device;
    CUDA_CHECK(cudaMemcpy(d_max_roots, &max_roots, sizeof(*d_max_roots), cudaMemcpyHostToDevice));
    thrust::device_vector<I> roots(max_roots);

    I *d_roots = thrust::raw_pointer_cast(roots.data());
    bisection<T, max_depth><<<1, 1, 0, stream>>>(x, tolerance, d_roots, d_max_roots);

    CUDA_CHECK(cudaMemcpy(&max_roots, d_max_roots, sizeof(*d_max_roots), cudaMemcpyDeviceToHost));

    roots.resize(max_roots);
    thrust::host_vector<I> h_roots = roots;
    for (std::size_t i = 0; i < max_roots; i++) {
        expect(contains(h_roots[i], ref_roots[i]));
        expect(h_roots[i].ub - h_roots[i].lb < tolerance);
    }
}
