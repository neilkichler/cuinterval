#include <cuinterval/cuinterval.h>
#include <cuinterval/examples/bisection.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "test_ops.cuh"
#include "tests.h"
#include "tests_common.cuh"

template<typename T>
void tests_bisect(cuda_buffers buffers, cuda_streams streams)
{
    using namespace boost::ut;

    using I = interval<T>;
    using B = bool;
    using N = int;

    T infinity = std::numeric_limits<T>::infinity();
    I empty    = { infinity, -infinity };
    I entire   = { -infinity, infinity };
    T NaN      = ::nan("");

    char *d_buffer                       = buffers.device;
    char *h_buffer                       = buffers.host;
    const int n                          = 8; // count of largest test array
    const int n_bytes                    = n * sizeof(I);
    const int blockSize                  = 256;
    [[maybe_unused]] const int numBlocks = (n + blockSize - 1) / blockSize;

    I *d_xs_, *d_ys_, *d_zs_, *d_res_;

    d_xs_  = (I *)d_buffer;
    d_ys_  = (I *)d_buffer + 1 * n_bytes;
    d_zs_  = (I *)d_buffer + 2 * n_bytes;
    d_res_ = (I *)d_buffer + 3 * n_bytes;

    "bisection"_test = [&] {
        constexpr int n = 8;
        I *h_xs         = new (h_buffer) I[n] {
            empty,
            entire,
            entire,
            entire,
            { 0.0, 2.0 },
            { 1.0, 1.0 },
            { 0.0, 1.0 },
            { 0.0, 1.0 },
        };
        h_buffer += n * sizeof(I);

        T *h_ys = new (h_buffer) T[n] {
            0.5,
            0.5,
            0.25,
            0.75,
            0.5,
            0.5,
            0.5,
            0.25,
        };
        h_buffer += n * sizeof(I);
        split<T> *h_res = new (h_buffer) split<T>[n] {};
        h_buffer += n * sizeof(split<T>);
        split<T> *d_res    = (split<T> *)d_res_;
        I *d_xs            = (I *)d_xs_;
        T *d_ys            = (T *)d_ys_;
        int n_result_bytes = n * sizeof(*d_res);

        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n_bytes, cudaMemcpyHostToDevice, streams[0]));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n_bytes, cudaMemcpyHostToDevice, streams[1]));
        CUDA_CHECK(cudaMemcpyAsync(d_res, h_res, n_result_bytes, cudaMemcpyHostToDevice, streams[2]));
        test_bisect<<<numBlocks, blockSize, 0, streams[3]>>>(n, d_xs, d_ys, d_res);
      
        std::array<split<T>, n> h_ref { {
            { empty, empty },
            { { entire.lb, 0.0 }, { 0.0, entire.ub } },
            { { entire.lb, -0x1.fffffffffffffp+1023 }, { -0x1.fffffffffffffp+1023, entire.ub } },
            { { entire.lb, 0x1.fffffffffffffp+1023 }, { 0x1.fffffffffffffp+1023, entire.ub } },
            { { 0.0, 1.0 }, { 1.0, 2.0 } },
            { { 1.0, 1.0 }, empty },
            { { 0.0, 0.5 }, { 0.5, 1.0 } },
            { { 0.0, 0.25 }, { 0.25, 1.0 } },
        } };

        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n_result_bytes, cudaMemcpyDeviceToHost, streams[0]));
        CUDA_CHECK(cudaDeviceSynchronize());
        int max_ulp_diff = 0;
        check_all_equal<split<T>, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };
}

template<typename I>
__device__ I f(I x)
{
    return exp(I { -3.0, -3.0 } * x) - sin(x) * sin(x) * sin(x);
};

template<typename T>
void tests_bisection(cuda_buffers buffers, cudaStream_t stream)
{
    using namespace boost::ut;
    using I = interval<T>;

    I x = { -10.0, 10.0 };

    T ref_roots[4] = {
        0.588532743981861,
        3.09636393241065,
        6.28504927338259,
        9.42469725473852
    };

    constexpr double tolerance      = 1e-7;
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
    for (std::size_t i = 0; i < max_roots; i++) {
        expect(contains(h_roots[i], ref_roots[i]));
        expect(h_roots[i].ub - h_roots[i].lb < tolerance);
    }
}
