#include <cuinterval/cuinterval.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <functional>

#include "test_ops.cuh"
#include "tests.h"

template<typename T>
void tests_bisect()
{
    using namespace boost::ut;

    using I = interval<T>;
    using B = bool;
    using N = int;

    T infinity = std::numeric_limits<T>::infinity();
    I empty    = { infinity, -infinity };
    I entire   = { -infinity, infinity };
    T NaN      = ::nan("");

    const int n                          = 8; // count of largest test array
    const int n_bytes                    = n * sizeof(I);
    const int blockSize                  = 256;
    [[maybe_unused]] const int numBlocks = (n + blockSize - 1) / blockSize;

    I *d_xs_, *d_ys_, *d_zs_, *d_res_;

    CUDA_CHECK(cudaMalloc(&d_xs_, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_ys_, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_zs_, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_res_, 2 * n_bytes));

    "bisection"_test = [&] {
        constexpr int n = 8;
        std::array<I, n> h_xs { {
            empty,
            entire,
            entire,
            entire,
            { 0.0, 2.0 },
            { 1.0, 1.0 },
            { 0.0, 1.0 },
            { 0.0, 1.0 },
        } };

        std::array<T, n> h_ys { {
            0.5,
            0.5,
            0.25,
            0.75,
            0.5,
            0.5,
            0.5,
            0.25,
        } };

        std::array<split<T>, n> h_res {};
        split<T> *d_res    = (split<T> *)d_res_;
        I *d_xs            = (I *)d_xs_;
        T *d_ys            = (T *)d_ys_;
        int n_result_bytes = n * sizeof(*d_res);
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

        // CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        // CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        // CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_bisect<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        check_all_equal<split<T>, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    CUDA_CHECK(cudaFree(d_xs_));
    CUDA_CHECK(cudaFree(d_ys_));
    CUDA_CHECK(cudaFree(d_zs_));
    CUDA_CHECK(cudaFree(d_res_));
}

// Stack in local memory. Managed independently for each thread.
template<class T, std::size_t N>
struct local_stack
{
    using size_type = std::size_t;

    __device__ T const &top() const { return buf[len - 1]; }
    __device__ T &top() { return buf[len - 1]; }
    __device__ void push(T const &v) { buf[len++] = v; }
    __device__ T pop() { return buf[--len]; }
    __device__ bool full() { return len == N; }
    __device__ bool empty() { return len == 0; }

    T buf[N];
    size_type len {};
};

template<typename I>
__device__ I f(I x)
{
    return exp(I { -3.0, -3.0 } * x) - sin(x) * sin(x) * sin(x);
};

template<typename T, int max_depth>
__global__ void bisection(interval<T> x_init, double tol, interval<T> *roots, std::size_t *max_roots)
{
    using I = interval<T>;

    std::size_t n_roots = 0;
    local_stack<I, max_depth> intervals;
    intervals.push(x_init);

    for (int depth = 0; !intervals.empty() && depth < max_depth; depth++) {
        I x = intervals.pop();
        I y = f(x);

        if (!contains(y, 0.0)) {
            continue; // no roots in this interval -> no further splitting
        }

        T m = mid(x);
        if (width(x) < tol || m == inf(x) || m == sup(x)) {
            // found a root
            roots[n_roots] = x;
            n_roots++;
            if (n_roots == *max_roots) {
                break; // reached max roots we can store
            }
        } else {
            // interval could still contain a root -> bisect
            split<T> c = bisect(x, 0.5);
            intervals.push(c.upper_half);
            intervals.push(c.lower_half);
        }
    }

    *max_roots = n_roots;
}

template<typename T>
void tests_bisection()
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

    std::size_t *d_max_roots;
    CUDA_CHECK(cudaMalloc(&d_max_roots, sizeof(*d_max_roots)));
    CUDA_CHECK(cudaMemcpy(d_max_roots, &max_roots, sizeof(*d_max_roots), cudaMemcpyHostToDevice));
    thrust::device_vector<I> roots(max_roots);

    I *d_roots = thrust::raw_pointer_cast(roots.data());
    bisection<T, max_depth><<<1, 1>>>(x, tolerance, d_roots, d_max_roots);
    CUDA_CHECK(cudaMemcpy(&max_roots, d_max_roots, sizeof(*d_max_roots), cudaMemcpyDeviceToHost));

    roots.resize(max_roots);
    thrust::host_vector<I> h_roots = roots;
    for (std::size_t i = 0; i < max_roots; i++) {
        expect(contains(h_roots[i], ref_roots[i]));
        expect(le(h_roots[i].lb, ref_roots[i]));
        expect(ge(h_roots[i].ub, ref_roots[i]));
    }
}
