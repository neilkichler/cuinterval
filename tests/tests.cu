
#include <cuinterval/cuinterval.h>

#include <stdio.h>
#include <stdlib.h>

// compiler bug fix; TODO: remove when fixed
#ifdef __CUDACC__
#pragma push_macro("__cpp_consteval")
#define consteval constexpr
#include <boost/ut.hpp>
#undef consteval
#pragma pop_macro("__cpp_consteval")
#else
#include <boost/ut.hpp>
#endif

#define CUDA_CHECK(x)                                                                \
    do {                                                                             \
        cudaError_t err = x;                                                         \
        if (err != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, \
                    __FILE__, __LINE__, cudaGetErrorString(err),                     \
                    cudaGetErrorName(err), err);                                     \
            abort();                                                                 \
        }                                                                            \
    } while (0)

template<typename T>
__global__ void test_neg(int n, interval<T> *x)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        x[i] = -x[i];
    }
}

template<typename T>
__global__ void test_add(int n, interval<T> *x, interval<T> *y)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        x[i] = x[i] + y[i];
    }
}

template<typename T>
__global__ void test_sub(int n, interval<T> *x, interval<T> *y)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        x[i] = x[i] - y[i];
    }
}

int main()
{
    using namespace boost::ut;

    using I = interval<double>;

    suite arith_tests = [] {
        I empty = ::empty<double>();
        // I entire = ::entire<double>();

        const int n = 8;
        int n_bytes = n * sizeof(I);

        interval<double> *d_vec;
        CUDA_CHECK(cudaMalloc(&d_vec, n_bytes));

        "neg"_test = [&] {
            I h_xs[n] = {
                { 0, 1 },
                { 1, 2 },
                // empty,
                { 0, 2 },
                { -0, 2 },
                { -2, 0 },
                { -2, -0 },
                { 0, 0 },
                { -0, 0 },
            };

            I h_ref[n] = {
                { -1, -0 },
                { -2, -1 },
                // empty,
                { -2, 0 },
                { -2, 0 },
                { 0, 2 },
                { 0, 2 },
                { 0, 0 },
                { 0, 0 },
            };

            CUDA_CHECK(cudaMemcpy(d_vec, h_xs, n_bytes, cudaMemcpyHostToDevice));

            int blockSize = 256;
            int numBlocks = (n + blockSize - 1) / blockSize;

            test_neg<<<numBlocks, blockSize>>>(n, d_vec);

            CUDA_CHECK(cudaMemcpy(h_xs, d_vec, n_bytes, cudaMemcpyDeviceToHost));

            for (int i = 0; i < n; ++i) {
                expect(h_xs[i] == h_ref[i]);
                // printf("h_in: %lf, h_ref: %lf\n", h_xs[i].lb, h_ref[i].lb);
                // printf("h_in: %lf, h_ref: %lf\n", h_xs[i].ub, h_ref[i].ub);
            }
        };

        // "add"_test = [&] {
        //     for (int i = 0; i < n; ++i) {
        //         expect(h_xs[i] == h_ref[i]);
        //     }
        // };

        CUDA_CHECK(cudaFree(d_vec));
    };

    return 0;
}
