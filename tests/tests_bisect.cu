#include <cuinterval/cuinterval.h>

#include "tests.h"
#include "test_ops.cuh"

template<typename T>
void tests_bisect() {
    using namespace boost::ut;

    using I = interval<T>;
    using B = bool;
    using N = int;

    T infinity = std::numeric_limits<T>::infinity();
    I empty    = { infinity, -infinity };
    I entire   = { -infinity, infinity };
    T NaN = ::nan("");

    const int n = 8; // count of largest test array
    const int n_bytes   = n * sizeof(I);
    const int blockSize = 256;
    [[maybe_unused]] const int numBlocks = (n + blockSize - 1) / blockSize;

    I *d_xs_, *d_ys_, *d_zs_, *d_res_;

    CUDA_CHECK(cudaMalloc(&d_xs_, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_ys_, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_zs_, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_res_, 2*n_bytes));

    "bisection"_test = [&] {
        constexpr int n = 8;
        std::array<I, n> h_xs {{
            empty,
            entire,
            entire,
            entire,
            {0.0, 2.0},
            {1.0, 1.0},
            {0.0, 1.0},
            {0.0, 1.0},
        }};

        std::array<T, n> h_ys {{
            0.5,
            0.5,
            0.25,
            0.75,
            0.5,
            0.5,
            0.5,
            0.25,
        }};

        std::array<split<T>, n> h_res{};
        split<T> *d_res = (split<T> *)d_res_;
        I *d_xs = (I *)d_xs_;
        T *d_ys = (T *)d_ys_;
        int n_result_bytes = n * sizeof(*d_res);
        std::array<split<T>, n> h_ref {{
            {empty, empty},
            {{entire.lb,0.0}, {0.0,entire.ub}},
            {{entire.lb,-0x1.fffffffffffffp+1023}, {-0x1.fffffffffffffp+1023,entire.ub}},
            {{entire.lb,0x1.fffffffffffffp+1023}, {0x1.fffffffffffffp+1023,entire.ub}},
            {{0.0,1.0}, {1.0,2.0}},
            {{1.0,1.0}, empty},
            {{0.0,0.5}, {0.5,1.0}},
            {{0.0,0.25}, {0.25,1.0}},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_bisect<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        check_all_equal<split<T>, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    CUDA_CHECK(cudaFree(d_xs_));
    CUDA_CHECK(cudaFree(d_ys_));
    CUDA_CHECK(cudaFree(d_zs_));
    CUDA_CHECK(cudaFree(d_res_));
}
