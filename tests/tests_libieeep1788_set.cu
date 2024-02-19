
#include <cuinterval/cuinterval.h>

#include "tests.h"
#include "test_ops.cuh"

#include <stdio.h>

template<typename T>
void tests_libieeep1788_set() {
    using namespace boost::ut;

    using I = interval<T>;
    using B = bool;

    I empty         = ::empty<T>();
    I entire        = ::entire<T>();
    T infinity = std::numeric_limits<T>::infinity();
    T NaN = ::nan("");

    const int n = 5; // count of largest test array
    const int n_bytes   = n * sizeof(I);
    const int blockSize = 256;
    [[maybe_unused]] const int numBlocks = (n + blockSize - 1) / blockSize;

    I *d_xs, *d_ys, *d_zs, *d_res_;

    CUDA_CHECK(cudaMalloc(&d_xs, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_ys, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_zs, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_res_, n_bytes));

    "minimal_intersection_intersection"_test = [&] {
        constexpr int n = 5;
        std::array<I, n> h_xs {{
            {1.0,3.0},
            {1.0,3.0},
            {1.0,3.0},
            {1.0,3.0},
            entire,
        }};

        std::array<I, n> h_ys {{
            {2.1,4.0},
            {3.0,4.0},
            empty,
            entire,
            empty,
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {2.1,3.0},
            {3.0,3.0},
            empty,
            {1.0,3.0},
            empty,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_intersection<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<I, n>(h_res, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "minimal_convex_hull_convexHull"_test = [&] {
        constexpr int n = 5;
        std::array<I, n> h_xs {{
            {1.0,1.0},
            {1.0,3.0},
            {1.0,3.0},
            {1.0,3.0},
            empty,
        }};

        std::array<I, n> h_ys {{
            {2.1,4.0},
            {2.1,4.0},
            empty,
            entire,
            empty,
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {1.0,4.0},
            {1.0,4.0},
            {1.0,3.0},
            entire,
            empty,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_convexHull<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<I, n>(h_res, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };


    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_ys));
    CUDA_CHECK(cudaFree(d_zs));
    CUDA_CHECK(cudaFree(d_res_));
}
