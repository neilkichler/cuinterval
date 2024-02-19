
#include <cuinterval/cuinterval.h>

#include "tests.h"
#include "test_ops.cuh"

#include <stdio.h>

template<typename T>
void tests_libieeep1788_rec_bool() {
    using namespace boost::ut;

    using I = interval<T>;
    using B = bool;

    I empty         = ::empty<T>();
    I entire        = ::entire<T>();
    T infinity = std::numeric_limits<T>::infinity();
    T NaN = ::nan("");

    const int n = 15; // count of largest test array
    const int n_bytes   = n * sizeof(I);
    const int blockSize = 256;
    [[maybe_unused]] const int numBlocks = (n + blockSize - 1) / blockSize;

    I *d_xs, *d_ys, *d_zs, *d_res_;

    CUDA_CHECK(cudaMalloc(&d_xs, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_ys, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_zs, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_res_, n_bytes));

    "minimal_is_common_interval_isCommonInterval"_test = [&] {
        constexpr int n = 12;
        std::array<I, n> h_xs {{
            {-0.0,-0.0},
            {-0.0,0.0},
            {-0x1.FFFFFFFFFFFFFp1023,0x1.FFFFFFFFFFFFFp1023},
            {-27.0,-27.0},
            {-27.0,0.0},
            {-infinity,0.0},
            {0.0,-0.0},
            {0.0,0.0},
            {0.0,infinity},
            {5.0,12.4},
            empty,
            entire,
        }};

        std::array<B, n> h_res{};
        B *d_res = (B *)d_res_;
        int n_result_bytes = n * sizeof(B);
        std::array<B, n> h_ref {{
            true,
            true,
            true,
            true,
            true,
            false,
            true,
            true,
            false,
            true,
            false,
            false,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_isCommonInterval<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<B, n>(h_res, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "minimal_is_singleton_isSingleton"_test = [&] {
        constexpr int n = 15;
        std::array<I, n> h_xs {{
            {-0.0,-0.0},
            {-0.0,0.0},
            {-1.0,-0.5},
            {-1.0,0.0},
            {-1.0,infinity},
            {-2.0,-2.0},
            {-27.0,-27.0},
            {-infinity,-0x1.FFFFFFFFFFFFFp1023},
            {0.0,-0.0},
            {0.0,0.0},
            {1.0,2.0},
            {12.0,12.0},
            {17.1,17.1},
            empty,
            entire,
        }};

        std::array<B, n> h_res{};
        B *d_res = (B *)d_res_;
        int n_result_bytes = n * sizeof(B);
        std::array<B, n> h_ref {{
            true,
            true,
            false,
            false,
            false,
            true,
            true,
            false,
            true,
            true,
            false,
            true,
            true,
            false,
            false,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_isSingleton<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<B, n>(h_res, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };


    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_ys));
    CUDA_CHECK(cudaFree(d_zs));
    CUDA_CHECK(cudaFree(d_res_));
}
