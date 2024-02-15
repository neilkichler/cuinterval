
#include <cuinterval/cuinterval.h>

#include "tests.h"
#include "test_ops.cuh"

#include <stdio.h>

template<typename T>
void tests_libieeep1788_num() {
    using namespace boost::ut;

    using I = interval<T>;

    I empty         = ::empty<T>();
    I entire        = ::entire<T>();
    T infinity = std::numeric_limits<T>::infinity();
    T NaN = ::nan("");

    const int n = 11; // count of largest test array
    const int n_bytes   = n * sizeof(I);
    const int blockSize = 256;
    [[maybe_unused]] const int numBlocks = (n + blockSize - 1) / blockSize;

    I *d_xs, *d_ys, *d_zs, *d_res_;

    CUDA_CHECK(cudaMalloc(&d_xs, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_ys, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_zs, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_res_, n_bytes));

    "minimal_wid_wid"_test = [&] {
        constexpr int n = 8;
        std::array<I, n> h_xs {{
            {-infinity,+infinity},
            {-infinity,2.0},
            {0X1P+0,0X1.0000000000001P+0},
            {0X1P-1022,0X1.0000000000001P-1022},
            {1.0,2.0},
            {1.0,infinity},
            {2.0,2.0},
            empty,
        }};

        std::array<T, n> h_res{};
        T *d_res = (T *)d_res_;
        int n_result_bytes = n * sizeof(T);
        std::array<T, n> h_ref {{
            infinity,
            infinity,
            0X1P-52,
            0X0.0000000000001P-1022,
            1.0,
            infinity,
            0.0,
            NaN,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_wid<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<T, n>(h_res, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("r = [%a]\n", h_ref[fail_id]);
        }
    };

    "minimal_mag_mag"_test = [&] {
        constexpr int n = 8;
        std::array<I, n> h_xs {{
            {-0.0,-0.0},
            {-0.0,0.0},
            {-4.0,2.0},
            {-infinity,+infinity},
            {-infinity,2.0},
            {1.0,2.0},
            {1.0,infinity},
            empty,
        }};

        std::array<T, n> h_res{};
        T *d_res = (T *)d_res_;
        int n_result_bytes = n * sizeof(T);
        std::array<T, n> h_ref {{
            0.0,
            0.0,
            4.0,
            infinity,
            infinity,
            2.0,
            infinity,
            NaN,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_mag<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<T, n>(h_res, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("r = [%a]\n", h_ref[fail_id]);
        }
    };

    "minimal_mig_mig"_test = [&] {
        constexpr int n = 11;
        std::array<I, n> h_xs {{
            {-0.0,-0.0},
            {-0.0,0.0},
            {-1.0,infinity},
            {-4.0,-2.0},
            {-4.0,2.0},
            {-infinity,+infinity},
            {-infinity,-2.0},
            {-infinity,2.0},
            {1.0,2.0},
            {1.0,infinity},
            empty,
        }};

        std::array<T, n> h_res{};
        T *d_res = (T *)d_res_;
        int n_result_bytes = n * sizeof(T);
        std::array<T, n> h_ref {{
            0.0,
            0.0,
            0.0,
            2.0,
            0.0,
            0.0,
            2.0,
            0.0,
            1.0,
            1.0,
            NaN,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_mig<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<T, n>(h_res, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("r = [%a]\n", h_ref[fail_id]);
        }
    };


    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_ys));
    CUDA_CHECK(cudaFree(d_zs));
    CUDA_CHECK(cudaFree(d_res_));
}
