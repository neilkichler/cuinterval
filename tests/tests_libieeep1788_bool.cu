
#include <cuinterval/cuinterval.h>

#include "tests.h"
#include "test_ops.cuh"

#include <stdio.h>

template<typename T>
void tests_libieeep1788_bool() {
    using namespace boost::ut;

    using I = interval<T>;
    using B = bool;

    I empty         = ::empty<T>();
    I entire        = ::entire<T>();
    T infinity = std::numeric_limits<T>::infinity();
    T NaN = ::nan("");

    const int n = 27; // count of largest test array
    const int n_bytes   = n * sizeof(I);
    const int blockSize = 256;
    [[maybe_unused]] const int numBlocks = (n + blockSize - 1) / blockSize;

    I *d_xs, *d_ys, *d_zs, *d_res_;

    CUDA_CHECK(cudaMalloc(&d_xs, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_ys, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_zs, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_res_, n_bytes));

    "minimal_is_empty_isEmpty"_test = [&] {
        constexpr int n = 14;
        std::array<I, n> h_xs {{
            {-0.0,-0.0},
            {-0.0,0.0},
            {-0.0,infinity},
            {-1.0,2.0},
            {-3.0,-2.0},
            {-infinity,+infinity},
            {-infinity,-0.0},
            {-infinity,0.0},
            {-infinity,2.0},
            {0.0,-0.0},
            {0.0,0.0},
            {0.0,infinity},
            {1.0,2.0},
            empty,
        }};

        std::array<B, n> h_res{};
        B *d_res = (B *)d_res_;
        int n_result_bytes = n * sizeof(B);
        std::array<B, n> h_ref {{
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            true,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_isEmpty<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<B, n>(h_res, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "minimal_is_entire_isEntire"_test = [&] {
        constexpr int n = 14;
        std::array<I, n> h_xs {{
            {-0.0,-0.0},
            {-0.0,0.0},
            {-0.0,infinity},
            {-1.0,2.0},
            {-3.0,-2.0},
            {-infinity,+infinity},
            {-infinity,-0.0},
            {-infinity,0.0},
            {-infinity,2.0},
            {0.0,-0.0},
            {0.0,0.0},
            {0.0,infinity},
            {1.0,2.0},
            empty,
        }};

        std::array<B, n> h_res{};
        B *d_res = (B *)d_res_;
        int n_result_bytes = n * sizeof(B);
        std::array<B, n> h_ref {{
            false,
            false,
            false,
            false,
            false,
            true,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_isEntire<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<B, n>(h_res, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "minimal_equal_equal"_test = [&] {
        constexpr int n = 15;
        std::array<I, n> h_xs {{
            {-0.0,-0.0},
            {-0.0,0.0},
            {-0.0,2.0},
            {-2.0,0.0},
            {-infinity,+infinity},
            {-infinity,2.0},
            {-infinity,2.4},
            {0.0,-0.0},
            {1.0,2.0},
            {1.0,2.1},
            {1.0,2.4},
            {1.0,2.4},
            {1.0,infinity},
            empty,
            empty,
        }};

        std::array<I, n> h_ys {{
            {0.0,0.0},
            {0.0,0.0},
            {0.0,2.0},
            {-2.0,0.0},
            {-infinity,+infinity},
            {-infinity,2.0},
            {-infinity,2.0},
            {0.0,0.0},
            {1.0,2.0},
            {1.0,2.0},
            {-infinity,+infinity},
            {1.0,infinity},
            {1.0,infinity},
            {1.0,2.0},
            empty,
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
            true,
            false,
            true,
            true,
            false,
            false,
            false,
            true,
            false,
            true,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_equal<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<B, n>(h_res, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "minimal_subset_subset"_test = [&] {
        constexpr int n = 27;
        std::array<I, n> h_xs {{
            {-0.0,-0.0},
            {-0.0,0.0},
            {-0.0,0.0},
            {-0.0,4.0},
            {-0.0,4.0},
            {-0.1,-0.1},
            {-0.1,1.0},
            {-0.1,1.0},
            {-infinity,+infinity},
            {-infinity,+infinity},
            {0.0,-0.0},
            {0.0,-0.0},
            {0.0,0.0},
            {0.0,4.0},
            {0.0,4.0},
            {0.1,0.2},
            {0.1,0.2},
            {1.0,2.0},
            {1.0,2.0},
            {1.0,2.0},
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
        }};

        std::array<I, n> h_ys {{
            {0.0,0.0},
            {0.0,-0.0},
            {0.0,0.0},
            {-infinity,+infinity},
            empty,
            {-4.0,3.4},
            {-infinity,+infinity},
            empty,
            {-infinity,+infinity},
            empty,
            {-0.0,0.0},
            {0.0,0.0},
            {-0.0,-0.0},
            {-infinity,+infinity},
            empty,
            {-0.0,4.0},
            {0.0,4.0},
            {-0.0,4.0},
            {0.0,4.0},
            {1.0,2.0},
            {-0.0,4.0},
            {-0.1,-0.0},
            {-0.1,0.0},
            {-0.1,1.0},
            {-infinity,+infinity},
            {0.0,4.0},
            empty,
        }};

        std::array<B, n> h_res{};
        B *d_res = (B *)d_res_;
        int n_result_bytes = n * sizeof(B);
        std::array<B, n> h_ref {{
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
            true,
            true,
            true,
            true,
            false,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            true,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_subset<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<B, n>(h_res, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "minimal_less_less"_test = [&] {
        constexpr int n = 26;
        std::array<I, n> h_xs {{
            {-0.0,-0.0},
            {-0.0,0.0},
            {-0.0,0.0},
            {-0.0,2.0},
            {-0.0,2.0},
            {-2.0,-1.0},
            {-3.0,-1.5},
            {-infinity,+infinity},
            {-infinity,+infinity},
            {-infinity,+infinity},
            {-infinity,+infinity},
            {0.0,-0.0},
            {0.0,-0.0},
            {0.0,0.0},
            {0.0,2.0},
            {0.0,2.0},
            {0.0,2.0},
            {0.0,2.0},
            {1.0,2.0},
            {1.0,2.0},
            {1.0,2.0},
            {1.0,2.0},
            {1.0,3.5},
            {1.0,4.0},
            empty,
            empty,
        }};

        std::array<I, n> h_ys {{
            {0.0,0.0},
            {0.0,-0.0},
            {0.0,0.0},
            {-infinity,+infinity},
            {1.0,2.0},
            {-2.0,-1.0},
            {-2.0,-1.0},
            {-0.0,2.0},
            {-infinity,+infinity},
            {0.0,2.0},
            {1.0,2.0},
            {-0.0,0.0},
            {0.0,0.0},
            {-0.0,-0.0},
            {-0.0,2.0},
            {-infinity,+infinity},
            {0.0,2.0},
            {1.0,2.0},
            {-infinity,+infinity},
            {1.0,2.0},
            {3.0,4.0},
            empty,
            {3.0,4.0},
            {3.0,4.0},
            {1.0,2.0},
            empty,
        }};

        std::array<B, n> h_res{};
        B *d_res = (B *)d_res_;
        int n_result_bytes = n * sizeof(B);
        std::array<B, n> h_ref {{
            true,
            true,
            true,
            false,
            true,
            true,
            true,
            false,
            true,
            false,
            false,
            true,
            true,
            true,
            true,
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
            true,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_less<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<B, n>(h_res, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "minimal_precedes_precedes"_test = [&] {
        constexpr int n = 21;
        std::array<I, n> h_xs {{
            {-0.0,-0.0},
            {-0.0,0.0},
            {-0.0,0.0},
            {-0.0,2.0},
            {-3.0,-0.1},
            {-3.0,-1.0},
            {-3.0,-1.0},
            {-infinity,+infinity},
            {-infinity,+infinity},
            {0.0,-0.0},
            {0.0,-0.0},
            {0.0,0.0},
            {0.0,2.0},
            {1.0,2.0},
            {1.0,2.0},
            {1.0,3.0},
            {1.0,3.5},
            {1.0,4.0},
            {3.0,4.0},
            empty,
            empty,
        }};

        std::array<I, n> h_ys {{
            {0.0,0.0},
            {0.0,-0.0},
            {0.0,0.0},
            {-infinity,+infinity},
            {-1.0,0.0},
            {-1.0,-0.0},
            {-1.0,0.0},
            {-infinity,+infinity},
            {1.0,2.0},
            {-0.0,0.0},
            {0.0,0.0},
            {-0.0,-0.0},
            {-infinity,+infinity},
            {-infinity,+infinity},
            {3.0,4.0},
            {3.0,4.0},
            {3.0,4.0},
            {3.0,4.0},
            empty,
            {3.0,4.0},
            empty,
        }};

        std::array<B, n> h_res{};
        B *d_res = (B *)d_res_;
        int n_result_bytes = n * sizeof(B);
        std::array<B, n> h_ref {{
            true,
            true,
            true,
            false,
            false,
            true,
            true,
            false,
            false,
            true,
            true,
            true,
            false,
            false,
            true,
            true,
            false,
            false,
            true,
            true,
            true,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_precedes<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<B, n>(h_res, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "minimal_interior_interior"_test = [&] {
        constexpr int n = 16;
        std::array<I, n> h_xs {{
            {-0.0,-0.0},
            {-1.0,-1.0},
            {-2.0,2.0},
            {-infinity,+infinity},
            {-infinity,+infinity},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,4.0},
            {0.0,4.0},
            {0.0,4.0},
            {0.0,4.4},
            {1.0,2.0},
            {2.0,2.0},
            empty,
            empty,
            empty,
        }};

        std::array<I, n> h_ys {{
            {-2.0,4.0},
            {0.0,4.0},
            {-2.0,4.0},
            {-infinity,+infinity},
            {0.0,4.0},
            {-0.0,-0.0},
            {-2.0,4.0},
            {-infinity,+infinity},
            {0.0,4.0},
            empty,
            {0.0,4.0},
            {0.0,4.0},
            {-2.0,-1.0},
            {-infinity,+infinity},
            {0.0,4.0},
            empty,
        }};

        std::array<B, n> h_res{};
        B *d_res = (B *)d_res_;
        int n_result_bytes = n * sizeof(B);
        std::array<B, n> h_ref {{
            true,
            false,
            false,
            true,
            false,
            false,
            true,
            true,
            false,
            false,
            false,
            true,
            false,
            true,
            true,
            true,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_interior<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<B, n>(h_res, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "minimal_strictly_less_strictLess"_test = [&] {
        constexpr int n = 14;
        std::array<I, n> h_xs {{
            {-0.0,4.0},
            {-2.0,-1.0},
            {-3.0,-1.5},
            {-infinity,+infinity},
            {-infinity,+infinity},
            {0.0,4.0},
            {1.0,2.0},
            {1.0,2.0},
            {1.0,2.0},
            {1.0,2.0},
            {1.0,3.5},
            {1.0,4.0},
            empty,
            empty,
        }};

        std::array<I, n> h_ys {{
            {0.0,4.0},
            {-2.0,-1.0},
            {-2.0,-1.0},
            {-infinity,+infinity},
            {1.0,2.0},
            {0.0,4.0},
            {-infinity,+infinity},
            {1.0,2.0},
            {3.0,4.0},
            empty,
            {3.0,4.0},
            {3.0,4.0},
            {1.0,2.0},
            empty,
        }};

        std::array<B, n> h_res{};
        B *d_res = (B *)d_res_;
        int n_result_bytes = n * sizeof(B);
        std::array<B, n> h_ref {{
            false,
            false,
            true,
            true,
            false,
            false,
            false,
            false,
            true,
            false,
            true,
            false,
            false,
            true,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_strictLess<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<B, n>(h_res, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "minimal_strictly_precedes_strictPrecedes"_test = [&] {
        constexpr int n = 14;
        std::array<I, n> h_xs {{
            {-3.0,-0.0},
            {-3.0,-0.1},
            {-3.0,-1.0},
            {-3.0,0.0},
            {-infinity,+infinity},
            {-infinity,+infinity},
            {1.0,2.0},
            {1.0,2.0},
            {1.0,3.0},
            {1.0,3.5},
            {1.0,4.0},
            {3.0,4.0},
            empty,
            empty,
        }};

        std::array<I, n> h_ys {{
            {0.0,1.0},
            {-1.0,0.0},
            {-1.0,0.0},
            {-0.0,1.0},
            {-infinity,+infinity},
            {1.0,2.0},
            {-infinity,+infinity},
            {3.0,4.0},
            {3.0,4.0},
            {3.0,4.0},
            {3.0,4.0},
            empty,
            {3.0,4.0},
            empty,
        }};

        std::array<B, n> h_res{};
        B *d_res = (B *)d_res_;
        int n_result_bytes = n * sizeof(B);
        std::array<B, n> h_ref {{
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            true,
            false,
            false,
            false,
            true,
            true,
            true,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_strictPrecedes<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<B, n>(h_res, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "minimal_disjoint_disjoint"_test = [&] {
        constexpr int n = 10;
        std::array<I, n> h_xs {{
            {-infinity,+infinity},
            {-infinity,+infinity},
            {0.0,-0.0},
            {0.0,0.0},
            {3.0,4.0},
            {3.0,4.0},
            {3.0,4.0},
            {3.0,4.0},
            empty,
            empty,
        }};

        std::array<I, n> h_ys {{
            {-infinity,+infinity},
            {1.0,7.0},
            {-0.0,0.0},
            {-0.0,-0.0},
            {-infinity,+infinity},
            {1.0,2.0},
            {1.0,7.0},
            empty,
            {3.0,4.0},
            empty,
        }};

        std::array<B, n> h_res{};
        B *d_res = (B *)d_res_;
        int n_result_bytes = n * sizeof(B);
        std::array<B, n> h_ref {{
            false,
            false,
            false,
            false,
            false,
            true,
            false,
            true,
            true,
            true,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_disjoint<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<B, n>(h_res, h_ref);
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
