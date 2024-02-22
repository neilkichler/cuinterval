
#include <cuinterval/cuinterval.h>

#include "tests.h"
#include "test_ops.cuh"

#include <stdio.h>

template<typename T>
void tests_c_xsc() {
    using namespace boost::ut;

    using I = interval<T>;
    using B = bool;

    I empty         = ::empty<T>();
    I entire        = ::entire<T>();
    T infinity = std::numeric_limits<T>::infinity();
    T NaN = ::nan("");

    const int n = 16; // count of largest test array
    const int n_bytes   = n * sizeof(I);
    const int blockSize = 256;
    [[maybe_unused]] const int numBlocks = (n + blockSize - 1) / blockSize;

    I *d_xs_, *d_ys_, *d_zs_, *d_res_;

    CUDA_CHECK(cudaMalloc(&d_xs_, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_ys_, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_zs_, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_res_, n_bytes));

    "cxsc.intervaladdsub_add"_test = [&] {
        constexpr int n = 2;
        std::array<I, n> h_xs {{
            {10.0,20.0},
            {13.0,17.0},
        }};

        std::array<I, n> h_ys {{
            {13.0,17.0},
            {10.0,20.0},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {23.0,37.0},
            {23.0,37.0},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_add<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "cxsc.intervaladdsub_neg"_test = [&] {
        constexpr int n = 1;
        std::array<I, n> h_xs {{
            {10.0,20.0},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-20.0,-10.0},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_neg<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "cxsc.intervaladdsub_pos"_test = [&] {
        constexpr int n = 1;
        std::array<I, n> h_xs {{
            {10.0,20.0},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {10.0,20.0},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_pos<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "cxsc.intervaladdsub_sub"_test = [&] {
        constexpr int n = 2;
        std::array<I, n> h_xs {{
            {10.0,20.0},
            {13.0,16.0},
        }};

        std::array<I, n> h_ys {{
            {13.0,16.0},
            {10.0,20.0},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-6.0,7.0},
            {-7.0,6.0},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_sub<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "cxsc.intervalmuldiv_div"_test = [&] {
        constexpr int n = 16;
        std::array<I, n> h_xs {{
            {-1.0,2.0},
            {-1.0,2.0},
            {-1.0,2.0},
            {-1.0,2.0},
            {-2.0,-1.0},
            {-2.0,-1.0},
            {-2.0,-1.0},
            {-2.0,-1.0},
            {-2.0,1.0},
            {-2.0,1.0},
            {-2.0,1.0},
            {-2.0,1.0},
            {1.0,2.0},
            {1.0,2.0},
            {1.0,2.0},
            {1.0,2.0},
        }};

        std::array<I, n> h_ys {{
            {-4.0,8.0},
            {-8.0,-4.0},
            {-8.0,4.0},
            {4.0,8.0},
            {-4.0,8.0},
            {-8.0,-4.0},
            {-8.0,4.0},
            {4.0,8.0},
            {-4.0,8.0},
            {-8.0,-4.0},
            {-8.0,4.0},
            {4.0,8.0},
            {-4.0,8.0},
            {-8.0,-4.0},
            {-8.0,4.0},
            {4.0,8.0},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            entire,
            {-0.5,0.25},
            entire,
            {-0.25,0.5},
            entire,
            {0.125,0.5},
            entire,
            {-0.5,-0.125},
            entire,
            {-0.25,0.5},
            entire,
            {-0.5,0.25},
            entire,
            {-0.5,-0.125},
            entire,
            {0.125,0.5},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_div<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "cxsc.intervalmuldiv_mul"_test = [&] {
        constexpr int n = 15;
        std::array<I, n> h_xs {{
            {-1.0,2.0},
            {-1.0,2.0},
            {-1.0,2.0},
            {-1.0,2.0},
            {-2.0,-1.0},
            {-2.0,-1.0},
            {-2.0,-1.0},
            {-2.0,-1.0},
            {-2.0,1.0},
            {-2.0,1.0},
            {-2.0,1.0},
            {1.0,2.0},
            {1.0,2.0},
            {1.0,2.0},
            {1.0,2.0},
        }};

        std::array<I, n> h_ys {{
            {-3.0,4.0},
            {-4.0,-3.0},
            {-4.0,3.0},
            {3.0,4.0},
            {-3.0,4.0},
            {-4.0,-3.0},
            {-4.0,3.0},
            {3.0,4.0},
            {-3.0,4.0},
            {-4.0,3.0},
            {3.0,4.0},
            {-3.0,4.0},
            {-4.0,-3.0},
            {-4.0,3.0},
            {3.0,4.0},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-6.0,8.0},
            {-8.0,4.0},
            {-8.0,6.0},
            {-4.0,8.0},
            {-8.0,6.0},
            {3.0,8.0},
            {-6.0,8.0},
            {-8.0,-3.0},
            {-8.0,6.0},
            {-6.0,8.0},
            {-8.0,4.0},
            {-6.0,8.0},
            {-8.0,-3.0},
            {-8.0,6.0},
            {3.0,8.0},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_mul<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "cxsc.intervalsetop_convexHull"_test = [&] {
        constexpr int n = 12;
        std::array<I, n> h_xs {{
            {-1.0,1.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-4.0,-1.0},
            {-4.0,-3.0},
            {-4.0,4.0},
            {1.0,4.0},
            {3.0,4.0},
        }};

        std::array<I, n> h_ys {{
            {-2.0,2.0},
            {-1.0,1.0},
            {-4.0,-1.0},
            {-4.0,-3.0},
            {-4.0,4.0},
            {1.0,4.0},
            {3.0,4.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-2.0,2.0},
            {-2.0,2.0},
            {-4.0,2.0},
            {-4.0,2.0},
            {-4.0,4.0},
            {-2.0,4.0},
            {-2.0,4.0},
            {-4.0,2.0},
            {-4.0,2.0},
            {-4.0,4.0},
            {-2.0,4.0},
            {-2.0,4.0},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_convexHull<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "cxsc.intervalsetop_intersection"_test = [&] {
        constexpr int n = 12;
        std::array<I, n> h_xs {{
            {-1.0,1.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-4.0,-1.0},
            {-4.0,-3.0},
            {-4.0,4.0},
            {1.0,4.0},
            {3.0,4.0},
        }};

        std::array<I, n> h_ys {{
            {-2.0,2.0},
            {-1.0,1.0},
            {-4.0,-1.0},
            {-4.0,-3.0},
            {-4.0,4.0},
            {1.0,4.0},
            {3.0,4.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-1.0,1.0},
            {-1.0,1.0},
            {-2.0,-1.0},
            empty,
            {-2.0,2.0},
            {1.0,2.0},
            empty,
            {-2.0,-1.0},
            empty,
            {-2.0,2.0},
            {1.0,2.0},
            empty,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_intersection<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "cxsc.intervalmixsetop_convexHull"_test = [&] {
        constexpr int n = 6;
        std::array<I, n> h_xs {{
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-4.0,-4.0},
            {1.0,1.0},
            {4.0,4.0},
        }};

        std::array<I, n> h_ys {{
            {-4.0,-4.0},
            {1.0,1.0},
            {4.0,4.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-4.0,2.0},
            {-2.0,2.0},
            {-2.0,4.0},
            {-4.0,2.0},
            {-2.0,2.0},
            {-2.0,4.0},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_convexHull<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "cxsc.intervalmixsetop_intersection"_test = [&] {
        constexpr int n = 6;
        std::array<I, n> h_xs {{
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-4.0,-4.0},
            {1.0,1.0},
            {4.0,4.0},
        }};

        std::array<I, n> h_ys {{
            {-4.0,-4.0},
            {1.0,1.0},
            {4.0,4.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            empty,
            {1.0,1.0},
            empty,
            empty,
            {1.0,1.0},
            empty,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_intersection<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "cxsc.scalarmixsetop_convexHull"_test = [&] {
        constexpr int n = 6;
        std::array<I, n> h_xs {{
            {-2.0,-2.0},
            {-2.0,-2.0},
            {-2.0,-2.0},
            {-2.0,-2.0},
            {-4.0,-4.0},
            {2.0,2.0},
        }};

        std::array<I, n> h_ys {{
            {-2.0,-2.0},
            {-2.0,-2.0},
            {-4.0,-4.0},
            {2.0,2.0},
            {-2.0,-2.0},
            {-2.0,-2.0},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-2.0,-2.0},
            {-2.0,-2.0},
            {-4.0,-2.0},
            {-2.0,2.0},
            {-4.0,-2.0},
            {-2.0,2.0},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_convexHull<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "cxsc.intervalsetcompop_equal"_test = [&] {
        constexpr int n = 7;
        std::array<I, n> h_xs {{
            {-1.0,2.0},
            {-2.0,1.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
        }};

        std::array<I, n> h_ys {{
            {-1.0,2.0},
            {-3.0,2.0},
            {-1.0,1.0},
            {-1.0,2.0},
            {-2.0,1.0},
            {-2.0,3.0},
            {-3.0,2.0},
        }};

        std::array<B, n> h_res{};
        B *d_res = (B *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(B);
        std::array<B, n> h_ref {{
            true,
            false,
            false,
            false,
            false,
            false,
            false,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_equal<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<B, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "cxsc.intervalsetcompop_interior"_test = [&] {
        constexpr int n = 14;
        std::array<I, n> h_xs {{
            {-1.0,1.0},
            {-1.0,2.0},
            {-1.0,2.0},
            {-1.0,2.0},
            {-2.0,1.0},
            {-2.0,1.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,3.0},
            {-3.0,2.0},
            {-3.0,2.0},
        }};

        std::array<I, n> h_ys {{
            {-2.0,2.0},
            {-1.0,2.0},
            {-1.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-3.0,2.0},
            {-1.0,1.0},
            {-1.0,2.0},
            {-2.0,1.0},
            {-2.0,3.0},
            {-3.0,2.0},
            {-2.0,2.0},
            {-2.0,1.0},
            {-2.0,2.0},
        }};

        std::array<B, n> h_res{};
        B *d_res = (B *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(B);
        std::array<B, n> h_ref {{
            true,
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
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_interior<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<B, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "cxsc.intervalsetcompop_subset"_test = [&] {
        constexpr int n = 13;
        std::array<I, n> h_xs {{
            {-1.0,1.0},
            {-1.0,2.0},
            {-1.0,2.0},
            {-2.0,1.0},
            {-2.0,1.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,3.0},
            {-3.0,2.0},
            {-3.0,2.0},
        }};

        std::array<I, n> h_ys {{
            {-2.0,2.0},
            {-1.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-3.0,2.0},
            {-1.0,1.0},
            {-1.0,2.0},
            {-2.0,1.0},
            {-2.0,3.0},
            {-3.0,2.0},
            {-2.0,2.0},
            {-2.0,1.0},
            {-2.0,2.0},
        }};

        std::array<B, n> h_res{};
        B *d_res = (B *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(B);
        std::array<B, n> h_ref {{
            true,
            true,
            true,
            true,
            true,
            false,
            false,
            false,
            true,
            true,
            false,
            false,
            false,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_subset<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<B, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "cxsc.intervalscalarsetcompop_equal"_test = [&] {
        constexpr int n = 7;
        std::array<I, n> h_xs {{
            {-1.0,-1.0},
            {-1.0,-1.0},
            {-1.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
        }};

        std::array<I, n> h_ys {{
            {-1.0,-1.0},
            {1.0,1.0},
            {-2.0,-2.0},
            {-2.0,-2.0},
            {0.0,0.0},
            {2.0,2.0},
            {3.0,3.0},
        }};

        std::array<B, n> h_res{};
        B *d_res = (B *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(B);
        std::array<B, n> h_ref {{
            true,
            false,
            false,
            false,
            false,
            false,
            false,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_equal<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<B, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "cxsc.intervalscalarsetcompop_interior"_test = [&] {
        constexpr int n = 14;
        std::array<I, n> h_xs {{
            {-1.0,-1.0},
            {-1.0,-1.0},
            {-1.0,-1.0},
            {-1.0,2.0},
            {-2.0,-2.0},
            {-2.0,-2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {0.0,0.0},
            {1.0,1.0},
            {2.0,2.0},
            {3.0,3.0},
        }};

        std::array<I, n> h_ys {{
            {-1.0,-1.0},
            {-1.0,-1.0},
            {1.0,1.0},
            {-2.0,-2.0},
            {-1.0,2.0},
            {-2.0,2.0},
            {-2.0,-2.0},
            {0.0,0.0},
            {2.0,2.0},
            {3.0,3.0},
            {-2.0,2.0},
            {-1.0,-1.0},
            {-2.0,2.0},
            {-2.0,2.0},
        }};

        std::array<B, n> h_res{};
        B *d_res = (B *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
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
            true,
            false,
            false,
            false,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_interior<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<B, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "cxsc.intervalscalarsetcompop_subset"_test = [&] {
        constexpr int n = 14;
        std::array<I, n> h_xs {{
            {-1.0,-1.0},
            {-1.0,-1.0},
            {-1.0,-1.0},
            {-1.0,2.0},
            {-2.0,-2.0},
            {-2.0,-2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {0.0,0.0},
            {1.0,1.0},
            {2.0,2.0},
            {3.0,3.0},
        }};

        std::array<I, n> h_ys {{
            {-1.0,-1.0},
            {-1.0,-1.0},
            {1.0,1.0},
            {-2.0,-2.0},
            {-1.0,2.0},
            {-2.0,2.0},
            {-2.0,-2.0},
            {0.0,0.0},
            {2.0,2.0},
            {3.0,3.0},
            {-2.0,2.0},
            {-1.0,-1.0},
            {-2.0,2.0},
            {-2.0,2.0},
        }};

        std::array<B, n> h_res{};
        B *d_res = (B *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(B);
        std::array<B, n> h_ref {{
            true,
            true,
            false,
            false,
            false,
            true,
            false,
            false,
            false,
            false,
            true,
            false,
            true,
            false,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_subset<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<B, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "cxsc.intervalstdfunc_sqr"_test = [&] {
        constexpr int n = 3;
        std::array<I, n> h_xs {{
            {-9.0,-9.0},
            {0.0,0.0},
            {11.0,11.0},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {81.0,81.0},
            {0.0,0.0},
            {121.0,121.0},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_sqr<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "cxsc.intervalstdfunc_sqrt"_test = [&] {
        constexpr int n = 3;
        std::array<I, n> h_xs {{
            {0.0,0.0},
            {121.0,121.0},
            {81.0,81.0},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {0.0,0.0},
            {11.0,11.0},
            {9.0,9.0},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_sqrt<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };


    CUDA_CHECK(cudaFree(d_xs_));
    CUDA_CHECK(cudaFree(d_ys_));
    CUDA_CHECK(cudaFree(d_zs_));
    CUDA_CHECK(cudaFree(d_res_));
}
