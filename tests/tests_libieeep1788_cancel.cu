
#include <cuinterval/cuinterval.h>

#include "tests.h"
#include "test_ops.cuh"

#include <stdio.h>

template<typename T>
void tests_libieeep1788_cancel() {
    using namespace boost::ut;

    using I = interval<T>;
    using B = bool;

    I empty         = ::empty<T>();
    I entire        = ::entire<T>();
    T infinity = std::numeric_limits<T>::infinity();
    T NaN = ::nan("");

    const int n = 63; // count of largest test array
    const int n_bytes   = n * sizeof(I);
    const int blockSize = 256;
    [[maybe_unused]] const int numBlocks = (n + blockSize - 1) / blockSize;

    I *d_xs, *d_ys, *d_zs, *d_res_;

    CUDA_CHECK(cudaMalloc(&d_xs, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_ys, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_zs, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_res_, n_bytes));

    "minimal_cancel_plus_cancelPlus"_test = [&] {
        constexpr int n = 58;
        std::array<I, n> h_xs {{
            {-0X1.999999999999AP-4,0X1.FFFFFFFFFFFFP+0},
            {-0X1.FFFFFFFFFFFFEP+1023,0x1.FFFFFFFFFFFFFp1023},
            {-0X1P+0,0X1.FFFFFFFFFFFFEP-53},
            {-0X1P+0,0X1.FFFFFFFFFFFFFP-53},
            {-0x1.FFFFFFFFFFFFFp1023,0X1.FFFFFFFFFFFFEP+1023},
            {-0x1.FFFFFFFFFFFFFp1023,0x1.FFFFFFFFFFFFFp1023},
            {-0x1.FFFFFFFFFFFFFp1023,0x1.FFFFFFFFFFFFFp1023},
            {-0x1.FFFFFFFFFFFFFp1023,0x1.FFFFFFFFFFFFFp1023},
            {-1.0,5.0},
            {-1.0,5.0},
            {-1.0,5.0},
            {-1.0,infinity},
            {-1.0,infinity},
            {-1.0,infinity},
            {-10.0,-1.0},
            {-10.0,5.0},
            {-10.0,5.0},
            {-10.0,5.0},
            {-10.0,5.0},
            {-10.0,5.0},
            {-10.0,5.1},
            {-10.1,5.0},
            {-10.1,5.1},
            {-5.0,-0.9},
            {-5.0,-1.0},
            {-5.0,-1.0},
            {-5.0,-1.0},
            {-5.0,-1.0},
            {-5.1,-0.0},
            {-5.1,-0.9},
            {-5.1,-1.0},
            {-infinity,-1.0},
            {-infinity,-1.0},
            {-infinity,-1.0},
            {0.0,5.0},
            {0.0,5.1},
            {0.9,5.0},
            {0.9,5.1},
            {0X1.FFFFFFFFFFFFP+0,0X1.FFFFFFFFFFFFP+0},
            {0x1.FFFFFFFFFFFFFp1023,0x1.FFFFFFFFFFFFFp1023},
            {1.0,5.0},
            {1.0,5.0},
            {1.0,5.0},
            {1.0,5.0},
            {1.0,5.0},
            {1.0,5.1},
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            entire,
            entire,
            entire,
            entire,
            entire,
        }};

        std::array<I, n> h_ys {{
            {-0X1.999999999999AP-4,0.01},
            {-0x1.FFFFFFFFFFFFFp1023,0x1.FFFFFFFFFFFFFp1023},
            {-0X1P+0,0X1.FFFFFFFFFFFFFP-53},
            {-0X1P+0,0X1.FFFFFFFFFFFFEP-53},
            {-0x1.FFFFFFFFFFFFFp1023,0x1.FFFFFFFFFFFFFp1023},
            {-0X1.FFFFFFFFFFFFEP+1023,0x1.FFFFFFFFFFFFFp1023},
            {-0x1.FFFFFFFFFFFFFp1023,0X1.FFFFFFFFFFFFEP+1023},
            {-0x1.FFFFFFFFFFFFFp1023,0x1.FFFFFFFFFFFFFp1023},
            {-infinity,1.0},
            {1.0,infinity},
            entire,
            {-5.0,1.0},
            empty,
            entire,
            empty,
            {-5.0,10.0},
            {-5.0,10.1},
            {-5.1,10.0},
            {-5.1,10.1},
            empty,
            {-5.0,10.0},
            {-5.0,10.0},
            {-5.0,10.0},
            {1.0,5.0},
            {0.9,5.0},
            {0.9,5.1},
            {1.0,5.0},
            {1.0,5.1},
            {0.0,5.0},
            {1.0,5.0},
            {1.0,5.0},
            {-5.0,1.0},
            empty,
            entire,
            {-5.0,-0.0},
            {-5.0,-0.0},
            {-5.0,-1.0},
            {-5.0,-1.0},
            {-0X1.999999999999AP-4,-0X1.999999999999AP-4},
            {0x1.FFFFFFFFFFFFFp1023,0x1.FFFFFFFFFFFFFp1023},
            {-5.0,-0.9},
            {-5.0,-1.0},
            {-5.1,-0.9},
            {-5.1,-1.0},
            empty,
            {-5.0,-1.0},
            {-5.0,-1.0},
            {-5.0,10.0},
            {-infinity,1.0},
            {1.0,10.0},
            {1.0,infinity},
            empty,
            entire,
            {-5.0,1.0},
            {-infinity,1.0},
            {1.0,infinity},
            empty,
            entire,
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-0X1.70A3D70A3D70BP-4,0X1.E666666666657P+0},
            entire,
            entire,
            {-0X1.FFFFFFFFFFFFFP-1,-0X1.FFFFFFFFFFFFEP-1},
            entire,
            {0.0,0X1P+971},
            {-0X1P+971,0.0},
            {0.0,0.0},
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            {0.0,0.0},
            entire,
            entire,
            entire,
            entire,
            {0.0,0X1.999999999998P-4},
            {-0X1.999999999998P-4,0.0},
            {-0X1.999999999998P-4,0X1.999999999998P-4},
            {0.0,0X1.9999999999998P-4},
            entire,
            entire,
            {0.0,0.0},
            entire,
            {-0X1.999999999998P-4,0.0},
            {-0X1.999999999998P-4,0X1.9999999999998P-4},
            {-0X1.999999999998P-4,0.0},
            entire,
            entire,
            entire,
            {0.0,0.0},
            {0.0,0X1.999999999998P-4},
            {-0X1.9999999999998P-4,0.0},
            {-0X1.9999999999998P-4,0X1.999999999998P-4},
            {0X1.E666666666656P+0,0X1.E666666666657P+0},
            {0x1.FFFFFFFFFFFFFp1023,infinity},
            entire,
            {0.0,0.0},
            entire,
            entire,
            entire,
            {0.0,0X1.999999999998P-4},
            empty,
            empty,
            entire,
            empty,
            entire,
            empty,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_cancelPlus<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<I, n>(h_res, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "minimal_cancel_minus_cancelMinus"_test = [&] {
        constexpr int n = 63;
        std::array<I, n> h_xs {{
            {-0.0,5.1},
            {-0X1.999999999999AP-4,0X1.FFFFFFFFFFFFP+0},
            {-0X1.FFFFFFFFFFFFEP+1023,0x1.FFFFFFFFFFFFFp1023},
            {-0X1P+0,0X1.FFFFFFFFFFFFEP-53},
            {-0X1P+0,0X1.FFFFFFFFFFFFFP-53},
            {-0x1.FFFFFFFFFFFFFp1023,0X1.FFFFFFFFFFFFEP+1023},
            {-0x1.FFFFFFFFFFFFFp1023,0x1.FFFFFFFFFFFFFp1023},
            {-0x1.FFFFFFFFFFFFFp1023,0x1.FFFFFFFFFFFFFp1023},
            {-0x1.FFFFFFFFFFFFFp1023,0x1.FFFFFFFFFFFFFp1023},
            {-1.0,5.0},
            {-1.0,5.0},
            {-1.0,5.0},
            {-1.0,infinity},
            {-1.0,infinity},
            {-1.0,infinity},
            {-10.0,-1.0},
            {-10.0,5.0},
            {-10.0,5.0},
            {-10.0,5.0},
            {-10.0,5.0},
            {-10.0,5.0},
            {-10.0,5.1},
            {-10.1,5.0},
            {-10.1,5.1},
            {-5.0,-0.9},
            {-5.0,-1.0},
            {-5.0,-1.0},
            {-5.0,-1.0},
            {-5.0,-1.0},
            {-5.0,0.0},
            {-5.0,1.0},
            {-5.1,-0.0},
            {-5.1,-0.9},
            {-5.1,-1.0},
            {-infinity,-1.0},
            {-infinity,-1.0},
            {-infinity,-1.0},
            {0.9,5.0},
            {0.9,5.1},
            {0X1.FFFFFFFFFFFFP+0,0X1.FFFFFFFFFFFFP+0},
            {0X1P-1022,0X1.0000000000001P-1022},
            {0X1P-1022,0X1.0000000000002P-1022},
            {0x0.0000000000001p-1022,0x0.0000000000001p-1022},
            {0x0.0000000000001p-1022,0x0.0000000000001p-1022},
            {0x1.FFFFFFFFFFFFFp1023,0x1.FFFFFFFFFFFFFp1023},
            {1.0,5.0},
            {1.0,5.0},
            {1.0,5.0},
            {1.0,5.0},
            {1.0,5.0},
            {1.0,5.1},
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            entire,
            entire,
            entire,
            entire,
            entire,
        }};

        std::array<I, n> h_ys {{
            {0.0,5.0},
            {-0.01,0X1.999999999999AP-4},
            {-0x1.FFFFFFFFFFFFFp1023,0x1.FFFFFFFFFFFFFp1023},
            {-0X1.FFFFFFFFFFFFFP-53,0X1P+0},
            {-0X1.FFFFFFFFFFFFEP-53,0X1P+0},
            {-0x1.FFFFFFFFFFFFFp1023,0x1.FFFFFFFFFFFFFp1023},
            {-0X1.FFFFFFFFFFFFEP+1023,0x1.FFFFFFFFFFFFFp1023},
            {-0x1.FFFFFFFFFFFFFp1023,0X1.FFFFFFFFFFFFEP+1023},
            {-0x1.FFFFFFFFFFFFFp1023,0x1.FFFFFFFFFFFFFp1023},
            {-1.0,infinity},
            {-infinity,-1.0},
            entire,
            {-1.0,5.0},
            empty,
            entire,
            empty,
            {-10.0,5.0},
            {-10.0,5.1},
            {-10.1,5.0},
            {-10.1,5.1},
            empty,
            {-10.0,5.0},
            {-10.0,5.0},
            {-10.0,5.0},
            {-5.0,-1.0},
            {-5.0,-0.9},
            {-5.0,-1.0},
            {-5.1,-0.9},
            {-5.1,-1.0},
            {-0.0,5.0},
            {-1.0,5.0},
            {-5.0,0.0},
            {-5.0,-1.0},
            {-5.0,-1.0},
            {-1.0,5.0},
            empty,
            entire,
            {1.0,5.0},
            {1.0,5.0},
            {0X1.999999999999AP-4,0X1.999999999999AP-4},
            {0X1P-1022,0X1.0000000000002P-1022},
            {0X1P-1022,0X1.0000000000001P-1022},
            {-0x0.0000000000001p-1022,-0x0.0000000000001p-1022},
            {0x0.0000000000001p-1022,0x0.0000000000001p-1022},
            {-0x1.FFFFFFFFFFFFFp1023,-0x1.FFFFFFFFFFFFFp1023},
            {0.9,5.0},
            {0.9,5.1},
            {1.0,5.0},
            {1.0,5.1},
            empty,
            {1.0,5.0},
            {-1.0,infinity},
            {-10.0,-1.0},
            {-10.0,5.0},
            {-infinity,-1.0},
            {1.0,5.0},
            empty,
            entire,
            {-1.0,5.0},
            {-1.0,infinity},
            {-infinity,-1.0},
            empty,
            entire,
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {0.0,0X1.999999999998P-4},
            {-0X1.70A3D70A3D70BP-4,0X1.E666666666657P+0},
            entire,
            entire,
            {-0X1.FFFFFFFFFFFFFP-1,-0X1.FFFFFFFFFFFFEP-1},
            entire,
            {-0X1P+971,0.0},
            {0.0,0X1P+971},
            {0.0,0.0},
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            {0.0,0.0},
            entire,
            entire,
            entire,
            entire,
            {0.0,0X1.999999999998P-4},
            {-0X1.999999999998P-4,0.0},
            {-0X1.999999999998P-4,0X1.999999999998P-4},
            {0.0,0X1.9999999999998P-4},
            entire,
            {0.0,0.0},
            entire,
            entire,
            {-5.0,-5.0},
            {-4.0,-4.0},
            {-0X1.999999999998P-4,0.0},
            {-0X1.999999999998P-4,0X1.9999999999998P-4},
            {-0X1.999999999998P-4,0.0},
            entire,
            entire,
            entire,
            {-0X1.9999999999998P-4,0.0},
            {-0X1.9999999999998P-4,0X1.999999999998P-4},
            {0X1.E666666666656P+0,0X1.E666666666657P+0},
            entire,
            {0.0,0X0.0000000000001P-1022},
            {0x0.0000000000002p-1022,0x0.0000000000002p-1022},
            {0.0,0.0},
            {0x1.FFFFFFFFFFFFFp1023,infinity},
            entire,
            entire,
            {0.0,0.0},
            entire,
            entire,
            {0.0,0X1.999999999998P-4},
            entire,
            empty,
            empty,
            entire,
            empty,
            empty,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_cancelMinus<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
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
