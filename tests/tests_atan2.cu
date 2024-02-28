
#include <cuinterval/cuinterval.h>

#include "tests.h"
#include "test_ops.cuh"

#include <stdio.h>

template<typename T>
void tests_atan2() {
    using namespace boost::ut;

    using I = interval<T>;
    using B = bool;
    using N = int;

    I empty         = ::empty<T>();
    I entire        = ::entire<T>();
    T infinity = std::numeric_limits<T>::infinity();
    T NaN = ::nan("");

    const int n = 37; // count of largest test array
    const int n_bytes   = n * sizeof(I);
    const int blockSize = 256;
    [[maybe_unused]] const int numBlocks = (n + blockSize - 1) / blockSize;

    I *d_xs_, *d_ys_, *d_zs_, *d_res_;

    CUDA_CHECK(cudaMalloc(&d_xs_, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_ys_, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_zs_, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_res_, n_bytes));

    "minimal.atan2_atan2"_test = [&] {
        constexpr int n = 37;
        std::array<I, n> h_xs {{
            {-0x1p-1022,-0x1p-1022},
            {-0x1p-1022,0.0},
            {-0x1p-1022,0x1p-1022},
            {-0x1p-1022,0x1p-1022},
            {-1.0,-1.0},
            {-1.0,-1.0},
            {-2.0,0.0},
            {-2.0,0.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-3.0,-1.0},
            {-3.0,-1.0},
            {-3.0,-1.0},
            {-3.0,-1.0},
            {-3.0,-1.0},
            {-5.0,0.0},
            {-infinity,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,2.0},
            {0.0,2.0},
            {0.0,5.0},
            {0.0,5.0},
            {0.0,infinity},
            {0x1p-1022,0x1p-1022},
            {1.0,1.0},
            {1.0,1.0},
            {1.0,3.0},
            {1.0,3.0},
            {1.0,3.0},
            {1.0,3.0},
            {1.0,3.0},
            empty,
            empty,
            entire,
            entire,
        }};

        std::array<I, n> h_ys {{
            {-0x1p-1022,0x1p-1022},
            {-0x1p-1022,-0x1p-1022},
            {-0x1p-1022,-0x1p-1022},
            {0x1p-1022,0x1p-1022},
            {-1.0,-1.0},
            {1.0,1.0},
            {-3.0,-1.0},
            {1.0,3.0},
            {-3.0,-1.0},
            {1.0,3.0},
            {-2.0,0.0},
            {-2.0,2.0},
            {-3.0,-1.0},
            {0.0,2.0},
            {1.0,3.0},
            {-5.0,0.0},
            {0.0,0.0},
            {-infinity,0.0},
            {0.0,0.0},
            {0.0,infinity},
            {-3.0,-1.0},
            {1.0,3.0},
            {-5.0,0.0},
            {0.0,5.0},
            {0.0,0.0},
            {-0x1p-1022,0x1p-1022},
            {-1.0,-1.0},
            {1.0,1.0},
            {-2.0,0.0},
            {-2.0,2.0},
            {-3.0,-1.0},
            {0.0,2.0},
            {1.0,3.0},
            empty,
            entire,
            empty,
            entire,
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-0x1.2D97C7F3321D3p1,-0x1.921FB54442D18p-1},
            {-0x1.921FB54442D19p1,+0x1.921FB54442D19p1},
            {-0x1.921FB54442D19p1,+0x1.921FB54442D19p1},
            {-0x1.921FB54442D19p-1,+0x1.921FB54442D19p-1},
            {-0x1.2D97C7F3321D3p1,-0x1.2D97C7F3321D2p1},
            {-0x1.921FB54442D19p-1,-0x1.921FB54442D18p-1},
            {-0x1.921FB54442D19p1,+0x1.921FB54442D19p1},
            {-0x1.1B6E192EBBE45p0,0x0p0},
            {-0x1.921FB54442D19p1,+0x1.921FB54442D19p1},
            {-0x1.1B6E192EBBE45p0,+0x1.1B6E192EBBE45p0},
            {-0x1.56C6E7397F5AFp1,-0x1.921FB54442D18p0},
            {-0x1.56C6E7397F5AFp1,-0x1.DAC670561BB4Fp-2},
            {-0x1.68F095FDF593Dp1,-0x1.E47DF3D0DD4Dp0},
            {-0x1.921FB54442D19p0,-0x1.DAC670561BB4Fp-2},
            {-0x1.3FC176B7A856p0,-0x1.4978FA3269EE1p-2},
            {-0x1.921FB54442D19p1,+0x1.921FB54442D19p1},
            {-0x1.921FB54442D19p0,-0x1.921FB54442D18p0},
            {0x1.921FB54442D18p1,0x1.921FB54442D19p1},
            empty,
            {0.0,0.0},
            {0x1.0468A8ACE4DF6p1,0x1.921FB54442D19p1},
            {0x0p0,0x1.1B6E192EBBE45p0},
            {0x1.921FB54442D18p0,0x1.921FB54442D19p1},
            {0x0p0,0x1.921FB54442D19p0},
            {0x1.921FB54442D18p0,0x1.921FB54442D19p0},
            {0x1.921FB54442D18p-1,0x1.2D97C7F3321D3p1},
            {0x1.2D97C7F3321D2p1,0x1.2D97C7F3321D3p1},
            {0x1.921FB54442D18p-1,0x1.921FB54442D19p-1},
            {0x1.921FB54442D18p0,0x1.56C6E7397F5AFp1},
            {0x1.DAC670561BB4Fp-2,0x1.56C6E7397F5AFp1},
            {0x1.E47DF3D0DD4Dp0,0x1.68F095FDF593Dp1},
            {0x1.DAC670561BB4Fp-2,0x1.921FB54442D19p0},
            {0x1.4978FA3269EE1p-2,0x1.3FC176B7A856p0},
            empty,
            empty,
            empty,
            {-0x1.921FB54442D19p1,+0x1.921FB54442D19p1},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_atan2<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 3;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };


    CUDA_CHECK(cudaFree(d_xs_));
    CUDA_CHECK(cudaFree(d_ys_));
    CUDA_CHECK(cudaFree(d_zs_));
    CUDA_CHECK(cudaFree(d_res_));
}
