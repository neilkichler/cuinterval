
#include <cuinterval/cuinterval.h>

#include "tests.h"
#include "test_ops.cuh"

#include <stdio.h>

template<typename T>
void tests_mpfi() {
    using namespace boost::ut;

    using I = interval<T>;
    using B = bool;

    I empty         = ::empty<T>();
    I entire        = ::entire<T>();
    T infinity = std::numeric_limits<T>::infinity();
    T NaN = ::nan("");

    const int n = 128; // count of largest test array
    const int n_bytes   = n * sizeof(I);
    const int blockSize = 256;
    [[maybe_unused]] const int numBlocks = (n + blockSize - 1) / blockSize;

    I *d_xs_, *d_ys_, *d_zs_, *d_res_;

    CUDA_CHECK(cudaMalloc(&d_xs_, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_ys_, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_zs_, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_res_, n_bytes));

    "mpfi_ab_abs"_test = [&] {
        constexpr int n = 12;
        std::array<I, n> h_xs {{
            {-0x123456789p-16,0x123456799p-16},
            {-infinity,+8.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {-infinity,0.0},
            {0.0,+8.0},
            {0.0,+8.0},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,0.0},
            {0x123456789p-16,0x123456799p-16},
            entire,
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {0.0,0x123456799p-16},
            {0.0,+infinity},
            {+7.0,+infinity},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,+8.0},
            {0.0,+8.0},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,0.0},
            {0x123456789p-16,0x123456799p-16},
            {0.0,+infinity},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_abs<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "mpfi_aco_acos"_test = [&] {
        constexpr int n = 8;
        std::array<I, n> h_xs {{
            {-0.5,0.5},
            {-0.75,-0.25},
            {-1.0,-0.5},
            {-1.0,0.0},
            {-1.0,1.0},
            {0.0,+1.0},
            {0.0,0.0},
            {0.25,0.625},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {0x10c152382d7365p-52,0x860a91c16b9b3p-50},
            {0x1d2cf5c7c70f0bp-52,0x4d6749be4edb1p-49},
            {0x10c152382d7365p-51,0x1921fb54442d19p-51},
            {0x3243f6a8885a3p-49,0x1921fb54442d19p-51},
            {0.0,0x1921fb54442d19p-51},
            {0.0,0x1921fb54442d19p-52},
            {0x3243f6a8885a3p-49,0x1921fb54442d19p-52},
            {0x1ca94936b98a21p-53,0x151700e0c14b25p-52},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_acos<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 3;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "mpfi_acosh_acosh"_test = [&] {
        constexpr int n = 5;
        std::array<I, n> h_xs {{
            {+1.0,+infinity},
            {+1.5,+infinity},
            {1.0,1.5},
            {1.5,1.5},
            {2.0,1000.0},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {0.0,+infinity},
            {0x1ecc2caec51609p-53,+infinity},
            {0.0,0xf661657628b05p-52},
            {0x1ecc2caec51609p-53,0xf661657628b05p-52},
            {0x544909c66010dp-50,0x799d4ba2a13b5p-48},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_acosh<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 3;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "mpfi_add_add"_test = [&] {
        constexpr int n = 19;
        std::array<I, n> h_xs {{
            {+4.0,+8.0},
            {+4.0,+8.0},
            {-0.375,-0x10187p-256},
            {-0x1p-300,0x123456p+28},
            {-4.0,+7.0},
            {-infinity,+8.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {0.0,+8.0},
            {0.0,+8.0},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0x1000100010001p+8,0x1p+60},
            entire,
        }};

        std::array<I, n> h_ys {{
            {-4.0,-2.0},
            {-9.0,-8.0},
            {-0.125,0x1p-240},
            {-0x10000000000000p-93,0x789abcdp0},
            {-0x123456789abcdp-17,3e300},
            {0.0,+8.0},
            {-1.0,+8.0},
            {+8.0,+infinity},
            {-7.0,0.0},
            {0.0,+8.0},
            {-7.0,+8.0},
            {0.0,+8.0},
            {+8.0,+infinity},
            {-infinity,-7.0},
            {0.0,+8.0},
            {0.0,0.0},
            entire,
            {0x1000100010001p0,3.0e300},
            {0.0,+8.0},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {0.0,+6.0},
            {-5.0,0.0},
            {-0x1p-1,-0x187p-256},
            {-0x10000000000001p-93,0x123456789abcdp0},
            {-0x123456791abcdp-17,0x8f596b3002c1bp+947},
            {-infinity,+16.0},
            {-infinity,+1.0},
            entire,
            {-7.0,+8.0},
            {0.0,+16.0},
            {-7.0,+infinity},
            {0.0,+infinity},
            {+8.0,+infinity},
            {-infinity,-7.0},
            {0.0,+8.0},
            {0.0,0.0},
            entire,
            {+0x1010101010101p+8,0x8f596b3002c1bp+947},
            entire,
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

    "mpfi_add_d_add"_test = [&] {
        constexpr int n = 32;
        std::array<I, n> h_xs {{
            {-0x1fffffffffffffp-52,-0x1p-550},
            {-0x1fffffffffffffp-52,-0x1p-550},
            {-0xfb53d14aa9c2fp-47,-17.0},
            {-0xffp0,0x123456789abcdfp-52},
            {-0xffp0,0x123456789abcdfp-52},
            {-32.0,-0xfb53d14aa9c2fp-48},
            {-32.0,-17.0},
            {-infinity,-7.0},
            {-infinity,-7.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {-infinity,0.0},
            {-infinity,0.0},
            {-infinity,8.0},
            {-infinity,8.0},
            {-infinity,8.0},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,8.0},
            {0.0,8.0},
            {0.0,8.0},
            {0x123456789abcdfp-48,0x123456789abcdfp-4},
            {0x123456789abcdfp-48,0x123456789abcdfp-4},
            {0x123456789abcdfp-56,0x123456789abcdfp-4},
            {0x123456789abcdfp-56,0x123456789abcdfp-4},
            entire,
            entire,
            entire,
        }};

        std::array<I, n> h_ys {{
            {-4097.5,-4097.5},
            {4097.5,4097.5},
            {0xfb53d14aa9c2fp-47,0xfb53d14aa9c2fp-47},
            {-256.5,-256.5},
            {256.5,256.5},
            {0xfb53d14aa9c2fp-48,0xfb53d14aa9c2fp-48},
            {-0xfb53d14aa9c2fp-47,-0xfb53d14aa9c2fp-47},
            {-0x170ef54646d497p-107,-0x170ef54646d497p-107},
            {0.0,0.0},
            {0x170ef54646d497p-107,0x170ef54646d497p-107},
            {-0x170ef54646d497p-106,-0x170ef54646d497p-106},
            {0.0,0.0},
            {0x170ef54646d497p-106,0x170ef54646d497p-106},
            {-0x16345785d8a00000p0,-0x16345785d8a00000p0},
            {0.0,0.0},
            {0x16345785d8a00000p0,0x16345785d8a00000p0},
            {-0x50b45a75f7e81p-104,-0x50b45a75f7e81p-104},
            {0.0,0.0},
            {0x142d169d7dfa03p-106,0x142d169d7dfa03p-106},
            {-0x170ef54646d497p-109,-0x170ef54646d497p-109},
            {0.0,0.0},
            {0x170ef54646d497p-109,0x170ef54646d497p-109},
            {-0x114b37f4b51f71p-107,-0x114b37f4b51f71p-107},
            {0.0,0.0},
            {0x114b37f4b51f7p-103,0x114b37f4b51f7p-103},
            {-3.5,-3.5},
            {3.5,3.5},
            {-3.5,-3.5},
            {3.5,3.5},
            {+0x170ef54646d497p-105,+0x170ef54646d497p-105},
            {-0x170ef54646d497p-105,-0x170ef54646d497p-105},
            {0.0e-17,0.0e-17},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-0x10038p-4,-0x10018p-4},
            {0xfff8p-4,0x10018p-4},
            {0.0,0x7353d14aa9c2fp-47},
            {-0x1ff8p-4,-0xff5cba9876543p-44},
            {0x18p-4,0x101a3456789abdp-44},
            {-0x104ac2eb5563d1p-48,0.0},
            {-0x1fb53d14aa9c2fp-47,-0x18353d14aa9c2fp-47},
            {-infinity,-7.0},
            {-infinity,-7.0},
            {-infinity,-0x1bffffffffffffp-50},
            {-infinity,-8.0e-17},
            {-infinity,0.0},
            {-infinity,0x170ef54646d497p-106},
            {-infinity,-0x16345785d89fff00p0},
            {-infinity,8.0},
            {-infinity,0x16345785d8a00100p0},
            {-0x50b45a75f7e81p-104,+infinity},
            {0.0,+infinity},
            {0x142d169d7dfa03p-106,+infinity},
            {-0x170ef54646d497p-109,-0x170ef54646d497p-109},
            {0.0,0.0},
            {0x170ef54646d497p-109,0x170ef54646d497p-109},
            {-0x114b37f4b51f71p-107,8.0},
            {0.0,8.0},
            {0x114b37f4b51f7p-103,0x10000000000001p-49},
            {0xeb456789abcdfp-48,0x123456789abca7p-4},
            {0x15b456789abcdfp-48,0x123456789abd17p-4},
            {-0x36dcba98765434p-52,0x123456789abca7p-4},
            {0x3923456789abcdp-52,0x123456789abd17p-4},
            entire,
            entire,
            entire,
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

    "mpfi_asin_asin"_test = [&] {
        constexpr int n = 8;
        std::array<I, n> h_xs {{
            {-0.5,0.5},
            {-0.75,-0.25},
            {-1.0,-0.5},
            {-1.0,0.0},
            {-1.0,1.0},
            {0.0,+1.0},
            {0.0,0.0},
            {0.25,0.625},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-0x860a91c16b9b3p-52,0x860a91c16b9b3p-52},
            {-0x1b235315c680ddp-53,-0x102be9ce0b87cdp-54},
            {-0x1921fb54442d19p-52,-0x10c152382d7365p-53},
            {-0x1921fb54442d19p-52,0.0},
            {-0x1921fb54442d19p-52,0x1921fb54442d19p-52},
            {0.0,0x1921fb54442d19p-52},
            {0.0,0.0},
            {0x102be9ce0b87cdp-54,0x159aad71ced00fp-53},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_asin<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 3;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "mpfi_asinh_asinh"_test = [&] {
        constexpr int n = 19;
        std::array<I, n> h_xs {{
            {-0.5,0.5},
            {-0.75,-0.25},
            {-1.0,-0.5},
            {-1.0,0.0},
            {-1.0,1.0},
            {-2.0,-0.5},
            {-42.0,17.0},
            {-6.0,-4.0},
            {-infinity,+8.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {0.0,+1.0},
            {0.0,+8.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.125,17.0},
            {0.25,0.625},
            {17.0,42.0},
            entire,
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-0xf661657628b05p-53,0xf661657628b05p-53},
            {-0x162e42fefa39fp-49,-0xfd67d91ccf31bp-54},
            {-0x1c34366179d427p-53,-0x1ecc2caec51609p-54},
            {-0x1c34366179d427p-53,0.0},
            {-0x1c34366179d427p-53,0x1c34366179d427p-53},
            {-0x2e32430627a11p-49,-0x1ecc2caec51609p-54},
            {-0x8dca6976ad6bdp-49,0xe1be0ba541ef7p-50},
            {-0x4fbca919fe219p-49,-0x10c1f8a6e80eebp-51},
            {-infinity,0x58d8dc657eaf5p-49},
            {-infinity,-0x152728c91b5f1dp-51},
            {-infinity,0.0},
            {0.0,0x1c34366179d427p-53},
            {0.0,0x58d8dc657eaf5p-49},
            {0.0,+infinity},
            {0.0,0.0},
            {0xff5685b4cb4b9p-55,0xe1be0ba541ef7p-50},
            {0xfd67d91ccf31bp-54,0x4b89d40b2fecdp-51},
            {0x1c37c174a83dedp-51,0x8dca6976ad6bdp-49},
            entire,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_asinh<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 3;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "mpfi_atan_atan"_test = [&] {
        constexpr int n = 19;
        std::array<I, n> h_xs {{
            {-0.5,0.5},
            {-0.75,-0.25},
            {-1.0,-0.5},
            {-1.0,0.0},
            {-1.0,1.0},
            {-2.0,-0.5},
            {-42.0,17.0},
            {-6.0,-4.0},
            {-infinity,+8.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {0.0,+1.0},
            {0.0,+8.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.125,17.0},
            {0.25,0.625},
            {17.0,42.0},
            entire,
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-0x1dac670561bb5p-50,0x1dac670561bb5p-50},
            {-0xa4bc7d1934f71p-52,-0x1f5b75f92c80ddp-55},
            {-0x1921fb54442d19p-53,-0x1dac670561bb4fp-54},
            {-0x1921fb54442d19p-53,0.0},
            {-0x1921fb54442d19p-53,0x1921fb54442d19p-53},
            {-0x11b6e192ebbe45p-52,-0x1dac670561bb4fp-54},
            {-0x18c079f3350d27p-52,0x1831516233f561p-52},
            {-0x167d8863bc99bdp-52,-0x54da32547a73fp-50},
            {-0x1921fb54442d19p-52,0xb924fd54cb511p-51},
            {-0x1921fb54442d19p-52,-0x5b7315eed597fp-50},
            {-0x1921fb54442d19p-52,0.0},
            {0.0,0x1921fb54442d19p-53},
            {0.0,0xb924fd54cb511p-51},
            {0.0,0x1921fb54442d19p-52},
            {0.0,0.0},
            {0x1fd5ba9aac2f6dp-56,0x1831516233f561p-52},
            {0x1f5b75f92c80ddp-55,0x47802eaf7bfadp-51},
            {0xc18a8b119fabp-47,0x18c079f3350d27p-52},
            {-0x1921fb54442d19p-52,0x1921fb54442d19p-52},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_atan<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 3;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "mpfi_atan2_atan2"_test = [&] {
        constexpr int n = 18;
        std::array<I, n> h_xs {{
            {-17.0,-5.0},
            {-17.0,-5.0},
            {-17.0,5.0},
            {-infinity,+8.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {0.0,+8.0},
            {0.0,+8.0},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {5.0,17.0},
            {5.0,17.0},
            entire,
        }};

        std::array<I, n> h_ys {{
            {-4002.0,-1.0},
            {1.0,4002.0},
            {-4002.0,1.0},
            {0.0,+8.0},
            {-1.0,+8.0},
            {+8.0,+infinity},
            {-7.0,+8.0},
            {-7.0,0.0},
            {0.0,+8.0},
            {0.0,+8.0},
            {+8.0,+infinity},
            {-infinity,-7.0},
            {0.0,+8.0},
            {0.0,0.0},
            entire,
            {-4002.0,-1.0},
            {1.0,4002.0},
            {0.0,+8.0},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-0x191f6c4c09a81bp-51,-0x1a12a5465464cfp-52},
            {-0x1831516233f561p-52,-0xa3c20ea13f5e5p-61},
            {-0x1921fb54442d19p-51,0x1921fb54442d19p-51},
            {-0x1921fb54442d19p-52,0x1921fb54442d19p-52},
            {-0x6d9cc4b34bd0dp-50,-0x1700a7c5784633p-53},
            {-0x1921fb54442d19p-52,0.0},
            {0.0,0x1921fb54442d19p-51},
            {0x3243f6a8885a3p-49,0x1921fb54442d19p-51},
            {0.0,0x1921fb54442d19p-52},
            {0.0,0x1921fb54442d19p-52},
            {0.0,0.0},
            {0x1921fb54442d18p-51,0x1921fb54442d19p-51},
            {0.0,0.0},
            empty,
            {0.0,0x1921fb54442d19p-51},
            {0x1a12a5465464cfp-52,0x191f6c4c09a81bp-51},
            {0xa3c20ea13f5e5p-61,0x1831516233f561p-52},
            {-0x1921fb54442d19p-52,0x1921fb54442d19p-52},
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

    "mpfi_atanh_atanh"_test = [&] {
        constexpr int n = 9;
        std::array<I, n> h_xs {{
            {-0.5,0.5},
            {-0.75,-0.25},
            {-1.0,-0.5},
            {-1.0,0.0},
            {-1.0,1.0},
            {0.0,+1.0},
            {0.0,0.0},
            {0.125,1.0},
            {0.25,0.625},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-0x1193ea7aad030bp-53,0x1193ea7aad030bp-53},
            {-0x3e44e55c64b4bp-50,-0x1058aefa811451p-54},
            {-infinity,-0x8c9f53d568185p-52},
            {-infinity,0.0},
            entire,
            {0.0,+infinity},
            {0.0,0.0},
            {0x1015891c9eaef7p-55,+infinity},
            {0x1058aefa811451p-54,0x2eec3bb76c2b3p-50},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_atanh<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 3;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "mpfi_bounded_p_isCommonInterval"_test = [&] {
        constexpr int n = 16;
        std::array<I, n> h_xs {{
            {+0x1fffffffffffffp-53,2.0},
            {+8.0,+0x7fffffffffffbp+51},
            {-0x1921fb54442d18p-51,0x1921fb54442d19p-51},
            {-34.0,-17.0},
            {-34.0,17.0},
            {-8.0,-1.0},
            {-8.0,0.0},
            {-infinity,-8.0},
            {-infinity,0.0},
            {-infinity,5.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,5.0},
            {0x1921fb54442d18p-51,0x1921fb54442d19p-51},
            {5.0,+infinity},
            entire,
        }};

        std::array<B, n> h_res{};
        B *d_res = (B *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(B);
        std::array<B, n> h_ref {{
            true,
            true,
            true,
            true,
            true,
            true,
            true,
            false,
            false,
            false,
            false,
            true,
            true,
            true,
            false,
            false,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_isCommonInterval<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<B, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "mpfi_co_cos"_test = [&] {
        constexpr int n = 46;
        std::array<I, n> h_xs {{
            {-0.5,0.5},
            {-1.0,-0.25},
            {-1.0,0.0},
            {-2.0,-0.5},
            {-4.0,-1.0},
            {-4.0,-2.0},
            {-4.0,-3.0},
            {-4.0,-4.0},
            {-4.0,0.0},
            {-4.0,1.0},
            {-4.5,0.625},
            {-5.0,-1.0},
            {-5.0,-2.0},
            {-5.0,-3.0},
            {-5.0,-4.0},
            {-5.0,-5.0},
            {-5.0,0.0},
            {-5.0,1.0},
            {-6.0,-1.0},
            {-6.0,-2.0},
            {-6.0,-3.0},
            {-6.0,-4.0},
            {-6.0,-5.0},
            {-6.0,-6.0},
            {-6.0,0.0},
            {-6.0,1.0},
            {-7.0,-1.0},
            {-7.0,-2.0},
            {-7.0,-3.0},
            {-7.0,-4.0},
            {-7.0,-5.0},
            {-7.0,-6.0},
            {-7.0,-7.0},
            {-7.0,0.0},
            {-7.0,1.0},
            {-infinity,+8.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {0.0,+1.0},
            {0.0,+8.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.125,17.0},
            {1.0,0x3243f6a8885a3p-48},
            {17.0,42.0},
            entire,
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {0x1c1528065b7d4fp-53,1.0},
            {0x114a280fb5068bp-53,0xf80aa4fbef751p-52},
            {0x114a280fb5068bp-53,1.0},
            {-0x1aa22657537205p-54,0x1c1528065b7d5p-49},
            {-1.0,0x114a280fb5068cp-53},
            {-1.0,-0x1aa22657537204p-54},
            {-1.0,-0x14eaa606db24c0p-53},
            {-0x14eaa606db24c1p-53,-0x14eaa606db24c0p-53},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,0x114a280fb5068cp-53},
            {-1.0,0x122785706b4adap-54},
            {-1.0,0x122785706b4adap-54},
            {-0x14eaa606db24c1p-53,0x122785706b4adap-54},
            {0x122785706b4ad9p-54,0x122785706b4adap-54},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,0x1eb9b7097822f6p-53},
            {-1.0,0x1eb9b7097822f6p-53},
            {-1.0,0x1eb9b7097822f6p-53},
            {-0x14eaa606db24c1p-53,0x1eb9b7097822f6p-53},
            {0x122785706b4ad9p-54,0x1eb9b7097822f6p-53},
            {0x1eb9b7097822f5p-53,0x1eb9b7097822f6p-53},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-0x14eaa606db24c1p-53,1.0},
            {0x122785706b4ad9p-54,1.0},
            {0x181ff79ed92017p-53,1.0},
            {0x181ff79ed92017p-53,0x181ff79ed92018p-53},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {0x114a280fb5068bp-53,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {1.0,1.0},
            {-1.0,1.0},
            {-1.0,0x4528a03ed41a3p-51},
            {-1.0,1.0},
            {-1.0,1.0},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_cos<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 2;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "mpfi_cosh_cosh"_test = [&] {
        constexpr int n = 14;
        std::array<I, n> h_xs {{
            {-0.125,0.0},
            {-1.0,0.0},
            {-4.5,-0.625},
            {-infinity,+8.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {0.0,+1.0},
            {0.0,+8.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0x10000000000001p-53},
            {1.0,3.0},
            {17.0,0xb145bb71d3dbp-38},
            entire,
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {1.0,0x10200aac16db6fp-52},
            {1.0,0x18b07551d9f551p-52},
            {0x99d310a496b6dp-51,0x1681ceb0641359p-47},
            {1.0,+infinity},
            {0x11228949ba3a8bp-43,+infinity},
            {1.0,+infinity},
            {1.0,0x18b07551d9f551p-52},
            {1.0,0x1749eaa93f4e77p-42},
            {1.0,+infinity},
            {1.0,1.0},
            {1.0,0x120ac1862ae8d1p-52},
            {0x18b07551d9f55p-48,0x1422a497d6185fp-49},
            {0x1709348c0ea503p-29,0x3ffffffffffa34p+968},
            {1.0,+infinity},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_cosh<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 2;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "mpfi_d_div_div"_test = [&] {
        constexpr int n = 30;
        std::array<I, n> h_xs {{
            {+0x170ef54646d497p-105,+0x170ef54646d497p-105},
            {-0x114b37f4b51f71p-107,-0x114b37f4b51f71p-107},
            {-0x16345785d8a00000p0,-0x16345785d8a00000p0},
            {-0x170ef54646d496p-107,-0x170ef54646d496p-107},
            {-0x170ef54646d497p-105,-0x170ef54646d497p-105},
            {-0x170ef54646d497p-106,-0x170ef54646d497p-106},
            {-0x170ef54646d497p-109,-0x170ef54646d497p-109},
            {-0x50b45a75f7e81p-104,-0x50b45a75f7e81p-104},
            {-2.5,-2.5},
            {-2.5,-2.5},
            {-2.5,-2.5},
            {-2.5,-2.5},
            {-2.5,-2.5},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0e-17,0.0e-17},
            {0x114b37f4b51f71p-107,0x114b37f4b51f71p-107},
            {0x142d169d7dfa03p-106,0x142d169d7dfa03p-106},
            {0x16345785d8a00000p0,0x16345785d8a00000p0},
            {0x170ef54646d496p-107,0x170ef54646d496p-107},
            {0x170ef54646d497p-106,0x170ef54646d497p-106},
            {0x170ef54646d497p-109,0x170ef54646d497p-109},
            {33.125,33.125},
            {33.125,33.125},
            {33.125,33.125},
            {33.125,33.125},
        }};

        std::array<I, n> h_ys {{
            entire,
            {0.0,7.0},
            {-infinity,8.0},
            {-infinity,-7.0},
            entire,
            {-infinity,0.0},
            {0.0,0.0},
            {0.0,+infinity},
            {-16.0,-7.0},
            {-8.0,-5.0},
            {-8.0,8.0},
            {11.0,143.0},
            {25.0,40.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {-infinity,8.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,7.0},
            entire,
            {0.0,7.0},
            {0.0,+infinity},
            {-infinity,8.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {0.0,0.0},
            {-530.0,-496.875},
            {52.0,54.0},
            {54.0,265.0},
            {8.28125,530.0},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            entire,
            {-infinity,-0x13c3ada9f391a5p-110},
            entire,
            {0.0,0x1a5a3ce29a1787p-110},
            entire,
            {0.0,+infinity},
            empty,
            {-infinity,0.0},
            {0x5p-5,0x16db6db6db6db7p-54},
            {0x5p-4,0.5},
            entire,
            {-0x1d1745d1745d18p-55,-0x11e6efe35b4cfap-58},
            {-0x1999999999999ap-56,-0x1p-4},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            empty,
            {0.0,0.0},
            {0.0,0.0},
            {0x13c3ada9f391a5p-110,+infinity},
            {0.0,+infinity},
            entire,
            {-0x1a5a3ce29a1787p-110,0.0},
            {-infinity,0.0},
            empty,
            {-0x11111111111112p-56,-0x1p-4},
            {0x13a12f684bda12p-53,0x14627627627628p-53},
            {0.125,0x13a12f684bda13p-53},
            {0x1p-4,4.0},
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

    "mpfi_diam_ab_wid"_test = [&] {
        constexpr int n = 10;
        std::array<I, n> h_xs {{
            {-34.0,-17.0},
            {-8.0,0.0},
            {-infinity,-8.0},
            {-infinity,0.0},
            {-infinity,0.0},
            {-infinity,5.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,5.0},
            entire,
        }};

        std::array<T, n> h_res{};
        T *d_res = (T *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(T);
        std::array<T, n> h_ref {{
            17,
            +8,
            +infinity,
            +infinity,
            +infinity,
            +infinity,
            +infinity,
            -0,
            +5,
            +infinity,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_wid<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<T, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "mpfi_div_div"_test = [&] {
        constexpr int n = 62;
        std::array<I, n> h_xs {{
            {-0x1.02f0415f9f596p+0,-0x1.489c07caba163p-4},
            {-0x1.02f0415f9f596p+0,-0x754ep-16},
            {-0x1.18333622af827p+0,0x2.14b836907297p+0},
            {-0x1.25f2d73472753p+0,+0x9.9a19fd3c1fc18p-4},
            {-0x1.25f2d73472753p+0,-0x9.9a19fd3c1fc18p-4},
            {-0x1.25f2d73472753p+0,0.0},
            {-0x1.4298b2138f2a7p-4,0.0},
            {-0x1.4298b2138f2a7p-4,0.0},
            {-0x100p0,-0xe.bb80d0a0824ep-4},
            {-0x10p0,0xd0e9dc4p+12},
            {-0x123456789p0,-0x1.b0a62934c76e9p+0},
            {-0x123456789p0,-0x754ep+4},
            {-0x12p0,0x10p0},
            {-0x1p0,0x754ep-16},
            {-0x754ep0,0x1p+10},
            {-0x754ep0,0xd0e9dc4p+12},
            {-0x75bcd15p0,-0x1.489c07caba163p-4},
            {-0x75bcd15p0,-0x754ep0},
            {-0x75bcd15p0,0.0},
            {-0x75bcd15p0,0.0},
            {-0x75bcd15p0,0xa680p0},
            {-0xacbp+256,-0x6f9p0},
            {-0xacbp+256,0x6f9p0},
            {-0xb.5b90b4d32136p-4,0x6.e694ac6767394p+0},
            {-0xd.67775e4b8588p+0,-0x1.b0a62934c76e9p+0},
            {-0xd.67775e4b8588p-4,-0x754ep-53},
            {-0xeeeeeeeeep0,0.0},
            {-0xeeeeeeeeep0,0.0},
            {-100.0,-15.0},
            {-2.0,-0x1.25f2d73472753p+0},
            {-infinity,+8.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {0.0,+15.0},
            {0.0,+8.0},
            {0.0,+8.0},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0x1.5f6b03dc8c66fp+0},
            {0.0,0x1.acbf1702af6edp+0},
            {0.0,0x75bcd15p0},
            {0.0,0x75bcd15p0},
            {0.0,0xap0},
            {0.0,0xap0},
            {0x1.7f03f2a978865p+0,0xeeeeep0},
            {0x1.a9016514490e6p-4,0xeeeep0},
            {0x1.d7c06f9ff0706p-8,0x1ba2dc763p0},
            {0x5.efc1492p-4,0x1.008a3accc766dp+0},
            {0x5efc1492p0,0x1ba2dc763p0},
            {0x754ep-16,0x1.008a3accc766dp+4},
            {0x754ep0,+0xeeeeep0},
            {0x754ep0,0x75bcd15p0},
            {0x754ep0,0xeeeep0},
            {0x8.440e7d65be6bp-8,0x3.99982e9eae09ep+0},
            {0x9.ac412ff1f1478p-4,0x75bcd15p0},
            {0xe.1552a314d629p-4,0x1.064c5adfd0042p+0},
            {5.0,6.0},
            entire,
        }};

        std::array<I, n> h_ys {{
            {-0x2.e8e36e560704ap+0,-0x7.62ce64fbacd2cp-8},
            {-0x11ep0,-0x7.62ce64fbacd2cp-8},
            {0x1.263147d1f4bcbp+0,0x111p0},
            {-0x9.3b0c8074ccc18p-4,+0x4.788df5d72af78p-4},
            {-0x9.3b0c8074ccc18p-4,+0x4.788df5d72af78p-4},
            {-0x9.3b0c8074ccc18p-4,+0x4.788df5d72af78p-4},
            {-0x1p-8,-0xf.5e4900c9c19fp-12},
            {0xf.5e4900c9c19fp-12,0x9p0},
            {-0x1.7c6d760a831fap+0,0.0},
            {0x11ep0,0xbbbp0},
            {0x40bp-17,0x2.761ec797697a4p-4},
            {0x40bp0,0x11ep+4},
            {-0xbbbbbbbbbbp0,-0x9p0},
            {-0xccccccccccp0,-0x11ep0},
            {0x11ep0,0xbbbp0},
            {0x11ep0,0xbbbp0},
            {-0x2.e8e36e560704ap+4,-0x9p0},
            {-0x11ep0,-0x9p0},
            {-0x90p0,-0x9p0},
            {0x9p0,0x90p0},
            {-0xaf6p0,-0x9p0},
            {-0x7p0,0.0},
            {-0x7p0,0.0},
            {-0xdddddddddddp0,-0xc.f459be9e80108p-4},
            {0x4.887091874ffc8p-4,0x2.761ec797697a4p+4},
            {0x4.887091874ffc8p+0,0x11ep+201},
            {-0xaaaaaaaaap0,0.0},
            {0.0,+0x3p0},
            {0.0,+3.0},
            {0.0,+0x9.3b0c8074ccc18p-4},
            {0.0,+8.0},
            {-1.0,+8.0},
            {+8.0,+infinity},
            {-3.0,+3.0},
            {-7.0,+8.0},
            {-7.0,0.0},
            {0.0,+8.0},
            {0.0,+8.0},
            {+8.0,+infinity},
            {-infinity,-7.0},
            {0.0,+8.0},
            entire,
            {+0x2.39ad24e812dcep+0,0xap0},
            {-0x0.fp0,-0xe.3d7a59e2bdacp-4},
            {+0x9p0,+0xap0},
            {-0xap0,-0x9p0},
            {-0x9p0,0.0},
            {-1.0,+1.0},
            {0.0,0x1.48b08624606b9p+0},
            {-0xe.316e87be0b24p-4,0.0},
            {0x2fdd1fp-20,0xe.3d7a59e2bdacp+0},
            {0x2.497403b31d32ap+0,0x8.89b71p+0},
            {0x2fdd1fp0,0x889b71p0},
            {-0x11ep0,-0x2.497403b31d32ap+0},
            {0.0,+0x11ep0},
            {-0x11ep0,-0x9p0},
            {-0x11ep0,0.0},
            {0x8.29fa8d0659e48p-4,0xc.13d2fd762e4a8p-4},
            {-0x1.5232c83a0e726p+4,-0x9p0},
            {-0x5.0d4d319a50b04p-4,-0x2.d8f51df1e322ep-4},
            {-0x5.0d4d319a50b04p-4,0x2.d8f51df1e322ep-4},
            {0.0,+8.0},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {0x7.0ef61537b1704p-8,0x2.30ee5eef9c36cp+4},
            {0x69p-16,0x2.30ee5eef9c36cp+4},
            {-0xf.3d2f5db8ec728p-4,0x1.cf8fa732de129p+0},
            entire,
            entire,
            entire,
            {0.0,0x1.4fdb41a33d6cep+4},
            {-0x1.4fdb41a33d6cep+4,0.0},
            {0x9.e9f24790445fp-4,+infinity},
            {-0xe.525982af70c9p-8,0xbaffep+12},
            {-0x480b3bp+17,-0xa.fc5e7338f3e4p+0},
            {-0x480b3bp0,-0x69p0},
            {-0x1.c71c71c71c71dp0,2.0},
            {-0x69p-16,0xe.525982af70c9p-12},
            {-0x69p0,0xe.525982af70c9p-2},
            {-0x69p0,0xbaffep+12},
            {0x7.0ef61537b1704p-12,0xd14fadp0},
            {0x69p0,0xd14fadp0},
            {0.0,0xd14fadp0},
            {-0xd14fadp0,0.0},
            {-0x1280p0,0xd14fadp0},
            {0xffp0,+infinity},
            entire,
            {-0x8.85e40b3c3f63p+0,0xe.071cbfa1de788p-4},
            {-0x2.f5008d2df94ccp+4,-0xa.fc5e7338f3e4p-8},
            {-0x2.f5008d2df94ccp-4,-0x69p-254},
            {0.0,+infinity},
            {-infinity,0.0},
            {-infinity,-5.0},
            {-infinity,-0x1.fd8457415f917p+0},
            entire,
            entire,
            {-infinity,0.0},
            entire,
            entire,
            {-infinity,0.0},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0x9.deb65b02baep-4},
            {-0x1.e1bb896bfda07p+0,0.0},
            {0.0,0xd14fadp0},
            {-0xd14fadp0,0.0},
            {-infinity,0.0},
            entire,
            {0x1.2a4fcda56843p+0,+infinity},
            {-infinity,-0x1.df1cc82e6a583p-4},
            {0x2.120d75be74b54p-12,0x93dp+20},
            {0xb.2p-8,0x7.02d3edfbc8b6p-4},
            {0xb2p0,0x93dp0},
            {-0x7.02d3edfbc8b6p+0,-0x69p-16},
            {0x69p0,+infinity},
            {-0xd14fadp0,-0x69p0},
            {-infinity,-0x69p0},
            {0xa.f3518768b206p-8,0x7.0e2acad54859cp+0},
            {-0xd14fadp0,-0x7.52680a49e5d68p-8},
            {-0x5.c1d97d57d81ccp+0,-0x2.c9a600c455f5ap+0},
            entire,
            entire,
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

    "mpfi_div_d_div"_test = [&] {
        constexpr int n = 25;
        std::array<I, n> h_xs {{
            {-0x10000000000001p-20,-0x10000000000001p-53},
            {-0x10000000000001p-20,-0x10000020000001p-53},
            {-0x10000000000002p-20,-0x10000000000001p-53},
            {-0x10000000000002p-20,-0x10000020000001p-53},
            {-0x123456789abcdfp-53,0x10000000000001p-53},
            {-0x123456789abcdfp-53,0x123456789abcdfp-7},
            {-1.0,0x10000000000001p-53},
            {-1.0,0x123456789abcdfp-7},
            {-infinity,-7.0},
            {-infinity,-7.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {-infinity,0.0},
            {-infinity,8.0},
            {-infinity,8.0},
            {-infinity,8.0},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,8.0},
            {0.0,8.0},
            entire,
            entire,
            entire,
        }};

        std::array<I, n> h_ys {{
            {-1.0,-1.0},
            {0x10000000000001p-53,0x10000000000001p-53},
            {0x10000000000001p-53,0x10000000000001p-53},
            {0x10000000000001p-53,0x10000000000001p-53},
            {-0x123456789abcdfp0,-0x123456789abcdfp0},
            {-0x123456789abcdfp0,-0x123456789abcdfp0},
            {-0x123456789abcdfp0,-0x123456789abcdfp0},
            {-0x123456789abcdfp0,-0x123456789abcdfp0},
            {-7.0,-7.0},
            {0.0,0.0},
            {7.0,7.0},
            {-0x170ef54646d497p-106,-0x170ef54646d497p-106},
            {0x170ef54646d497p-106,0x170ef54646d497p-106},
            {-3.0,-3.0},
            {0.0,0.0},
            {3.0,3.0},
            {-0x50b45a75f7e81p-104,-0x50b45a75f7e81p-104},
            {0x142d169d7dfa03p-106,0x142d169d7dfa03p-106},
            {-0x170ef54646d497p-109,-0x170ef54646d497p-109},
            {0x170ef54646d497p-109,0x170ef54646d497p-109},
            {-0x114b37f4b51f71p-107,-0x114b37f4b51f71p-107},
            {0x114b37f4b51f71p-107,0x114b37f4b51f71p-107},
            {+0x170ef54646d497p-105,+0x170ef54646d497p-105},
            {-0x170ef54646d497p-105,-0x170ef54646d497p-105},
            {0.0e-17,0.0e-17},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {0x10000000000001p-53,0x10000000000001p-20},
            {-0x1p+33,-0x1000001fffffffp-52},
            {-0x10000000000001p-19,-1.0},
            {-0x10000000000001p-19,-0x1000001fffffffp-52},
            {-0x1c200000000002p-106,0x1p-53},
            {-0x1p-7,0x1p-53},
            {-0x1c200000000002p-106,0x1c200000000001p-105},
            {-0x1p-7,0x1c200000000001p-105},
            {1.0,+infinity},
            empty,
            {-infinity,-1.0},
            {0.0,+infinity},
            {-infinity,0.0},
            {-0x15555555555556p-51,+infinity},
            empty,
            {-infinity,0x15555555555556p-51},
            {-infinity,0.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.0},
            {-0x1d9b1f5d20d556p+5,0.0},
            {0.0,0x1d9b1f5d20d556p+5},
            entire,
            entire,
            empty,
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

    "mpfi_d_sub_sub"_test = [&] {
        constexpr int n = 32;
        std::array<I, n> h_xs {{
            {-0x114b37f4b51f71p-107,-0x114b37f4b51f71p-107},
            {-0x142d169d7dfa03p-106,-0x142d169d7dfa03p-106},
            {-0x16345785d8a00000p0,-0x16345785d8a00000p0},
            {-0x170ef54646d497p-105,-0x170ef54646d497p-105},
            {-0x170ef54646d497p-107,-0x170ef54646d497p-107},
            {-0x170ef54646d497p-109,-0x170ef54646d497p-109},
            {-0x170ef54646d497p-96,-0x170ef54646d497p-96},
            {-0x50b45a75f7e81p-104,-0x50b45a75f7e81p-104},
            {-0xfb53d14aa9c2fp-47,-0xfb53d14aa9c2fp-47},
            {-256.5,-256.5},
            {-3.5,-3.5},
            {-3.5,-3.5},
            {-4097.5,-4097.5},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0e-17,0.0e-17},
            {0x114b37f4b51f71p-107,0x114b37f4b51f71p-107},
            {0x16345785d8a00000p0,0x16345785d8a00000p0},
            {0x170ef54646d497p-105,0x170ef54646d497p-105},
            {0x170ef54646d497p-107,0x170ef54646d497p-107},
            {0x170ef54646d497p-109,0x170ef54646d497p-109},
            {0x170ef54646d497p-96,0x170ef54646d497p-96},
            {0xfb53d14aa9c2fp-47,0xfb53d14aa9c2fp-47},
            {0xfb53d14aa9c2fp-48,0xfb53d14aa9c2fp-48},
            {256.5,256.5},
            {3.5,3.5},
            {3.5,3.5},
            {4097.5,4097.5},
        }};

        std::array<I, n> h_ys {{
            {0.0,8.0},
            {0.0,+infinity},
            {-infinity,8.0},
            entire,
            {-infinity,-7.0},
            {0.0,0.0},
            {-infinity,0.0},
            {0.0,+infinity},
            {17.0,32.0},
            {-0x123456789abcdfp-52,0xffp0},
            {-0x123456789abcdfp-4,-0x123456789abcdfp-48},
            {-0x123456789abcdfp-4,-0x123456789abcdfp-56},
            {0x1p-550,0x1fffffffffffffp-52},
            {-infinity,-7.0},
            {-infinity,0.0},
            {-infinity,8.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,8.0},
            entire,
            {0.0,8.0},
            {-infinity,8.0},
            entire,
            {-infinity,-7.0},
            {0.0,0.0},
            {-infinity,0.0},
            {17.0,0xfb53d14aa9c2fp-47},
            {0xfb53d14aa9c2fp-48,32.0},
            {-0x123456789abcdfp-52,0xffp0},
            {-0x123456789abcdfp-4,-0x123456789abcdfp-48},
            {-0x123456789abcdfp-4,-0x123456789abcdfp-56},
            {0x1p-550,0x1fffffffffffffp-52},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-0x10000000000001p-49,-0x114b37f4b51f71p-107},
            {-infinity,-0x142d169d7dfa03p-106},
            {-0x16345785d8a00100p0,+infinity},
            entire,
            {0x1bffffffffffffp-50,+infinity},
            {-0x170ef54646d497p-109,-0x170ef54646d497p-109},
            {-0x170ef54646d497p-96,+infinity},
            {-infinity,-0x50b45a75f7e81p-104},
            {-0x1fb53d14aa9c2fp-47,-0x18353d14aa9c2fp-47},
            {-0x1ff8p-4,-0xff5cba9876543p-44},
            {0xeb456789abcdfp-48,0x123456789abca7p-4},
            {-0x36dcba98765434p-52,0x123456789abca7p-4},
            {-0x10038p-4,-0x10018p-4},
            {7.0,+infinity},
            {0.0,+infinity},
            {-8.0,+infinity},
            {-infinity,0.0},
            {0.0,0.0},
            {-8.0,0.0},
            entire,
            {-8.0,0x114b37f4b51f71p-107},
            {0x16345785d89fff00p0,+infinity},
            entire,
            {7.0,+infinity},
            {0x170ef54646d497p-109,0x170ef54646d497p-109},
            {0x170ef54646d497p-96,+infinity},
            {0.0,0x7353d14aa9c2fp-47},
            {-0x104ac2eb5563d1p-48,0.0},
            {0x18p-4,0x101a3456789abdp-44},
            {0x15b456789abcdfp-48,0x123456789abd17p-4},
            {0x3923456789abcdp-52,0x123456789abd17p-4},
            {0xfff8p-4,0x10018p-4},
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

    "mpfi_exp_exp"_test = [&] {
        constexpr int n = 12;
        std::array<I, n> h_xs {{
            {-0.125,0.0},
            {-0.125,0.25},
            {-123.0,-17.0},
            {-infinity,+1.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {0.0,+1.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.25},
            {0xap-47,0xbp-47},
            entire,
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {0x1c3d6a24ed8221p-53,1.0},
            {0x1c3d6a24ed8221p-53,0x148b5e3c3e8187p-52},
            {0x1766b45dd84f17p-230,0x1639e3175a689dp-77},
            {0.0,0x15bf0a8b14576ap-51},
            {0.0,0x1de16b9c24a98fp-63},
            {0.0,1.0},
            {1.0,0x15bf0a8b14576ap-51},
            {1.0,+infinity},
            {1.0,1.0},
            {1.0,0x148b5e3c3e8187p-52},
            {0x10000000000140p-52,0x10000000000161p-52},
            {0.0,+infinity},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_exp<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 3;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "mpfi_exp2_exp2"_test = [&] {
        constexpr int n = 13;
        std::array<I, n> h_xs {{
            {-0.125,0.0},
            {-0.125,0.25},
            {-123.0,-17.0},
            {-7.0,7.0},
            {-infinity,-1.0},
            {-infinity,0.0},
            {-infinity,1.0},
            {0.0,+1.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.25},
            {0xap-47,0xbp-47},
            entire,
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {0x1d5818dcfba487p-53,1.0},
            {0x1d5818dcfba487p-53,0x1306fe0a31b716p-52},
            {0x1p-123,0x1p-17},
            {0x1p-7,0x1p+7},
            {0.0,0.5},
            {0.0,1.0},
            {0.0,2.0},
            {1.0,2.0},
            {1.0,+infinity},
            {1.0,1.0},
            {1.0,0x1306fe0a31b716p-52},
            {0x100000000000ddp-52,0x100000000000f4p-52},
            {0.0,+infinity},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_exp2<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 3;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "mpfi_expm1_expm1"_test = [&] {
        constexpr int n = 12;
        std::array<I, n> h_xs {{
            {-0.125,0.0},
            {-0.125,0.25},
            {-36.0,-36.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {-infinity,1.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.25},
            {0.0,1.0},
            {0xap-47,0xbp-47},
            entire,
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-0x1e14aed893eef4p-56,0.0},
            {-0x1e14aed893eef4p-56,0x122d78f0fa061ap-54},
            {-0x1ffffffffffffep-53,-0x1ffffffffffffdp-53},
            {-1.0,-0x1ff887a518f6d5p-53},
            {-1.0,0.0},
            {-1.0,0x1b7e151628aed3p-52},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0x122d78f0fa061ap-54},
            {0.0,0x1b7e151628aed3p-52},
            {0x140000000000c8p-96,0x160000000000f3p-96},
            {-1.0,+infinity},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_expm1<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 3;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "mpfi_intersec_intersection"_test = [&] {
        constexpr int n = 14;
        std::array<I, n> h_xs {{
            {-infinity,+8.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {0.0,+8.0},
            {0.0,+8.0},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0x12p0,0x90p0},
            entire,
        }};

        std::array<I, n> h_ys {{
            {0.0,+8.0},
            {-1.0,+8.0},
            {+8.0,+infinity},
            {-7.0,+8.0},
            {-7.0,0.0},
            {0.0,+8.0},
            {0.0,+8.0},
            {+8.0,+infinity},
            {-infinity,-7.0},
            {0.0,+8.0},
            {0.0,0.0},
            entire,
            {-0x0dp0,0x34p0},
            {0.0,+8.0},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {0.0,+8.0},
            empty,
            empty,
            {0.0,+8.0},
            {0.0,0.0},
            {0.0,+8.0},
            {0.0,+8.0},
            empty,
            empty,
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0x12p0,0x34p0},
            {0.0,+8.0},
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

    "mpfi_inv_recip"_test = [&] {
        constexpr int n = 11;
        std::array<I, n> h_xs {{
            {-0xae83b95effd69p-52,-0x63e3cb4ed72a3p-53},
            {-8.0,-2.0},
            {-infinity,+4.0},
            {-infinity,-.25},
            {-infinity,0.0},
            {0.0,+2.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0x19f1a539c91fddp-55,+64.0},
            {0x1p-4,0x1440c131282cd9p-53},
            entire,
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-0x1480a9b5772a23p-50,-0x177887d65484c9p-52},
            {-.5,-0.125},
            entire,
            {-4.0,0.0},
            {-infinity,0.0},
            {+.5,+infinity},
            {0.0,+infinity},
            empty,
            {0.015625,0x13bc205a76b3fdp-50},
            {0x1947bfce1bc417p-52,0x10p0},
            entire,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_recip<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "mpfi_is_neg_precedes"_test = [&] {
        constexpr int n = 16;
        std::array<I, n> h_xs {{
            {+0x1fffffffffffffp-53,2.0},
            {+8.0,+0x7fffffffffffbp+51},
            {-0x1921fb54442d18p-51,0x1921fb54442d19p-51},
            {-34.0,-17.0},
            {-34.0,17.0},
            {-8.0,-1.0},
            {-8.0,0.0},
            {-infinity,-8.0},
            {-infinity,0.0},
            {-infinity,5.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,5.0},
            {0x1921fb54442d18p-51,0x1921fb54442d19p-51},
            {5.0,+infinity},
            entire,
        }};

        std::array<I, n> h_ys {{
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
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
            true,
            false,
            true,
            true,
            true,
            true,
            false,
            false,
            true,
            false,
            false,
            false,
            false,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_precedes<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<B, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "mpfi_is_nonneg_less"_test = [&] {
        constexpr int n = 16;
        std::array<I, n> h_xs {{
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
        }};

        std::array<I, n> h_ys {{
            {+0x1fffffffffffffp-53,2.0},
            {+8.0,+0x7fffffffffffbp+51},
            {-0x1921fb54442d18p-51,0x1921fb54442d19p-51},
            {-34.0,-17.0},
            {-34.0,17.0},
            {-8.0,-1.0},
            {-8.0,0.0},
            {-infinity,-8.0},
            {-infinity,0.0},
            {-infinity,5.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,5.0},
            {0x1921fb54442d18p-51,0x1921fb54442d19p-51},
            {5.0,+infinity},
            entire,
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
            false,
            false,
            false,
            false,
            false,
            true,
            true,
            true,
            true,
            true,
            false,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_less<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<B, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "mpfi_is_nonpo_less"_test = [&] {
        constexpr int n = 16;
        std::array<I, n> h_xs {{
            {-0x1921fb54442d18p-51,0x1921fb54442d19p-51},
            {-34.0,-17.0},
            {-34.0,17.0},
            {-8.0,-1.0},
            {-8.0,0.0},
            {-infinity,-8.0},
            {-infinity,0.0},
            {-infinity,5.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,5.0},
            {0x1921fb54442d18p-51,0x1921fb54442d19p-51},
            {0x1fffffffffffffp-53,2.0},
            {5.0,+infinity},
            {8.0,0x7fffffffffffbp+51},
            entire,
        }};

        std::array<I, n> h_ys {{
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
        }};

        std::array<B, n> h_res{};
        B *d_res = (B *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(B);
        std::array<B, n> h_ref {{
            false,
            true,
            false,
            true,
            true,
            true,
            true,
            false,
            false,
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
        test_less<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<B, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "mpfi_is_po_precedes"_test = [&] {
        constexpr int n = 16;
        std::array<I, n> h_xs {{
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
        }};

        std::array<I, n> h_ys {{
            {+0x1fffffffffffffp-53,2.0},
            {+8.0,+0x7fffffffffffbp+51},
            {-0x1921fb54442d18p-51,0x1921fb54442d19p-51},
            {-34.0,-17.0},
            {-34.0,17.0},
            {-8.0,-1.0},
            {-8.0,0.0},
            {-infinity,-8.0},
            {-infinity,0.0},
            {-infinity,5.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,5.0},
            {0x1921fb54442d18p-51,0x1921fb54442d19p-51},
            {5.0,+infinity},
            entire,
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
            false,
            false,
            false,
            false,
            false,
            true,
            true,
            true,
            true,
            true,
            false,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_precedes<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<B, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "mpfi_is_strictly_neg_strictPrecedes"_test = [&] {
        constexpr int n = 16;
        std::array<I, n> h_xs {{
            {+0x1fffffffffffffp-53,2.0},
            {+8.0,+0x7fffffffffffbp+51},
            {-0x1921fb54442d18p-51,0x1921fb54442d19p-51},
            {-34.0,-17.0},
            {-34.0,17.0},
            {-8.0,-1.0},
            {-8.0,0.0},
            {-infinity,-8.0},
            {-infinity,0.0},
            {-infinity,5.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,5.0},
            {0x1921fb54442d18p-51,0x1921fb54442d19p-51},
            {5.0,+infinity},
            entire,
        }};

        std::array<I, n> h_ys {{
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
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
            true,
            false,
            true,
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
        test_strictPrecedes<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<B, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "mpfi_is_strictly_po_strictPrecedes"_test = [&] {
        constexpr int n = 16;
        std::array<I, n> h_xs {{
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
        }};

        std::array<I, n> h_ys {{
            {+0x1fffffffffffffp-53,2.0},
            {+8.0,+0x7fffffffffffbp+51},
            {-0x1921fb54442d18p-51,0x1921fb54442d19p-51},
            {-34.0,-17.0},
            {-34.0,17.0},
            {-8.0,-1.0},
            {-8.0,0.0},
            {-infinity,-8.0},
            {-infinity,0.0},
            {-infinity,5.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,5.0},
            {0x1921fb54442d18p-51,0x1921fb54442d19p-51},
            {5.0,+infinity},
            entire,
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
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            false,
            true,
            true,
            false,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_strictPrecedes<<<numBlocks, blockSize>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<B, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\ny = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub, h_ys[fail_id].lb, h_ys[fail_id].ub);
        }
    };

    "mpfi_log_log"_test = [&] {
        constexpr int n = 7;
        std::array<I, n> h_xs {{
            {+1.0,+1.0},
            {+1.0,0x8ac74d932fae3p-21},
            {0.0,+1.0},
            {0.0,+infinity},
            {0x3a2a08c23afe3p-14,0x1463ceb440d6bdp-14},
            {0x4c322657ec89bp-16,0x4d68ba5f26bf1p-11},
            {0xb616ab8b683b5p-52,+1.0},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {0.0,0.0},
            {0.0,0x5380455576989p-46},
            {-infinity,0.0},
            entire,
            {0xc6dc8a2928579p-47,0x1a9500bc7ffcc5p-48},
            {0xbdee7228cfedfp-47,0x1b3913fc99f555p-48},
            {-0x2b9b8b1fb2fb9p-51,0.0},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_log<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 3;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "mpfi_log1p_log1p"_test = [&] {
        constexpr int n = 7;
        std::array<I, n> h_xs {{
            {-0xb616ab8b683b5p-52,0.0},
            {-1.0,0.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0x8ac74d932fae3p-21},
            {0.0,1.0},
            {0x4c322657ec89bp-16,0x4d68ba5f26bf1p-11},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-0x13e080325bab7bp-52,0.0},
            {-infinity,0.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0x14e0115561569cp-48},
            {0.0,0x162e42fefa39f0p-53},
            {0x17bdce451a337fp-48,0x1b3913fc99f6fcp-48},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_log1p<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 3;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "mpfi_log2_log2"_test = [&] {
        constexpr int n = 6;
        std::array<I, n> h_xs {{
            {0.0,+1.0},
            {0.0,+infinity},
            {0x4c322657ec89bp-16,0x4d68ba5f26bf1p-11},
            {0xb616ab8b683b5p-52,1.0},
            {1.0,0x8ac74d932fae3p-21},
            {1.0,1.0},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-infinity,0.0},
            entire,
            {0x112035c9390c07p-47,0x13a3208f61f10cp-47},
            {-0x1f74cb5d105b3ap-54,0.0},
            {0.0,0x1e1ddc27c2c70fp-48},
            {0.0,0.0},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_log2<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 3;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "mpfi_log10_log10"_test = [&] {
        constexpr int n = 7;
        std::array<I, n> h_xs {{
            {0.0,+infinity},
            {0.0,1.0},
            {0x3a2a08c23afe3p-14,0x1463ceb440d6bdp-14},
            {0x4c322657ec89bp-16,0x4d68ba5f26bf1p-11},
            {0xb616ab8b683b5p-52,1.0},
            {1.0,1.0},
            {100.0,0x8ac74d932fae3p-21},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            entire,
            {-infinity,0.0},
            {0x159753104a9401p-49,0x1716c01a04b570p-49},
            {0x149f1d70168f49p-49,0x17a543a94fb65ep-49},
            {-0x12f043ec00f8d6p-55,0.0},
            {0.0,0.0},
            {2.0,0x1221cc590b9946p-49},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_log10<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 3;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "mpfi_mag_mag"_test = [&] {
        constexpr int n = 10;
        std::array<I, n> h_xs {{
            {-34.0,-17.0},
            {-8.0,0.0},
            {-infinity,-8.0},
            {-infinity,0.0},
            {-infinity,0.0},
            {-infinity,5.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,5.0},
            entire,
        }};

        std::array<T, n> h_res{};
        T *d_res = (T *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(T);
        std::array<T, n> h_ref {{
            34,
            +8,
            +infinity,
            +infinity,
            +infinity,
            +infinity,
            +infinity,
            +0,
            +5,
            +infinity,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_mag<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<T, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "mpfi_mid_mid"_test = [&] {
        constexpr int n = 11;
        std::array<I, n> h_xs {{
            {-0x1921fb54442d19p-51,-0x1921fb54442d18p-51},
            {-0x1fffffffffffffp-53,2.0},
            {-34.0,-17.0},
            {-34.0,17.0},
            {-4.0,-0x7fffffffffffdp-51},
            {-8.0,-0x7fffffffffffbp-51},
            {-8.0,0.0},
            {0.0,+0x123456789abcdp-2},
            {0.0,0.0},
            {0.0,5.0},
            {0x1921fb54442d18p-51,0x1921fb54442d19p-51},
        }};

        std::array<T, n> h_res{};
        T *d_res = (T *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(T);
        std::array<T, n> h_ref {{
            -0x1921fb54442d18p-51,
            0.5,
            -0x33p-1,
            -8.5,
            -0x27fffffffffffbp-52,
            -0x47fffffffffffbp-52,
            -4,
            +0x123456789abcdp-3,
            +0,
            +2.5,
            0x1921fb54442d18p-51,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_mid<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<T, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "mpfi_mig_mig"_test = [&] {
        constexpr int n = 10;
        std::array<I, n> h_xs {{
            {-34.0,-17.0},
            {-8.0,0.0},
            {-infinity,-8.0},
            {-infinity,0.0},
            {-infinity,0.0},
            {-infinity,5.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,5.0},
            entire,
        }};

        std::array<T, n> h_res{};
        T *d_res = (T *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(T);
        std::array<T, n> h_ref {{
            17,
            +0,
            8,
            +0,
            +0,
            +0,
            +0,
            +0,
            +0,
            +0,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_mig<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 0;
        auto failed = check_all_equal<T, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "mpfi_mul_mul"_test = [&] {
        constexpr int n = 50;
        std::array<I, n> h_xs {{
            {-0x01p0,0x1.90aa487ecf153p+0},
            {-0x01p0,0x10p0},
            {-0x01p0,0x11p0},
            {-0x01p0,0x2.db091cea593fap-4},
            {-0x01p0,0x6.e211fefc216ap-4},
            {-0x01p0,0xe.ca7ddfdb8572p-4},
            {-0x04p0,-0xa.497d533c3b2ep-8},
            {-0x07p0,0x07p0},
            {-0x0dp0,-0x09p0},
            {-0x0dp0,-0xd.f0e7927d247cp-4},
            {-0x1.15e079e49a0ddp+0,0x1p-8},
            {-0x1.1d069e75e8741p+8,0x01p0},
            {-0x1.c40db77f2f6fcp+0,0x1.8eb70bbd94478p+0},
            {-0x1.cb540b71699a8p+4,-0x0.33p0},
            {-0x1.cb540b71699a8p+4,-0x0.33p0},
            {-0x123456789ap0,-0x01p0},
            {-0x37p0,-0x07p0},
            {-0xa.8071f870126cp-4,0x10p0},
            {-0xb.6c67d3a37d54p-4,-0x0.8p0},
            {-0xb.6c67d3a37d54p-4,-0xa.497d533c3b2ep-8},
            {-0xe.063f267ed51ap-4,-0x0.33p0},
            {-0xe.26c9e9eb67b48p-4,-0x8.237d2eb8b1178p-4},
            {-0xe.ca7ddfdb8572p-4,0x1.1d069e75e8741p+8},
            {-3.0,+7.0},
            {-infinity,+8.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {0.0,+8.0},
            {0.0,+8.0},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0x0.3p0,0xa.a97267f56a9b8p-4},
            {0x01p0,0xcp0},
            {0x03p0,0x7.2bea531ef4098p+0},
            {0x123p-52,0x1.ec24910ac6aecp+0},
            {0x2.48380232f6c16p+0,0x7p0},
            {0x3.10e8a605572p-4,0x2.48380232f6c16p+0},
            {0x3p0,0x3.71cb6c53e68eep+0},
            {0x3p0,0x7p0},
            {0xb.38f1fb0ef4308p+0,0x2dp0},
            {0xcp0,0x1.1833fdcab4c4ap+10},
            {0xcp0,0x2dp0},
            {0xf.08367984ca1cp-4,0xa.bcf6c6cbe341p+0},
            entire,
            entire,
        }};

        std::array<I, n> h_ys {{
            {0x01p-53,0x1.442e2695ac81ap+0},
            {-0x02p0,0x03p0},
            {-0x07p0,-0x04p0},
            {-0x2.6bff2625fb71cp-4,0x1p-8},
            {-0x1p-4,0x1.8e3fe93a4ea52p+0},
            {-0x2.3b46226145234p+0,-0x0.1p0},
            {0xb.d248df3373e68p-4,0x04p0},
            {0x13p0,0x24p0},
            {-0x04p0,-0x02p0},
            {-0x04p0,-0xa.41084aff48f8p-8},
            {-0x2.77fc84629a602p+0,0x8.3885932f13fp-4},
            {-0x2.3b46226145234p+0,-0x0.1p0},
            {0x02p0,0x3.45118635235c6p+0},
            {-0x1.64dcaaa101f18p+0,0x01p0},
            {-0x1.64dcaaa101f18p+0,0x1.eb67a1a6ef725p+4},
            {0x01p0,0x10p0},
            {-0x01p0,0x22p0},
            {0x02p0,0x2.3381083e7d3b4p+0},
            {0x02p0,0x2.0bee4e8bb3dfp+0},
            {0xb.d248df3373e68p-4,0x2.0bee4e8bb3dfp+0},
            {-0x01p0,0x1.777ab178b4a1ep+0},
            {-0x5.8c899a0706d5p-4,-0x3.344e57a37b5e8p-4},
            {-0x2.3b46226145234p+0,-0x0.1p0},
            {0.0,0.0},
            {0.0,+8.0},
            {-1.0,+8.0},
            {+8.0,+infinity},
            {-7.0,+8.0},
            {-7.0,0.0},
            {0.0,+8.0},
            {0.0,+8.0},
            {+8.0,+infinity},
            {-infinity,-7.0},
            {0.0,+8.0},
            {0.0,0.0},
            entire,
            {-0x1.ec24910ac6aecp+0,0x7.2bea531ef4098p+0},
            {-0xe5p0,0x01p0},
            {-0x01p0,0xa.a97267f56a9b8p-4},
            {-0xa.a97267f56a9b8p-4,0x1p+32},
            {0x3.71cb6c53e68eep+0,0xbp0},
            {0xc.3d8e305214ecp-4,0x2.9e7db05203c88p+0},
            {0x5p-25,0x2.48380232f6c16p+0},
            {0x5p0,0xbp0},
            {-0x679p0,-0xa.4771d7d0c604p+0},
            {-0x2.4c0afc50522ccp+40,-0xe5p0},
            {-0x679p0,-0xe5p0},
            {-0x5.cbc445e9952c4p+0,-0x2.8ad05a7b988fep-8},
            {0.0,+8.0},
            {0.0,0.0},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-0x1.442e2695ac81ap+0,0x1.fb5fbebd0cbc6p+0},
            {-0x20p0,0x30p0},
            {-0x77p0,0x07p0},
            {-0x6.ea77a3ee43de8p-8,0x2.6bff2625fb71cp-4},
            {-0x1.8e3fe93a4ea52p+0,0xa.b52fe22d72788p-4},
            {-0x2.101b41d3d48b8p+0,0x2.3b46226145234p+0},
            {-0x10p0,-0x7.99b990532d434p-8},
            {-0xfcp0,0xfcp0},
            {0x12p0,0x34p0},
            {0x8.ef3aa21dba748p-8,0x34p0},
            {-0x8.ec5de73125be8p-4,0x2.adfe651d3b19ap+0},
            {-0x2.3b46226145234p+0,0x2.7c0bd9877f404p+8},
            {-0x5.c61fcad908df4p+0,0x5.17b7c49130824p+0},
            {-0x1.cb540b71699a8p+4,0x2.804cce4a3f42ep+4},
            {-0x3.71b422ce817f4p+8,0x2.804cce4a3f42ep+4},
            {-0x123456789a0p0,-0x01p0},
            {-0x74ep0,0x37p0},
            {-0x1.71dc5b5607781p+0,0x2.3381083e7d3b4p+4},
            {-0x1.7611a672948a5p+0,-0x01p0},
            {-0x1.7611a672948a5p+0,-0x7.99b990532d434p-8},
            {-0x1.491df346a9f15p+0,0xe.063f267ed51ap-4},
            {0x1.a142a930de328p-4,0x4.e86c3434cd924p-4},
            {-0x2.7c0bd9877f404p+8,0x2.101b41d3d48b8p+0},
            {0.0,0.0},
            {-infinity,+64.0},
            entire,
            {-infinity,0.0},
            {-56.0,+64.0},
            {-56.0,0.0},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {-0x1.47f2dbe4ef916p+0,0x4.c765967f9468p+0},
            {-0xabcp0,0xcp0},
            {-0x7.2bea531ef4098p+0,0x4.c765967f9468p+0},
            {-0x1.47f2dbe4ef916p+0,0x1.ec24910ac6aecp+32},
            {0x7.dc58fb323ad78p+0,0x4dp0},
            {0x2.587a32d02bc04p-4,0x5.fa216b7c20c6cp+0},
            {0xfp-25,0x7.dc58fb323ad7cp+0},
            {0xfp0,0x4dp0},
            {-0x12345p0,-0x7.35b3c8400ade4p+4},
            {-0x2.83a3712099234p+50,-0xabcp0},
            {-0x12345p0,-0xabcp0},
            {-0x3.e3ce52d4a139cp+4,-0x2.637164cf2f346p-8},
            entire,
            {0.0,0.0},
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

    "mpfi_mul_d_mul"_test = [&] {
        constexpr int n = 45;
        std::array<I, n> h_xs {{
            {-0x10000000000001p0,-0x1aaaaaaaaaaaaap-123},
            {-0x10000000000001p0,-0xaaaaaaaaaaaaap-123},
            {-0x10000000000001p0,0x10000000000001p0},
            {-0x10000000000001p0,0x1717170p+401},
            {-0x11717171717171p0,-0x10000000000001p0},
            {-0x11717171717171p0,-0xaaaaaaaaaaaaap-123},
            {-0x1717170p+36,-0x10000000000001p0},
            {-0x1717170p0,-0x1aaaaaaaaaaaaap-123},
            {-0x1717170p0,-0xaaaaaaaaaaaaap-123},
            {-0x1717170p0,-0xaaaaaaaaaaaaap-123},
            {-0xaaaaaaaaaaaaap0,0x10000000000001p0},
            {-0xaaaaaaaaaaaaap0,0x11717171717171p0},
            {-0xaaaaaaaaaaaaap0,0x1717170p+401},
            {-0xaaaaaaaaaaaaap0,0x1717170p+401},
            {-0xaaaaaaaaaaaabp0,0x11717171717171p0},
            {-0xaaaaaaaaaaaabp0,0x1717170p+401},
            {-infinity,-7.0},
            {-infinity,-7.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {-infinity,0.0},
            {-infinity,0.0},
            {-infinity,8.0},
            {-infinity,8.0},
            {-infinity,8.0},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,7.0},
            {0.0,8.0},
            {0.0,9.0},
            {0x10000000000001p0,0x11111111111111p0},
            {0x10000000000001p0,0x18888888888889p0},
            {0x10000000000001p0,0x888888888888p+654},
            {0x10000000000001p0,0x888888888888p+654},
            {0x10000000000010p0,0x11111111111111p0},
            {0x10000000000010p0,0x18888888888889p0},
            {0x10000000000010p0,0x888888888888p+654},
            {0x10000000000010p0,0x888888888888p+654},
            entire,
            entire,
            entire,
        }};

        std::array<I, n> h_ys {{
            {1.5,1.5},
            {1.5,1.5},
            {-1.5,-1.5},
            {-1.5,-1.5},
            {-1.5,-1.5},
            {-1.5,-1.5},
            {-1.5,-1.5},
            {1.5,1.5},
            {-1.5,-1.5},
            {1.5,1.5},
            {-1.5,-1.5},
            {1.5,1.5},
            {-1.5,-1.5},
            {1.5,1.5},
            {1.5,1.5},
            {1.5,1.5},
            {-0x17p0,-0x17p0},
            {0.0,0.0},
            {0x170ef54646d497p-107,0x170ef54646d497p-107},
            {-0x170ef54646d497p-106,-0x170ef54646d497p-106},
            {0.0,0.0},
            {0x170ef54646d497p-106,0x170ef54646d497p-106},
            {-0x16345785d8a00000p0,-0x16345785d8a00000p0},
            {0.0,0.0},
            {0x16345785d8a00000p0,0x16345785d8a00000p0},
            {-0x50b45a75f7e81p-104,-0x50b45a75f7e81p-104},
            {0.0,0.0},
            {0x142d169d7dfa03p-106,0x142d169d7dfa03p-106},
            {-0x170ef54646d497p-109,-0x170ef54646d497p-109},
            {0.0,0.0},
            {0x170ef54646d497p-109,0x170ef54646d497p-109},
            {-0x114b37f4b51f71p-107,-0x114b37f4b51f71p-107},
            {0.0,0.0},
            {0x114b37f4b51f71p-103,0x114b37f4b51f71p-103},
            {-2.125,-2.125},
            {2.125,2.125},
            {-2.125,-2.125},
            {2.125,2.125},
            {-2.125,-2.125},
            {2.125,2.125},
            {-2.125,-2.125},
            {2.125,2.125},
            {+0x170ef54646d497p-105,+0x170ef54646d497p-105},
            {-0x170ef54646d497p-105,-0x170ef54646d497p-105},
            {0.0e-17,0.0e-17},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-0x18000000000002p0,-0x27fffffffffffep-123},
            {-0x18000000000002p0,-0xfffffffffffffp-123},
            {-0x18000000000002p0,0x18000000000002p0},
            {-0x22a2a28p+401,0x18000000000002p0},
            {0x18000000000001p0,0x1a2a2a2a2a2a2ap0},
            {0xfffffffffffffp-123,0x1a2a2a2a2a2a2ap0},
            {0x18000000000001p0,0x22a2a28p+36},
            {-0x22a2a28p0,-0x27fffffffffffep-123},
            {0xfffffffffffffp-123,0x22a2a28p0},
            {-0x22a2a28p0,-0xfffffffffffffp-123},
            {-0x18000000000002p0,0xfffffffffffffp0},
            {-0xfffffffffffffp0,0x1a2a2a2a2a2a2ap0},
            {-0x22a2a28p+401,0xfffffffffffffp0},
            {-0xfffffffffffffp0,0x22a2a28p+401},
            {-0x10000000000001p0,0x1a2a2a2a2a2a2ap0},
            {-0x10000000000001p0,0x22a2a28p+401},
            {+0xa1p0,+infinity},
            {0.0,0.0},
            {-infinity,-0xa168b4ebefd020p-107},
            {0.0,+infinity},
            {0.0,0.0},
            {-infinity,0.0},
            {-0xb1a2bc2ec5000000p0,+infinity},
            {0.0,0.0},
            {-infinity,0xb1a2bc2ec5000000p0},
            {-infinity,0.0},
            {0.0,0.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {-0x790e87b0f3dc18p-107,0.0},
            {0.0,0.0},
            {0.0,0x9ba4f79a5e1b00p-103},
            {-0x12222222222223p+1,-0x22000000000002p0},
            {0x22000000000002p0,0x34222222222224p0},
            {-0x1222222222221p+654,-0x22000000000002p0},
            {0x22000000000002p0,0x1222222222221p+654},
            {-0x12222222222223p+1,-0x22000000000022p0},
            {0x22000000000022p0,0x34222222222224p0},
            {-0x1222222222221p+654,-0x22000000000022p0},
            {0x22000000000022p0,0x1222222222221p+654},
            entire,
            entire,
            {0.0,0.0},
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

    "mpfi_neg_neg"_test = [&] {
        constexpr int n = 8;
        std::array<I, n> h_xs {{
            {-infinity,+8.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {0.0,+8.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0x123456789p-16,0x123456799p-16},
            entire,
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-8.0,+infinity},
            {+7.0,+infinity},
            {0.0,+infinity},
            {-8.0,0.0},
            {-infinity,0.0},
            {0.0,0.0},
            {-0x123456799p-16,-0x123456789p-16},
            entire,
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

    "mpfi_put_d_convexHull"_test = [&] {
        constexpr int n = 3;
        std::array<I, n> h_xs {{
            {+5.0,+5.0},
            {0.0,0.0},
            {0.0,0.0},
        }};

        std::array<I, n> h_ys {{
            {0.0,0.0},
            {-8.0,-8.0},
            {0.0,0.0},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {0.0,+5.0},
            {-8.0,0.0},
            {0.0,0.0},
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

    "mpfi_sin_sin"_test = [&] {
        constexpr int n = 128;
        std::array<I, n> h_xs {{
            {-0.5,0.5},
            {-1.0,-0.25},
            {-1.0,-1.0},
            {-1.0,0.0},
            {-1.0,0.0},
            {-1.0,1.0},
            {-1.0,2.0},
            {-1.0,3.0},
            {-1.0,4.0},
            {-1.0,5.0},
            {-1.0,6.0},
            {-1.0,7.0},
            {-2.0,-0.5},
            {-2.0,-1.0},
            {-2.0,-2.0},
            {-2.0,0.0},
            {-2.0,1.0},
            {-2.0,2.0},
            {-2.0,3.0},
            {-2.0,4.0},
            {-2.0,5.0},
            {-2.0,6.0},
            {-2.0,7.0},
            {-3.0,-1.0},
            {-3.0,-2.0},
            {-3.0,-3.0},
            {-3.0,0.0},
            {-3.0,1.0},
            {-3.0,2.0},
            {-3.0,3.0},
            {-3.0,4.0},
            {-3.0,5.0},
            {-3.0,6.0},
            {-3.0,7.0},
            {-4.0,-1.0},
            {-4.0,-2.0},
            {-4.0,-3.0},
            {-4.0,-4.0},
            {-4.0,0.0},
            {-4.0,1.0},
            {-4.0,2.0},
            {-4.0,3.0},
            {-4.0,4.0},
            {-4.0,5.0},
            {-4.0,6.0},
            {-4.0,7.0},
            {-4.5,0.625},
            {-5.0,-1.0},
            {-5.0,-2.0},
            {-5.0,-3.0},
            {-5.0,-4.0},
            {-5.0,-5.0},
            {-5.0,0.0},
            {-5.0,1.0},
            {-5.0,2.0},
            {-5.0,3.0},
            {-5.0,4.0},
            {-5.0,5.0},
            {-5.0,6.0},
            {-5.0,7.0},
            {-6.0,-1.0},
            {-6.0,-2.0},
            {-6.0,-3.0},
            {-6.0,-4.0},
            {-6.0,-5.0},
            {-6.0,-6.0},
            {-6.0,0.0},
            {-6.0,1.0},
            {-6.0,2.0},
            {-6.0,3.0},
            {-6.0,4.0},
            {-6.0,5.0},
            {-6.0,6.0},
            {-6.0,7.0},
            {-7.0,-1.0},
            {-7.0,-2.0},
            {-7.0,-3.0},
            {-7.0,-4.0},
            {-7.0,-5.0},
            {-7.0,-6.0},
            {-7.0,-7.0},
            {-7.0,0.0},
            {-7.0,1.0},
            {-7.0,2.0},
            {-7.0,3.0},
            {-7.0,4.0},
            {-7.0,5.0},
            {-7.0,6.0},
            {-7.0,7.0},
            {-infinity,+8.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {0.0,+1.0},
            {0.0,+8.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.125,17.0},
            {0x1921fb54442d18p-52,0x1921fb54442d19p-52},
            {0x71p+76,0x71p+76},
            {1.0,1.0},
            {1.0,2.0},
            {1.0,3.0},
            {1.0,4.0},
            {1.0,5.0},
            {1.0,6.0},
            {1.0,7.0},
            {2.0,2.0},
            {2.0,3.0},
            {2.0,4.0},
            {2.0,5.0},
            {2.0,6.0},
            {2.0,7.0},
            {3.0,3.0},
            {3.0,4.0},
            {3.0,5.0},
            {3.0,6.0},
            {3.0,7.0},
            {4.0,4.0},
            {4.0,5.0},
            {4.0,6.0},
            {4.0,7.0},
            {5.0,5.0},
            {5.0,6.0},
            {5.0,7.0},
            {6.0,6.0},
            {6.0,7.0},
            {7.0,7.0},
            entire,
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-0x1eaee8744b05f0p-54,0x1eaee8744b05f0p-54},
            {-0x1aed548f090cefp-53,-0x1faaeed4f31576p-55},
            {-0x1aed548f090cefp-53,-0x1aed548f090ceep-53},
            {-0x1aed548f090cefp-53,0.0},
            {-0x1aed548f090cefp-53,0.0},
            {-0x1aed548f090cefp-53,0x1aed548f090cefp-53},
            {-0x1aed548f090cefp-53,1.0},
            {-0x1aed548f090cefp-53,1.0},
            {-0x1aed548f090cefp-53,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,-0x1eaee8744b05efp-54},
            {-1.0,-0x1aed548f090ceep-53},
            {-0x1d18f6ead1b446p-53,-0x1d18f6ead1b445p-53},
            {-1.0,0.0},
            {-1.0,0x1aed548f090cefp-53},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,-0x1210386db6d55bp-55},
            {-0x1d18f6ead1b446p-53,-0x1210386db6d55bp-55},
            {-0x1210386db6d55cp-55,-0x1210386db6d55bp-55},
            {-1.0,0.0},
            {-1.0,0x1aed548f090cefp-53},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,0x1837b9dddc1eafp-53},
            {-0x1d18f6ead1b446p-53,0x1837b9dddc1eafp-53},
            {-0x1210386db6d55cp-55,0x1837b9dddc1eafp-53},
            {0x1837b9dddc1eaep-53,0x1837b9dddc1eafp-53},
            {-1.0,0x1837b9dddc1eafp-53},
            {-1.0,0x1aed548f090cefp-53},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,0x1f47ed3dc74081p-53},
            {-1.0,1.0},
            {-0x1d18f6ead1b446p-53,1.0},
            {-0x1210386db6d55cp-55,1.0},
            {0x1837b9dddc1eaep-53,1.0},
            {0x1eaf81f5e09933p-53,0x1eaf81f5e09934p-53},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-0x1d18f6ead1b446p-53,1.0},
            {-0x1210386db6d55cp-55,1.0},
            {0x11e1f18ab0a2c0p-54,1.0},
            {0x11e1f18ab0a2c0p-54,0x1eaf81f5e09934p-53},
            {0x11e1f18ab0a2c0p-54,0x11e1f18ab0a2c1p-54},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-0x1d18f6ead1b446p-53,1.0},
            {-0x150608c26d0a09p-53,1.0},
            {-0x150608c26d0a09p-53,1.0},
            {-0x150608c26d0a09p-53,0x1eaf81f5e09934p-53},
            {-0x150608c26d0a09p-53,0x11e1f18ab0a2c1p-54},
            {-0x150608c26d0a09p-53,-0x150608c26d0a08p-53},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {0.0,0x1aed548f090cefp-53},
            {-1.0,1.0},
            {-1.0,1.0},
            {0.0,0.0},
            {-1.0,1.0},
            {0x1fffffffffffffp-53,1.0},
            {0x1bde6c11cbfc46p-55,0x1bde6c11cbfc47p-55},
            {0x1aed548f090ceep-53,0x1aed548f090cefp-53},
            {0x1aed548f090ceep-53,1.0},
            {0x1210386db6d55bp-55,1.0},
            {-0x1837b9dddc1eafp-53,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {-1.0,1.0},
            {0x1d18f6ead1b445p-53,0x1d18f6ead1b446p-53},
            {0x1210386db6d55bp-55,0x1d18f6ead1b446p-53},
            {-0x1837b9dddc1eafp-53,0x1d18f6ead1b446p-53},
            {-1.0,0x1d18f6ead1b446p-53},
            {-1.0,0x1d18f6ead1b446p-53},
            {-1.0,0x1d18f6ead1b446p-53},
            {0x1210386db6d55bp-55,0x1210386db6d55cp-55},
            {-0x1837b9dddc1eafp-53,0x1210386db6d55cp-55},
            {-1.0,0x1210386db6d55cp-55},
            {-1.0,0x1210386db6d55cp-55},
            {-1.0,0x150608c26d0a09p-53},
            {-0x1837b9dddc1eafp-53,-0x1837b9dddc1eaep-53},
            {-1.0,-0x1837b9dddc1eaep-53},
            {-1.0,-0x11e1f18ab0a2c0p-54},
            {-1.0,0x150608c26d0a09p-53},
            {-0x1eaf81f5e09934p-53,-0x1eaf81f5e09933p-53},
            {-0x1eaf81f5e09934p-53,-0x11e1f18ab0a2c0p-54},
            {-0x1eaf81f5e09934p-53,0x150608c26d0a09p-53},
            {-0x11e1f18ab0a2c1p-54,-0x11e1f18ab0a2c0p-54},
            {-0x11e1f18ab0a2c1p-54,0x150608c26d0a09p-53},
            {0x150608c26d0a08p-53,0x150608c26d0a09p-53},
            {-1.0,1.0},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_sin<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 2;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "mpfi_sinh_sinh"_test = [&] {
        constexpr int n = 13;
        std::array<I, n> h_xs {{
            {-0.125,0.0},
            {-1.0,0.0},
            {-4.5,-0.625},
            {-infinity,+8.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {0.0,+1.0},
            {0.0,+8.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0x10000000000001p-53},
            {1.0,3.0},
            entire,
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-0x100aaccd00d2f1p-55,0.0},
            {-0x12cd9fc44eb983p-52,0.0},
            {-0x168062ab5fa9fdp-47,-0x1553e795dc19ccp-53},
            {-infinity,0x1749ea514eca66p-42},
            {-infinity,-0x1122876ba380c9p-43},
            {-infinity,0.0},
            {0.0,0x12cd9fc44eb983p-52},
            {0.0,0x1749ea514eca66p-42},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0x10acd00fe63b98p-53},
            {0x12cd9fc44eb982p-52,0x140926e70949aep-49},
            entire,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_sinh<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 3;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "mpfi_sqr_sqr"_test = [&] {
        constexpr int n = 11;
        std::array<I, n> h_xs {{
            {-0x1.64722ad2480c9p+0,0x1p0},
            {-infinity,+8.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {0.0,+8.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0x1.6b079248747a2p+0,0x2.b041176d263f6p+0},
            {0x6.61485c33c0b14p+4,0x123456p0},
            {0x8.6374d8p-4,0x3.f1d929p+8},
            entire,
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {0.0,0x1.f04dba0302d4dp+0},
            {0.0,+infinity},
            {+49.0,+infinity},
            {0.0,+infinity},
            {0.0,+64.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0x2.02ce7912cddf6p+0,0x7.3a5dee779527p+0},
            {0x2.8b45c3cc03ea6p+12,0x14b66cb0ce4p0},
            {0x4.65df11464764p-4,0xf.8f918d688891p+16},
            {0.0,+infinity},
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

    "mpfi_sqr_sqrt"_test = [&] {
        constexpr int n = 7;
        std::array<I, n> h_xs {{
            {0.0,+9.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0xa.aa1p-4,0x1.0c348f804c7a9p+0},
            {0xaaa1p0,0x14b66cb0ce4p0},
            {0xe.49ae7969e41bp-4,0x1.0c348f804c7a9p+0},
            {0xe.49ae7969e41bp-4,0xaaa1p0},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {0.0,+3.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0xd.1p-4,0x1.06081714eef1dp+0},
            {0xd1p0,0x123456p0},
            {0xf.1ea42821b27a8p-4,0x1.06081714eef1dp+0},
            {0xf.1ea42821b27a8p-4,0xd1p0},
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

    "mpfi_sub_sub"_test = [&] {
        constexpr int n = 19;
        std::array<I, n> h_xs {{
            {-0x1000100010001p+8,0x1p+60},
            {-0x1p-300,0x123456p+28},
            {-4.0,7.0},
            {-5.0,1.0},
            {-5.0,59.0},
            {-infinity,+8.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {0.0,+8.0},
            {0.0,+8.0},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {5.0,0x1p+70},
            entire,
        }};

        std::array<I, n> h_ys {{
            {-3e300,0x1000100010001p0},
            {-0x789abcdp0,0x10000000000000p-93},
            {-3e300,0x123456789abcdp-17},
            {1.0,0x1p+70},
            {17.0,81.0},
            {0.0,+8.0},
            {-1.0,+8.0},
            {+8.0,+infinity},
            {-7.0,+8.0},
            {-7.0,0.0},
            {0.0,+8.0},
            {0.0,+8.0},
            {+8.0,+infinity},
            {-infinity,-7.0},
            {0.0,+8.0},
            {0.0,0.0},
            entire,
            {3.0,5.0},
            {0.0,+8.0},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-0x10101010101011p+4,0x8f596b3002c1bp+947},
            {-0x10000000000001p-93,0x123456789abcdp0},
            {-0x123456791abcdp-17,0x8f596b3002c1bp+947},
            {-0x10000000000001p+18,0.0},
            {-86.0,42.0},
            {-infinity,+8.0},
            {-infinity,-6.0},
            {-infinity,-8.0},
            {-8.0,+15.0},
            {0.0,+15.0},
            {-8.0,+infinity},
            {-8.0,+infinity},
            {-infinity,-8.0},
            {+7.0,+infinity},
            {-8.0,0.0},
            {0.0,0.0},
            entire,
            {0.0,0x1p+70},
            entire,
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

    "mpfi_sub_d_sub"_test = [&] {
        constexpr int n = 32;
        std::array<I, n> h_xs {{
            {-0x1fffffffffffffp-52,-0x1p-550},
            {-0x1fffffffffffffp-52,-0x1p-550},
            {-0xfb53d14aa9c2fp-47,-17.0},
            {-0xffp0,0x123456789abcdfp-52},
            {-0xffp0,0x123456789abcdfp-52},
            {-32.0,-0xfb53d14aa9c2fp-48},
            {-32.0,-17.0},
            {-infinity,-7.0},
            {-infinity,-7.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {-infinity,0.0},
            {-infinity,0.0},
            {-infinity,8.0},
            {-infinity,8.0},
            {-infinity,8.0},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,8.0},
            {0.0,8.0},
            {0.0,8.0},
            {0x123456789abcdfp-48,0x123456789abcdfp-4},
            {0x123456789abcdfp-48,0x123456789abcdfp-4},
            {0x123456789abcdfp-56,0x123456789abcdfp-4},
            {0x123456789abcdfp-56,0x123456789abcdfp-4},
            entire,
            entire,
            entire,
        }};

        std::array<I, n> h_ys {{
            {-4097.5,-4097.5},
            {4097.5,4097.5},
            {-0xfb53d14aa9c2fp-47,-0xfb53d14aa9c2fp-47},
            {-256.5,-256.5},
            {256.5,256.5},
            {-0xfb53d14aa9c2fp-48,-0xfb53d14aa9c2fp-48},
            {0xfb53d14aa9c2fp-47,0xfb53d14aa9c2fp-47},
            {-0x170ef54646d497p-107,-0x170ef54646d497p-107},
            {0.0,0.0},
            {0x170ef54646d497p-107,0x170ef54646d497p-107},
            {-0x170ef54646d497p-106,-0x170ef54646d497p-106},
            {0.0,0.0},
            {0x170ef54646d497p-106,0x170ef54646d497p-106},
            {-0x16345785d8a00000p0,-0x16345785d8a00000p0},
            {0.0,0.0},
            {0x16345785d8a00000p0,0x16345785d8a00000p0},
            {-0x50b45a75f7e81p-104,-0x50b45a75f7e81p-104},
            {0.0,0.0},
            {0x142d169d7dfa03p-106,0x142d169d7dfa03p-106},
            {-0x170ef54646d497p-109,-0x170ef54646d497p-109},
            {0.0,0.0},
            {0x170ef54646d497p-109,0x170ef54646d497p-109},
            {-0x114b37f4b51f71p-107,-0x114b37f4b51f71p-107},
            {0.0,0.0},
            {0x114b37f4b51f71p-107,0x114b37f4b51f71p-107},
            {-3.5,-3.5},
            {3.5,3.5},
            {-3.5,-3.5},
            {3.5,3.5},
            {+0x170ef54646d497p-105,+0x170ef54646d497p-105},
            {-0x170ef54646d497p-105,-0x170ef54646d497p-105},
            {0.0e-17,0.0e-17},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {0xfff8p-4,0x10018p-4},
            {-0x10038p-4,-0x10018p-4},
            {0.0,0x7353d14aa9c2fp-47},
            {0x18p-4,0x101a3456789abdp-44},
            {-0x1ff8p-4,-0xff5cba9876543p-44},
            {-0x104ac2eb5563d1p-48,0.0},
            {-0x1fb53d14aa9c2fp-47,-0x18353d14aa9c2fp-47},
            {-infinity,-0x1bffffffffffffp-50},
            {-infinity,-7.0},
            {-infinity,-7.0},
            {-infinity,0x170ef54646d497p-106},
            {-infinity,0.0},
            {-infinity,-8.0e-17},
            {-infinity,0x16345785d8a00100p0},
            {-infinity,8.0},
            {-infinity,-0x16345785d89fff00p0},
            {0x50b45a75f7e81p-104,+infinity},
            {0.0,+infinity},
            {-0x142d169d7dfa03p-106,+infinity},
            {+0x170ef54646d497p-109,+0x170ef54646d497p-109},
            {0.0,0.0},
            {-0x170ef54646d497p-109,-0x170ef54646d497p-109},
            {0x114b37f4b51f71p-107,0x10000000000001p-49},
            {0.0,8.0},
            {-0x114b37f4b51f71p-107,8.0},
            {0x15b456789abcdfp-48,0x123456789abd17p-4},
            {0xeb456789abcdfp-48,0x123456789abca7p-4},
            {0x3923456789abcdp-52,0x123456789abd17p-4},
            {-0x36dcba98765434p-52,0x123456789abca7p-4},
            entire,
            entire,
            entire,
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

    "mpfi_tan_tan"_test = [&] {
        constexpr int n = 128;
        std::array<I, n> h_xs {{
            {-0.5,0.5},
            {-1.0,-0.25},
            {-1.0,-1.0},
            {-1.0,0.0},
            {-1.0,0.0},
            {-1.0,1.0},
            {-1.0,2.0},
            {-1.0,3.0},
            {-1.0,4.0},
            {-1.0,5.0},
            {-1.0,6.0},
            {-1.0,7.0},
            {-2.0,-0.5},
            {-2.0,-1.0},
            {-2.0,-2.0},
            {-2.0,0.0},
            {-2.0,1.0},
            {-2.0,2.0},
            {-2.0,3.0},
            {-2.0,4.0},
            {-2.0,5.0},
            {-2.0,6.0},
            {-2.0,7.0},
            {-3.0,-1.0},
            {-3.0,-2.0},
            {-3.0,-3.0},
            {-3.0,0.0},
            {-3.0,1.0},
            {-3.0,2.0},
            {-3.0,3.0},
            {-3.0,4.0},
            {-3.0,5.0},
            {-3.0,6.0},
            {-3.0,7.0},
            {-4.0,-1.0},
            {-4.0,-2.0},
            {-4.0,-3.0},
            {-4.0,-4.0},
            {-4.0,0.0},
            {-4.0,1.0},
            {-4.0,2.0},
            {-4.0,3.0},
            {-4.0,4.0},
            {-4.0,5.0},
            {-4.0,6.0},
            {-4.0,7.0},
            {-4.5,0.625},
            {-5.0,-1.0},
            {-5.0,-2.0},
            {-5.0,-3.0},
            {-5.0,-4.0},
            {-5.0,-5.0},
            {-5.0,0.0},
            {-5.0,1.0},
            {-5.0,2.0},
            {-5.0,3.0},
            {-5.0,4.0},
            {-5.0,5.0},
            {-5.0,6.0},
            {-5.0,7.0},
            {-6.0,-1.0},
            {-6.0,-2.0},
            {-6.0,-3.0},
            {-6.0,-4.0},
            {-6.0,-5.0},
            {-6.0,-6.0},
            {-6.0,0.0},
            {-6.0,1.0},
            {-6.0,2.0},
            {-6.0,3.0},
            {-6.0,4.0},
            {-6.0,5.0},
            {-6.0,6.0},
            {-6.0,7.0},
            {-7.0,-1.0},
            {-7.0,-2.0},
            {-7.0,-3.0},
            {-7.0,-4.0},
            {-7.0,-5.0},
            {-7.0,-6.0},
            {-7.0,-7.0},
            {-7.0,0.0},
            {-7.0,1.0},
            {-7.0,2.0},
            {-7.0,3.0},
            {-7.0,4.0},
            {-7.0,5.0},
            {-7.0,6.0},
            {-7.0,7.0},
            {-infinity,+8.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {0.0,+1.0},
            {0.0,+8.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.125,17.0},
            {0x1921fb54442d18p-52,0x1921fb54442d19p-52},
            {0x71p+76,0x71p+76},
            {1.0,1.0},
            {1.0,2.0},
            {1.0,3.0},
            {1.0,4.0},
            {1.0,5.0},
            {1.0,6.0},
            {1.0,7.0},
            {2.0,2.0},
            {2.0,3.0},
            {2.0,4.0},
            {2.0,5.0},
            {2.0,6.0},
            {2.0,7.0},
            {3.0,3.0},
            {3.0,4.0},
            {3.0,5.0},
            {3.0,6.0},
            {3.0,7.0},
            {4.0,4.0},
            {4.0,5.0},
            {4.0,6.0},
            {4.0,7.0},
            {5.0,5.0},
            {5.0,6.0},
            {5.0,7.0},
            {6.0,6.0},
            {6.0,7.0},
            {7.0,7.0},
            entire,
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-0x117b4f5bf3474bp-53,0x117b4f5bf3474bp-53},
            {-0x18eb245cbee3a6p-52,-0x105785a43c4c55p-54},
            {-0x18eb245cbee3a6p-52,-0x18eb245cbee3a5p-52},
            {-0x18eb245cbee3a6p-52,0.0},
            {-0x18eb245cbee3a6p-52,0.0},
            {-0x18eb245cbee3a6p-52,0x18eb245cbee3a6p-52},
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            {0x117af62e0950f8p-51,0x117af62e0950f9p-51},
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            {0x123ef71254b86fp-55,0x117af62e0950f9p-51},
            {0x123ef71254b86fp-55,0x123ef71254b870p-55},
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            {-0x12866f9be4de14p-52,0x117af62e0950f9p-51},
            {-0x12866f9be4de14p-52,0x123ef71254b870p-55},
            {-0x12866f9be4de14p-52,-0x12866f9be4de13p-52},
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            {0x1b0b4b739bbb06p-51,0x1b0b4b739bbb07p-51},
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            {0x129fd86ebb95bep-54,0x1b0b4b739bbb07p-51},
            {0x129fd86ebb95bep-54,0x129fd86ebb95bfp-54},
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            {-0x1be2e6e13eea79p-53,0x1b0b4b739bbb07p-51},
            {-0x1be2e6e13eea79p-53,0x129fd86ebb95bfp-54},
            {-0x1be2e6e13eea79p-53,-0x1be2e6e13eea78p-53},
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            {0.0,0x18eb245cbee3a6p-52},
            entire,
            entire,
            {0.0,0.0},
            entire,
            entire,
            {-0x1c8dc87ddcc134p-55,-0x1c8dc87ddcc133p-55},
            {0x18eb245cbee3a5p-52,0x18eb245cbee3a6p-52},
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            {-0x117af62e0950f9p-51,-0x117af62e0950f8p-51},
            {-0x117af62e0950f9p-51,-0x123ef71254b86fp-55},
            {-0x117af62e0950f9p-51,0x12866f9be4de14p-52},
            entire,
            entire,
            entire,
            {-0x123ef71254b870p-55,-0x123ef71254b86fp-55},
            {-0x123ef71254b870p-55,0x12866f9be4de14p-52},
            entire,
            entire,
            entire,
            {0x12866f9be4de13p-52,0x12866f9be4de14p-52},
            entire,
            entire,
            entire,
            {-0x1b0b4b739bbb07p-51,-0x1b0b4b739bbb06p-51},
            {-0x1b0b4b739bbb07p-51,-0x129fd86ebb95bep-54},
            {-0x1b0b4b739bbb07p-51,0x1be2e6e13eea79p-53},
            {-0x129fd86ebb95bfp-54,-0x129fd86ebb95bep-54},
            {-0x129fd86ebb95bfp-54,0x1be2e6e13eea79p-53},
            {0x1be2e6e13eea78p-53,0x1be2e6e13eea79p-53},
            entire,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_tan<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 3;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "mpfi_tanh_tanh"_test = [&] {
        constexpr int n = 14;
        std::array<I, n> h_xs {{
            {-0.125,0.0},
            {-1.0,0.0},
            {-4.5,-0.625},
            {-infinity,-7.0},
            {-infinity,0.0},
            {-infinity,8.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0x10000000000001p-53},
            {0.0,1.0},
            {0.0,8.0},
            {1.0,3.0},
            {17.0,18.0},
            entire,
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-0x1fd5992bc4b835p-56,0.0},
            {-0x185efab514f395p-53,0.0},
            {-0x1ffdfa72153984p-53,-0x11bf47eabb8f95p-53},
            {-1.0,-0x1ffffc832750f1p-53},
            {-1.0,0.0},
            {-1.0,0x1fffff872a91f9p-53},
            {0.0,+1.0},
            {0.0,0.0},
            {0.0,0x1d9353d7568af5p-54},
            {0.0,0x185efab514f395p-53},
            {0.0,0x1fffff872a91f9p-53},
            {0x185efab514f394p-53,0x1fd77d111a0b00p-53},
            {0x1fffffffffffe1p-53,0x1ffffffffffffcp-53},
            {-1.0,+1.0},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));
        test_tanh<<<numBlocks, blockSize>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));
        int max_ulp_diff = 2;
        auto failed = check_all_equal<I, n>(h_res, h_ref, max_ulp_diff);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("x = [%a, %a]\n", h_xs[fail_id].lb, h_xs[fail_id].ub);
        }
    };

    "mpfi_union_convexHull"_test = [&] {
        constexpr int n = 13;
        std::array<I, n> h_xs {{
            {-infinity,+8.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {0.0,+8.0},
            {0.0,+8.0},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            entire,
        }};

        std::array<I, n> h_ys {{
            {0.0,+8.0},
            {-1.0,+8.0},
            {+8.0,+infinity},
            {-7.0,+8.0},
            {-7.0,0.0},
            {0.0,+8.0},
            {0.0,+8.0},
            {+8.0,+infinity},
            {-infinity,-7.0},
            {0.0,+8.0},
            {0.0,0.0},
            entire,
            {0.0,+8.0},
        }};

        std::array<I, n> h_res{};
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        I *d_ys = (I *)d_ys_;
        int n_result_bytes = n * sizeof(I);
        std::array<I, n> h_ref {{
            {-infinity,+8.0},
            {-infinity,+8.0},
            entire,
            {-7.0,+8.0},
            {-7.0,+8.0},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,+infinity},
            {-infinity,0.0},
            {0.0,+8.0},
            {0.0,0.0},
            entire,
            entire,
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


    CUDA_CHECK(cudaFree(d_xs_));
    CUDA_CHECK(cudaFree(d_ys_));
    CUDA_CHECK(cudaFree(d_zs_));
    CUDA_CHECK(cudaFree(d_res_));
}
