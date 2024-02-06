
#include <cuinterval/cuinterval.h>

#include "test_ops.cuh"

#include <span>
#include <ostream>

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

template<typename T, int N>
std::vector<size_t> check_all_equal(std::span<T, N> h_xs, std::span<T, N> h_ref, const std::source_location location = std::source_location::current())
{
    using namespace boost::ut;

    std::vector<size_t> failed_ids;

    for (size_t i = 0; i < h_xs.size(); ++i) {
        if (h_xs[i] != h_ref[i])
            failed_ids.push_back(i);

        expect(eq(h_xs[i], h_ref[i]), location);
    }

    return failed_ids;
}

template<typename T>
auto &operator<<(std::ostream &os, const interval<T> &x)
{
    return (os << '[' << std::hexfloat << x.lb << ',' << x.ub << ']');
}

template<typename T>
void tests() {
    using namespace boost::ut;

    using I = interval<T>;

    I empty         = ::empty<T>();
    I entire        = ::entire<T>();
    T infinity = std::numeric_limits<T>::infinity();

    const int n = 16; // count of largest test array
    const int n_bytes   = n * sizeof(I);
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;

    interval<T> *d_xs, *d_ys, *d_zs;
    CUDA_CHECK(cudaMalloc(&d_xs, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_ys, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_zs, n_bytes));

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

        std::array<I, n> h_ref {{
            {23.0,37.0},
            {23.0,37.0},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        test_add<<<numBlocks, blockSize>>>(n, d_xs, d_ys);
        CUDA_CHECK(cudaMemcpy(h_xs.data(), d_xs, n_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<I, n>(h_xs, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("y = [%a, %a]\nr = [%a, %a]\n", h_ys[fail_id].lb, h_ys[fail_id].ub, h_ref[fail_id].lb, h_ref[fail_id].ub);
        }
    };

    "cxsc.intervaladdsub_neg"_test = [&] {
        constexpr int n = 1;
        std::array<I, n> h_xs {{
            {10.0,20.0},
        }};

        std::array<I, n> h_ref {{
            {-20.0,-10.0},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        test_neg<<<numBlocks, blockSize>>>(n, d_xs);
        CUDA_CHECK(cudaMemcpy(h_xs.data(), d_xs, n_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<I, n>(h_xs, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("r = [%a, %a]\n", h_ref[fail_id].lb, h_ref[fail_id].ub);
        }
    };

    "cxsc.intervaladdsub_pos"_test = [&] {
        constexpr int n = 1;
        std::array<I, n> h_xs {{
            {10.0,20.0},
        }};

        std::array<I, n> h_ref {{
            {10.0,20.0},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        test_pos<<<numBlocks, blockSize>>>(n, d_xs);
        CUDA_CHECK(cudaMemcpy(h_xs.data(), d_xs, n_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<I, n>(h_xs, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("r = [%a, %a]\n", h_ref[fail_id].lb, h_ref[fail_id].ub);
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

        std::array<I, n> h_ref {{
            {-6.0,7.0},
            {-7.0,6.0},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        test_sub<<<numBlocks, blockSize>>>(n, d_xs, d_ys);
        CUDA_CHECK(cudaMemcpy(h_xs.data(), d_xs, n_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<I, n>(h_xs, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("y = [%a, %a]\nr = [%a, %a]\n", h_ys[fail_id].lb, h_ys[fail_id].ub, h_ref[fail_id].lb, h_ref[fail_id].ub);
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
        test_div<<<numBlocks, blockSize>>>(n, d_xs, d_ys);
        CUDA_CHECK(cudaMemcpy(h_xs.data(), d_xs, n_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<I, n>(h_xs, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("y = [%a, %a]\nr = [%a, %a]\n", h_ys[fail_id].lb, h_ys[fail_id].ub, h_ref[fail_id].lb, h_ref[fail_id].ub);
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
        test_mul<<<numBlocks, blockSize>>>(n, d_xs, d_ys);
        CUDA_CHECK(cudaMemcpy(h_xs.data(), d_xs, n_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<I, n>(h_xs, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("y = [%a, %a]\nr = [%a, %a]\n", h_ys[fail_id].lb, h_ys[fail_id].ub, h_ref[fail_id].lb, h_ref[fail_id].ub);
        }
    };

    "cxsc.intervalstdfunc_sqr"_test = [&] {
        constexpr int n = 3;
        std::array<I, n> h_xs {{
            {-9.0,-9.0},
            {0.0,0.0},
            {11.0,11.0},
        }};

        std::array<I, n> h_ref {{
            {81.0,81.0},
            {0.0,0.0},
            {121.0,121.0},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        test_sqr<<<numBlocks, blockSize>>>(n, d_xs);
        CUDA_CHECK(cudaMemcpy(h_xs.data(), d_xs, n_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<I, n>(h_xs, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("r = [%a, %a]\n", h_ref[fail_id].lb, h_ref[fail_id].ub);
        }
    };

    "cxsc.intervalstdfunc_sqrt"_test = [&] {
        constexpr int n = 3;
        std::array<I, n> h_xs {{
            {0.0,0.0},
            {121.0,121.0},
            {81.0,81.0},
        }};

        std::array<I, n> h_ref {{
            {0.0,0.0},
            {11.0,11.0},
            {9.0,9.0},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        test_sqrt<<<numBlocks, blockSize>>>(n, d_xs);
        CUDA_CHECK(cudaMemcpy(h_xs.data(), d_xs, n_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<I, n>(h_xs, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("r = [%a, %a]\n", h_ref[fail_id].lb, h_ref[fail_id].ub);
        }
    };


    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_ys));
    CUDA_CHECK(cudaFree(d_zs));
}

int main()
{
    tests<double>();
    return 0;
}
