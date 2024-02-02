
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
void check_all_equal(std::span<T, N> h_xs, std::span<T, N> h_ref)
{
    using namespace boost::ut;

    for (size_t i = 0; i < h_xs.size(); ++i) {
        expect(eq(h_xs[i], h_ref[i]));
    }
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

    const int n         = 100;
    const int n_bytes   = n * sizeof(I);
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;

    interval<T> *d_xs, *d_ys;
    CUDA_CHECK(cudaMalloc(&d_xs, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_ys, n_bytes));

    "pos"_test = [&] {
        std::array<I, n> h_xs {{
            {1.0,2.0},
            empty,
            entire,
            {1.0,infinity},
            {-infinity,-1.0},
            {0.0,2.0},
            {-0.0,2.0},
            {-2.5,-0.0},
            {-2.5,0.0},
            {-0.0,-0.0},
            {0.0,0.0},
        }};

        std::array<I, n> h_ref {{
            {1.0,2.0},
            empty,
            entire,
            {1.0,infinity},
            {-infinity,-1.0},
            {0.0,2.0},
            {0.0,2.0},
            {-2.5,0.0},
            {-2.5,0.0},
            {0.0,0.0},
            {0.0,0.0},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        test_pos<<<numBlocks, blockSize>>>(n, d_xs);
        CUDA_CHECK(cudaMemcpy(h_xs.data(), d_xs, n_bytes, cudaMemcpyDeviceToHost));
        check_all_equal<I, n>(h_xs, h_ref);
    };

    "neg"_test = [&] {
        std::array<I, n> h_xs {{
            {1.0,2.0},
            empty,
            entire,
            {1.0,infinity},
            {-infinity,1.0},
            {0.0,2.0},
            {-0.0,2.0},
            {-2.0,0.0},
            {-2.0,-0.0},
            {0.0,-0.0},
            {-0.0,-0.0},
        }};

        std::array<I, n> h_ref {{
            {-2.0,-1.0},
            empty,
            entire,
            {-infinity,-1.0},
            {-1.0,infinity},
            {-2.0,0.0},
            {-2.0,0.0},
            {0.0,2.0},
            {0.0,2.0},
            {0.0,0.0},
            {0.0,0.0},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        test_neg<<<numBlocks, blockSize>>>(n, d_xs);
        CUDA_CHECK(cudaMemcpy(h_xs.data(), d_xs, n_bytes, cudaMemcpyDeviceToHost));
        check_all_equal<I, n>(h_xs, h_ref);
    };

    "add"_test = [&] {
        std::array<I, n> h_xs {{
            empty,
            {-1.0,1.0},
            empty,
            empty,
            entire,
            entire,
            entire,
            entire,
            entire,
            {-infinity,1.0},
            {-1.0,1.0},
            {-1.0,infinity},
            {-infinity,2.0},
            {-infinity,2.0},
            {-infinity,2.0},
            {1.0,2.0},
            {1.0,2.0},
            {1.0,2.0},
            {1.0,infinity},
            {1.0,infinity},
            {1.0,infinity},
            {1.0,std::numeric_limits<T>::max()},
            {std::numeric_limits<T>::lowest(),2.0},
            {std::numeric_limits<T>::lowest(),2.0},
            {1.0,std::numeric_limits<T>::max()},
            {1.0,std::numeric_limits<T>::max()},
            {0.0,0.0},
            {-0.0,-0.0},
            {0X1.FFFFFFFFFFFFP+0,0X1.FFFFFFFFFFFFP+0},
            {0X1.FFFFFFFFFFFFP+0,0X1.FFFFFFFFFFFFP+0},
            {-0X1.FFFFFFFFFFFFP+0,0X1.FFFFFFFFFFFFP+0},
        }};

        std::array<I, n> h_ys {{
            empty,
            empty,
            {-1.0,1.0},
            entire,
            empty,
            {-infinity,1.0},
            {-1.0,1.0},
            {-1.0,infinity},
            entire,
            entire,
            entire,
            entire,
            {-infinity,4.0},
            {3.0,4.0},
            {3.0,infinity},
            {-infinity,4.0},
            {3.0,4.0},
            {3.0,infinity},
            {-infinity,4.0},
            {3.0,4.0},
            {3.0,infinity},
            {3.0,4.0},
            {-3.0,4.0},
            {-3.0,std::numeric_limits<T>::max()},
            {0.0,0.0},
            {-0.0,-0.0},
            {-3.0,4.0},
            {-3.0,std::numeric_limits<T>::max()},
            {0X1.999999999999AP-4,0X1.999999999999AP-4},
            {-0X1.999999999999AP-4,-0X1.999999999999AP-4},
            {0X1.999999999999AP-4,0X1.999999999999AP-4},
        }};

        std::array<I, n> h_ref {{
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
            entire,
            entire,
            {-infinity,6.0},
            {-infinity,6.0},
            entire,
            {-infinity,6.0},
            {4.0,6.0},
            {4.0,infinity},
            entire,
            {4.0,infinity},
            {4.0,infinity},
            {4.0,infinity},
            {-infinity,6.0},
            entire,
            {1.0,std::numeric_limits<T>::max()},
            {1.0,std::numeric_limits<T>::max()},
            {-3.0,4.0},
            {-3.0,std::numeric_limits<T>::max()},
            {0X1.0CCCCCCCCCCC4P+1,0X1.0CCCCCCCCCCC5P+1},
            {0X1.E666666666656P+0,0X1.E666666666657P+0},
            {-0X1.E666666666657P+0,0X1.0CCCCCCCCCCC5P+1},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        test_add<<<numBlocks, blockSize>>>(n, d_xs, d_ys);
        CUDA_CHECK(cudaMemcpy(h_xs.data(), d_xs, n_bytes, cudaMemcpyDeviceToHost));
        check_all_equal<I, n>(h_xs, h_ref);
    };

    "sub"_test = [&] {
        std::array<I, n> h_xs {{
            empty,
            {-1.0,1.0},
            empty,
            empty,
            entire,
            entire,
            entire,
            entire,
            entire,
            {-infinity,1.0},
            {-1.0,1.0},
            {-1.0,infinity},
            {-infinity,2.0},
            {-infinity,2.0},
            {-infinity,2.0},
            {1.0,2.0},
            {1.0,2.0},
            {1.0,2.0},
            {1.0,infinity},
            {1.0,infinity},
            {1.0,infinity},
            {1.0,std::numeric_limits<T>::max()},
            {std::numeric_limits<T>::lowest(),2.0},
            {std::numeric_limits<T>::lowest(),2.0},
            {1.0,std::numeric_limits<T>::max()},
            {1.0,std::numeric_limits<T>::max()},
            {0.0,0.0},
            {-0.0,-0.0},
            {0X1.FFFFFFFFFFFFP+0,0X1.FFFFFFFFFFFFP+0},
            {0X1.FFFFFFFFFFFFP+0,0X1.FFFFFFFFFFFFP+0},
            {-0X1.FFFFFFFFFFFFP+0,0X1.FFFFFFFFFFFFP+0},
        }};

        std::array<I, n> h_ys {{
            empty,
            empty,
            {-1.0,1.0},
            entire,
            empty,
            {-infinity,1.0},
            {-1.0,1.0},
            {-1.0,infinity},
            entire,
            entire,
            entire,
            entire,
            {-infinity,4.0},
            {3.0,4.0},
            {3.0,infinity},
            {-infinity,4.0},
            {3.0,4.0},
            {3.0,infinity},
            {-infinity,4.0},
            {3.0,4.0},
            {3.0,infinity},
            {-3.0,4.0},
            {3.0,4.0},
            {std::numeric_limits<T>::lowest(),4.0},
            {0.0,0.0},
            {-0.0,-0.0},
            {-3.0,4.0},
            {-3.0,std::numeric_limits<T>::max()},
            {0X1.999999999999AP-4,0X1.999999999999AP-4},
            {-0X1.999999999999AP-4,-0X1.999999999999AP-4},
            {0X1.999999999999AP-4,0X1.999999999999AP-4},
        }};

        std::array<I, n> h_ref {{
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
            entire,
            entire,
            entire,
            {-infinity,-1.0},
            {-infinity,-1.0},
            {-3.0,infinity},
            {-3.0,-1.0},
            {-infinity,-1.0},
            {-3.0,infinity},
            {-3.0,infinity},
            entire,
            {-3.0,infinity},
            {-infinity,-1.0},
            entire,
            {1.0,std::numeric_limits<T>::max()},
            {1.0,std::numeric_limits<T>::max()},
            {-4.0,3.0},
            {std::numeric_limits<T>::lowest(),3.0},
            {0X1.E666666666656P+0,0X1.E666666666657P+0},
            {0X1.0CCCCCCCCCCC4P+1,0X1.0CCCCCCCCCCC5P+1},
            {-0X1.0CCCCCCCCCCC5P+1,0X1.E666666666657P+0},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        test_sub<<<numBlocks, blockSize>>>(n, d_xs, d_ys);
        CUDA_CHECK(cudaMemcpy(h_xs.data(), d_xs, n_bytes, cudaMemcpyDeviceToHost));
        check_all_equal<I, n>(h_xs, h_ref);
    };

    "mul"_test = [&] {
        std::array<I, n> h_xs {{
            empty,
            {-1.0,1.0},
            empty,
            empty,
            entire,
            {0.0,0.0},
            empty,
            {-0.0,-0.0},
            empty,
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
            {1.0,infinity},
            {1.0,infinity},
            {1.0,infinity},
            {1.0,infinity},
            {1.0,infinity},
            {1.0,infinity},
            {1.0,infinity},
            {1.0,infinity},
            {1.0,infinity},
            {1.0,infinity},
            {-1.0,infinity},
            {-1.0,infinity},
            {-1.0,infinity},
            {-1.0,infinity},
            {-1.0,infinity},
            {-1.0,infinity},
            {-1.0,infinity},
            {-1.0,infinity},
            {-1.0,infinity},
            {-1.0,infinity},
            {-infinity,3.0},
            {-infinity,3.0},
            {-infinity,3.0},
            {-infinity,3.0},
            {-infinity,3.0},
            {-infinity,3.0},
            {-infinity,3.0},
            {-infinity,3.0},
            {-infinity,3.0},
            {-infinity,3.0},
            {-infinity,-3.0},
            {-infinity,-3.0},
            {-infinity,-3.0},
            {-infinity,-3.0},
            {-infinity,-3.0},
            {-infinity,-3.0},
            {-infinity,-3.0},
            {-infinity,-3.0},
            {-infinity,-3.0},
            {-infinity,-3.0},
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
            {-0.0,-0.0},
            {-0.0,-0.0},
            {-0.0,-0.0},
            {-0.0,-0.0},
            {-0.0,-0.0},
            {-0.0,-0.0},
            {-0.0,-0.0},
            {-0.0,-0.0},
            {-0.0,-0.0},
            {-0.0,-0.0},
            {1.0,5.0},
            {1.0,5.0},
            {1.0,5.0},
            {1.0,5.0},
            {1.0,5.0},
            {1.0,5.0},
            {1.0,5.0},
            {1.0,5.0},
            {1.0,5.0},
            {1.0,5.0},
            {-1.0,5.0},
            {-1.0,5.0},
            {-1.0,5.0},
            {-1.0,5.0},
            {-10.0,2.0},
            {-1.0,5.0},
            {-2.0,2.0},
            {-1.0,5.0},
            {-1.0,5.0},
            {-1.0,5.0},
            {-1.0,5.0},
            {-1.0,5.0},
            {-1.0,5.0},
            {-10.0,-5.0},
            {-10.0,-5.0},
            {-10.0,-5.0},
            {-10.0,-5.0},
            {-10.0,-5.0},
            {-10.0,-5.0},
            {-10.0,-5.0},
            {-10.0,-5.0},
            {-10.0,-5.0},
            {-10.0,-5.0},
            {0X1.999999999999AP-4,0X1.FFFFFFFFFFFFP+0},
            {-0X1.999999999999AP-4,0X1.FFFFFFFFFFFFP+0},
            {-0X1.999999999999AP-4,0X1.999999999999AP-4},
            {-0X1.FFFFFFFFFFFFP+0,-0X1.999999999999AP-4},
        }};

        std::array<I, n> h_ys {{
            empty,
            empty,
            {-1.0,1.0},
            entire,
            empty,
            empty,
            {0.0,0.0},
            empty,
            {-0.0,-0.0},
            {0.0,0.0},
            {-0.0,-0.0},
            {-5.0,-1.0},
            {-5.0,3.0},
            {1.0,3.0},
            {-infinity,-1.0},
            {-infinity,3.0},
            {-5.0,infinity},
            {1.0,infinity},
            entire,
            {0.0,0.0},
            {-0.0,-0.0},
            {-5.0,-1.0},
            {-5.0,3.0},
            {1.0,3.0},
            {-infinity,-1.0},
            {-infinity,3.0},
            {-5.0,infinity},
            {1.0,infinity},
            entire,
            {0.0,0.0},
            {-0.0,-0.0},
            {-5.0,-1.0},
            {-5.0,3.0},
            {1.0,3.0},
            {-infinity,-1.0},
            {-infinity,3.0},
            {-5.0,infinity},
            {1.0,infinity},
            entire,
            {0.0,0.0},
            {-0.0,-0.0},
            {-5.0,-1.0},
            {-5.0,3.0},
            {1.0,3.0},
            {-infinity,-1.0},
            {-infinity,3.0},
            {-5.0,infinity},
            {1.0,infinity},
            entire,
            {0.0,0.0},
            {-0.0,-0.0},
            {-5.0,-1.0},
            {-5.0,3.0},
            {1.0,3.0},
            {-infinity,-1.0},
            {-infinity,3.0},
            {-5.0,infinity},
            {1.0,infinity},
            entire,
            {0.0,0.0},
            {-0.0,-0.0},
            {-5.0,-1.0},
            {-5.0,3.0},
            {1.0,3.0},
            {-infinity,-1.0},
            {-infinity,3.0},
            {-5.0,infinity},
            {1.0,infinity},
            entire,
            {0.0,0.0},
            {-0.0,-0.0},
            {-5.0,-1.0},
            {-5.0,3.0},
            {1.0,3.0},
            {-infinity,-1.0},
            {-infinity,3.0},
            {-5.0,infinity},
            {1.0,infinity},
            entire,
            {0.0,0.0},
            {-0.0,-0.0},
            {-5.0,-1.0},
            {-5.0,3.0},
            {1.0,3.0},
            {-infinity,-1.0},
            {-infinity,3.0},
            {-5.0,infinity},
            {1.0,infinity},
            entire,
            {0.0,0.0},
            {-0.0,-0.0},
            {-5.0,-1.0},
            {-5.0,3.0},
            {-5.0,3.0},
            {-1.0,10.0},
            {-5.0,3.0},
            {1.0,3.0},
            {-infinity,-1.0},
            {-infinity,3.0},
            {-5.0,infinity},
            {1.0,infinity},
            entire,
            {0.0,0.0},
            {-0.0,-0.0},
            {-5.0,-1.0},
            {-5.0,3.0},
            {1.0,3.0},
            {-infinity,-1.0},
            {-infinity,3.0},
            {-5.0,infinity},
            {1.0,infinity},
            entire,
            {-0X1.FFFFFFFFFFFFP+0,infinity},
            {-0X1.FFFFFFFFFFFFP+0,-0X1.999999999999AP-4},
            {-0X1.FFFFFFFFFFFFP+0,0X1.999999999999AP-4},
            {0X1.999999999999AP-4,0X1.FFFFFFFFFFFFP+0},
        }};

        std::array<I, n> h_ref {{
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            {0.0,0.0},
            {0.0,0.0},
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            entire,
            {0.0,0.0},
            {0.0,0.0},
            {-infinity,-1.0},
            entire,
            {1.0,infinity},
            {-infinity,-1.0},
            entire,
            entire,
            {1.0,infinity},
            entire,
            {0.0,0.0},
            {0.0,0.0},
            {-infinity,5.0},
            entire,
            {-3.0,infinity},
            entire,
            entire,
            entire,
            entire,
            entire,
            {0.0,0.0},
            {0.0,0.0},
            {-15.0,infinity},
            entire,
            {-infinity,9.0},
            entire,
            entire,
            entire,
            entire,
            entire,
            {0.0,0.0},
            {0.0,0.0},
            {3.0,infinity},
            entire,
            {-infinity,-3.0},
            {3.0,infinity},
            entire,
            entire,
            {-infinity,-3.0},
            entire,
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
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {-25.0,-1.0},
            {-25.0,15.0},
            {1.0,15.0},
            {-infinity,-1.0},
            {-infinity,15.0},
            {-25.0,infinity},
            {1.0,infinity},
            entire,
            {0.0,0.0},
            {0.0,0.0},
            {-25.0,5.0},
            {-25.0,15.0},
            {-30.0,50.0},
            {-10.0,50.0},
            {-10.0,10.0},
            {-3.0,15.0},
            entire,
            entire,
            entire,
            entire,
            entire,
            {0.0,0.0},
            {0.0,0.0},
            {5.0,50.0},
            {-30.0,50.0},
            {-30.0,-5.0},
            {5.0,infinity},
            {-30.0,infinity},
            {-infinity,50.0},
            {-infinity,-5.0},
            entire,
            {-0X1.FFFFFFFFFFFE1P+1,infinity},
            {-0X1.FFFFFFFFFFFE1P+1,0X1.999999999998EP-3},
            {-0X1.999999999998EP-3,0X1.999999999998EP-3},
            {-0X1.FFFFFFFFFFFE1P+1,-0X1.47AE147AE147BP-7},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        test_mul<<<numBlocks, blockSize>>>(n, d_xs, d_ys);
        CUDA_CHECK(cudaMemcpy(h_xs.data(), d_xs, n_bytes, cudaMemcpyDeviceToHost));
        check_all_equal<I, n>(h_xs, h_ref);
    };

    "recip"_test = [&] {
        std::array<I, n> h_xs {{
            {-50.0,-10.0},
            {10.0,50.0},
            {-infinity,-10.0},
            {10.0,infinity},
            {0.0,0.0},
            {-0.0,-0.0},
            {-10.0,0.0},
            {-10.0,-0.0},
            {-10.0,10.0},
            {0.0,10.0},
            {-0.0,10.0},
            {-infinity,0.0},
            {-infinity,-0.0},
            {-infinity,10.0},
            {-10.0,infinity},
            {0.0,infinity},
            {-0.0,infinity},
            entire,
        }};

        std::array<I, n> h_ref {{
            {-0X1.999999999999AP-4,-0X1.47AE147AE147AP-6},
            {0X1.47AE147AE147AP-6,0X1.999999999999AP-4},
            {-0X1.999999999999AP-4,0.0},
            {0.0,0X1.999999999999AP-4},
            empty,
            empty,
            {-infinity,-0X1.9999999999999P-4},
            {-infinity,-0X1.9999999999999P-4},
            entire,
            {0X1.9999999999999P-4,infinity},
            {0X1.9999999999999P-4,infinity},
            {-infinity,0.0},
            {-infinity,0.0},
            entire,
            entire,
            {0.0,infinity},
            {0.0,infinity},
            entire,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        test_recip<<<numBlocks, blockSize>>>(n, d_xs);
        CUDA_CHECK(cudaMemcpy(h_xs.data(), d_xs, n_bytes, cudaMemcpyDeviceToHost));
        check_all_equal<I, n>(h_xs, h_ref);
    };


    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_ys));
}

int main()
{
    tests<float>();
    tests<double>();
    return 0;
}
