// NOTE: This file is automatically generated by test_converter.py using itl tests.


#include "../tests.h"
#include "../tests_common.h"
#include "../tests_ops.h"

#include <omp.h>

void tests_intervalarithmeticjl(cuda_buffer buffer, cudaStream_t stream) {
    using namespace boost::ut;

    using T = double;
    using I = interval<T>;
    using B = bool;
    using N = int;

    T infinity = std::numeric_limits<T>::infinity();
    I empty    = { infinity, -infinity };
    I entire   = { -infinity, infinity };
    T NaN = ::nan("");

    const int n = 12; // count of largest test array
    const int n_bytes   = n * sizeof(I);
    const int blockSize = 256;
    [[maybe_unused]] const int numBlocks = (n + blockSize - 1) / blockSize;

    char *d_buffer = buffer.device;
    char *h_buffer = buffer.host;

    I *d_xs_  = (I *) d_buffer;
    I *d_ys_  = (I *) d_buffer + 1 * n_bytes;
    I *d_zs_  = (I *) d_buffer + 2 * n_bytes;
    I *d_res_ = (I *) d_buffer + 3 * n_bytes;

    {
        constexpr int n = 12;
        I *h_xs = new (h_buffer) I[n]{
            {-0.25,0.25},
            {0.0,2.0},
            {0.25,0.75},
            {0.5,0.5},
            {0.5,1.5},
            {1.0,1.0},
            {1.0,2.0},
            {1.5,1.5},
            {2.0,2.0},
            {36.0,37.0},
            empty,
            entire,
        };

        h_buffer += n * sizeof(I);
        I *h_res = new (h_buffer) I[n]{};
        std::array<I, n> h_ref {{
            {-0x1.6a09e667f3bcdp-1,0x1.6a09e667f3bcdp-1},
            {-1.0,1.0},
            {0x1.6a09e667f3bcdp-1,1.0},
            {1.0,1.0},
            {-1.0,1.0},
            {0.0,0.0},
            {-1.0,0.0},
            {-1.0,-1.0},
            {0.0,0.0},
            {0.0,1.0},
            empty,
            {-1.0,1.0},
        }};

        h_buffer += n * sizeof(I);
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_res, h_res, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_sinpi_call(numBlocks, blockSize, stream, n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(I), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 3;
        check_all_equal<I, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs);
    };

    {
        constexpr int n = 12;
        I *h_xs = new (h_buffer) I[n]{
            {-0.25,0.25},
            {0.0,2.0},
            {0.25,0.75},
            {0.5,0.5},
            {0.5,1.5},
            {1.0,1.0},
            {1.0,2.0},
            {1.5,1.5},
            {2.0,2.0},
            {36.0,37.0},
            empty,
            entire,
        };

        h_buffer += n * sizeof(I);
        I *h_res = new (h_buffer) I[n]{};
        std::array<I, n> h_ref {{
            {0x1.6a09e667f3bcdp-1,1.0},
            {-1.0,1.0},
            {-0x1.6a09e667f3bcdp-1,0x1.6a09e667f3bcdp-1},
            {0.0,0.0},
            {-1.0,0.0},
            {-1.0,-1.0},
            {-1.0,1.0},
            {0.0,0.0},
            {1.0,1.0},
            {-1.0,1.0},
            empty,
            {-1.0,1.0},
        }};

        h_buffer += n * sizeof(I);
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_res, h_res, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_cospi_call(numBlocks, blockSize, stream, n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(I), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 3;
        check_all_equal<I, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs);
    };

    {
        constexpr int n = 4;
        I *h_xs = new (h_buffer) I[n]{
            {0.5,0.5},
            {0.5,1.67},
            {1.67,3.2},
            {6.638314112824137,8.38263151220128},
        };

        h_buffer += n * sizeof(I);
        I *h_res = new (h_buffer) I[n]{};
        std::array<I, n> h_ref {{
            {0.54630248984379048,0.5463024898437906},
            entire,
            {-10.047182299210307,0.05847385445957865},
            entire,
        }};

        h_buffer += n * sizeof(I);
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_res, h_res, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_tan_call(numBlocks, blockSize, stream, n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(I), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 3;
        check_all_equal<I, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs);
    };

    {
        constexpr int n = 11;
        I *h_xs = new (h_buffer) I[n]{
            {0,27},
            {0,81},
            {1,2},
            {1,7},
            {16,81},
            {5,8},
            {8,27},
            empty,
            empty,
            empty,
            empty,
        };

        h_buffer += n * sizeof(I);
        N *h_ys = new (h_buffer) N[n]{
            3,
            4,
            0,
            0,
            4,
            0,
            3,
            -3,
            -4,
            3,
            4,
        };

        h_buffer += n * sizeof(N);
        I *h_res = new (h_buffer) I[n]{};
        std::array<I, n> h_ref {{
            {0,3},
            {0,3},
            empty,
            empty,
            {2,3},
            empty,
            {2,3},
            empty,
            empty,
            empty,
            empty,
        }};

        h_buffer += n * sizeof(I);
        I *d_res = (I *)d_res_;
        N *d_ys = (N *)d_ys_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_res, h_res, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n*sizeof(N), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_rootn_call(numBlocks, blockSize, stream, n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(I), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 2;
        check_all_equal<I, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

}