// NOTE: This file is automatically generated by test_converter.py using itl tests.


#include "../tests.h"
#include "../tests_common.h"
#include "../tests_ops.h"

#include <omp.h>

void tests_c_xsc(cuda_buffer buffer, cudaStream_t stream) {
    using namespace boost::ut;

    using T = double;
    using I = interval<T>;
    using B = bool;
    using N = int;

    T infinity = std::numeric_limits<T>::infinity();
    I empty    = { infinity, -infinity };
    I entire   = { -infinity, infinity };
    T NaN = ::nan("");

    const int n = 16; // count of largest test array
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
        constexpr int n = 2;
        I *h_xs = new (h_buffer) I[n]{
            {10.0,20.0},
            {13.0,17.0},
        };

        h_buffer += n * sizeof(I);
        I *h_ys = new (h_buffer) I[n]{
            {13.0,17.0},
            {10.0,20.0},
        };

        h_buffer += n * sizeof(I);
        I *h_res = new (h_buffer) I[n]{};
        std::array<I, n> h_ref {{
            {23.0,37.0},
            {23.0,37.0},
        }};

        h_buffer += n * sizeof(I);
        I *d_res = (I *)d_res_;
        I *d_ys = (I *)d_ys_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_add_call(numBlocks, blockSize, stream, n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(I), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<I, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    {
        constexpr int n = 1;
        I *h_xs = new (h_buffer) I[n]{
            {10.0,20.0},
        };

        h_buffer += n * sizeof(I);
        I *h_res = new (h_buffer) I[n]{};
        std::array<I, n> h_ref {{
            {-20.0,-10.0},
        }};

        h_buffer += n * sizeof(I);
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_neg_call(numBlocks, blockSize, stream, n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(I), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<I, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs);
    };

    {
        constexpr int n = 1;
        I *h_xs = new (h_buffer) I[n]{
            {10.0,20.0},
        };

        h_buffer += n * sizeof(I);
        I *h_res = new (h_buffer) I[n]{};
        std::array<I, n> h_ref {{
            {10.0,20.0},
        }};

        h_buffer += n * sizeof(I);
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_pos_call(numBlocks, blockSize, stream, n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(I), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<I, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs);
    };

    {
        constexpr int n = 2;
        I *h_xs = new (h_buffer) I[n]{
            {10.0,20.0},
            {13.0,16.0},
        };

        h_buffer += n * sizeof(I);
        I *h_ys = new (h_buffer) I[n]{
            {13.0,16.0},
            {10.0,20.0},
        };

        h_buffer += n * sizeof(I);
        I *h_res = new (h_buffer) I[n]{};
        std::array<I, n> h_ref {{
            {-6.0,7.0},
            {-7.0,6.0},
        }};

        h_buffer += n * sizeof(I);
        I *d_res = (I *)d_res_;
        I *d_ys = (I *)d_ys_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_sub_call(numBlocks, blockSize, stream, n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(I), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<I, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    {
        constexpr int n = 16;
        I *h_xs = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        I *h_ys = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        I *h_res = new (h_buffer) I[n]{};
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

        h_buffer += n * sizeof(I);
        I *d_res = (I *)d_res_;
        I *d_ys = (I *)d_ys_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_div_call(numBlocks, blockSize, stream, n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(I), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<I, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    {
        constexpr int n = 15;
        I *h_xs = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        I *h_ys = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        I *h_res = new (h_buffer) I[n]{};
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

        h_buffer += n * sizeof(I);
        I *d_res = (I *)d_res_;
        I *d_ys = (I *)d_ys_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_mul_call(numBlocks, blockSize, stream, n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(I), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<I, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    {
        constexpr int n = 12;
        I *h_xs = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        I *h_ys = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        I *h_res = new (h_buffer) I[n]{};
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

        h_buffer += n * sizeof(I);
        I *d_res = (I *)d_res_;
        I *d_ys = (I *)d_ys_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_convexHull_call(numBlocks, blockSize, stream, n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(I), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<I, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    {
        constexpr int n = 12;
        I *h_xs = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        I *h_ys = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        I *h_res = new (h_buffer) I[n]{};
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

        h_buffer += n * sizeof(I);
        I *d_res = (I *)d_res_;
        I *d_ys = (I *)d_ys_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_intersection_call(numBlocks, blockSize, stream, n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(I), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<I, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    {
        constexpr int n = 6;
        I *h_xs = new (h_buffer) I[n]{
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-4.0,-4.0},
            {1.0,1.0},
            {4.0,4.0},
        };

        h_buffer += n * sizeof(I);
        I *h_ys = new (h_buffer) I[n]{
            {-4.0,-4.0},
            {1.0,1.0},
            {4.0,4.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
        };

        h_buffer += n * sizeof(I);
        I *h_res = new (h_buffer) I[n]{};
        std::array<I, n> h_ref {{
            {-4.0,2.0},
            {-2.0,2.0},
            {-2.0,4.0},
            {-4.0,2.0},
            {-2.0,2.0},
            {-2.0,4.0},
        }};

        h_buffer += n * sizeof(I);
        I *d_res = (I *)d_res_;
        I *d_ys = (I *)d_ys_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_convexHull_call(numBlocks, blockSize, stream, n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(I), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<I, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    {
        constexpr int n = 6;
        I *h_xs = new (h_buffer) I[n]{
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-4.0,-4.0},
            {1.0,1.0},
            {4.0,4.0},
        };

        h_buffer += n * sizeof(I);
        I *h_ys = new (h_buffer) I[n]{
            {-4.0,-4.0},
            {1.0,1.0},
            {4.0,4.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
        };

        h_buffer += n * sizeof(I);
        I *h_res = new (h_buffer) I[n]{};
        std::array<I, n> h_ref {{
            empty,
            {1.0,1.0},
            empty,
            empty,
            {1.0,1.0},
            empty,
        }};

        h_buffer += n * sizeof(I);
        I *d_res = (I *)d_res_;
        I *d_ys = (I *)d_ys_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_intersection_call(numBlocks, blockSize, stream, n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(I), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<I, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    {
        constexpr int n = 6;
        I *h_xs = new (h_buffer) I[n]{
            {-2.0,-2.0},
            {-2.0,-2.0},
            {-2.0,-2.0},
            {-2.0,-2.0},
            {-4.0,-4.0},
            {2.0,2.0},
        };

        h_buffer += n * sizeof(I);
        I *h_ys = new (h_buffer) I[n]{
            {-2.0,-2.0},
            {-2.0,-2.0},
            {-4.0,-4.0},
            {2.0,2.0},
            {-2.0,-2.0},
            {-2.0,-2.0},
        };

        h_buffer += n * sizeof(I);
        I *h_res = new (h_buffer) I[n]{};
        std::array<I, n> h_ref {{
            {-2.0,-2.0},
            {-2.0,-2.0},
            {-4.0,-2.0},
            {-2.0,2.0},
            {-4.0,-2.0},
            {-2.0,2.0},
        }};

        h_buffer += n * sizeof(I);
        I *d_res = (I *)d_res_;
        I *d_ys = (I *)d_ys_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_convexHull_call(numBlocks, blockSize, stream, n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(I), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<I, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    {
        constexpr int n = 7;
        I *h_xs = new (h_buffer) I[n]{
            {-1.0,2.0},
            {-2.0,1.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
        };

        h_buffer += n * sizeof(I);
        I *h_ys = new (h_buffer) I[n]{
            {-1.0,2.0},
            {-3.0,2.0},
            {-1.0,1.0},
            {-1.0,2.0},
            {-2.0,1.0},
            {-2.0,3.0},
            {-3.0,2.0},
        };

        h_buffer += n * sizeof(I);
        B *h_res = new (h_buffer) B[n]{};
        std::array<B, n> h_ref {{
            true,
            false,
            false,
            false,
            false,
            false,
            false,
        }};

        h_buffer += n * sizeof(B);
        B *d_res = (B *)d_res_;
        I *d_ys = (I *)d_ys_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_equal_call(numBlocks, blockSize, stream, n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(B), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<B, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    {
        constexpr int n = 14;
        I *h_xs = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        I *h_ys = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        B *h_res = new (h_buffer) B[n]{};
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

        h_buffer += n * sizeof(B);
        B *d_res = (B *)d_res_;
        I *d_ys = (I *)d_ys_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_interior_call(numBlocks, blockSize, stream, n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(B), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<B, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    {
        constexpr int n = 13;
        I *h_xs = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        I *h_ys = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        B *h_res = new (h_buffer) B[n]{};
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

        h_buffer += n * sizeof(B);
        B *d_res = (B *)d_res_;
        I *d_ys = (I *)d_ys_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_subset_call(numBlocks, blockSize, stream, n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(B), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<B, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    {
        constexpr int n = 7;
        I *h_xs = new (h_buffer) I[n]{
            {-1.0,-1.0},
            {-1.0,-1.0},
            {-1.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
            {-2.0,2.0},
        };

        h_buffer += n * sizeof(I);
        I *h_ys = new (h_buffer) I[n]{
            {-1.0,-1.0},
            {1.0,1.0},
            {-2.0,-2.0},
            {-2.0,-2.0},
            {0.0,0.0},
            {2.0,2.0},
            {3.0,3.0},
        };

        h_buffer += n * sizeof(I);
        B *h_res = new (h_buffer) B[n]{};
        std::array<B, n> h_ref {{
            true,
            false,
            false,
            false,
            false,
            false,
            false,
        }};

        h_buffer += n * sizeof(B);
        B *d_res = (B *)d_res_;
        I *d_ys = (I *)d_ys_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_equal_call(numBlocks, blockSize, stream, n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(B), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<B, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    {
        constexpr int n = 14;
        I *h_xs = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        I *h_ys = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        B *h_res = new (h_buffer) B[n]{};
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

        h_buffer += n * sizeof(B);
        B *d_res = (B *)d_res_;
        I *d_ys = (I *)d_ys_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_interior_call(numBlocks, blockSize, stream, n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(B), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<B, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    {
        constexpr int n = 14;
        I *h_xs = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        I *h_ys = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        B *h_res = new (h_buffer) B[n]{};
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

        h_buffer += n * sizeof(B);
        B *d_res = (B *)d_res_;
        I *d_ys = (I *)d_ys_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_subset_call(numBlocks, blockSize, stream, n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(B), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<B, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    {
        constexpr int n = 2;
        I *h_xs = new (h_buffer) I[n]{
            {2.0,2.0},
            {4.0,4.0},
        };

        h_buffer += n * sizeof(I);
        I *h_ys = new (h_buffer) I[n]{
            {2.0,2.0},
            {5.0,5.0},
        };

        h_buffer += n * sizeof(I);
        I *h_res = new (h_buffer) I[n]{};
        std::array<I, n> h_ref {{
            {4.0,4.0},
            {1024.0,1024.0},
        }};

        h_buffer += n * sizeof(I);
        I *d_res = (I *)d_res_;
        I *d_ys = (I *)d_ys_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_pow_call(numBlocks, blockSize, stream, n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(I), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 1;
        check_all_equal<I, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    {
        constexpr int n = 3;
        I *h_xs = new (h_buffer) I[n]{
            {0.0,0.0},
            {1024.0,1024.0},
            {27.0,27.0},
        };

        h_buffer += n * sizeof(I);
        N *h_ys = new (h_buffer) N[n]{
            4,
            10,
            3,
        };

        h_buffer += n * sizeof(N);
        I *h_res = new (h_buffer) I[n]{};
        std::array<I, n> h_ref {{
            {0.0,0.0},
            {2.0,2.0},
            {3.0,3.0},
        }};

        h_buffer += n * sizeof(I);
        I *d_res = (I *)d_res_;
        N *d_ys = (N *)d_ys_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n*sizeof(N), cudaMemcpyHostToDevice, stream));
        tests_rootn_call(numBlocks, blockSize, stream, n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(I), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 2;
        check_all_equal<I, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    {
        constexpr int n = 3;
        I *h_xs = new (h_buffer) I[n]{
            {-9.0,-9.0},
            {0.0,0.0},
            {11.0,11.0},
        };

        h_buffer += n * sizeof(I);
        I *h_res = new (h_buffer) I[n]{};
        std::array<I, n> h_ref {{
            {81.0,81.0},
            {0.0,0.0},
            {121.0,121.0},
        }};

        h_buffer += n * sizeof(I);
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_sqr_call(numBlocks, blockSize, stream, n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(I), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<I, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs);
    };

    {
        constexpr int n = 3;
        I *h_xs = new (h_buffer) I[n]{
            {0.0,0.0},
            {121.0,121.0},
            {81.0,81.0},
        };

        h_buffer += n * sizeof(I);
        I *h_res = new (h_buffer) I[n]{};
        std::array<I, n> h_ref {{
            {0.0,0.0},
            {11.0,11.0},
            {9.0,9.0},
        }};

        h_buffer += n * sizeof(I);
        I *d_res = (I *)d_res_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_sqrt_call(numBlocks, blockSize, stream, n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(I), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<I, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs);
    };

}