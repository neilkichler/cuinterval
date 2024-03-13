// NOTE: This file is automatically generated by test_converter.py using itl tests.


#include "../tests.h"
#include "../tests_common.h"
#include "../tests_ops.h"

#include <omp.h>

void tests_libieeep1788_bool(cuda_buffer buffer, cudaStream_t stream) {
    using namespace boost::ut;

    using T = double;
    using I = interval<T>;
    using B = bool;
    using N = int;

    T infinity = std::numeric_limits<T>::infinity();
    I empty    = { infinity, -infinity };
    I entire   = { -infinity, infinity };
    T NaN = ::nan("");

    const int n = 27; // count of largest test array
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
        constexpr int n = 14;
        I *h_xs = new (h_buffer) I[n]{
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
            false,
            false,
            false,
            true,
        }};

        h_buffer += n * sizeof(B);
        B *d_res = (B *)d_res_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_res, h_res, n*sizeof(B), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_isEmpty_call(numBlocks, blockSize, stream, n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(B), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<B, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs);
    };

    {
        constexpr int n = 14;
        I *h_xs = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        B *h_res = new (h_buffer) B[n]{};
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

        h_buffer += n * sizeof(B);
        B *d_res = (B *)d_res_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_res, h_res, n*sizeof(B), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_isEntire_call(numBlocks, blockSize, stream, n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(B), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<B, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs);
    };

    {
        constexpr int n = 15;
        I *h_xs = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        I *h_ys = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        B *h_res = new (h_buffer) B[n]{};
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

        h_buffer += n * sizeof(B);
        B *d_res = (B *)d_res_;
        I *d_ys = (I *)d_ys_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_res, h_res, n*sizeof(B), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_equal_call(numBlocks, blockSize, stream, n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(B), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<B, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    {
        constexpr int n = 27;
        I *h_xs = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        I *h_ys = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        B *h_res = new (h_buffer) B[n]{};
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

        h_buffer += n * sizeof(B);
        B *d_res = (B *)d_res_;
        I *d_ys = (I *)d_ys_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_res, h_res, n*sizeof(B), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_subset_call(numBlocks, blockSize, stream, n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(B), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<B, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    {
        constexpr int n = 26;
        I *h_xs = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        I *h_ys = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        B *h_res = new (h_buffer) B[n]{};
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

        h_buffer += n * sizeof(B);
        B *d_res = (B *)d_res_;
        I *d_ys = (I *)d_ys_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_res, h_res, n*sizeof(B), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_less_call(numBlocks, blockSize, stream, n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(B), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<B, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    {
        constexpr int n = 21;
        I *h_xs = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        I *h_ys = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        B *h_res = new (h_buffer) B[n]{};
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

        h_buffer += n * sizeof(B);
        B *d_res = (B *)d_res_;
        I *d_ys = (I *)d_ys_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_res, h_res, n*sizeof(B), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_precedes_call(numBlocks, blockSize, stream, n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(B), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<B, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    {
        constexpr int n = 16;
        I *h_xs = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        I *h_ys = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        B *h_res = new (h_buffer) B[n]{};
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

        h_buffer += n * sizeof(B);
        B *d_res = (B *)d_res_;
        I *d_ys = (I *)d_ys_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_res, h_res, n*sizeof(B), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_interior_call(numBlocks, blockSize, stream, n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(B), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<B, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    {
        constexpr int n = 14;
        I *h_xs = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        I *h_ys = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        B *h_res = new (h_buffer) B[n]{};
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

        h_buffer += n * sizeof(B);
        B *d_res = (B *)d_res_;
        I *d_ys = (I *)d_ys_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_res, h_res, n*sizeof(B), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_strictLess_call(numBlocks, blockSize, stream, n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(B), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<B, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    {
        constexpr int n = 14;
        I *h_xs = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        I *h_ys = new (h_buffer) I[n]{
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
            true,
            false,
            false,
            false,
            true,
            true,
            true,
        }};

        h_buffer += n * sizeof(B);
        B *d_res = (B *)d_res_;
        I *d_ys = (I *)d_ys_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_res, h_res, n*sizeof(B), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_strictPrecedes_call(numBlocks, blockSize, stream, n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(B), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<B, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    {
        constexpr int n = 10;
        I *h_xs = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        I *h_ys = new (h_buffer) I[n]{
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
        };

        h_buffer += n * sizeof(I);
        B *h_res = new (h_buffer) B[n]{};
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

        h_buffer += n * sizeof(B);
        B *d_res = (B *)d_res_;
        I *d_ys = (I *)d_ys_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_res, h_res, n*sizeof(B), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_disjoint_call(numBlocks, blockSize, stream, n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(B), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<B, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

}