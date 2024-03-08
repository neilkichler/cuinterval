// NOTE: This file is automatically generated by test_converter.py using itl tests.

#include <cuinterval/cuinterval.h>

#include "../test_ops.cuh"
#include "../tests.h"
#include "../tests_common.cuh"

template<typename T>
void tests_libieeep1788_bool(cuda_buffers buffers, cudaStream_t stream) {
    using namespace boost::ut;

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

    char *d_buffer = buffers.device;
    char *h_buffer = buffers.host;

    I *d_xs_  = (I *) d_buffer;
    I *d_ys_  = (I *) d_buffer + 1 * n_bytes;
    I *d_zs_  = (I *) d_buffer + 2 * n_bytes;
    I *d_res_ = (I *) d_buffer + 3 * n_bytes;

    "minimal_is_empty_isEmpty"_test = [&] {
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
        test_isEmpty<<<numBlocks, blockSize, 0, stream>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(B), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaDeviceSynchronize());        int max_ulp_diff = 0;
        check_all_equal<B, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs);
    };

    "minimal_is_entire_isEntire"_test = [&] {
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
        test_isEntire<<<numBlocks, blockSize, 0, stream>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(B), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaDeviceSynchronize());        int max_ulp_diff = 0;
        check_all_equal<B, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs);
    };

    "minimal_equal_equal"_test = [&] {
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
        test_equal<<<numBlocks, blockSize, 0, stream>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(B), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaDeviceSynchronize());        int max_ulp_diff = 0;
        check_all_equal<B, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    "minimal_subset_subset"_test = [&] {
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
        test_subset<<<numBlocks, blockSize, 0, stream>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(B), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaDeviceSynchronize());        int max_ulp_diff = 0;
        check_all_equal<B, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    "minimal_less_less"_test = [&] {
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
        test_less<<<numBlocks, blockSize, 0, stream>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(B), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaDeviceSynchronize());        int max_ulp_diff = 0;
        check_all_equal<B, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    "minimal_precedes_precedes"_test = [&] {
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
        test_precedes<<<numBlocks, blockSize, 0, stream>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(B), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaDeviceSynchronize());        int max_ulp_diff = 0;
        check_all_equal<B, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    "minimal_interior_interior"_test = [&] {
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
        test_interior<<<numBlocks, blockSize, 0, stream>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(B), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaDeviceSynchronize());        int max_ulp_diff = 0;
        check_all_equal<B, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    "minimal_strictly_less_strictLess"_test = [&] {
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
        test_strictLess<<<numBlocks, blockSize, 0, stream>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(B), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaDeviceSynchronize());        int max_ulp_diff = 0;
        check_all_equal<B, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    "minimal_strictly_precedes_strictPrecedes"_test = [&] {
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
        test_strictPrecedes<<<numBlocks, blockSize, 0, stream>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(B), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaDeviceSynchronize());        int max_ulp_diff = 0;
        check_all_equal<B, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    "minimal_disjoint_disjoint"_test = [&] {
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
        test_disjoint<<<numBlocks, blockSize, 0, stream>>>(n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(B), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaDeviceSynchronize());        int max_ulp_diff = 0;
        check_all_equal<B, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

}