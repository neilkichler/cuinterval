// NOTE: This file is automatically generated by test_converter.py using itl tests.

#include <cuinterval/cuinterval.h>

#include "../test_ops.cuh"
#include "../tests.h"
#include "../tests_common.cuh"

template<typename T>
void tests_libieeep1788_num(cuda_buffers buffers, cudaStream_t stream) {
    using namespace boost::ut;

    using I = interval<T>;
    using B = bool;
    using N = int;

    T infinity = std::numeric_limits<T>::infinity();
    I empty    = { infinity, -infinity };
    I entire   = { -infinity, infinity };
    T NaN = ::nan("");

    const int n = 14; // count of largest test array
    const int n_bytes   = n * sizeof(I);
    const int blockSize = 256;
    [[maybe_unused]] const int numBlocks = (n + blockSize - 1) / blockSize;

    char *d_buffer = buffers.device;
    char *h_buffer = buffers.host;

    I *d_xs_  = (I *) d_buffer;
    I *d_ys_  = (I *) d_buffer + 1 * n_bytes;
    I *d_zs_  = (I *) d_buffer + 2 * n_bytes;
    I *d_res_ = (I *) d_buffer + 3 * n_bytes;

    "minimal_inf_inf"_test = [&] {
        constexpr int n = 14;
        I *h_xs = new (h_buffer) I[n]{
            {-0.0,-0.0},
            {-0.0,0.0},
            {-0.0,infinity},
            {-2.0,infinity},
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
        T *h_res = new (h_buffer) T[n]{};
        std::array<T, n> h_ref {{
            -0.0,
            -0.0,
            -0.0,
            -2.0,
            -3.0,
            -infinity,
            -infinity,
            -infinity,
            -infinity,
            -0.0,
            -0.0,
            -0.0,
            1.0,
            +infinity,
        }};

        h_buffer += n * sizeof(T);
        T *d_res = (T *)d_res_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_res, h_res, n*sizeof(T), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        test_inf<<<numBlocks, blockSize, 0, stream>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<T, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs);
    };

    "minimal_sup_sup"_test = [&] {
        constexpr int n = 14;
        I *h_xs = new (h_buffer) I[n]{
            {-0.0,-0.0},
            {-0.0,0.0},
            {-0.0,infinity},
            {-2.0,infinity},
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
        T *h_res = new (h_buffer) T[n]{};
        std::array<T, n> h_ref {{
            0.0,
            0.0,
            infinity,
            infinity,
            -2.0,
            +infinity,
            0.0,
            0.0,
            2.0,
            0.0,
            0.0,
            infinity,
            2.0,
            -infinity,
        }};

        h_buffer += n * sizeof(T);
        T *d_res = (T *)d_res_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_res, h_res, n*sizeof(T), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        test_sup<<<numBlocks, blockSize, 0, stream>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<T, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs);
    };

    "minimal_mid_mid"_test = [&] {
        constexpr int n = 12;
        I *h_xs = new (h_buffer) I[n]{
            {-0X0.0000000000001P-1022,0X0.0000000000002P-1022},
            {-0X0.0000000000002P-1022,0X0.0000000000001P-1022},
            {-0x1.FFFFFFFFFFFFFp1023,+0x1.FFFFFFFFFFFFFp1023},
            {-2.0,2.0},
            {-infinity,+infinity},
            {-infinity,1.2},
            {0.0,2.0},
            {0.0,infinity},
            {0X0.0000000000001P-1022,0X0.0000000000003P-1022},
            {0X1.FFFFFFFFFFFFFP+1022,0X1.FFFFFFFFFFFFFP+1023},
            {2.0,2.0},
            empty,
        };

        h_buffer += n * sizeof(I);
        T *h_res = new (h_buffer) T[n]{};
        std::array<T, n> h_ref {{
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0x1.FFFFFFFFFFFFFp1023,
            1.0,
            0x1.FFFFFFFFFFFFFp1023,
            0X0.0000000000002P-1022,
            0X1.7FFFFFFFFFFFFP+1023,
            2.0,
            NaN,
        }};

        h_buffer += n * sizeof(T);
        T *d_res = (T *)d_res_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_res, h_res, n*sizeof(T), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        test_mid<<<numBlocks, blockSize, 0, stream>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<T, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs);
    };

    "minimal_rad_rad"_test = [&] {
        constexpr int n = 9;
        I *h_xs = new (h_buffer) I[n]{
            {-0X0.0000000000002P-1022,0X0.0000000000001P-1022},
            {-infinity,+infinity},
            {-infinity,1.2},
            {0.0,2.0},
            {0.0,infinity},
            {0X0.0000000000001P-1022,0X0.0000000000002P-1022},
            {0X1P+0,0X1.0000000000003P+0},
            {2.0,2.0},
            empty,
        };

        h_buffer += n * sizeof(I);
        T *h_res = new (h_buffer) T[n]{};
        std::array<T, n> h_ref {{
            0X0.0000000000002P-1022,
            infinity,
            infinity,
            1.0,
            infinity,
            0X0.0000000000001P-1022,
            0X1P-51,
            0.0,
            NaN,
        }};

        h_buffer += n * sizeof(T);
        T *d_res = (T *)d_res_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_res, h_res, n*sizeof(T), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        test_rad<<<numBlocks, blockSize, 0, stream>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<T, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs);
    };

    "minimal_wid_wid"_test = [&] {
        constexpr int n = 8;
        I *h_xs = new (h_buffer) I[n]{
            {-infinity,+infinity},
            {-infinity,2.0},
            {0X1P+0,0X1.0000000000001P+0},
            {0X1P-1022,0X1.0000000000001P-1022},
            {1.0,2.0},
            {1.0,infinity},
            {2.0,2.0},
            empty,
        };

        h_buffer += n * sizeof(I);
        T *h_res = new (h_buffer) T[n]{};
        std::array<T, n> h_ref {{
            infinity,
            infinity,
            0X1P-52,
            0X0.0000000000001P-1022,
            1.0,
            infinity,
            0.0,
            NaN,
        }};

        h_buffer += n * sizeof(T);
        T *d_res = (T *)d_res_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_res, h_res, n*sizeof(T), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        test_wid<<<numBlocks, blockSize, 0, stream>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<T, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs);
    };

    "minimal_mag_mag"_test = [&] {
        constexpr int n = 8;
        I *h_xs = new (h_buffer) I[n]{
            {-0.0,-0.0},
            {-0.0,0.0},
            {-4.0,2.0},
            {-infinity,+infinity},
            {-infinity,2.0},
            {1.0,2.0},
            {1.0,infinity},
            empty,
        };

        h_buffer += n * sizeof(I);
        T *h_res = new (h_buffer) T[n]{};
        std::array<T, n> h_ref {{
            0.0,
            0.0,
            4.0,
            infinity,
            infinity,
            2.0,
            infinity,
            NaN,
        }};

        h_buffer += n * sizeof(T);
        T *d_res = (T *)d_res_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_res, h_res, n*sizeof(T), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        test_mag<<<numBlocks, blockSize, 0, stream>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<T, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs);
    };

    "minimal_mig_mig"_test = [&] {
        constexpr int n = 11;
        I *h_xs = new (h_buffer) I[n]{
            {-0.0,-0.0},
            {-0.0,0.0},
            {-1.0,infinity},
            {-4.0,-2.0},
            {-4.0,2.0},
            {-infinity,+infinity},
            {-infinity,-2.0},
            {-infinity,2.0},
            {1.0,2.0},
            {1.0,infinity},
            empty,
        };

        h_buffer += n * sizeof(I);
        T *h_res = new (h_buffer) T[n]{};
        std::array<T, n> h_ref {{
            0.0,
            0.0,
            0.0,
            2.0,
            0.0,
            0.0,
            2.0,
            0.0,
            1.0,
            1.0,
            NaN,
        }};

        h_buffer += n * sizeof(T);
        T *d_res = (T *)d_res_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_res, h_res, n*sizeof(T), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        test_mig<<<numBlocks, blockSize, 0, stream>>>(n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<T, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs);
    };

}