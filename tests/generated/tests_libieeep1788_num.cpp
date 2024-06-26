// NOTE: This file is automatically generated by test_converter.py using itl tests.


#include "../tests.h"
#include "../tests_common.h"
#include "../tests_ops.h"
#include "../tests_utils.h"

#include <omp.h>

void tests_libieeep1788_num(cuda_buffer buffer, cudaStream_t stream, cudaEvent_t event) {
    using namespace boost::ut;

    using T = double;
    using I = cu::interval<T>;
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

    char *d_buffer = buffer.device;

    I *d_xs_  = (I *) d_buffer;
    I *d_ys_  = (I *) d_buffer + 1 * n_bytes;
    I *d_zs_  = (I *) d_buffer + 2 * n_bytes;
    I *d_res_ = (I *) d_buffer + 3 * n_bytes;

    {
        char *h_buffer = buffer.host;
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

        h_buffer += align_to(n * sizeof(I), alignof(T));
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

        T *d_res = (T *)d_res_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_inf_call(numBlocks, blockSize, stream, n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaEventRecord(event, stream));
        CUDA_CHECK(cudaEventSynchronize(event));
        int max_ulp_diff = 0;
        check_all_equal<T, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs);
    };

    {
        char *h_buffer = buffer.host;
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

        h_buffer += align_to(n * sizeof(I), alignof(T));
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

        T *d_res = (T *)d_res_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_sup_call(numBlocks, blockSize, stream, n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaEventRecord(event, stream));
        CUDA_CHECK(cudaEventSynchronize(event));
        int max_ulp_diff = 0;
        check_all_equal<T, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs);
    };

    {
        char *h_buffer = buffer.host;
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

        h_buffer += align_to(n * sizeof(I), alignof(T));
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

        T *d_res = (T *)d_res_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_mid_call(numBlocks, blockSize, stream, n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaEventRecord(event, stream));
        CUDA_CHECK(cudaEventSynchronize(event));
        int max_ulp_diff = 0;
        check_all_equal<T, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs);
    };

    {
        char *h_buffer = buffer.host;
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

        h_buffer += align_to(n * sizeof(I), alignof(T));
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

        T *d_res = (T *)d_res_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_rad_call(numBlocks, blockSize, stream, n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaEventRecord(event, stream));
        CUDA_CHECK(cudaEventSynchronize(event));
        int max_ulp_diff = 0;
        check_all_equal<T, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs);
    };

    {
        char *h_buffer = buffer.host;
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

        h_buffer += align_to(n * sizeof(I), alignof(T));
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

        T *d_res = (T *)d_res_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_wid_call(numBlocks, blockSize, stream, n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaEventRecord(event, stream));
        CUDA_CHECK(cudaEventSynchronize(event));
        int max_ulp_diff = 0;
        check_all_equal<T, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs);
    };

    {
        char *h_buffer = buffer.host;
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

        h_buffer += align_to(n * sizeof(I), alignof(T));
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

        T *d_res = (T *)d_res_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_mag_call(numBlocks, blockSize, stream, n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaEventRecord(event, stream));
        CUDA_CHECK(cudaEventSynchronize(event));
        int max_ulp_diff = 0;
        check_all_equal<T, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs);
    };

    {
        char *h_buffer = buffer.host;
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

        h_buffer += align_to(n * sizeof(I), alignof(T));
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

        T *d_res = (T *)d_res_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_mig_call(numBlocks, blockSize, stream, n, d_xs, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(T), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaEventRecord(event, stream));
        CUDA_CHECK(cudaEventSynchronize(event));
        int max_ulp_diff = 0;
        check_all_equal<T, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs);
    };

}