// NOTE: This file is automatically generated by test_converter.py using itl tests.


#include "../tests.h"
#include "../tests_common.h"
#include "../tests_ops.h"
#include "../tests_utils.h"

#include <omp.h>

void tests_libieeep1788_set(cuda_buffer buffer, cudaStream_t stream, cudaEvent_t event) {
    using namespace boost::ut;

    using T = double;
    using I = interval<T>;
    using B = bool;
    using N = int;

    T infinity = std::numeric_limits<T>::infinity();
    I empty    = { infinity, -infinity };
    I entire   = { -infinity, infinity };
    T NaN = ::nan("");

    const int n = 5; // count of largest test array
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
        constexpr int n = 5;
        I *h_xs = new (h_buffer) I[n]{
            {1.0,3.0},
            {1.0,3.0},
            {1.0,3.0},
            {1.0,3.0},
            entire,
        };

        h_buffer += align_to(n * sizeof(I), alignof(I));
        I *h_ys = new (h_buffer) I[n]{
            {2.1,4.0},
            {3.0,4.0},
            empty,
            entire,
            empty,
        };

        h_buffer += align_to(n * sizeof(I), alignof(I));
        I *h_res = new (h_buffer) I[n]{};
        std::array<I, n> h_ref {{
            {2.1,3.0},
            {3.0,3.0},
            empty,
            {1.0,3.0},
            empty,
        }};

        I *d_res = (I *)d_res_;
        I *d_ys = (I *)d_ys_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_intersection_call(numBlocks, blockSize, stream, n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(I), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaEventRecord(event, stream));
        CUDA_CHECK(cudaEventSynchronize(event));
        int max_ulp_diff = 0;
        check_all_equal<I, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

    {
        char *h_buffer = buffer.host;
        constexpr int n = 5;
        I *h_xs = new (h_buffer) I[n]{
            {1.0,1.0},
            {1.0,3.0},
            {1.0,3.0},
            {1.0,3.0},
            empty,
        };

        h_buffer += align_to(n * sizeof(I), alignof(I));
        I *h_ys = new (h_buffer) I[n]{
            {2.1,4.0},
            {2.1,4.0},
            empty,
            entire,
            empty,
        };

        h_buffer += align_to(n * sizeof(I), alignof(I));
        I *h_res = new (h_buffer) I[n]{};
        std::array<I, n> h_ref {{
            {1.0,4.0},
            {1.0,4.0},
            {1.0,3.0},
            entire,
            empty,
        }};

        I *d_res = (I *)d_res_;
        I *d_ys = (I *)d_ys_;
        I *d_xs = (I *)d_xs_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        tests_convexHull_call(numBlocks, blockSize, stream, n, d_xs, d_ys, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n*sizeof(I), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaEventRecord(event, stream));
        CUDA_CHECK(cudaEventSynchronize(event));
        int max_ulp_diff = 0;
        check_all_equal<I, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };

}