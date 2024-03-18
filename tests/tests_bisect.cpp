#include <cuda_runtime.h>
#include <omp.h>
#include <thrust/host_vector.h>

#include <cuinterval/arithmetic/interval.h>

#include "tests_bisect.h"
#include "tests_ops.h"
#include "tests_utils.h"

void test_bisect_call(cudaStream_t stream, int n,
                      interval<double> *x, double *y, split<double> *res);

void tests_bisect(cuda_buffer buffer, cuda_streams streams, cuda_events events)
{
    using namespace boost::ut;

    using T = double;
    using I = interval<T>;
    using B = bool;
    using N = int;

    T infinity = std::numeric_limits<T>::infinity();
    I empty    = { infinity, -infinity };
    I entire   = { -infinity, infinity };
    T NaN      = ::nan("");

    char *d_buffer    = buffer.device;
    char *h_buffer    = buffer.host;
    const int n       = 8; // count of largest test array
    const int n_bytes = n * sizeof(I);

    I *d_xs_, *d_ys_, *d_res_;

    d_xs_  = (I *)d_buffer;
    d_ys_  = (I *)d_buffer + 1 * n_bytes;
    d_res_ = (I *)d_buffer + 2 * n_bytes;

    printf("Bisect: Inside OpenMP thread %i\n", omp_get_thread_num());

    "bisection"_test = [&] {
        constexpr int n = 8;
        I *h_xs         = new (h_buffer) I[n] {
            empty,
            entire,
            entire,
            entire,
            { 0.0, 2.0 },
            { 1.0, 1.0 },
            { 0.0, 1.0 },
            { 0.0, 1.0 },
        };
        h_buffer += n * sizeof(I);

        T *h_ys = new (h_buffer) T[n] {
            0.5,
            0.5,
            0.25,
            0.75,
            0.5,
            0.5,
            0.5,
            0.25,
        };
        h_buffer += n * sizeof(I);
        split<T> *h_res = new (h_buffer) split<T>[n] {};
        h_buffer += n * sizeof(split<T>);
        split<T> *d_res    = (split<T> *)d_res_;
        I *d_xs            = (I *)d_xs_;
        T *d_ys            = (T *)d_ys_;
        int n_result_bytes = n * sizeof(*d_res);

        int n_chunk = n / 2;
        for (int i = 0; i < 2; i++) {
            auto stream = streams[i];
            CUDA_CHECK(cudaMemcpyAsync(d_xs + i * n_chunk, h_xs + i * n_chunk, n_chunk * sizeof(I), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_ys + i * n_chunk, h_ys + i * n_chunk, n_chunk * sizeof(I), cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(d_res + i * n_chunk, h_res + i * n_chunk, n_result_bytes / 2, cudaMemcpyHostToDevice, stream));
            test_bisect_call(stream, n_chunk, d_xs + i * n_chunk, d_ys + i * n_chunk, d_res + i * n_chunk);
            CUDA_CHECK(cudaMemcpyAsync(h_res + i * n_chunk, d_res + i * n_chunk, n_result_bytes / 2, cudaMemcpyDeviceToHost, stream));
        }

        std::array<split<T>, n> h_ref { {
            { empty, empty },
            { { entire.lb, 0.0 }, { 0.0, entire.ub } },
            { { entire.lb, -0x1.fffffffffffffp+1023 }, { -0x1.fffffffffffffp+1023, entire.ub } },
            { { entire.lb, 0x1.fffffffffffffp+1023 }, { 0x1.fffffffffffffp+1023, entire.ub } },
            { { 0.0, 1.0 }, { 1.0, 2.0 } },
            { { 1.0, 1.0 }, empty },
            { { 0.0, 0.5 }, { 0.5, 1.0 } },
            { { 0.0, 0.25 }, { 0.25, 1.0 } },
        } };

        for (int i = 0; i < 2; i++) {
            auto stream = streams[i];
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        int max_ulp_diff = 0;
        check_all_equal<split<T>, n>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs, h_ys);
    };
}

void tests_mince_call(int numBlocks, int blockSize, cudaStream_t stream,
                      int n, interval<double> *d_xs, int *d_offsets, interval<double> *d_res);

void tests_mince(cuda_buffer buffer, cudaStream_t stream, cudaEvent_t event)
{
    printf("Mince: Inside OpenMP thread %i\n", omp_get_thread_num());

    using namespace boost::ut;

    using T = double;
    using I = interval<T>;
    using B = bool;
    using N = int;

    T infinity = std::numeric_limits<T>::infinity();
    I empty    = { infinity, -infinity };
    I entire   = { -infinity, infinity };
    T NaN      = ::nan("");

    char *d_buffer = buffer.device;
    char *h_buffer = buffer.host;

    constexpr int n         = 5;
    constexpr int n_results = 16;
    const int n_bytes       = n * sizeof(I);
    const int blockSize     = 256;
    const int numBlocks     = (n + blockSize - 1) / blockSize;

    I *d_xs_  = (I *)d_buffer;
    I *d_ys_  = (I *)d_buffer + n_bytes;
    I *d_res_ = (I *)d_buffer + n_bytes + (n_results + 1) * sizeof(int);

    {
        I *h_xs = new (h_buffer) I[n] {
            { -0.0, 0.0 },
            { -0.0, -0.0 },
            { 0.0, 4.0 },
            empty,
            { -1.0, 1.0 },
        };

        h_buffer += n * sizeof(I);
        int *h_offsets = new (h_buffer) int[n_results] {
            0, 4, 8, 12, 14, 16
        };

        h_buffer += (n_results + 1) * sizeof(int);

        I *h_res = new (h_buffer) I[n_results] {};

        std::array<I, n_results> h_ref { {
            { 0.0, 0.0 },
            empty,
            empty,
            empty,
            { -0.0, -0.0 },
            empty,
            empty,
            empty,
            { 0.0, 1.0 },
            { 1.0, 2.0 },
            { 2.0, 3.0 },
            { 3.0, 4.0 },
            empty,
            empty,
            { -1.0, 0.0 },
            { 0.0, 1.0 },
        } };

        I *d_xs        = (I *)d_xs_;
        int *d_offsets = (int *)d_ys_;
        I *d_res       = (I *)d_res_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n * sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_offsets, h_offsets, (n_results + 1) * sizeof(int), cudaMemcpyHostToDevice, stream));
        tests_mince_call(numBlocks, blockSize, stream, n, d_xs, d_offsets, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n_results * sizeof(I), cudaMemcpyDeviceToHost, stream));
        // CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventRecord(event, stream));
        CUDA_CHECK(cudaEventSynchronize(event));
        int max_ulp_diff = 0;
        check_all_equal<I, n_results>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs);
    };
}

thrust::host_vector<interval<double>> test_bisection_kernel(cudaStream_t stream, cuda_buffer buffer, interval<double> x, double tolerance);

void tests_bisection(cuda_buffer buffer, cudaStream_t stream, cudaEvent_t event)
{
    printf("Bisection: Inside OpenMP thread %i\n", omp_get_thread_num());

    using namespace boost::ut;
    using T = double;
    using I = interval<T>;

    I x = { -5.0, 10.0 };

    T ref_roots[3] = {
        -3.0,
        -1.0,
        5.0
    };

    constexpr double tolerance = 1e-12;
    thrust::host_vector<I> h_roots = test_bisection_kernel(stream, buffer, x, tolerance);

    for (std::size_t i = 0; i < h_roots.size(); i++) {
        contains(h_roots[i], ref_roots[i]);
        // printf("root is: %.15f %.15f\n", h_roots[i].lb, h_roots[i].ub);
        // printf("diff is: %.15f\n", h_roots[i].ub - h_roots[i].lb);
        // printf("tolerance is: %.15f\n", tolerance);
        expect(le(h_roots[i].ub - h_roots[i].lb, tolerance));
    }
}
