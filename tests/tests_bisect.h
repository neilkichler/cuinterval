#pragma once

#include <cuda_runtime.h>
#include <omp.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuinterval/arithmetic/interval.h>

#include "tests.h"
#include "tests_common.h"

void test_bisect_call(cudaStream_t stream, int n,
                      interval<double> *x, double *y, split<double> *res);

void test_bisection_call(cudaStream_t stream, interval<double> x, double tolerance,
                         interval<double> *roots, std::size_t *max_roots);

template<typename T>
void tests_bisect(cuda_buffers buffers, cuda_streams streams)
{
    using namespace boost::ut;

    using I = interval<T>;
    using B = bool;
    using N = int;

    T infinity = std::numeric_limits<T>::infinity();
    I empty    = { infinity, -infinity };
    I entire   = { -infinity, infinity };
    T NaN      = ::nan("");

    char *d_buffer    = buffers.device;
    char *h_buffer    = buffers.host;
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

void tests_bisection(cuda_buffers buffers, cudaStream_t stream);
