#include "tests_bisect.h"
#include "tests_ops.h"
#include "tests_utils.h"

void tests_mince(cuda_buffer buffer, cudaStream_t stream)
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

    char *d_buffer      = buffer.device;
    char *h_buffer      = buffer.host;

    constexpr int n = 5;
    constexpr int n_results = 16;
    const int n_bytes   = n * sizeof(I);
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;

    I *d_xs_  = (I *) d_buffer;
    I *d_ys_  = (I *) d_buffer + n_bytes;
    I *d_res_ = (I *) d_buffer + n_bytes + (n_results + 1) * sizeof(int);

    {
        I *h_xs = new (h_buffer) I[n]{
            {-0.0,0.0},
            {-0.0,-0.0},
            {0.0,4.0},
            empty,
            {-1.0,1.0},
        };

        h_buffer += n * sizeof(I);
        int *h_offsets = new (h_buffer) int[n_results]{
            0, 4, 8, 12, 14, 16
        };

        h_buffer += (n_results + 1) * sizeof(int);

        I *h_res = new (h_buffer) I[n_results] {
        };

        std::array<I, n_results> h_ref {{
            { 0.0, 0.0 }, empty, empty, empty,
            { -0.0, -0.0 }, empty, empty, empty,
            { 0.0, 1.0 }, { 1.0, 2.0 }, { 2.0, 3.0 }, { 3.0, 4.0 },
            empty, empty,
            { -1.0, 0.0 }, { 0.0, 1.0 },
        }};

        I *d_xs = (I *)d_xs_;
        int *d_offsets = (int *)d_ys_;
        I *d_res = (I *)d_res_;
        CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n*sizeof(I), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_offsets, h_offsets, (n_results + 1)*sizeof(int), cudaMemcpyHostToDevice, stream));
        tests_mince_call(numBlocks, blockSize, stream, n, d_xs, d_offsets, d_res);
        CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n_results*sizeof(I), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        int max_ulp_diff = 0;
        check_all_equal<I, n_results>(h_res, h_ref, max_ulp_diff, std::source_location::current(), h_xs);
    };
}

thrust::host_vector<interval<double>> test_bisection_kernel(cudaStream_t stream, cuda_buffer buffer, interval<double> x, double tolerance);

void tests_bisection(cuda_buffer buffer, cudaStream_t stream)
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

    // TODO: we should have an example function that can be called from cpp
    // settings should be passed in a struct
    thrust::host_vector<I> h_roots = test_bisection_kernel(stream, buffer, x, tolerance);

    for (std::size_t i = 0; i < h_roots.size(); i++) {
        contains(h_roots[i], ref_roots[i]);
        // printf("root is: %.15f %.15f\n", h_roots[i].lb, h_roots[i].ub);
        // printf("diff is: %.15f\n", h_roots[i].ub - h_roots[i].lb);
        // printf("tolerance is: %.15f\n", tolerance);
        expect(le(h_roots[i].ub - h_roots[i].lb, 3 * tolerance)); // TODO: fix multiply
    }
}
