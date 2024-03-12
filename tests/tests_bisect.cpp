#include "tests_bisect.h"

thrust::host_vector<interval<double>> test_bisection_kernel(cudaStream_t stream, cuda_buffers buffers, interval<double> x, double tolerance);


void tests_bisection(cuda_buffers buffers, cudaStream_t stream)
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
    thrust::host_vector<I> h_roots = test_bisection_kernel(stream, buffers, x, tolerance);

    auto contains = [](interval<double> x, double y) {
        expect(le(x.lb, y));
        expect(le(y, x.ub));
    };

    for (std::size_t i = 0; i < h_roots.size(); i++) {
        contains(h_roots[i], ref_roots[i]);
        // printf("root is: %.15f %.15f\n", h_roots[i].lb, h_roots[i].ub);
        // printf("diff is: %.15f\n", h_roots[i].ub - h_roots[i].lb);
        // printf("tolerance is: %.15f\n", tolerance);
        expect(le(h_roots[i].ub - h_roots[i].lb, 3 * tolerance)); // TODO: fix multiply
    }
}
