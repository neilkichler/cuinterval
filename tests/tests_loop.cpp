#include "tests_loop.h"
#include "tests.h"
#include "tests_utils.h"

#include <cuinterval/interval.h>

#include <numbers>
#include <vector>

#include <omp.h>

using cu::interval;
using cu::split;

std::vector<interval<double>> compute_pi_approximation(cudaStream_t stream);

void tests_pi_approximation(cudaStream_t stream, cudaEvent_t event)
{
    using T = double;
    using I = interval<T>;

    using namespace boost::ut;

    printf("Pi Approx: Inside OpenMP thread %i\n", omp_get_thread_num());

    std::vector<I> h_pi = compute_pi_approximation(stream);

    for (I pi_approx : h_pi) {
        contains(pi_approx, std::numbers::pi);
        expect(le(pi_approx.lb, std::numbers::pi));
        expect(ge(pi_approx.ub, std::numbers::pi));
    }
}

std::vector<interval<double>> compute_horner(cudaStream_t stream);

void tests_horner(cudaStream_t stream, cudaEvent_t event)
{
    using T = double;
    using I = interval<T>;

    using namespace boost::ut;

    printf("Horner: Inside OpenMP thread %i\n", omp_get_thread_num());

    // thrust::host_vector<I> res = compute_horner(stream);
    std::vector<I> res = compute_horner(stream);

    I exp_approx = res[res.size() - 1];
    T exp_true   = std::numbers::e;

    contains(exp_approx, exp_true);
}
