#ifndef CUDA_INTERVAL_TESTS_H
#define CUDA_INTERVAL_TESTS_H

#include <cuinterval/cuinterval.h>

// compiler bug fix; TODO: remove when fixed
#ifdef __CUDACC__
#pragma push_macro("__cpp_consteval")
#define consteval constexpr
#include <boost/ut.hpp>
#undef consteval
#pragma pop_macro("__cpp_consteval")
#else
#include <boost/ut.hpp>
#endif

#include <span>
#include <vector>
#include <source_location>
#include <ostream>

#define CUDA_CHECK(x)                                                                \
    do {                                                                             \
        cudaError_t err = x;                                                         \
        if (err != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, \
                    __FILE__, __LINE__, cudaGetErrorString(err),                     \
                    cudaGetErrorName(err), err);                                     \
            abort();                                                                 \
        }                                                                            \
    } while (0)

template<typename T, int N>
std::vector<size_t> check_all_equal(std::span<T, N> h_xs, std::span<T, N> h_ref, const std::source_location location = std::source_location::current())
{
    using namespace boost::ut;

    std::vector<size_t> failed_ids;

    for (size_t i = 0; i < h_xs.size(); ++i) {
        if (h_xs[i] != h_xs[i] && h_ref[i] != h_ref[i]) // both are NaN
            continue;

        if (h_xs[i] != h_ref[i])
            failed_ids.push_back(i);

        expect(eq(h_xs[i], h_ref[i]), location);
    }

    return failed_ids;
}

template<typename T>
std::ostream &operator<<(std::ostream &os, const interval<T> &x)
{
    return (os << '[' << std::hexfloat << x.lb << ',' << x.ub << ']');
}

#endif // CUDA_INTERVAL_TESTS_H
