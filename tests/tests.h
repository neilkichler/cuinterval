#ifndef CUDA_INTERVAL_TESTS_H
#define CUDA_INTERVAL_TESTS_H

// compiler bug fix; TODO: remove when fixed
#ifdef __CUDACC__
#pragma nv_diagnostic push
#pragma nv_diag_suppress 2811
#pragma push_macro("__cpp_consteval")
#define consteval constexpr
#include <boost/ut.hpp>
#undef consteval
#pragma nv_diagnostic pop
#pragma nv_diag_default 2811
#pragma pop_macro("__cpp_consteval")
#else
#pragma message("No compiler detected")
#include <boost/ut.hpp>
#endif

#include <cuinterval/cuinterval.h>

#include <ostream>
#include <source_location>
#include <span>
#include <vector>

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

template<typename T>
std::ostream &operator<<(std::ostream &os, const interval<T> &x)
{
    return os << '[' << std::hexfloat << x.lb << ',' << x.ub << ']';
}

template<typename T>
std::ostream &operator<<(std::ostream &os, const split<T> &x)
{
    return os << '(' << x.lower_half << ',' << x.upper_half << ')';
}

template<typename T>
bool check_within_ulps(T x, T y, std::size_t n, T direction)
{
    if (x == y) {
        return true;
    }

    for (int i = 0; i < n; ++i) {
        y = std::nextafter(y, direction);

        if (x == y) {
            return true;
        }
    }

    return false;
}

template<typename T, int N>
std::vector<size_t> check_all_equal(std::span<T, N> h_res, std::span<T, N> h_ref, int max_ulps_diff, const std::source_location location = std::source_location::current())
{
    using namespace boost::ut;

    std::vector<size_t> failed_ids;

    for (size_t i = 0; i < h_res.size(); ++i) {
        if (h_res[i] != h_res[i] && h_ref[i] != h_ref[i]) // both are NaN
            continue;

        if constexpr (std::is_same_v<T, interval<double>>) {
            if (!empty(h_res[i]) || !empty(h_ref[i])) {
                bool lb_within_ulps = check_within_ulps(h_res[i].lb, h_ref[i].lb, max_ulps_diff, -std::numeric_limits<double>::infinity());
                bool ub_within_ulps = check_within_ulps(h_res[i].ub, h_ref[i].ub, max_ulps_diff, std::numeric_limits<double>::infinity());

                expect(lb_within_ulps && ub_within_ulps, location)
                    << "Failed at [" << i << "]: " << h_res[i] << "!= " << h_ref[i] << "\n"
                    << "Delta  at [" << i << "]: [" << std::fabs(h_res[i].lb - h_ref[i].lb) 
                                            << ", " << std::fabs(h_res[i].ub - h_ref[i].ub) << "]\n";
            }
        } else {
            expect(eq(h_res[i], h_ref[i]), location);
        }
    }

    return failed_ids;
}

#endif // CUDA_INTERVAL_TESTS_H
