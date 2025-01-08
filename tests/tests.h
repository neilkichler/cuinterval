#ifndef CUDA_INTERVAL_TESTS_H
#define CUDA_INTERVAL_TESTS_H

#ifdef __CUDACC__
#pragma nv_diagnostic push
#pragma nv_diag_suppress 2811
// TODO: remove when static_asserts are fixed upstream for nvcc
#ifdef _MSC_VER
#define static_assert(...)
#endif
#include <boost/ut.hpp>
#ifdef _MSC_VER
#undef static_assert
#endif
#pragma nv_diagnostic pop
#pragma nv_diag_default 2811
#else
#include <boost/ut.hpp>
#endif

#include <cuinterval/format.h>
#include <cuinterval/interval.h>

#include <cmath>
#include <source_location>
#include <span>
#include <vector>

template<typename T>
bool check_within_ulps(T x, T y, std::size_t n, T direction)
{
    if (x == y) {
        return true;
    }

    for (auto i = 0u; i < n; ++i) {
        y = std::nextafter(y, direction);

        if (x == y) {
            return true;
        }
    }

    return false;
}

template<typename T>
struct is_interval : std::false_type
{
};

template<typename T>
struct is_interval<cu::interval<T>> : std::true_type
{
};

template<typename T>
inline constexpr bool is_interval_v = is_interval<T>::value;

template<typename T, int N, typename... Args>
void check_all_equal(T *h_res, std::span<T, N> h_ref, int max_ulps_diff, std::source_location location, Args &&...args)
{
    using namespace boost::ut;

    std::vector<size_t> failed_ids;

    auto show_inputs = [](auto &out, auto &&...args) {
        out << "with input:";
        ((out << "\n\t"
              << args),
         ...);
    };

    auto empty = [](auto &&x) {
        return !(x.lb <= x.ub);
    };

    for (size_t i = 0; i < h_ref.size(); ++i) {
        if (h_res[i] != h_res[i] && h_ref[i] != h_ref[i]) // both are NaN
            continue;

        if constexpr (is_interval_v<T>) {

            if (!empty(h_res[i]) || !empty(h_ref[i])) {
                using TV            = T::value_type;
                auto inf            = std::numeric_limits<TV>::infinity();
                bool lb_within_ulps = check_within_ulps(h_res[i].lb, h_ref[i].lb, max_ulps_diff, -inf);
                bool ub_within_ulps = check_within_ulps(h_res[i].ub, h_ref[i].ub, max_ulps_diff, inf);

                auto out = expect(eq(lb_within_ulps && ub_within_ulps, true), location)
                    << std::hexfloat
                    << "Failed at case" << i << ": " << h_res[i] << "!= " << h_ref[i] << "\n"
                    << '\t' << "with delta: [" << std::fabs(h_res[i].lb - h_ref[i].lb)
                    << ", " << std::fabs(h_res[i].ub - h_ref[i].ub) << "]\n\t";
                show_inputs(out, std::forward<Args>(args)[i]...);
            }
        } else {
            auto out = expect(eq(h_res[i], h_ref[i]), location);
            out << "Failed at case" << i << ":\n";
            out << std::hexfloat << '\t';
            show_inputs(out, std::forward<Args>(args)[i]...);
        }
    }
}

#endif // CUDA_INTERVAL_TESTS_H
