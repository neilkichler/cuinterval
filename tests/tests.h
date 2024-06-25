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

    for (int i = 0; i < n; ++i) {
        y = std::nextafter(y, direction);

        if (x == y) {
            return true;
        }
    }

    return false;
}

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

        if constexpr (std::is_same_v<T, cu::interval<double>>) {
            if (!empty(h_res[i]) || !empty(h_ref[i])) {
                bool lb_within_ulps = check_within_ulps(h_res[i].lb, h_ref[i].lb, max_ulps_diff, -std::numeric_limits<double>::infinity());
                bool ub_within_ulps = check_within_ulps(h_res[i].ub, h_ref[i].ub, max_ulps_diff, std::numeric_limits<double>::infinity());

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
