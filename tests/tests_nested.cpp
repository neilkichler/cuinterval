#include "tests_nested.h"
#include "tests.h"

#include <cuinterval/cuinterval.h>

#include <array>
#include <source_location>
#include <vector>

template<typename T, typename U = T>
std::vector<T> compute_nested(cudaStream_t stream, std::vector<T> xs, std::vector<U> ys);

void tests_nested(cudaStream_t stream, cudaEvent_t event)
{
    using T = cu::interval<float>;
    using I = cu::interval<T>;

    constexpr int n            = 6;
    constexpr int max_ulp_diff = 0;
    constexpr T infinity       = std::numeric_limits<T>::infinity();

    { // I<I<T>> + I<I<T>>
        std::vector<I> xs = { { +1.0, +1.0 },
                              { +0.0, +1.0 },
                              { +1.0, +2.0 },
                              { +0.0, +0.0 },
                              { infinity, infinity },
                              { 0.0, infinity } };

        std::vector<I> ys = { { +0.0, +1.0 },
                              { +1.0, +1.0 },
                              { +1.0, +2.0 },
                              { +0.0, +0.0 },
                              { +0.0, +1.0 },
                              { +1.0, +1.0 } };

        std::vector<I> out = compute_nested(stream, xs, ys);

        // check lower and upper bounds separately
        std::vector<T> out_lb(n);
        std::vector<T> out_ub(n);
        for (int i = 0; i < n; i++) {
            out_lb[i] = out[i].lb;
            out_ub[i] = out[i].ub;
        }

        // results are point intervals
        std::array<T, n> ref_lb = { 1.0, 1.0, 2.0, 0.0, infinity, 1.0 };
        std::array<T, n> ref_ub = { 2.0, 2.0, 4.0, 0.0, infinity, infinity };

        check_all_equal<T, n>(out_lb.data(), ref_lb, max_ulp_diff, "+ (lb)", std::source_location::current(), xs.data(), ys.data());
        check_all_equal<T, n>(out_ub.data(), ref_ub, max_ulp_diff, "+ (ub)", std::source_location::current(), xs.data(), ys.data());
    }

    { // I<I<T>> + I<T>
        std::vector<I> xs = { { +1.0, +1.0 },
                              { +0.0, +1.0 },
                              { +1.0, +2.0 },
                              { +0.0, +0.0 },
                              { infinity, infinity },
                              { 0.0, infinity } };

        std::vector<T> ys = { { +0.0, +1.0 },
                              { +1.0, +1.0 },
                              { +1.0, +2.0 },
                              { +0.0, +0.0 },
                              { +0.0, +1.0 },
                              { +1.0, +1.0 } };

        std::vector<I> out = compute_nested(stream, xs, ys);

        // for [[a, b], [c, d]] create [a, d] and check against ref
        std::vector<T> out_outer(n);
        for (int i = 0; i < n; i++) {
            out_outer[i] = { out[i].lb.lb, out[i].ub.ub };
        }

        std::array<T, n> ref = { T { 1.0, 2.0 },
                                 { 1.0, 2.0 },
                                 { 2.0, 4.0 },
                                 { 0.0, 0.0 },
                                 { infinity },
                                 { 1.0, infinity.ub } };

        check_all_equal<T, n>(out_outer.data(), ref, max_ulp_diff, "I<I<T>> + I<T>", std::source_location::current(), xs.data(), ys.data());
    }
}
