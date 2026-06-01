#include "tests_rounding.h"
#include "tests.h"

#include <cuinterval/interval.h>

#include <source_location>
#include <span>
#include <vector>

using cu::interval;

std::vector<interval<float>> compute_rounding(cudaStream_t stream, std::vector<interval<float>> xs, std::vector<int> exponents);
std::vector<interval<float>> compute_rounding_ref(cudaStream_t stream, std::vector<interval<float>> xs, std::vector<int> exponents);

void tests_rounding(cudaStream_t stream, cudaEvent_t event)
{
    using T = float;
    using I = interval<T>;

    constexpr int n = 25;
    constexpr T inf = std::numeric_limits<T>::infinity();

    // we want the custom rounding to match n times applying std::nextafter
    constexpr int max_ulp_diff = 0;

    {
        std::vector<I> xs = {
            { +0.f, +0.f },
            { +0.f, -0.f },
            { -0.f, +0.f },
            { -0.f, -0.f },
            { +0.f, +1.f },
            { +0.f, -1.f },
            { -0.f, +1.f },
            { -0.f, -1.f },
            { +0.f, +inf },
            { +0.f, -inf },
            { -0.f, +inf },
            { -0.f, -inf },
            { +inf, +inf },
            { +inf, -inf },
            { -inf, +inf },
            { -inf, -inf },
            { +1.f, +1.f },
            { +1.f, -1.f },
            { -1.f, +1.f },
            { -1.f, -1.f },
            { std::nextafter(+0.f, 0.f), +0.f },
            { std::nextafter(+0.f, 0.f), -0.f },
            { std::nextafter(-0.f, 0.f), +0.f },
            { std::nextafter(-0.f, 0.f), -0.f },
            { std::nextafter(+2.f, 0.f), +2.f },
        };

        std::vector<int> n_ulps(n);
        std::vector<I> out = compute_rounding(stream, xs, n_ulps);
        std::vector<I> ref = compute_rounding_ref(stream, xs, n_ulps);
        std::span<I, n> ref_span { ref.data(), ref.size() };
        check_all_equal<I, n>(out.data(), ref_span, max_ulp_diff, "rounding", std::source_location::current(), xs.data(), n_ulps.data());
    }
}
