#include "tests_pow.h"
#include "tests.h"

#include <cuinterval/interval.h>

#include <array>
#include <source_location>
#include <vector>

using cu::interval;

template<typename T>
std::vector<interval<float>> compute_pow(cudaStream_t stream, std::vector<interval<float>> xs, std::vector<T> exponents);

void tests_pow(cudaStream_t stream, cudaEvent_t event)
{
    using T = float;
    using I = interval<T>;

    constexpr int n            = 6;
    constexpr int max_ulp_diff = 4;
    constexpr T infinity       = std::numeric_limits<T>::infinity();

    { // Integer exponents
        std::vector<I> xs = { { +1.0, +1.0 },
                              { +0.0, +1.0 },
                              { +1.0, +2.0 },
                              { +2.0, +3.0 },
                              { +3.0, +4.0 },
                              { +0.0, +0.0 } };

        std::vector<int> exponents = { 0, 1, 2, 3, 4, 5 };
        std::vector<I> out         = compute_pow(stream, xs, exponents);
        std::array<I, n> ref { I { 1.0, 1.0 },
                               { 0.0, 1.0 },
                               { 1.0, 4.0 },
                               { 8.0, 27.0 },
                               { 81.0, 256.0 },
                               { 0.0, 0.0 } };

        check_all_equal<I, n>(out.data(), ref, max_ulp_diff, "pow", std::source_location::current(), xs.data(), exponents.data());

        std::vector<float> fexponents = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0 };
        out                           = compute_pow(stream, xs, fexponents);
        check_all_equal<I, n>(out.data(), ref, max_ulp_diff, "pow", std::source_location::current(), xs.data(), exponents.data());
    }

    { // Float exponents
        std::vector<I> xs = { { +1.0, +1.0 },
                              { +0.0, +1.0 },
                              { +1.0, +4.0 },
                              { +2.0, +3.0 },
                              { +1.0, +4.0 },
                              { +4.0, +5.0 } };

        std::vector<float> exponents = { 0.0, 0.5, 0.5, 1.0, 1.5, 2.0 };
        std::vector<I> out           = compute_pow(stream, xs, exponents);
        std::array<I, n> ref { I { 1.0, 1.0 },
                               { 0.0, 1.0 },
                               { 1.0, 2.0 },
                               { 2.0, 3.0 },
                               { 1.0, 8.0 },
                               { 16.0, 25.0 } };

        check_all_equal<I, n>(out.data(), ref, max_ulp_diff, "pow", std::source_location::current(), xs.data(), exponents.data());
    }

    { // Negative float exponents
        std::vector<I> xs = { { +1.0, +1.0 },
                              { +0.0, +1.0 },
                              { +4.0, +16.0 },
                              { +0.0, +4.0 },
                              { +0.0, +1.0 },
                              { +0.0, +2.0 } };

        std::vector<float> exponents = { -0.0, -0.5, -0.5, -1.0, -1.5, -2.0 };
        std::vector<I> out           = compute_pow(stream, xs, exponents);
        std::array<I, n> ref { I { 1.0, 1.0 },
                               { 1.0, infinity },
                               { 0.25, 0.5 },
                               { 0.25, infinity },
                               { 1.0, infinity },
                               { 0.25, infinity } };

        check_all_equal<I, n>(out.data(), ref, max_ulp_diff, "pow", std::source_location::current(), xs.data(), exponents.data());
    }
}
