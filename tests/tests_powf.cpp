#include "tests_powf.h"
#include "tests.h"

#include <cuinterval/interval.h>

#include <array>
#include <source_location>
#include <vector>

using cu::interval;

std::vector<interval<float>> compute_powf(cudaStream_t stream, std::vector<interval<float>> xs, std::vector<int> exponents);

void tests_powf(cudaStream_t stream, cudaEvent_t event)
{
    using T = float;
    using I = interval<T>;

    using namespace boost::ut;

    constexpr int n            = 5;
    constexpr int max_ulp_diff = 4;
    std::vector<I> xs          = { { -1.0, 0.0 }, { 0.0, 1.0 }, { 1.0, 2.0 }, { 2.0, 3.0 }, { 3.0, 4.0 } };
    std::vector<int> exponents = { 0, 1, 2, 3, 4 };
    std::vector<I> out         = compute_powf(stream, xs, exponents);
    std::array<I, n> ref { I { 1.0, 1.0 }, { 0.0, 1.0 }, { 1.0, 4.0 }, { 8.0, 27.0 }, { 81.0, 256.0 } };

    check_all_equal<I, n>(out.data(), ref, max_ulp_diff, "powf", std::source_location::current(), xs.data(), exponents.data());
}
