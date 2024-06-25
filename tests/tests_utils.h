#ifndef CUDA_INTERVAL_TESTS_UTILS_H
#define CUDA_INTERVAL_TESTS_UTILS_H

#include "tests.h"

#include <cassert>

template<typename T>
void contains(cu::interval<T> x, T y)
{
    using namespace boost::ut;

    expect(le(x.lb, y));
    expect(le(y, x.ub));
};

constexpr bool
is_power_of_two(std::size_t x) noexcept
{
    assert(x > 0);
    return (x & (x - 1)) == 0;
}

constexpr std::size_t align_to(std::size_t p, std::size_t align) noexcept
{
    assert(is_power_of_two(align));
    return (p + align - 1) & ~(align - 1);
}

#endif // CUDA_INTERVAL_TESTS_UTILS_H
