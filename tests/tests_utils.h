#ifndef CUDA_INTERVAL_TESTS_UTILS_H
#define CUDA_INTERVAL_TESTS_UTILS_H

#include "tests.h"

template<typename T>
void contains(interval<T> x, T y)
{
    using namespace boost::ut;

    expect(le(x.lb, y));
    expect(le(y, x.ub));
};

#endif // CUDA_INTERVAL_TESTS_UTILS_H
