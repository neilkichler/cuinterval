#ifndef CUINTERVAL_CUH
#define CUINTERVAL_CUH

#include <cuinterval/arithmetic/basic.cuh>
#include <cuinterval/arithmetic/interval.h>
#include <cuinterval/arithmetic/intrinsic.cuh>

template<typename T>
bool operator==(interval<T> lhs, interval<T> rhs)
{
    if (empty(lhs) && empty(rhs)) {
        return true;
    }

    return lhs.lb == rhs.lb && lhs.ub == rhs.ub;
}

#endif // CUINTERVAL_CUH
