#ifndef CUINTERVAL_FORMAT_H
#define CUINTERVAL_FORMAT_H

#include <cuinterval/interval.h>

#include <ostream>

namespace cu
{

template<typename T>
std::ostream &operator<<(std::ostream &os, interval<T> x)
{
    return os << "[" << x.lb << ", " << x.ub << "]";
}

template<typename T>
std::ostream &operator<<(std::ostream &os, split<T> x)
{
    return os << "[" << x.lower_half << ", " << x.upper_half << "]";
}

} // namespace cu

#endif // CUINTERVAL_FORMAT_H
