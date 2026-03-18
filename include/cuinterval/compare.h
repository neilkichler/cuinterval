#ifndef CUINTERVAL_COMPARE_H
#define CUINTERVAL_COMPARE_H

#include <cuinterval/interval.h>

namespace cu
{

template<typename T>
inline constexpr bool operator==(interval<T> a, auto b)
{
    return a.lb == b && a.ub == b;
}

template<typename T>
inline constexpr bool operator!=(interval<T> a, auto b)
{
    return !(a == b);
}

template<typename T>
inline constexpr bool operator==(auto a, interval<T> b)
{
    return a == b.lb && a == b.ub;
}

template<typename T>
inline constexpr bool operator!=(auto a, interval<T> b)
{
    return !(a == b);
}

template<typename T>
inline constexpr bool operator<=(interval<T> a, auto b)
{
    return a.ub <= b;
}

template<typename T>
inline constexpr bool operator<=(auto a, interval<T> b)
{
    return a <= b.lb;
}

template<typename T>
inline constexpr bool operator<(interval<T> a, auto b)
{
    return a.ub < b;
}

template<typename T>
inline constexpr bool operator<(auto a, interval<T> b)
{
    return a < b.lb;
}

template<typename T>
inline constexpr bool operator>=(interval<T> a, auto b)
{
    return a.lb >= b;
}

template<typename T>
inline constexpr bool operator>=(auto a, interval<T> b)
{
    return a >= b.ub;
}

template<typename T>
inline constexpr bool operator>(interval<T> a, auto b)
{
    return a.lb > b;
}

template<typename T>
inline constexpr bool operator>(auto a, interval<T> b)
{
    return a > b.ub;
}

} // namespace cu

#endif // CUINTERVAL_COMPARE_H
