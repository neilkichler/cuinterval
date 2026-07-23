#ifndef CUINTERVAL_COMPARE_H
#define CUINTERVAL_COMPARE_H

#include <cuinterval/interval.h>

#include <compare>
#include <concepts>

namespace cu
{

template<typename T, typename U>
concept numeric_or = std::integral<T> or std::floating_point<T> or std::same_as<T, U>;

template<typename T>
inline constexpr bool operator==(interval<T> a, numeric_or<T> auto b)
{
    return a.lb == b && a.ub == b;
}

template<typename T>
inline constexpr bool operator==(numeric_or<T> auto a, interval<T> b)
{
    return a == b.lb && a == b.ub;
}

template<typename T>
inline constexpr bool operator!=(interval<T> a, numeric_or<T> auto b)
{
    return !(a == b);
}

template<typename T>
inline constexpr bool operator!=(numeric_or<T> auto a, interval<T> b)
{
    return !(a == b);
}

template<typename T>
constexpr std::partial_ordering
operator<=>(interval<T> a, interval<T> b)
{
    if (a.ub <= b.lb)
        return std::partial_ordering::less;

    if (b.ub <= a.lb)
        return std::partial_ordering::greater;

    return std::partial_ordering::unordered;
}

template<typename T>
inline constexpr bool operator<=(interval<T> a, numeric_or<T> auto b)
{
    return a.ub <= b;
}

template<typename T>
inline constexpr bool operator<=(numeric_or<T> auto a, interval<T> b)
{
    return a <= b.lb;
}

template<typename T>
inline constexpr bool operator<(interval<T> a, numeric_or<T> auto b)
{
    return a.ub < b;
}

template<typename T>
inline constexpr bool operator<(numeric_or<T> auto a, interval<T> b)
{
    return a < b.lb;
}

template<typename T>
inline constexpr bool operator>=(interval<T> a, numeric_or<T> auto b)
{
    return a.lb >= b;
}

template<typename T>
inline constexpr bool operator>=(numeric_or<T> auto a, interval<T> b)
{
    return a >= b.ub;
}

template<typename T>
inline constexpr bool operator>(interval<T> a, numeric_or<T> auto b)
{
    return a.lb > b;
}

template<typename T>
inline constexpr bool operator>(numeric_or<T> auto a, interval<T> b)
{
    return a > b.ub;
}

} // namespace cu

#endif // CUINTERVAL_COMPARE_H
