#ifndef CUINTERVAL_INTERVAL_H
#define CUINTERVAL_INTERVAL_H

namespace cu
{

template<typename T>
struct interval
{
    using value_type = T;

    // to support designated initializers: return {{ .lb = lb, .ub = ub }} -> interval
    struct initializer
    {
        T lb;
        T ub;
    };

    constexpr interval() = default;
    constexpr interval(T p) : lb(p), ub(p) { } // point interval
    constexpr interval(T lb, T ub) : lb(lb), ub(ub) { }
    constexpr interval(initializer init) : lb(init.lb), ub(init.ub) { }

    constexpr bool operator==(const interval &rhs) const = default;

    T lb;
    T ub;
};

template<typename T>
struct split
{
    interval<T> lower_half;
    interval<T> upper_half;

    auto operator<=>(const split &) const = default;
};

} // namespace cu

#endif // CUINTERVAL_INTERVAL_H
