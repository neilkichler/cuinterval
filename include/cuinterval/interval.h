#ifndef CUINTERVAL_INTERVAL_H
#define CUINTERVAL_INTERVAL_H

namespace cu
{

template<typename T>
struct interval
{
    using value_type = T;

    T lb;
    T ub;
};

template<typename T>
constexpr bool operator==(interval<T> lhs, interval<T> rhs)
{
    auto empty = [](interval<T> x) { return !(x.lb <= x.ub); };


    return (empty(lhs) && empty(rhs)) || (lhs.lb == rhs.lb && lhs.ub == rhs.ub);
}

template<typename T>
struct split
{
    interval<T> lower_half;
    interval<T> upper_half;

    auto operator<=>(const split &) const = default;
};

} // namespace cu

#endif // CUINTERVAL_INTERVAL_H
