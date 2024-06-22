#ifndef CUINTERVAL_INTERVAL_H
#define CUINTERVAL_INTERVAL_H

template<typename T>
struct interval
{
    T lb;
    T ub;
};

template<typename T>
bool operator==(interval<T> lhs, interval<T> rhs)
{

    auto empty = [](interval<T> x) { return !(x.lb <= x.ub); };

    if (empty(lhs) && empty(rhs)) {
        return true;
    }

    return lhs.lb == rhs.lb && lhs.ub == rhs.ub;
}

template<typename T>
struct split
{
    interval<T> lower_half;
    interval<T> upper_half;

    auto operator<=>(const split &) const = default;
};

#endif // CUINTERVAL_INTERVAL_H
