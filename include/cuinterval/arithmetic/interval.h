#ifndef CUINTERVAL_ARITHMETIC_INTERVAL_H
#define CUINTERVAL_ARITHMETIC_INTERVAL_H

template<typename T>
struct interval
{
    T lb;
    T ub;
};

template<typename T>
bool operator==(interval<T> lhs, interval<T> rhs)
{
    if (empty(lhs) && empty(rhs)) {
        return true;
    }

    return lhs.lb == rhs.lb && lhs.ub == rhs.ub;
}

#endif // CUINTERVAL_ARITHMETIC_INTERVAL_H
