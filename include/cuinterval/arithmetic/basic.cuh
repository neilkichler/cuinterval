#ifndef CUINTERVAL_ARITHMETIC_BASIC_CUH
#define CUINTERVAL_ARITHMETIC_BASIC_CUH

#include "interval.h"
#include "intrinsic.cuh"

#include <assert.h>

// IEEE Std 1788.1-2017, Table 4.1

template<typename T>
__device__ interval<T> pos_inf()
{
    return { intrinsic::pos_inf<T>(), intrinsic::pos_inf<T>() };
}

template<typename T>
__device__ __host__ interval<T> empty()
{
    // return { intrinsic::nan<T>(), intrinsic::nan<T>() };
    // return { intrinsic::pos_inf<T>(), intrinsic::neg_inf<T>() };
    return { std::numeric_limits<T>::infinity(), -std::numeric_limits<T>::infinity() };
}

template<typename T>
__device__ __host__ interval<T> entire()
{
    return { -std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity() };
}

// template<typename T>
// __device__ /*__host__*/ interval<T> entire()
// {
//     return { intrinsic::neg_inf<T>(), intrinsic::pos_inf<T>() };
//     // return { -std::numeric_limits<T>::infinity(), std::numeric_limits<T>::infinity() };
// }

template<typename T>
__device__ __host__ bool empty(interval<T> x)
{
    // return isnan(x.lb) || isnan(x.ub);
    return !(x.lb <= x.ub);
    // return (x.lb == std::numeric_limits<T>::infinity() && x.ub == -std::numeric_limits<T>::infinity());
}

template<typename T>
__device__ bool just_zero(interval<T> x)
{
    return x.lb == 0 && x.ub == 0;
}

// Basic operations

template<typename T>
__device__ interval<T> neg(interval<T> x)
{
    return { -x.ub, -x.lb };
}

template<typename T>
__device__ interval<T> add(interval<T> a, interval<T> b)
{
    return { intrinsic::add_down(a.lb, b.lb), intrinsic::add_up(a.ub, b.ub) };
}

template<typename T>
__device__ interval<T> sub(interval<T> a, interval<T> b)
{
    return { intrinsic::sub_down(a.lb, b.ub), intrinsic::sub_up(a.ub, b.lb) };
}

template<typename T>
__device__ interval<T> mul(interval<T> a, interval<T> b)
{
    if (empty(a) || empty(b)) {
        return empty<T>();
    }

    if (just_zero(a) || just_zero(b)) {
        return {};
    }

    interval<T> c;
    c.lb = min(
        min(intrinsic::mul_down(a.lb, b.lb), intrinsic::mul_down(a.lb, b.ub)),
        min(intrinsic::mul_down(a.ub, b.lb), intrinsic::mul_down(a.ub, b.ub)));

    c.ub = max(max(intrinsic::mul_up(a.lb, b.lb), intrinsic::mul_up(a.lb, b.ub)),
               max(intrinsic::mul_up(a.ub, b.lb), intrinsic::mul_up(a.ub, b.ub)));
    return c;
}

template<typename T>
__device__ interval<T> fma(interval<T> x, interval<T> y, interval<T> z)
{
    return (x * y) + z;
    // interval<T> res;
    // res.lb = min(min(intrinsic::fma_down(x.lb, y.lb, z.lb),
    //                  intrinsic::fma_down(x.lb, y.ub, z.lb)),
    //              min(intrinsic::fma_down(x.ub, y.lb, z.lb),
    //                  intrinsic::fma_down(x.ub, y.ub, z.lb)));
    //
    // res.ub = max(max(intrinsic::fma_up(x.lb, y.lb, z.ub),
    //                  intrinsic::fma_up(x.lb, y.ub, z.ub)),
    //              max(intrinsic::fma_up(x.ub, y.lb, z.ub),
    //                  intrinsic::fma_up(x.ub, y.ub, z.ub)));
    // return res;
}

template<typename T>
__device__ interval<T> sqr(interval<T> x)
{
    if (empty(x)) {
        return x;
    } else if (x.lb >= 0) {
        return { intrinsic::mul_down(x.lb, x.lb), intrinsic::mul_up(x.ub, x.ub) };
    } else if (x.ub <= 0) {
        return { intrinsic::mul_down(x.ub, x.ub), intrinsic::mul_up(x.lb, x.lb) };
    } else {
        return { 0,
                 max(intrinsic::mul_up(x.lb, x.lb), intrinsic::mul_up(x.ub, x.ub)) };
    }
}

template<typename T>
__device__ interval<T> sqrt(interval<T> x)
{
    return { x.lb <= 0 && x.ub > 0 ? 0 : intrinsic::sqrt_down(x.lb),
             intrinsic::sqrt_up(x.ub) };
}

template<typename T>
__device__ bool contains(interval<T> x, T y)
{
    return x.lb <= y && y <= x.ub;
}

template<typename T>
__device__ interval<T> recip(interval<T> a)
{

    if (empty(a)) {
        return a;
    }

    if (contains(a, T {})) {
        if (a.lb < 0 && 0 == a.ub) {
            return { intrinsic::neg_inf<T>(), intrinsic::rcp_up(a.lb) };
            // return { intrinsic::neg_inf<T>(), __drcp_ru(a.lb) };
        } else if (a.lb == 0 && 0 < a.ub) {
            return { intrinsic::rcp_down(a.ub), intrinsic::pos_inf<T>() };
        } else if (a.lb < 0 && 0 < a.ub) {
            return ::entire<T>();
        } else if (a.lb == 0 && 0 == a.ub) {
            return ::empty<T>();
        }
    }

    interval<T> b;
    b.lb = intrinsic::rcp_down(a.ub);
    b.ub = intrinsic::rcp_up(a.lb);
    return b;
}

template<typename T>
__device__ interval<T> div(interval<T> x, interval<T> y)
{
    // return mul(a, recip(b));

    if (empty(x) || empty(y) || (y.lb == 0 && y.ub == 0)) {
        return empty<T>();
    }

    // test_equal(div(I{-infinity,-15.0}, {-3.0, 0.0}), I{5.0,infinity});

    if (y.lb > 0) {
        if (x.lb >= 0) {
            return { intrinsic::div_down(x.lb, y.ub), intrinsic::div_up(x.ub, y.lb) };
        } else if (sup(x) <= 0) {
            return { intrinsic::div_down(x.lb, y.lb), intrinsic::div_up(x.ub, y.ub) };
        } else {
            return { intrinsic::div_down(x.lb, y.lb), intrinsic::div_up(x.ub, y.lb) };
        }
    } else if (y.ub < 0) {
        if (x.lb >= 0) {
            return { intrinsic::div_down(x.ub, y.ub), intrinsic::div_up(x.lb, y.lb) };
        } else if (sup(x) <= 0) {
            return { intrinsic::div_down(x.ub, y.lb), intrinsic::div_up(x.lb, y.ub) };
        } else {
            return { intrinsic::div_down(x.ub, y.ub), intrinsic::div_up(x.lb, y.ub) };
        }
    } else {
        if (x.lb == 0 && x.ub == 0) {
            return x;
        }

        if (y.lb == 0) {
            if (x.lb >= 0) {
                return { intrinsic::div_down(x.lb, y.ub), intrinsic::pos_inf<T>() };
            } else if (x.ub <= 0) {
                return { intrinsic::neg_inf<T>(), intrinsic::div_up(x.ub, y.ub) };
            } else {
                return entire<T>();
            }
        } else if (y.ub == 0) {
            if (x.lb >= 0) {
                return { intrinsic::neg_inf<T>(), intrinsic::div_up(x.lb, y.lb) };
            } else if (x.ub <= 0) {
                return { intrinsic::div_down(x.ub, y.lb), intrinsic::pos_inf<T>() };
            } else {
                return entire<T>();
            }
        } else {
            return entire<T>();
        }
    }
    return empty<T>(); // unreachable
}

template<typename T>
__device__ T mag(interval<T> x)
{
    if (empty(x)) {
        return intrinsic::nan<T>();
    }
    return max(abs(x.lb), abs(x.ub));
}

template<typename T>
__device__ T mig(interval<T> x)
{
    // TODO: we might want to split up the function into the bare interval operation and this part.
    if (empty(x)) {
        return intrinsic::nan<T>();
    }

    if (contains(x, static_cast<T>(0))) {
        return {};
    }

    return min(abs(x.lb), abs(x.ub));
}

template<typename T>
__device__ T rad(interval<T> x)
{
    if (empty(x)) {
        return intrinsic::nan<T>();
    } else if (entire(x)) {
        return intrinsic::pos_inf<T>();
    } else {
        return (x.ub - x.lb) / 2;
    }
}

// abs(x) = [mig(x), mag(x)]
//
// extended cases:
//
// abs(x) = empty if x is empty
// abs(x) = inf   if x is +-inf
template<typename T>
__device__ interval<T> abs(interval<T> x)
{
    return { mig(x.lb), mag(x.ub) };
}

template<typename T>
__device__ interval<T> max(interval<T> x, interval<T> y)
{
    return { max(x.lb, y.lb), max(x.ub, y.ub) };
}

template<typename T>
__device__ interval<T> min(interval<T> x, interval<T> y)
{
    return { min(x.lb, y.lb), min(x.ub, y.ub) };
}

template<typename T>
__device__ interval<T> operator+(interval<T> x)
{
    return x;
}

template<typename T>
__device__ interval<T> operator-(interval<T> x)
{
    return neg(x);
}

template<typename T>
__device__ interval<T> operator+(interval<T> a, interval<T> b)
{
    return add(a, b);
}

template<typename T>
__device__ interval<T> operator-(interval<T> a, interval<T> b)
{
    return sub(a, b);
}

template<typename T>
__device__ interval<T> operator*(interval<T> a, interval<T> b)
{
    return mul(a, b);
}

__device__ interval<double> operator/(interval<double> a, interval<double> b)
{
    return div(a, b);
}

template<typename T>
__device__ bool entire(interval<T> x)
{
    return intrinsic::neg_inf<T>() == x.lb && intrinsic::pos_inf<T>() == x.ub;
}

template<typename T>
__device__ bool bounded(interval<T> x)
{
    return !entire(x) || empty(x);
}

template<typename T>
__device__ T width(interval<T> x)
{
    if (empty(x)) {
        return intrinsic::nan<T>();
    }
    return intrinsic::sub_up(x.ub, x.lb);
}

template<typename T>
__device__ T inf(interval<T> x) { return x.lb; }

template<typename T>
__device__ T sup(interval<T> x) { return x.ub; }

template<typename T>
__device__ T mid(interval<T> x)
{
    // if (x.lb == x.ub) {
    //     return x.lb;
    // } else if (abs(x.lb) == abs(x.ub)) {
    //     return 0;
    // } else {
    //     return 0.5 * x.lb + 0.5 * x.ub;
    // }

    return (x.lb == x.ub) * x.lb + (abs(x.lb) != abs(x.ub)) * (0.5 * x.lb + 0.5 * x.ub);
}

template<typename T>
__device__ bool equal(interval<T> a, interval<T> b)
{
    return (empty(a) && empty(b)) || (a.lb == b.lb && a.ub == b.ub);
}

template<typename T>
__device__ bool subset(interval<T> a, interval<T> b)
{
    return empty(a) || ((a.lb <= b.lb) && (b.ub <= a.ub));
}

template<typename T>
__device__ bool interior(interval<T> a, interval<T> b)
{
    return empty(a) || ((a.lb < b.lb) && (b.ub < a.ub));
}

template<typename T>
__device__ bool disjoint(interval<T> a, interval<T> b)
{
    return !(a.lb <= b.ub && b.lb <= a.ub);
}

template<typename T>
__device__ interval<T> cancel_minus(interval<T> x, interval<T> y)
{
    if (empty(x) && bounded(y)) {
        return interval<T>::empty();
    } else if (!bounded(x) || !bounded(y) || empty(y)) {
        return interval<T>::entire();
    } else if (width(y) <= width(x)) {
        return { x.lb - y.lb, x.ub - y.ub };
    } else {
        assert(0 && "TODO");
    }
}

template<typename T>
__device__ interval<T> cancel_plus(interval<T> x, interval<T> y)
{
    return cancel_minus(x, -y);
}

template<typename T>
__device__ interval<T> intersection(interval<T> x, interval<T> y)
{
    // extended
    if (empty(x) || empty(y)) {
        return interval<T>::empty();
    }

    return { max(x.lb, y.lb), min(x.ub, y.ub) };
}

template<typename T>
__device__ interval<T> convex_hull(interval<T> x, interval<T> y)
{
    // extended
    if (empty(x)) {
        return y;
    } else if (empty(y)) {
        return x;
    }

    return { min(x.lb, y.lb), max(x.ub, y.ub) };
}

template<typename T>
__device__ interval<T> ceil(interval<T> x)
{
    return { intrinsic::int_up(x.lb), intrinsic::int_up(x.ub) };
}

template<typename T>
__device__ interval<T> floor(interval<T> x)
{
    return { intrinsic::int_down(x.lb), intrinsic::int_down(x.ub) };
}

template<typename T>
__device__ interval<T> trunc(interval<T> x)
{
    return { intrinsic::int_nearest(x.lb), intrinsic::int_nearest(x.ub) };
}

#endif // CUINTERVAL_ARITHMETIC_BASIC_CUH
