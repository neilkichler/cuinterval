#ifndef CUINTERVAL_ARITHMETIC_BASIC_CUH
#define CUINTERVAL_ARITHMETIC_BASIC_CUH

#include "interval.h"
#include "intrinsic.cuh"

#include <cassert>
#include <numbers>

// IEEE Std 1788.1-2017, Table 4.1

//
// Constant intervals
//

template<typename T>
inline __device__ interval<T> pos_inf()
{
    return { intrinsic::pos_inf<T>(), intrinsic::pos_inf<T>() };
}

template<typename T>
constexpr inline __device__ interval<T> empty()
{
    return { intrinsic::pos_inf<T>(), intrinsic::neg_inf<T>() };
}

template<typename T>
constexpr inline __device__ interval<T> entire()
{
    return { intrinsic::neg_inf<T>(), intrinsic::pos_inf<T>() };
}

template<typename T>
inline __device__ __host__ bool empty(interval<T> x)
{
    return !(x.lb <= x.ub);
}

template<typename T>
inline __device__ bool just_zero(interval<T> x)
{
    return x.lb == 0 && x.ub == 0;
}

//
// Basic arithmetic operations
//

template<typename T>
inline __device__ interval<T> neg(interval<T> x)
{
    return { -x.ub, -x.lb };
}

template<typename T>
inline __device__ interval<T> add(interval<T> a, interval<T> b)
{
    return { intrinsic::add_down(a.lb, b.lb), intrinsic::add_up(a.ub, b.ub) };
}

template<typename T>
inline __device__ interval<T> sub(interval<T> a, interval<T> b)
{
    return { intrinsic::sub_down(a.lb, b.ub), intrinsic::sub_up(a.ub, b.lb) };
}

template<typename T>
inline __device__ interval<T> mul(interval<T> a, interval<T> b)
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
inline __device__ interval<T> fma(interval<T> x, interval<T> y, interval<T> z)
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
inline __device__ interval<T> sqr(interval<T> x)
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
inline __device__ interval<T> sqrt(interval<T> x)
{
    return { x.lb <= 0 && x.ub > 0 ? 0 : intrinsic::sqrt_down(x.lb),
             intrinsic::sqrt_up(x.ub) };
}

template<typename T>
inline __device__ interval<T> cbrt(interval<T> x)
{
    if (empty(x)) {
        return x;
    }

    return { intrinsic::prev_floating(cbrt(x.lb)),
             intrinsic::next_floating(cbrt(x.ub)) };
}

template<typename T>
inline __device__ interval<T> recip(interval<T> a)
{

    if (empty(a)) {
        return a;
    }

    constexpr auto zero = static_cast<T>(0);

    if (contains(a, zero)) {
        if (a.lb < zero && zero == a.ub) {
            return { intrinsic::neg_inf<T>(), intrinsic::rcp_up(a.lb) };
        } else if (a.lb == zero && zero < a.ub) {
            return { intrinsic::rcp_down(a.ub), intrinsic::pos_inf<T>() };
        } else if (a.lb < zero && zero < a.ub) {
            return ::entire<T>();
        } else if (a.lb == zero && zero == a.ub) {
            return ::empty<T>();
        }
    }

    interval<T> b;
    b.lb = intrinsic::rcp_down(a.ub);
    b.ub = intrinsic::rcp_up(a.lb);
    return b;
}

template<typename T>
inline __device__ interval<T> div(interval<T> x, interval<T> y)
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
inline __device__ T mag(interval<T> x)
{
    if (empty(x)) {
        return intrinsic::nan<T>();
    }
    return max(abs(x.lb), abs(x.ub));
}

template<typename T>
inline __device__ T mig(interval<T> x)
{
    // TODO: we might want to split up the function into the bare interval operation and this part.
    //       we could perhaps use a monad for either result or empty using expected?
    if (empty(x)) {
        return intrinsic::nan<T>();
    }

    if (contains(x, static_cast<T>(0))) {
        return {};
    }

    return min(abs(x.lb), abs(x.ub));
}

template<typename T>
inline __device__ T rad(interval<T> x)
{
    if (empty(x)) {
        return intrinsic::nan<T>();
    } else if (entire(x)) {
        return intrinsic::pos_inf<T>();
    } else {
        auto m = mid(x);
        return max(m - x.lb, x.ub - m);
    }
}

// abs(x) = [mig(x), mag(x)]
//
// extended cases:
//
// abs(x) = empty if x is empty
// abs(x) = inf   if x is +-inf
template<typename T>
inline __device__ interval<T> abs(interval<T> x)
{
    return { mig(x), mag(x) };
}

template<typename T>
inline __device__ interval<T> max(interval<T> x, interval<T> y)
{
    if (empty(x) || empty(y)) {
        return empty<T>();
    }

    return { max(x.lb, y.lb), max(x.ub, y.ub) };
}

template<typename T>
inline __device__ interval<T> min(interval<T> x, interval<T> y)
{
    if (empty(x) || empty(y)) {
        return empty<T>();
    }

    return { min(x.lb, y.lb), min(x.ub, y.ub) };
}

template<typename T>
inline __device__ interval<T> operator+(interval<T> x)
{
    return x;
}

template<typename T>
inline __device__ interval<T> operator-(interval<T> x)
{
    return neg(x);
}

template<typename T>
inline __device__ interval<T> operator+(interval<T> a, interval<T> b)
{
    return add(a, b);
}

template<typename T>
inline __device__ interval<T> operator+(T a, interval<T> b)
{
    if (isnan(a) || empty(b)) {
        return empty<T>();
    }

    return { intrinsic::add_down(a, b.lb), intrinsic::add_up(a, b.ub) };
}

template<typename T>
inline __device__ interval<T> operator+(interval<T> a, T b)
{
    return b + a;
}

template<typename T>
inline __device__ interval<T> operator-(interval<T> a, interval<T> b)
{
    return sub(a, b);
}

template<typename T>
inline __device__ interval<T> operator-(T a, interval<T> b)
{
    if (isnan(a) || empty(b)) {
        return empty<T>();
    }

    return { intrinsic::sub_down(a, b.ub), intrinsic::sub_up(a, b.lb) };
}

template<typename T>
inline __device__ interval<T> operator-(interval<T> a, T b)
{
    if (empty(a) || isnan(b)) {
        return empty<T>();
    }

    return { intrinsic::sub_down(a.lb, b), intrinsic::sub_up(a.ub, b) };
}

template<typename T>
inline __device__ interval<T> operator*(interval<T> a, interval<T> b)
{
    return mul(a, b);
}

template<typename T>
inline __device__ interval<T> operator*(T a, interval<T> b)
{
    if (isnan(a) || empty(b)) {
        return empty<T>();
    }

    constexpr auto zero = static_cast<T>(0);
    // TODO: maybe use ternary instead?
    if (a < zero) {
        return { intrinsic::mul_down(a, b.ub), intrinsic::mul_up(a, b.lb) };
    } else if (a == zero) {
        return { zero, zero };
    } else {
        return { intrinsic::mul_down(a, b.lb), intrinsic::mul_up(a, b.ub) };
    }
}

template<typename T>
inline __device__ interval<T> operator*(interval<T> a, T b)
{
    return b * a;
}

template<typename T>
inline __device__ interval<T> operator/(interval<T> a, interval<T> b)
{
    return div(a, b);
}

template<typename T>
inline __device__ interval<T> operator/(T a, interval<T> b)
{
    return div({ a, a }, b);
}

template<typename T>
inline __device__ interval<T> operator/(interval<T> a, T b)
{
    constexpr auto zero = static_cast<T>(0);
    if (empty(a) || isnan(b) || b == zero) {
        return empty<T>();
    }

    if (just_zero(a)) {
        return { zero, zero };
    }

    bool neg = b < zero;
    return { intrinsic::div_down(neg ? a.ub : a.lb, b),
             intrinsic::div_up(neg ? a.lb : a.ub, b) };
}

template<typename T>
__host__ inline __device__ bool contains(interval<T> x, T y)
{
    return x.lb <= y && y <= x.ub;
}

template<typename T>
inline __device__ bool entire(interval<T> x)
{
    return intrinsic::neg_inf<T>() == x.lb && intrinsic::pos_inf<T>() == x.ub;
}

template<typename T>
inline __device__ bool bounded(interval<T> x)
{
    // return (isfinite(x.lb) && isfinite(x.ub)) || empty(x);
    // if empty is given by +inf,-inf then the below is true
    return x.lb > intrinsic::neg_inf<T>() && x.ub < intrinsic::pos_inf<T>();
}

template<typename T>
inline __device__ T width(interval<T> x)
{
    if (empty(x)) {
        return intrinsic::nan<T>();
    }
    return intrinsic::sub_up(x.ub, x.lb);
}

template<typename T>
inline __device__ T inf(interval<T> x) { return x.lb; }

template<typename T>
inline __device__ T sup(interval<T> x) { return x.ub; }

template<typename T>
inline __device__ T mid(interval<T> x)
{
    if (empty(x)) {
        return intrinsic::nan<T>();
    } else if (entire(x)) {
        return static_cast<T>(0);
    } else if (x.lb == intrinsic::neg_inf<T>()) {
        // return std::numeric_limits<T>::lowest();
        return -0x1.fffffffffffffp+1023;
    } else if (x.ub == intrinsic::pos_inf<T>()) {
        // return std::numeric_limits<T>::max();
        return 0x1.fffffffffffffp+1023;
    } else {
        return __dmul_rd(0.5, x.lb) + __dmul_ru(0.5, x.ub);
    }

    // return (x.lb == x.ub) * x.lb + (abs(x.lb) != abs(x.ub)) * (0.5 * (x.lb + x.ub));
}

template<typename T>
inline __device__ bool equal(interval<T> a, interval<T> b)
{
    return (empty(a) && empty(b)) || (a.lb == b.lb && a.ub == b.ub);
}

template<typename T>
inline __device__ bool strict_less_or_both_inf(T x, T y)
{
    return (x < y) || ((isinf(x) || isinf(y)) && (x == y));
}

template<typename T>
inline __device__ bool subset(interval<T> a, interval<T> b)
{
    return empty(a) || ((b.lb <= a.lb) && (a.ub <= b.ub));
}

template<typename T>
inline __device__ bool interior(interval<T> a, interval<T> b)
{
    return empty(a) || (strict_less_or_both_inf(b.lb, a.lb) && strict_less_or_both_inf(a.ub, b.ub));
}

template<typename T>
inline __device__ bool disjoint(interval<T> a, interval<T> b)
{
    // return !(a.lb <= b.ub && b.lb <= a.ub);
    return empty(a)
        || empty(b)
        || strict_less_or_both_inf(b.ub, a.lb)
        || strict_less_or_both_inf(a.ub, b.lb);
}

template<typename T>
inline __device__ bool less(interval<T> a, interval<T> b)
{
    return (a.lb <= b.lb && a.ub <= b.ub);
}

template<typename T>
inline __device__ bool strict_less(interval<T> a, interval<T> b)
{
    return strict_less_or_both_inf(a.lb, b.lb) && strict_less_or_both_inf(a.ub, b.ub);
}

template<typename T>
inline __device__ bool precedes(interval<T> a, interval<T> b)
{
    return a.ub <= b.lb;
}

template<typename T>
inline __device__ bool strict_precedes(interval<T> a, interval<T> b)
{
    return empty(a) || empty(b) || a.ub < b.lb;
}

template<typename T>
inline __device__ interval<T> cancel_minus(interval<T> x, interval<T> y)
{
    if (empty(x) && bounded(y)) {
        return empty<T>();
    } else if (!bounded(x) || !bounded(y) || empty(y) || (width(x) < width(y))) {
        return entire<T>();
    } else if (width(y) <= width(x)) {
        interval<T> z { intrinsic::sub_down(x.lb, y.lb), intrinsic::sub_up(x.ub, y.ub) };

        if (z.lb > z.ub) {
            return entire<T>();
        }

        if (!bounded(z)) {
            return z;
        }

        // corner case if width(x) == width(y) in finite precision. See 12.12.5 of IA standard.
        T w_lb = intrinsic::add_down(y.lb, z.lb);
        T w_ub = intrinsic::add_up(y.ub, z.ub);

        if (width(x) == width(y) && (intrinsic::prev_floating(x.lb) > w_lb || intrinsic::next_floating(x.ub) < w_ub)) {
            return entire<T>();
        }

        return z;
    }
    return {};
}

template<typename T>
inline __device__ interval<T> cancel_plus(interval<T> x, interval<T> y)
{
    return cancel_minus(x, -y);
}

template<typename T>
inline __device__ interval<T> intersection(interval<T> x, interval<T> y)
{
    // extended
    if (disjoint(x, y)) {
        return empty<T>();
    }

    return { max(x.lb, y.lb), min(x.ub, y.ub) };
}

template<typename T>
inline __device__ interval<T> convex_hull(interval<T> x, interval<T> y)
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
inline __device__ interval<T> ceil(interval<T> x)
{
    return { intrinsic::int_up(x.lb), intrinsic::int_up(x.ub) };
}

template<typename T>
inline __device__ interval<T> floor(interval<T> x)
{
    return { intrinsic::int_down(x.lb), intrinsic::int_down(x.ub) };
}

template<typename T>
inline __device__ interval<T> trunc(interval<T> x)
{
    if (empty(x)) {
        return x;
    }

    return { intrinsic::trunc(x.lb), intrinsic::trunc(x.ub) };
}

template<typename T>
inline __device__ interval<T> sign(interval<T> x)
{
    if (empty(x)) {
        return x;
    }

    return { (x.lb != static_cast<T>(0)) * intrinsic::copy_sign(static_cast<T>(1), x.lb),
             (x.ub != static_cast<T>(0)) * intrinsic::copy_sign(static_cast<T>(1), x.ub) };
}

template<typename T>
__host__ inline __device__ bool isnai(interval<T> x)
{
    return x.lb != x.lb && x.ub != x.ub;
}

template<typename T>
inline __device__ bool is_member(T x, interval<T> y)
{
    return isfinite(x) && inf(y) <= x && x <= sup(y);
}

template<typename T>
inline __device__ bool is_singleton(interval<T> x)
{
    return x.lb == x.ub;
}

template<typename T>
inline __device__ bool is_common_interval(interval<T> x)
{
    return !empty(x) && bounded(x);
}

template<typename T>
inline __device__ bool is_atomic(interval<T> x)
{
    return empty(x) || is_singleton(x) || (intrinsic::next_floating(inf(x)) == sup(x));
}

template<typename T>
inline __device__ interval<T> round_to_nearest_even(interval<T> x)
{
    return { intrinsic::round_even(x.lb), intrinsic::round_even(x.ub) };
}

template<typename T>
inline __device__ interval<T> round_ties_to_away(interval<T> x)
{
    return { intrinsic::round_away(x.lb), intrinsic::round_away(x.ub) };
}

template<typename T>
inline __device__ interval<T> exp(interval<T> x)
{
    // NOTE: would not be needed if empty was using nan instead of inf
    if (empty(x)) {
        return x;
    }

    return { intrinsic::next_after(intrinsic::exp(x.lb), static_cast<T>(0)),
             intrinsic::next_floating(intrinsic::exp(x.ub)) };
}

template<typename T>
inline __device__ interval<T> exp2(interval<T> x)
{
    if (empty(x)) {
        return x;
    }

    return { intrinsic::next_after(intrinsic::exp2(x.lb), static_cast<T>(0)),
             intrinsic::next_floating(intrinsic::exp2(x.ub)) };
}

template<typename T>
inline __device__ interval<T> exp10(interval<T> x)
{
    if (empty(x)) {
        return x;
    }

    return { intrinsic::next_after(intrinsic::exp10(x.lb), static_cast<T>(0)),
             intrinsic::next_floating(intrinsic::exp10(x.ub)) };
}

template<typename T>
inline __device__ interval<T> expm1(interval<T> x)
{
    if (empty(x)) {
        return x;
    }

    return { intrinsic::next_after(std::expm1(x.lb), static_cast<T>(-1)), intrinsic::next_floating(std::expm1(x.ub)) };
}

template<typename T>
inline __device__ interval<T> log(interval<T> x)
{
    if (empty(x) || sup(x) == 0) {
        return empty<T>();
    }

    auto xx = intersection(x, { static_cast<T>(0), intrinsic::pos_inf<T>() });

    return { intrinsic::prev_floating(std::log(xx.lb)), intrinsic::next_floating(std::log(xx.ub)) };
}

// NOTE: The overestimation on the lower and upper bound is at most 2 ulps (unit in the last place)
//       (due to underlying function having error of at most 1 ulp).
template<typename T>
inline __device__ interval<T> log2(interval<T> x)
{
    if (empty(x) || sup(x) == 0) {
        return empty<T>();
    }

    auto xx = intersection(x, { static_cast<T>(0), intrinsic::pos_inf<T>() });
    // return { intrinsic::prev_floating(std::log2(x.lb)), intrinsic::next_floating(std::log2(x.ub)) };
    return { (xx.lb != 1) * intrinsic::prev_floating(intrinsic::prev_floating(std::log2(xx.lb))),
             (xx.ub != 1) * intrinsic::next_floating(intrinsic::next_floating(std::log2(xx.ub))) };
}

template<typename T>
inline __device__ interval<T> log10(interval<T> x)
{
    if (empty(x) || sup(x) == 0) {
        return empty<T>();
    }

    auto xx = intersection(x, { static_cast<T>(0), intrinsic::pos_inf<T>() });
    return { (xx.lb != 1) * intrinsic::prev_floating(intrinsic::prev_floating(std::log10(xx.lb))),
             (xx.ub != 1) * intrinsic::next_floating(intrinsic::next_floating(std::log10(xx.ub))) };
}

template<typename T>
inline __device__ interval<T> log1p(interval<T> x)
{
    if (empty(x) || sup(x) == -1) {
        return x;
    }

    auto xx = intersection(x, { static_cast<T>(-1), intrinsic::pos_inf<T>() });
    return { intrinsic::prev_floating(std::log1p(x.lb)), intrinsic::next_floating(std::log1p(x.ub)) };
}

template<typename T>
inline __device__ interval<T> pown(interval<T> x, std::integral auto n)
{
    if (empty(x)) {
        return x;
    } else if (n == 0) {
        return { 1, 1 };
    } else if (n == 1) {
        return x;
    } else if (n == 2) {
        return sqr(x);
    } else if (n < 0 && just_zero(x)) {
        return empty<T>();
    }

    using intrinsic::next_floating;
    using intrinsic::prev_floating;
    using intrinsic::next_after;

    if (n % 2) { // odd power
        if (entire(x)) {
            return x;
        }

        if (n > 0) {
            if (inf(x) == 0) {
                return { 0, next_floating(pow(sup(x), n)) };
            } else if (sup(x) == 0) {
                return { prev_floating(pow(inf(x), n)), 0 };
            } else {
                return { prev_floating(pow(inf(x), n)), next_floating(pow(sup(x), n)) };
            }
        } else {
            if (inf(x) >= 0) {
                if (inf(x) == 0) {
                    return { prev_floating(pow(sup(x), n)), next_floating(intrinsic::pos_inf<T>()) };
                } else {
                    return { prev_floating(pow(sup(x), n)), next_floating(pow(inf(x), n)) };
                }
            } else if (sup(x) <= 0) {
                if (sup(x) == 0) {
                    return { prev_floating(intrinsic::neg_inf<T>()), next_floating(pow(inf(x), n)) };
                } else {
                    return { prev_floating(pow(sup(x), n)), next_floating(pow(inf(x), n)) };
                }
            } else {
                return entire<T>();
            }
        }
    } else { // even power
        if (n > 0) {
            if (inf(x) >= 0) {
                return { next_after(pow(inf(x), n), 0.0), next_floating(pow(sup(x), n)) };
            } else if (sup(x) <= 0) {
                return { next_after(pow(sup(x), n), 0.0), next_floating(pow(inf(x), n)) };
            } else {
                return { next_after(pow(mig(x), n), 0.0), next_floating(pow(mag(x), n)) };
            }
        } else {
            if (inf(x) >= 0) {
                return { prev_floating(pow(sup(x), n)), next_floating(pow(inf(x), n)) };
            } else if (sup(x) <= 0) {
                return { prev_floating(pow(inf(x), n)), next_floating(pow(sup(x), n)) };
            } else {
                return { prev_floating(pow(mag(x), n)), next_floating(pow(mig(x), n)) };
            }
        }
    }
}

template<typename T>
inline __device__ interval<T> pow_(interval<T> x, T y)
{
    assert(inf(x) >= 0);

    using intrinsic::next_floating;
    using intrinsic::prev_floating;

    if (sup(x) == 0) {
        if (y > 0) {
            return { 0, 0 };
        } else {
            return empty<T>();
        }
    } else {
        if (rint(y) == y) {
            return pown(x, lrint(y));
        } else if (y == 0.5) {
            return sqrt(x);
        } else {
            interval<T> lb { prev_floating(pow(inf(x), y)), next_floating(pow(inf(x), y)) };
            interval<T> ub { prev_floating(pow(sup(x), y)), next_floating(pow(sup(x), y)) };
            return convex_hull(lb, ub);
        }
    }

    return {};
}

template<typename T>
inline __device__ interval<T> rootn(interval<T> x, std::integral auto n)
{
    if (empty(x)) {
        return x;
    }

    auto rootn_pos_n = [](interval<T> x, std::integral auto n) -> interval<T> {
        if (n == 0) {
            return empty<T>();
        } else if (n == 1) {
            return x;
        } else if (n == 2) {
            return sqrt(x);
        } else {
            bool is_odd = n % 2;
            interval<T> domain { is_odd ? intrinsic::neg_inf<T>() : static_cast<T>(0),
                                 intrinsic::pos_inf<T>() };

            x = intersection(x, domain);
            if (empty(x)) {
                return empty<T>();
            }

            return { intrinsic::next_after(pow(inf(x), 1.0 / n), domain.lb),
                     intrinsic::next_after(pow(sup(x), 1.0 / n), domain.ub) };
        }
    };

    if (n < 0) {
        return recip(rootn_pos_n(x, -n));
    } else {
        return rootn_pos_n(x, n);
    }
}

template<typename T>
inline __device__ interval<T> pow(interval<T> x, std::integral auto y)
{
    return pown(x, y);
}

template<typename T>
inline __device__ interval<T> pow(interval<T> x, interval<T> y)
{
    if (empty(y)) {
        return empty<T>();
    }

    interval<T> domain { static_cast<T>(0), intrinsic::pos_inf<T>() };
    x = intersection(x, domain);

    if (empty(x)) {
        return empty<T>();
    } else if (y.lb == y.ub) {
        return pow_(x, y.ub);
    } else {
        return convex_hull(pow_(x, y.lb), pow_(x, y.ub));
    }
}

//
// Trigonometric functions
//

template<typename T>
inline __device__ unsigned int quadrant(T v)
{
    int quotient;
    T vv  = intrinsic::next_after(intrinsic::sub_down(v, std::numbers::pi / 4), static_cast<T>(0));
    T rem = remquo(vv, std::numbers::pi / 2, &quotient);
    return static_cast<unsigned>(quotient) % 4;
};

template<typename T>
inline __device__ unsigned int quadrant_pi(T v)
{
    int quotient;
    T vv  = intrinsic::next_after(intrinsic::sub_down(v, 0.25), static_cast<T>(0));
    T rem = remquo(vv, 0.5, &quotient);
    return static_cast<unsigned>(quotient) % 4;
};

// NOTE: Prefer sinpi whenever possible to avoid immediate rounding error of pi during calculation.
template<typename T>
inline __device__ interval<T> sin(interval<T> x)
{
    if (empty(x)) {
        return x;
    }

    constexpr interval<T> pi { 0x1.921fb54442d18p+1, 0x1.921fb54442d19p+1 };
    constexpr interval<T> tau { 0x1.921fb54442d18p+2, 0x1.921fb54442d19p+2 };

    T sin_min = static_cast<T>(-1);
    T sin_max = static_cast<T>(1);

    T w           = width(x);
    T full_period = sup(tau);
    T half_period = sup(pi);

    if (w >= full_period) {
        // interval contains at least one full period -> return range of sin
        return { -1, 1 };
    }

    /*
       Determine the quadrant where x resides. We have

       q0: [0, pi/2)
       q1: [pi/2, pi)
       q2: [pi, 3pi/2)
       q3: [3pi/2, 2pi).

        NOTE: In floating point we have float64(pi/2) < pi/2 < float64(pi) < pi. So, e.g., float64(pi) is in q1 not q2!

         1 -|         ,-'''-.
            |      ,-'   |   `-.
            |    ,'      |      `.
            |  ,'        |        `.
            | /    q0    |    q1    \
            |/           |           \
        ----+-------------------------\--------------------------
            |          __           __ \           |           /  __
            |          ||/2         ||  \    q2    |    q3    /  2||
            |                            `.        |        ,'
            |                              `.      |      ,'
            |                                `-.   |   ,-'
        -1 -|                                   `-,,,-'
    */

    auto quadrant_lb = quadrant(x.lb);
    auto quadrant_ub = quadrant(x.ub);

    if (quadrant_lb == quadrant_ub) {
        if (w >= half_period) { // beyond single quadrant -> full range
            return { -1, 1 };
        } else if (quadrant_lb == 1 || quadrant_lb == 2) { // decreasing
            return { intrinsic::next_after(sin(x.ub), sin_min),
                     intrinsic::next_after(sin(x.lb), sin_max) };
        } else { // increasing
            return { intrinsic::next_after(sin(x.lb), sin_min),
                     intrinsic::next_after(sin(x.ub), sin_max) };
        }
    } else if (quadrant_lb == 3 && quadrant_ub == 0) { // increasing
        return { intrinsic::next_after(sin(x.lb), sin_min),
                 intrinsic::next_after(sin(x.ub), sin_max) };
    } else if (quadrant_lb == 1 && quadrant_ub == 2) { // decreasing
        return { intrinsic::next_after(sin(x.ub), sin_min),
                 intrinsic::next_after(sin(x.lb), sin_max) };
    } else if ((quadrant_lb == 3 || quadrant_lb == 0) && (quadrant_ub == 1 || quadrant_ub == 2)) {
        return { intrinsic::next_after(min(sin(x.lb), sin(x.ub)), sin_min), 1 };
    } else if ((quadrant_lb == 1 || quadrant_lb == 2) && (quadrant_ub == 3 || quadrant_ub == 0)) {
        return { -1, intrinsic::next_after(max(sin(x.lb), sin(x.ub)), sin_max) };
    } else {
        return { -1, 1 };
    }
}

template<typename T>
inline __device__ interval<T> sinpi(interval<T> x)
{
    if (empty(x)) {
        return x;
    }

    T sin_min = static_cast<T>(-1);
    T sin_max = static_cast<T>(1);

    T w           = width(x);
    T full_period = 2;
    T half_period = 1;

    if (w >= full_period) {
        // interval contains at least one full period -> return range of sin
        return { -1, 1 };
    }

    auto quadrant_lb = quadrant_pi(x.lb);
    auto quadrant_ub = quadrant_pi(x.ub);

    if (quadrant_lb == quadrant_ub) {
        if (w >= half_period) { // beyond single quadrant -> full range
            return { -1, 1 };
        } else if (quadrant_lb == 1 || quadrant_lb == 2) { // decreasing
            return { intrinsic::next_after(sinpi(x.ub), sin_min),
                     intrinsic::next_after(sinpi(x.lb), sin_max) };
        } else { // increasing
            return { intrinsic::next_after(sinpi(x.lb), sin_min),
                     intrinsic::next_after(sinpi(x.ub), sin_max) };
        }
    } else if (quadrant_lb == 3 && quadrant_ub == 0) { // increasing
        return { intrinsic::next_after(sinpi(x.lb), sin_min),
                 intrinsic::next_after(sinpi(x.ub), sin_max) };
    } else if (quadrant_lb == 1 && quadrant_ub == 2) { // decreasing
        return { intrinsic::next_after(sinpi(x.ub), sin_min),
                 intrinsic::next_after(sinpi(x.lb), sin_max) };
    } else if ((quadrant_lb == 3 || quadrant_lb == 0) && (quadrant_ub == 1 || quadrant_ub == 2)) {
        return { intrinsic::next_after(min(sinpi(x.lb), sinpi(x.ub)), sin_min), 1 };
    } else if ((quadrant_lb == 1 || quadrant_lb == 2) && (quadrant_ub == 3 || quadrant_ub == 0)) {
        return { -1, intrinsic::next_after(max(sinpi(x.lb), sinpi(x.ub)), sin_max) };
    } else {
        return { -1, 1 };
    }
}

// NOTE: Prefer cospi whenever possible to avoid immediate rounding error of pi during calculation.
template<typename T>
inline __device__ interval<T> cos(interval<T> x)
{
    if (empty(x)) {
        return x;
    }

    constexpr interval<T> pi { 0x1.921fb54442d18p+1, 0x1.921fb54442d19p+1 };
    constexpr interval<T> tau { 0x1.921fb54442d18p+2, 0x1.921fb54442d19p+2 };

    T cos_min = static_cast<T>(-1);
    T cos_max = static_cast<T>(1);

    T w           = width(x);
    T full_period = sup(tau);
    T half_period = sup(pi);

    if (w >= full_period) {
        // interval contains at least one full period -> return range of cos
        return { -1, 1 };
    }

    auto quadrant_lb = quadrant(x.lb);
    auto quadrant_ub = quadrant(x.ub);

    if (quadrant_lb == quadrant_ub) {
        if (w >= half_period) { // beyond single quadrant -> full range
            return { -1, 1 };
        } else if (quadrant_lb == 2 || quadrant_lb == 3) { // increasing
            return { intrinsic::next_after(cos(x.lb), cos_min),
                     intrinsic::next_after(cos(x.ub), cos_max) };
        } else { // decreasing
            return { intrinsic::next_after(cos(x.ub), cos_min),
                     intrinsic::next_after(cos(x.lb), cos_max) };
        }
    } else if (quadrant_lb == 2 && quadrant_ub == 3) { // increasing
        return { intrinsic::next_after(cos(x.lb), cos_min),
                 intrinsic::next_after(cos(x.ub), cos_max) };
    } else if (quadrant_lb == 0 && quadrant_ub == 1) { // decreasing
        return { intrinsic::next_after(cos(x.ub), cos_min),
                 intrinsic::next_after(cos(x.lb), cos_max) };
    } else if ((quadrant_lb == 2 || quadrant_lb == 3) && (quadrant_ub == 0 || quadrant_ub == 1)) {
        return { intrinsic::next_after(min(cos(x.lb), cos(x.ub)), cos_min), 1 };
    } else if ((quadrant_lb == 0 || quadrant_lb == 1) && (quadrant_ub == 2 || quadrant_ub == 3)) {
        return { -1, intrinsic::next_after(max(cos(x.lb), cos(x.ub)), cos_max) };
    } else {
        return { -1, 1 };
    }
}

template<typename T>
inline __device__ interval<T> cospi(interval<T> x)
{
    if (empty(x)) {
        return x;
    }

    T cos_min = static_cast<T>(-1);
    T cos_max = static_cast<T>(1);

    T w           = width(x);
    T full_period = 2;
    T half_period = 1;

    if (w >= full_period) {
        // interval contains at least one full period -> return range of cos
        return { -1, 1 };
    }

    auto quadrant_lb = quadrant_pi(x.lb);
    auto quadrant_ub = quadrant_pi(x.ub);

    if (quadrant_lb == quadrant_ub) {
        if (w >= half_period) { // beyond single quadrant -> full range
            return { -1, 1 };
        } else if (quadrant_lb == 2 || quadrant_lb == 3) { // increasing
            return { intrinsic::next_after(cospi(x.lb), cos_min),
                     intrinsic::next_after(cospi(x.ub), cos_max) };
        } else { // decreasing
            return { intrinsic::next_after(cospi(x.ub), cos_min),
                     intrinsic::next_after(cospi(x.lb), cos_max) };
        }
    } else if (quadrant_lb == 2 && quadrant_ub == 3) { // increasing
        return { intrinsic::next_after(cospi(x.lb), cos_min),
                 intrinsic::next_after(cospi(x.ub), cos_max) };
    } else if (quadrant_lb == 0 && quadrant_ub == 1) { // decreasing
        return { intrinsic::next_after(cospi(x.ub), cos_min),
                 intrinsic::next_after(cospi(x.lb), cos_max) };
    } else if ((quadrant_lb == 2 || quadrant_lb == 3) && (quadrant_ub == 0 || quadrant_ub == 1)) {
        return { intrinsic::next_after(min(cospi(x.lb), cospi(x.ub)), cos_min), 1 };
    } else if ((quadrant_lb == 0 || quadrant_lb == 1) && (quadrant_ub == 2 || quadrant_ub == 3)) {
        return { -1, intrinsic::next_after(max(cospi(x.lb), cospi(x.ub)), cos_max) };
    } else {
        return { -1, 1 };
    }
}

template<typename T>
inline __device__ interval<T> tan(interval<T> x)
{
    if (empty(x)) {
        return x;
    }

    constexpr interval<T> pi { 0x1.921fb54442d18p+1, 0x1.921fb54442d19p+1 };

    T w = width(x);

    if (w > sup(pi)) {
        // interval contains at least one full period -> return range of tan
        return entire<T>();
    }

    auto quadrant_lb     = quadrant(x.lb);
    auto quadrant_ub     = quadrant(x.ub);
    auto quadrant_lb_mod = quadrant_lb % 2;
    auto quadrant_ub_mod = quadrant_ub % 2;

    if ((quadrant_lb_mod == 0 && quadrant_ub_mod == 1)
        || (quadrant_lb_mod == quadrant_ub_mod && quadrant_lb != quadrant_ub)) {
        // crossing an asymptote -> return range of tan
        return entire<T>();
    } else {
        return { intrinsic::prev_floating(intrinsic::prev_floating(tan(x.lb))),
                 intrinsic::next_floating(intrinsic::next_floating(tan(x.ub))) };
    }
}

template<typename T>
inline __device__ interval<T> asin(interval<T> x)
{
    if (empty(x)) {
        return x;
    }

    constexpr interval<T> pi_2 { 0x1.921fb54442d18p+0, 0x1.921fb54442d19p+0 };
    constexpr interval<T> domain { static_cast<T>(-1), static_cast<T>(1) };

    auto xx = intersection(x, domain);
    return { (xx.lb != 0) * intrinsic::next_after(intrinsic::next_after(asin(xx.lb), -pi_2.ub), -pi_2.ub),
             (xx.ub != 0) * intrinsic::next_after(intrinsic::next_after(asin(xx.ub), pi_2.ub), pi_2.ub) };
}

template<typename T>
inline __device__ interval<T> acos(interval<T> x)
{
    if (empty(x)) {
        return x;
    }

    constexpr interval<T> pi { 0x1.921fb54442d18p+1, 0x1.921fb54442d19p+1 };
    constexpr interval<T> domain { static_cast<T>(-1), static_cast<T>(1) };

    auto xx = intersection(x, domain);
    return { intrinsic::next_after(intrinsic::next_after(acos(xx.ub), static_cast<T>(0)), static_cast<T>(0)),
             intrinsic::next_after(intrinsic::next_after(acos(xx.lb), pi.ub), pi.ub) };
}

template<typename T>
inline __device__ interval<T> atan(interval<T> x)
{
    if (empty(x)) {
        return x;
    }

    constexpr interval<T> pi_2 { 0x1.921fb54442d18p+0, 0x1.921fb54442d19p+0 };

    return { intrinsic::next_after(intrinsic::next_after(atan(x.lb), -pi_2.ub), -pi_2.ub),
             intrinsic::next_after(intrinsic::next_after(atan(x.ub), pi_2.ub), pi_2.ub) };
}

template<typename T>
inline __device__ interval<T> atan2(interval<T> y, interval<T> x)
{
    if (empty(x) || empty(y)) {
        return empty<T>();
    }

    constexpr interval<T> pi_2 { 0x1.921fb54442d18p+0, 0x1.921fb54442d19p+0 };
    constexpr interval<T> pi { 0x1.921fb54442d18p+1, 0x1.921fb54442d19p+1 };
    interval<T> range { -pi.ub, pi.ub };
    interval<T> half_range { -pi_2.ub, pi_2.ub };

    using intrinsic::next_after;

    if (just_zero(x)) {
        if (just_zero(y)) {
            return empty<T>();
        } else if (y.lb >= 0) {
            return pi_2;
        } else if (y.ub <= 0) {
            return -pi_2;
        } else {
            return half_range;
        }
    } else if (x.lb > 0) {
        if (just_zero(y)) {
            return y;
        } else if (y.lb >= 0) {
            return { next_after(next_after(atan2(y.lb, x.ub), range.lb), range.lb),
                     next_after(next_after(atan2(y.ub, x.lb), range.ub), range.ub) };
        } else if (y.ub <= 0) {
            return { next_after(next_after(atan2(y.lb, x.lb), range.lb), range.lb),
                     next_after(next_after(atan2(y.ub, x.ub), range.ub), range.ub) };
        } else {
            return { next_after(next_after(atan2(y.lb, x.lb), range.lb), range.lb),
                     next_after(next_after(atan2(y.ub, x.lb), range.ub), range.ub) };
        }
    } else if (x.ub < 0) {
        if (just_zero(y)) {
            return pi;
        } else if (y.lb >= 0) {
            return { next_after(next_after(atan2(y.ub, x.ub), range.lb), range.lb),
                     next_after(next_after(abs(atan2(y.lb, x.lb)), range.ub), range.ub) };
        } else if (y.ub < 0) {
            return { next_after(next_after(atan2(y.ub, x.lb), range.lb), range.lb),
                     next_after(next_after(atan2(y.lb, x.ub), range.ub), range.ub) };
        } else {
            return range;
        }
    } else {
        if (x.lb == 0) {
            if (just_zero(y)) {
                return y;
            } else if (y.lb >= 0) {
                return { next_after(next_after(atan2(y.lb, x.ub), range.lb), range.lb),
                         pi_2.ub };
            } else if (y.ub <= 0) {
                return { -pi_2.ub, next_after(next_after(atan2(y.ub, x.ub), range.ub), range.ub) };
            } else {
                return half_range;
            }
        } else if (x.ub == 0) {
            if (just_zero(y)) {
                return pi;
            } else if (y.lb >= 0) {
                return { pi_2.lb, next_after(next_after(abs(atan2(y.lb, x.lb)), range.ub), range.ub) };
            } else if (y.ub < 0) {
                return { next_after(next_after(atan2(y.ub, x.lb), range.lb), range.lb), -pi_2.lb };
            } else {
                return range;
            }
        } else {
            if (y.lb >= 0) {
                return { next_after(next_after(atan2(y.lb, x.ub), range.lb), range.lb),
                         next_after(next_after(abs(atan2(y.lb, x.lb)), range.ub), range.ub) };
            } else if (y.ub < 0) {
                return { next_after(next_after(atan2(y.ub, x.lb), range.lb), range.lb),
                         next_after(next_after(atan2(y.ub, x.ub), range.ub), range.ub) };
            } else {
                return range;
            }
        }
    }
}

//
// Hyperbolic functions
//

template<typename T>
inline __device__ interval<T> sinh(interval<T> x)
{
    if (empty(x)) {
        return x;
    }

    return { intrinsic::next_after(intrinsic::next_after(sinh(x.lb), intrinsic::neg_inf<T>()), intrinsic::neg_inf<T>()),
             intrinsic::next_after(intrinsic::next_after(sinh(x.ub), intrinsic::pos_inf<T>()), intrinsic::pos_inf<T>()) };
}

template<typename T>
inline __device__ interval<T> cosh(interval<T> x)
{
    if (empty(x)) {
        return x;
    }

    interval<T> range { static_cast<T>(1), intrinsic::pos_inf<T>() };

    return { intrinsic::next_after(cosh(mig(x)), range.lb),
             intrinsic::next_after(cosh(mag(x)), range.ub) };
}

template<typename T>
inline __device__ interval<T> tanh(interval<T> x)
{
    if (empty(x)) {
        return x;
    }

    interval<T> range { static_cast<T>(-1), static_cast<T>(1) };

    return { intrinsic::next_after(tanh(x.lb), range.lb),
             intrinsic::next_after(tanh(x.ub), range.ub) };
}

template<typename T>
inline __device__ interval<T> asinh(interval<T> x)
{
    if (empty(x)) {
        return x;
    }

    return { intrinsic::next_after(intrinsic::next_after(asinh(x.lb), intrinsic::neg_inf<T>()), intrinsic::neg_inf<T>()),
             intrinsic::next_after(intrinsic::next_after(asinh(x.ub), intrinsic::pos_inf<T>()), intrinsic::pos_inf<T>()) };
}

template<typename T>
inline __device__ interval<T> acosh(interval<T> x)
{
    if (empty(x)) {
        return x;
    }

    interval<T> range { static_cast<T>(0), intrinsic::pos_inf<T>() };
    interval<T> domain { static_cast<T>(1), intrinsic::pos_inf<T>() };

    auto xx = intersection(x, domain);

    return { intrinsic::next_after(intrinsic::next_after(acosh(inf(xx)), range.lb), range.lb),
             intrinsic::next_after(intrinsic::next_after(acosh(sup(xx)), range.ub), range.ub) };
}

template<typename T>
inline __device__ interval<T> atanh(interval<T> x)
{
    if (empty(x)) {
        return x;
    }

    interval<T> range { intrinsic::neg_inf<T>(), intrinsic::pos_inf<T>() };
    interval<T> domain { static_cast<T>(-1), static_cast<T>(1) };

    auto xx = intersection(x, domain);

    // TODO: this should not be needed and is kind of a hack for now.
    if (xx.lb == xx.ub && (xx.lb == domain.lb || xx.lb == domain.ub)) {
        return empty<T>();
    }

    return { intrinsic::next_after(intrinsic::next_after(atanh(inf(xx)), range.lb), range.lb),
             intrinsic::next_after(intrinsic::next_after(atanh(sup(xx)), range.ub), range.ub) };
}

template<typename T>
inline __device__ interval<T> cot(interval<T> x)
{
    // return cos(x) / sin(x);
    return {};
}

template<typename T>
inline __device__ split<T> bisect(interval<T> x, T split_ratio)
{
    assert(0 <= split_ratio && split_ratio <= 1);

    if (is_atomic(x)) {
        return { x, empty<T>() };
    }

    T split_point;
    T type_min = intrinsic::neg_inf<T>();
    T type_max = intrinsic::pos_inf<T>();

    using intrinsic::next_floating;
    using intrinsic::prev_floating;

    if (entire(x)) {
        if (split_ratio == 0.5) {
            split_point = 0;
        } else if (split_ratio > 0.5) {
            split_point = prev_floating(type_max);
        } else {
            split_point = next_floating(type_min);
        }
    } else {
        if (x.lb == type_min) {
            split_point = next_floating(x.lb);
        } else if (x.ub == type_max) {
            split_point = prev_floating(x.ub);
        } else {
            split_point = split_ratio * (x.ub + x.lb * (1 / split_ratio - 1));

            if (split_point == type_min || split_point == type_max) {
                split_point = (1 - split_ratio) * x.lb + split_ratio * x.ub;
            }

            split_point = (split_point != 0) * split_point; // turn -0 to 0
        }
    }

    return { { x.lb, split_point }, { split_point, x.ub } };
}

template<typename T>
inline __device__ void mince(interval<T> x, interval<T> *xs, std::size_t out_xs_size)
{
    if (is_atomic(x)) {
        xs[0] = x;
        for (std::size_t i = 1; i < out_xs_size; i++) {
            xs[i] = empty<T>();
        }
    } else {
        T lb = x.lb;
        T ub = x.ub;

        T step = (ub - lb) / static_cast<T>(out_xs_size);

        for (std::size_t i = 0; i < out_xs_size; i++) {
            xs[i].lb = lb + i * step;
            xs[i].ub = lb + (i + 1) * step;
        }
    }
}

#endif // CUINTERVAL_ARITHMETIC_BASIC_CUH
