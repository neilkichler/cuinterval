#ifndef CUINTERVAL_ARITHMETIC_BASIC_CUH
#define CUINTERVAL_ARITHMETIC_BASIC_CUH

#include <cuinterval/arithmetic/info.cuh>
#include <cuinterval/arithmetic/intrinsic.cuh>
#include <cuinterval/interval.h>
#include <cuinterval/numbers.h>

#include <cassert>
#include <cmath>
#include <limits>
#include <numbers>

namespace cu
{

//
// Constant intervals
//

template<typename T>
inline constexpr __device__ interval<T> empty()
{
    using intrinsic::neg_inf, intrinsic::pos_inf;
    return { pos_inf<T>(), neg_inf<T>() };
}

template<typename T>
inline constexpr __device__ interval<T> entire()
{
    using intrinsic::neg_inf, intrinsic::pos_inf;
    return { neg_inf<T>(), pos_inf<T>() };
}

//
// Basic arithmetic operations
//

template<typename T>
inline constexpr __device__ interval<T> neg(interval<T> x)
{
    return { -x.ub, -x.lb };
}

template<typename T>
inline constexpr __device__ interval<T> add(interval<T> a, interval<T> b)
{
    using intrinsic::add_down, intrinsic::add_up;
    return { add_down(a.lb, b.lb), add_up(a.ub, b.ub) };
}

template<typename T>
inline constexpr __device__ interval<T> sub(interval<T> a, interval<T> b)
{
    using intrinsic::sub_down, intrinsic::sub_up;
    return { sub_down(a.lb, b.ub), sub_up(a.ub, b.lb) };
}

template<typename T>
inline constexpr __device__ interval<T> mul(interval<T> a, interval<T> b)
{
    using intrinsic::mul_down, intrinsic::mul_up;

    if (empty(a) || empty(b)) {
        return empty<T>();
    }

    if (just_zero(a) || just_zero(b)) {
        return {};
    }

    interval<T> c;
    c.lb = min(
        min(mul_down(a.lb, b.lb), mul_down(a.lb, b.ub)),
        min(mul_down(a.ub, b.lb), mul_down(a.ub, b.ub)));

    c.ub = max(max(mul_up(a.lb, b.lb), mul_up(a.lb, b.ub)),
               max(mul_up(a.ub, b.lb), mul_up(a.ub, b.ub)));
    return c;
}

template<typename T>
inline constexpr __device__ interval<T> fma(interval<T> x, interval<T> y, interval<T> z)
{
    using intrinsic::fma_down, intrinsic::fma_up;

    if (empty(x) || empty(y) || empty(z)) {
        return empty<T>();
    }

    if (just_zero(x) || just_zero(y)) {
        return z;
    }

    interval<T> res;
    res.lb = min(min(fma_down(x.lb, y.lb, z.lb), fma_down(x.lb, y.ub, z.lb)),
                 min(fma_down(x.ub, y.lb, z.lb), fma_down(x.ub, y.ub, z.lb)));

    res.ub = max(max(fma_up(x.lb, y.lb, z.ub), fma_up(x.lb, y.ub, z.ub)),
                 max(fma_up(x.ub, y.lb, z.ub), fma_up(x.ub, y.ub, z.ub)));
    return res;
}

template<typename T>
inline constexpr __device__ interval<T> sqr(interval<T> x)
{
    using intrinsic::mul_down, intrinsic::mul_up;

    if (empty(x)) {
        return x;
    } else if (x.lb >= 0) {
        return { mul_down(x.lb, x.lb), mul_up(x.ub, x.ub) };
    } else if (x.ub <= 0) {
        return { mul_down(x.ub, x.ub), mul_up(x.lb, x.lb) };
    } else {
        return { 0,
                 max(mul_up(x.lb, x.lb), mul_up(x.ub, x.ub)) };
    }
}

template<typename T>
inline constexpr __device__ interval<T> sqrt(interval<T> x)
{
    return { x.lb <= 0 && x.ub > 0 ? 0 : intrinsic::sqrt_down(x.lb), intrinsic::sqrt_up(x.ub) };
}

template<typename T>
inline constexpr __device__ interval<T> cbrt(interval<T> x)
{
    using std::cbrt, intrinsic::round_down, intrinsic::round_up;

    if (empty(x)) {
        return x;
    }

    constexpr int n = info::cbrt<T>::max_ulp_error;
    return { round_down<n>(cbrt(x.lb)), round_up<n>(cbrt(x.ub)) };
}

template<typename T>
inline constexpr __device__ interval<T> recip(interval<T> a)
{
    using intrinsic::neg_inf, intrinsic::pos_inf, intrinsic::rcp_up, intrinsic::rcp_down;

    if (empty(a)) {
        return a;
    }

    constexpr auto zero = zero_v<T>;

    if (contains(a, zero)) {
        if (a.lb < zero && zero == a.ub) {
            return { neg_inf<T>(), rcp_up(a.lb) };
        } else if (a.lb == zero && zero < a.ub) {
            return { rcp_down(a.ub), pos_inf<T>() };
        } else if (a.lb < zero && zero < a.ub) {
            return entire<T>();
        } else if (a.lb == zero && zero == a.ub) {
            return empty<T>();
        }
    }

    return { rcp_down(a.ub), rcp_up(a.lb) };
}

template<typename T>
inline constexpr __device__ interval<T> div(interval<T> x, interval<T> y)
{
    using intrinsic::div_down, intrinsic::div_up, intrinsic::pos_inf, intrinsic::neg_inf;

    if (empty(x) || empty(y) || (y.lb == 0 && y.ub == 0)) {
        return empty<T>();
    }

    if (y.lb > 0) {
        if (x.lb >= 0) {
            return { div_down(x.lb, y.ub), div_up(x.ub, y.lb) };
        } else if (sup(x) <= 0) {
            return { div_down(x.lb, y.lb), div_up(x.ub, y.ub) };
        } else {
            return { div_down(x.lb, y.lb), div_up(x.ub, y.lb) };
        }
    } else if (y.ub < 0) {
        if (x.lb >= 0) {
            return { div_down(x.ub, y.ub), div_up(x.lb, y.lb) };
        } else if (sup(x) <= 0) {
            return { div_down(x.ub, y.lb), div_up(x.lb, y.ub) };
        } else {
            return { div_down(x.ub, y.ub), div_up(x.lb, y.ub) };
        }
    } else {
        if (x.lb == 0 && x.ub == 0) {
            return x;
        }

        if (y.lb == 0) {
            if (x.lb >= 0) {
                return { div_down(x.lb, y.ub), pos_inf<T>() };
            } else if (x.ub <= 0) {
                return { neg_inf<T>(), div_up(x.ub, y.ub) };
            } else {
                return entire<T>();
            }
        } else if (y.ub == 0) {
            if (x.lb >= 0) {
                return { neg_inf<T>(), div_up(x.lb, y.lb) };
            } else if (x.ub <= 0) {
                return { div_down(x.ub, y.lb), pos_inf<T>() };
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
inline constexpr __device__ T mag(interval<T> x)
{
    using std::max, std::abs, intrinsic::nan;

    if (empty(x)) {
        return nan<T>();
    }

    return max(abs(x.lb), abs(x.ub));
}

template<typename T>
inline constexpr __device__ T mig(interval<T> x)
{
    using std::min, intrinsic::nan;

    if (empty(x)) {
        return nan<T>();
    }

    if (contains(x, zero_v<T>)) {
        return {};
    }

    return min(abs(x.lb), abs(x.ub));
}

template<typename T>
inline constexpr __device__ T rad(interval<T> x)
{
    using intrinsic::nan, intrinsic::pos_inf;

    if (empty(x)) {
        return nan<T>();
    } else if (entire(x)) {
        return pos_inf<T>();
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
inline constexpr __device__ interval<T> abs(interval<T> x)
{
    return { mig(x), mag(x) };
}

template<typename T>
inline constexpr __device__ interval<T> fabs(interval<T> x)
{
    return abs(x);
}

template<typename T>
inline constexpr __device__ interval<T> max(interval<T> x, interval<T> y)
{
    using std::max;

    if (empty(x) || empty(y)) {
        return empty<T>();
    }

    return { max(x.lb, y.lb), max(x.ub, y.ub) };
}

template<typename T>
inline constexpr __device__ interval<T> fmax(interval<T> x, interval<T> y)
{
    using std::fmax;

    if (empty(x) || empty(y)) {
        return empty<T>();
    }

    return { fmax(x.lb, y.lb), fmax(x.ub, y.ub) };
}

template<typename T>
inline constexpr __device__ interval<T> fmin(interval<T> x, interval<T> y)
{
    using std::fmin;

    if (empty(x) || empty(y)) {
        return empty<T>();
    }

    return { fmin(x.lb, y.lb), fmin(x.ub, y.ub) };
}

template<typename T>
inline constexpr __device__ interval<T> min(interval<T> x, interval<T> y)
{
    using std::min;

    if (empty(x) || empty(y)) {
        return empty<T>();
    }

    return { min(x.lb, y.lb), min(x.ub, y.ub) };
}

template<typename T>
inline constexpr __device__ interval<T> operator+(interval<T> x)
{
    return x;
}

template<typename T>
inline constexpr __device__ interval<T> operator-(interval<T> x)
{
    return neg(x);
}

template<typename T>
inline constexpr __device__ interval<T> operator+(interval<T> a, interval<T> b)
{
    return add(a, b);
}

template<typename T>
inline constexpr __device__ interval<T> operator+(T a, interval<T> b)
{
    using intrinsic::add_down, intrinsic::add_up;

    if (isnan(a) || empty(b)) {
        return empty<T>();
    }

    return { add_down(a, b.lb), add_up(a, b.ub) };
}

template<typename T>
inline constexpr __device__ interval<T> operator+(std::floating_point auto a, interval<T> b)
{
    return static_cast<T>(a) + b;
}

template<typename T>
inline constexpr __device__ interval<T> operator+(std::integral auto a, interval<T> b)
{
    using intrinsic::add_down, intrinsic::add_up;

    if (empty(b)) {
        return empty<T>();
    }

    return { add_down(static_cast<T>(a), b.lb), add_up(static_cast<T>(a), b.ub) };
}

template<typename T>
inline constexpr __device__ interval<T> operator+(interval<T> a, T b)
{
    return b + a;
}

template<typename T>
inline constexpr __device__ interval<T> operator+(interval<T> a, std::floating_point auto b)
{
    return b + a;
}

template<typename T>
inline constexpr __device__ interval<T> operator+(interval<T> a, std::integral auto b)
{
    return b + a;
}

template<typename T>
inline constexpr __device__ interval<T> operator-(interval<T> a, interval<T> b)
{
    return sub(a, b);
}

template<typename T>
inline constexpr __device__ interval<T> operator-(T a, interval<T> b)
{
    using intrinsic::sub_down, intrinsic::sub_up;

    if (isnan(a) || empty(b)) {
        return empty<T>();
    }

    return { sub_down(a, b.ub), sub_up(a, b.lb) };
}

template<typename T>
inline constexpr __device__ interval<T> operator-(std::floating_point auto a, interval<T> b)
{
    return static_cast<T>(a) - b;
}

template<typename T>
inline constexpr __device__ interval<T> operator-(std::integral auto a, interval<T> b)
{
    return static_cast<T>(a) - b;
}

template<typename T>
inline constexpr __device__ interval<T> operator-(interval<T> a, T b)
{
    using intrinsic::sub_down, intrinsic::sub_up;

    if (empty(a) || isnan(b)) {
        return empty<T>();
    }

    return { sub_down(a.lb, b), sub_up(a.ub, b) };
}

template<typename T>
inline constexpr __device__ interval<T> operator-(interval<T> a, std::floating_point auto b)
{
    return a - static_cast<T>(b);
}

template<typename T>
inline constexpr __device__ interval<T> operator-(interval<T> a, std::integral auto b)
{
    using namespace intrinsic;
    if (empty(a)) {
        return empty<T>();
    }

    return { sub_down(a.lb, static_cast<T>(b)), sub_up(a.ub, static_cast<T>(b)) };
}

template<typename T>
inline constexpr __device__ interval<T> operator*(interval<T> a, interval<T> b)
{
    return mul(a, b);
}

template<typename T>
inline constexpr __device__ interval<T> operator*(T a, interval<T> b)
{
    using intrinsic::mul_down, intrinsic::mul_up;

    if (isnan(a) || empty(b)) {
        return empty<T>();
    }

    constexpr auto zero = zero_v<T>;

    if (a < zero) {
        return { mul_down(a, b.ub), mul_up(a, b.lb) };
    } else if (a == zero) {
        return { zero, zero };
    } else {
        return { mul_down(a, b.lb), mul_up(a, b.ub) };
    }
}

template<typename T>
inline constexpr __device__ interval<T> operator*(std::floating_point auto a, interval<T> b)
{
    return static_cast<T>(a) * b;
}

template<typename T>
inline constexpr __device__ interval<T> operator*(std::integral auto a, interval<T> b)
{
    return static_cast<T>(a) * b;
}

template<typename T>
inline constexpr __device__ interval<T> operator*(interval<T> a, T b)
{
    return b * a;
}

template<typename T>
inline constexpr __device__ interval<T> operator*(interval<T> a, std::floating_point auto b)
{
    return a * static_cast<T>(b);
}

template<typename T>
inline constexpr __device__ interval<T> operator*(interval<T> a, std::integral auto b)
{
    return a * static_cast<T>(b);
}

template<typename T>
inline constexpr __device__ interval<T> operator/(interval<T> a, interval<T> b)
{
    return div(a, b);
}

template<typename T>
inline constexpr __device__ interval<T> operator/(T a, interval<T> b)
{
    return div({ a, a }, b);
}

template<typename T>
inline constexpr __device__ interval<T> operator/(std::floating_point auto a, interval<T> b)
{
    return static_cast<T>(a) / b;
}

template<typename T>
inline constexpr __device__ interval<T> operator/(std::integral auto a, interval<T> b)
{
    return static_cast<T>(a) / b;
}

template<typename T>
inline constexpr __device__ interval<T> operator/(interval<T> a, T b)
{
    using intrinsic::div_down, intrinsic::div_up;
    constexpr auto zero = zero_v<T>;
    if (empty(a) || isnan(b) || b == zero) {
        return empty<T>();
    }

    if (just_zero(a)) {
        return { zero, zero };
    }

    bool neg = b < zero;
    return { div_down(neg ? a.ub : a.lb, b), div_up(neg ? a.lb : a.ub, b) };
}

template<typename T>
inline constexpr __device__ interval<T> operator/(interval<T> a, std::floating_point auto b)
{
    return a / static_cast<T>(b);
}

template<typename T>
inline constexpr __device__ interval<T> operator/(interval<T> a, std::integral auto b)
{
    return a / static_cast<T>(b);
}

template<typename T>
inline constexpr __device__ interval<T> &operator+=(interval<T> &a, auto b)
{
    a = a + b;
    return a;
}

template<typename T>
inline constexpr __device__ interval<T> &operator-=(interval<T> &a, auto b)
{
    a = a - b;
    return a;
}

template<typename T>
inline constexpr __device__ interval<T> &operator*=(interval<T> &a, auto b)
{
    a = a * b;
    return a;
}

template<typename T>
inline constexpr __device__ interval<T> &operator/=(interval<T> &a, auto b)
{
    a = a / b;
    return a;
}

//
// Boolean operations
//

template<typename T>
inline constexpr __device__ __host__ bool empty(interval<T> x)
{
    return !(x.lb <= x.ub);
}

template<typename T>
inline constexpr __device__ bool just_zero(interval<T> x)
{
    return x.lb == 0 && x.ub == 0;
}

template<typename T>
inline constexpr __device__ __host__ bool contains(interval<T> x, T y)
{
    return x.lb <= y && y <= x.ub;
}

template<typename T>
inline constexpr __device__ bool entire(interval<T> x)
{
    using intrinsic::neg_inf, intrinsic::pos_inf;
    return neg_inf<T>() == x.lb && pos_inf<T>() == x.ub;
}

template<typename T>
inline constexpr __device__ bool bounded(interval<T> x)
{
    using intrinsic::neg_inf, intrinsic::pos_inf;
    // return (isfinite(x.lb) && isfinite(x.ub)) || empty(x);
    // if empty is given by +inf,-inf then the below is true
    return x.lb > neg_inf<T>() && x.ub < pos_inf<T>();
}

template<typename T>
inline constexpr __device__ bool isfinite(interval<T> x)
{
    return bounded(x);
}

template<typename T>
inline constexpr __device__ bool equal(interval<T> a, interval<T> b)
{
    return a == b;
}

template<typename T>
inline constexpr __device__ bool strict_less_or_both_inf(T x, T y)
{
    return (x < y) || ((isinf(x) || isinf(y)) && (x == y));
}

template<typename T>
inline constexpr __device__ bool subset(interval<T> a, interval<T> b)
{
    return empty(a) || ((b.lb <= a.lb) && (a.ub <= b.ub));
}

template<typename T>
inline constexpr __device__ bool interior(interval<T> a, interval<T> b)
{
    return empty(a) || (strict_less_or_both_inf(b.lb, a.lb) && strict_less_or_both_inf(a.ub, b.ub));
}

template<typename T>
inline constexpr __device__ bool disjoint(interval<T> a, interval<T> b)
{
    // return !(a.lb <= b.ub && b.lb <= a.ub);
    return empty(a)
        || empty(b)
        || strict_less_or_both_inf(b.ub, a.lb)
        || strict_less_or_both_inf(a.ub, b.lb);
}

template<typename T>
inline constexpr __device__ bool less(interval<T> a, interval<T> b)
{
    return (a.lb <= b.lb && a.ub <= b.ub);
}

template<typename T>
inline constexpr __device__ bool strict_less(interval<T> a, interval<T> b)
{
    return strict_less_or_both_inf(a.lb, b.lb) && strict_less_or_both_inf(a.ub, b.ub);
}

template<typename T>
inline constexpr __device__ bool precedes(interval<T> a, interval<T> b)
{
    return a.ub <= b.lb;
}

template<typename T>
inline constexpr __device__ bool strict_precedes(interval<T> a, interval<T> b)
{
    return empty(a) || empty(b) || a.ub < b.lb;
}

// we define isinf for an interval to mean that one of its bounds is infinity.
template<typename T>
inline constexpr __device__ __host__ bool isinf(interval<T> x)
{
    using intrinsic::neg_inf, intrinsic::pos_inf;
    return x.lb == neg_inf<T>() || x.ub == pos_inf<T>();
}

// is not an interval
template<typename T>
inline constexpr __device__ __host__ bool isnai(interval<T> x)
{
    return x.lb != x.lb && x.ub != x.ub;
}

// is not a number is equivalent to isnai
template<typename T>
inline constexpr __device__ __host__ bool isnan(interval<T> x)
{
    return isnai(x);
}

template<typename T>
inline constexpr __device__ bool is_member(T x, interval<T> y)
{
    using ::isfinite;

    return isfinite(x) && inf(y) <= x && x <= sup(y);
}

template<typename T>
inline constexpr __device__ bool is_singleton(interval<T> x)
{
    return x.lb == x.ub;
}

template<typename T>
inline constexpr __device__ bool is_common_interval(interval<T> x)
{
    return !empty(x) && bounded(x);
}

template<typename T>
inline constexpr __device__ bool isnormal(interval<T> x)
{
    return is_common_interval(x);
}

template<typename T>
inline constexpr __device__ bool is_atomic(interval<T> x)
{
    using intrinsic::next_floating;
    return empty(x) || is_singleton(x) || (next_floating(inf(x)) == sup(x));
}

//
// Cancellative functions
//

template<typename T>
inline constexpr __device__ interval<T> cancel_minus(interval<T> x, interval<T> y)
{
    using namespace intrinsic;

    if (empty(x) && bounded(y)) {
        return empty<T>();
    } else if (!bounded(x) || !bounded(y) || empty(y) || (width(x) < width(y))) {
        return entire<T>();
    } else if (width(y) <= width(x)) {
        interval<T> z { sub_down(x.lb, y.lb), sub_up(x.ub, y.ub) };

        if (z.lb > z.ub) {
            return entire<T>();
        }

        if (!bounded(z)) {
            return z;
        }

        // corner case if width(x) == width(y) in finite precision. See 12.12.5 of IA standard.
        T w_lb = add_down(y.lb, z.lb);
        T w_ub = add_up(y.ub, z.ub);

        if (width(x) == width(y) && (prev_floating(x.lb) > w_lb || next_floating(x.ub) < w_ub)) {
            return entire<T>();
        }

        return z;
    }
    return {};
}

template<typename T>
inline constexpr __device__ interval<T> cancel_plus(interval<T> x, interval<T> y)
{
    return cancel_minus(x, -y);
}

//
// Utility functions
//

template<typename T>
inline constexpr __device__ T width(interval<T> x)
{
    using intrinsic::nan, intrinsic::sub_up;

    if (empty(x)) {
        return nan<T>();
    }

    return sub_up(x.ub, x.lb);
}

template<typename T>
inline constexpr __device__ T inf(interval<T> x) { return x.lb; }

template<typename T>
inline constexpr __device__ T sup(interval<T> x) { return x.ub; }

template<typename T>
inline constexpr __device__ T mid(interval<T> x)
{
    using namespace intrinsic;

    if (empty(x)) {
        return nan<T>();
    } else if (entire(x)) {
        return zero_v<T>;
    } else if (x.lb == neg_inf<T>()) {
        return std::numeric_limits<T>::lowest();
    } else if (x.ub == pos_inf<T>()) {
        return std::numeric_limits<T>::max();
    } else {
        constexpr T one_half = 0.5;
        return mul_down(one_half, x.lb) + mul_up(one_half, x.ub);
    }
}

//
// Set functions
//

template<typename T>
inline constexpr __device__ interval<T> intersection(interval<T> x, interval<T> y)
{
    using std::max, std::min;

    // extended
    if (disjoint(x, y)) {
        return empty<T>();
    }

    return { max(x.lb, y.lb), min(x.ub, y.ub) };
}

template<typename T>
inline constexpr __device__ interval<T> convex_hull(interval<T> x, interval<T> y)
{
    using std::max, std::min;

    // extended
    if (empty(x)) {
        return y;
    } else if (empty(y)) {
        return x;
    }

    return { min(x.lb, y.lb), max(x.ub, y.ub) };
}

template<typename T>
inline constexpr __device__ interval<T> hull(interval<T> x, interval<T> y)
{
    return convex_hull(x, y);
}

//
// Integer functions
//

template<typename T>
inline constexpr __device__ interval<T> ceil(interval<T> x)
{
    using intrinsic::int_up;
    return { int_up(x.lb), int_up(x.ub) };
}

template<typename T>
inline constexpr __device__ interval<T> floor(interval<T> x)
{
    using intrinsic::int_down;
    return { int_down(x.lb), int_down(x.ub) };
}

template<typename T>
inline constexpr __device__ interval<T> trunc(interval<T> x)
{
    using intrinsic::trunc;

    if (empty(x)) {
        return x;
    }

    return { trunc(x.lb), trunc(x.ub) };
}

template<typename T>
inline constexpr __device__ interval<T> round(interval<T> x)
{
    using intrinsic::round_away;
    return { round_away(x.lb), round_away(x.ub) };
}

template<typename T>
inline constexpr __device__ interval<T> nearbyint(interval<T> x)
{
    using intrinsic::round_even;
    // NOTE: The CUDA nearbyint always rounds to nearest even, regardless of the current rounding mode
    return { round_even(x.lb), round_even(x.ub) };
}

template<typename T>
inline constexpr __device__ interval<T> rint(interval<T> x)
{
    using std::rint;

    return { rint(x.lb), rint(x.ub) };
}

template<typename T>
inline constexpr __device__ interval<T> fdim(interval<T> x, interval<T> y)
{
    using std::max;

    constexpr T zero = zero_v<T>;
    auto xmy         = x - y;
    return { max(xmy.lb, zero), max(xmy.ub, zero) };
}

template<typename T>
inline constexpr __device__ interval<T> sign(interval<T> x)
{
    using std::copysign;

    if (empty(x)) {
        return x;
    }

    return { (x.lb != zero_v<T>)*copysign(one_v<T>, x.lb),
             (x.ub != zero_v<T>)*copysign(one_v<T>, x.ub) };
}

template<typename T>
inline constexpr __device__ interval<T> copysign(std::floating_point auto mag, interval<T> sgn)
{
    using std::copysign;

    return { copysign(mag, sgn.lb), copysign(mag, sgn.ub) };
}

template<typename T>
inline constexpr __device__ interval<T> copysign(interval<T> mag, std::floating_point auto sgn)
{
    using std::copysign, std::min, std::max;

    T a = copysign(mag.lb, sgn);
    T b = copysign(mag.ub, sgn);

    return { min(a, b), max(a, b) };
}

template<typename T>
inline constexpr __device__ interval<T> round_to_nearest_even(interval<T> x)
{
    using intrinsic::round_even;
    return { round_even(x.lb), round_even(x.ub) };
}

template<typename T>
inline constexpr __device__ interval<T> round_ties_to_away(interval<T> x)
{
    using intrinsic::round_away;
    return { round_away(x.lb), round_away(x.ub) };
}

//
// Power functions
//

template<typename T>
inline constexpr __device__ interval<T> exp(interval<T> x)
{
    using std::exp, intrinsic::round_down, intrinsic::round_up;

    // NOTE: would not be needed if empty was using nan instead of inf
    if (empty(x)) {
        return x;
    }

    constexpr int n = info::exp<T>::max_ulp_error;
    return { round_down<n>(exp(x.lb), zero_v<T>), round_up<n>(exp(x.ub)) };
}

template<typename T>
inline constexpr __device__ interval<T> exp2(interval<T> x)
{
    using std::exp2, intrinsic::round_down, intrinsic::round_up;

    if (empty(x)) {
        return x;
    }

    constexpr int n = info::exp2<T>::max_ulp_error;
    return { round_down<n>(exp2(x.lb), zero_v<T>), round_up<n>(exp2(x.ub)) };
}

template<typename T>
inline constexpr __device__ interval<T> exp10(interval<T> x)
{
    using intrinsic::exp10, intrinsic::round_down, intrinsic::round_up;

    if (empty(x)) {
        return x;
    }

    constexpr int n = info::exp10<T>::max_ulp_error;
    return { round_down<n>(exp10(x.lb), zero_v<T>), round_up<n>(exp10(x.ub)) };
}

template<typename T>
inline constexpr __device__ interval<T> expm1(interval<T> x)
{
    using std::expm1, intrinsic::round_down, intrinsic::round_up;

    if (empty(x)) {
        return x;
    }

    constexpr int n = info::expm1<T>::max_ulp_error;
    return { round_down<n>(expm1(x.lb), -one_v<T>), round_up<n>(expm1(x.ub)) };
}

template<typename T>
inline constexpr __device__ interval<T> ldexp(interval<T> x, int exp)
{
    using std::ldexp;

    return { ldexp(x.lb, exp), ldexp(x.ub, exp) };
}

template<typename T>
inline constexpr __device__ interval<T> scalbln(interval<T> x, long int n)
{
    using std::scalbln;

    return { scalbln(x.lb, n), scalbln(x.ub, n) };
}

template<typename T>
inline constexpr __device__ interval<T> scalbn(interval<T> x, int n)
{
    using std::scalbn;

    return { scalbn(x.lb, n), scalbn(x.ub, n) };
}

template<typename T>
inline constexpr __device__ interval<T> log(interval<T> x)
{
    using std::log, intrinsic::pos_inf, intrinsic::round_down, intrinsic::round_up;

    if (empty(x) || sup(x) == 0) {
        return empty<T>();
    }

    auto z = intersection(x, { zero_v<T>, pos_inf<T>() });

    constexpr int n = info::log<T>::max_ulp_error;
    return { round_down<n>(log(z.lb)), round_up<n>(log(z.ub)) };
}

template<typename T>
inline constexpr __device__ interval<T> log2(interval<T> x)
{
    using std::log2, intrinsic::round_down, intrinsic::round_up;

    if (empty(x) || sup(x) == 0) {
        return empty<T>();
    }

    auto z = intersection(x, { zero_v<T>, intrinsic::pos_inf<T>() });

    constexpr int n = info::log2<T>::max_ulp_error;
    return { (x.lb != 1) * round_down<n>(log2(z.lb)),
             (x.ub != 1) * round_up<n>(log2(z.ub)) };
}

template<typename T>
inline constexpr __device__ interval<T> log10(interval<T> x)
{
    using std::log10, intrinsic::round_down, intrinsic::round_up;

    if (empty(x) || sup(x) == 0) {
        return empty<T>();
    }

    auto z = intersection(x, { zero_v<T>, intrinsic::pos_inf<T>() });

    constexpr int n = info::log10<T>::max_ulp_error;
    return { (x.lb != 1) * round_down<n>(log10(z.lb)),
             (x.ub != 1) * round_up<n>(log10(z.ub)) };
}

template<typename T>
inline constexpr __device__ interval<T> log1p(interval<T> x)
{
    using std::log1p, intrinsic::pos_inf, intrinsic::round_down, intrinsic::round_up;

    if (empty(x) || sup(x) == -1) {
        return empty<T>();
    }

    auto z = intersection(x, { -one_v<T>, pos_inf<T>() });

    constexpr int n = info::log1p<T>::max_ulp_error;
    return { round_down<n>(log1p(z.lb)), round_up<n>(log1p(z.ub)) };
}

template<typename T>
inline constexpr __device__ interval<T> logb(interval<T> x)
{
    using std::logb, std::max, intrinsic::neg_inf;

    constexpr T zero = zero_v<T>;

    if (empty(x)) {
        return x;
    } else if (x.lb >= zero) {
        return { logb(x.lb), logb(x.ub) };
    } else if (x.ub <= zero) {
        return { logb(x.ub), logb(x.lb) };
    } else {
        return { neg_inf<T>(), max(logb(x.lb), logb(x.ub)) };
    }
}

template<typename T>
inline constexpr __device__ interval<T> pown(interval<T> x, std::integral auto n)
{
    using intrinsic::neg_inf, intrinsic::pos_inf, intrinsic::round_down, intrinsic::round_up;

    auto pow = [](T x, std::integral auto n) -> T {
        // The default std::pow implementation returns a double for std::pow(float, int).
        // We want a float.
        if constexpr (std::is_same_v<T, float>) {
            return powf(x, n);
        } else {
            using std::pow;
            return pow(x, n);
        }
    };

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

    constexpr int e = info::pow<T>::max_ulp_error;

    if (n % 2) { // odd power
        if (entire(x)) {
            return x;
        }

        if (n > 0) {
            if (inf(x) == 0) {
                return { 0, round_up<e>(pow(sup(x), n)) };
            } else if (sup(x) == 0) {
                return { round_down<e>(pow(inf(x), n)), 0 };
            } else {
                return { round_down<e>(pow(inf(x), n)), round_up<e>(pow(sup(x), n)) };
            }
        } else {
            if (inf(x) >= 0) {
                if (inf(x) == 0) {
                    return { round_down<e>(pow(sup(x), n)), round_up<e>(pos_inf<T>()) };
                } else {
                    return { round_down<e>(pow(sup(x), n)), round_up<e>(pow(inf(x), n)) };
                }
            } else if (sup(x) <= 0) {
                if (sup(x) == 0) {
                    return { round_down<e>(neg_inf<T>()), round_up<e>(pow(inf(x), n)) };
                } else {
                    return { round_down<e>(pow(sup(x), n)), round_up<e>(pow(inf(x), n)) };
                }
            } else {
                return entire<T>();
            }
        }
    } else { // even power
        if (n > 0) {
            if (inf(x) >= 0) {
                return { round_down<e>(pow(inf(x), n), T { 0.0 }), round_up<e>(pow(sup(x), n)) };
            } else if (sup(x) <= 0) {
                return { round_down<e>(pow(sup(x), n), T { 0.0 }), round_up<e>(pow(inf(x), n)) };
            } else {
                return { round_down<e>(pow(mig(x), n), T { 0.0 }), round_up<e>(pow(mag(x), n)) };
            }
        } else {
            if (inf(x) >= 0) {
                return { round_down<e>(pow(sup(x), n)), round_up<e>(pow(inf(x), n)) };
            } else if (sup(x) <= 0) {
                return { round_down<e>(pow(inf(x), n)), round_up<e>(pow(sup(x), n)) };
            } else {
                return { round_down<e>(pow(mag(x), n)), round_up<e>(pow(mig(x), n)) };
            }
        }
    }
}

template<typename T>
inline constexpr __device__ interval<T> pow_(interval<T> x, T y)
{
    assert(inf(x) >= 0);

    using intrinsic::round_down, intrinsic::round_up;
    using std::rint, std::lrint, std::pow, std::sqrt;

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
            constexpr int n = info::pow<T>::max_ulp_error;
            interval<T> lb { pow(inf(x), y), pow(inf(x), y) };
            interval<T> ub { pow(sup(x), y), pow(sup(x), y) };
            interval<T> res = convex_hull(lb, ub);
            return { round_down<n>(res.lb), round_up<n>(res.ub) };
        }
    }

    return {}; // unreachable
}

template<typename T>
inline constexpr __device__ interval<T> rootn(interval<T> x, std::integral auto n)
{
    using std::pow, intrinsic::neg_inf, intrinsic::pos_inf, intrinsic::round_down, intrinsic::round_up;

    if (empty(x)) {
        return x;
    }

    auto rootn_pos_n = [](interval<T> y, std::integral auto m) -> interval<T> {
        if (m == 0) {
            return empty<T>();
        } else if (m == 1) {
            return y;
        } else if (m == 2) {
            return sqrt(y);
        } else {
            bool is_odd = m % 2;
            interval<T> domain { is_odd ? neg_inf<T>() : zero_v<T>, pos_inf<T>() };

            y = intersection(y, domain);
            if (empty(y)) {
                return empty<T>();
            }

            return { round_down(pow(inf(y), 1.0 / m), domain.lb),
                     round_up(pow(sup(y), 1.0 / m), domain.ub) };
        }
    };

    if (n < 0) {
        return recip(rootn_pos_n(x, -n));
    } else {
        return rootn_pos_n(x, n);
    }
}

template<typename T>
inline constexpr __device__ interval<T> pow(interval<T> x, interval<T> y)
{
    using intrinsic::pos_inf;

    if (empty(y)) {
        return empty<T>();
    }

    interval<T> domain { zero_v<T>, pos_inf<T>() };
    x = intersection(x, domain);

    if (empty(x)) {
        return empty<T>();
    } else if (y.lb == y.ub) {
        return pow_(x, y.ub);
    } else {
        return convex_hull(pow_(x, y.lb), pow_(x, y.ub));
    }
}

template<typename T>
inline constexpr __device__ interval<T> pow(interval<T> x, std::integral auto y)
{
    return pown(x, y);
}

template<typename T>
inline constexpr __device__ interval<T> pow(interval<T> x, std::floating_point auto y)
{
    return pow(x, { y, y });
}

//
// Trigonometric functions
//

template<typename T>
inline constexpr __device__ unsigned int quadrant(T v)
{
    using intrinsic::next_after, intrinsic::sub_down;

    int quotient;
    T pi_4 { std::numbers::pi_v<T> / 4 };
    T pi_2 { std::numbers::pi_v<T> / 2 };
    T vv  = next_after(sub_down(v, pi_4), zero_v<T>);
    T rem = remquo(vv, pi_2, &quotient);
    return static_cast<unsigned>(quotient) % 4;
};

template<typename T>
inline constexpr __device__ unsigned int quadrant_pi(T v)
{
    using intrinsic::next_after, intrinsic::sub_down;

    int quotient;
    T vv  = next_after(sub_down(v, 0.25), zero_v<T>);
    T rem = remquo(vv, 0.5, &quotient);
    return static_cast<unsigned>(quotient) % 4;
};

// NOTE: Prefer sinpi whenever possible to avoid immediate rounding error of pi during calculation.
template<typename T>
inline constexpr __device__ interval<T> sin(interval<T> x)
{
    using std::max, std::min, std::sin, intrinsic::round_down, intrinsic::round_up;

    if (empty(x)) {
        return x;
    }

    constexpr auto pi  = pi_v<interval<T>>;
    constexpr auto tau = tau_v<interval<T>>;

    T sin_min = -one_v<T>;
    T sin_max = one_v<T>;

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

    constexpr auto n = info::sin<T>::max_ulp_error;

    if (quadrant_lb == quadrant_ub) {
        if (w >= half_period) { // beyond single quadrant -> full range
            return { -1, 1 };
        } else if (quadrant_lb == 1 || quadrant_lb == 2) { // decreasing
            return { round_down<n>(sin(x.ub), sin_min),
                     round_up<n>(sin(x.lb), sin_max) };
        } else { // increasing
            return { round_down<n>(sin(x.lb), sin_min),
                     round_up<n>(sin(x.ub), sin_max) };
        }
    } else if (quadrant_lb == 3 && quadrant_ub == 0) { // increasing
        return { round_down<n>(sin(x.lb), sin_min),
                 round_up<n>(sin(x.ub), sin_max) };
    } else if (quadrant_lb == 1 && quadrant_ub == 2) { // decreasing
        return { round_down<n>(sin(x.ub), sin_min),
                 round_up<n>(sin(x.lb), sin_max) };
    } else if ((quadrant_lb == 3 || quadrant_lb == 0) && (quadrant_ub == 1 || quadrant_ub == 2)) {
        return { round_down<n>(min(sin(x.lb), sin(x.ub)), sin_min), 1 };
    } else if ((quadrant_lb == 1 || quadrant_lb == 2) && (quadrant_ub == 3 || quadrant_ub == 0)) {
        return { -1, round_up<n>(max(sin(x.lb), sin(x.ub)), sin_max) };
    } else {
        return { -1, 1 };
    }
}

template<typename T>
inline constexpr __device__ interval<T> sinpi(interval<T> x)
{
    using ::sinpi, std::max, std::min, intrinsic::round_down, intrinsic::round_up;

    if (empty(x)) {
        return x;
    }

    T sin_min = -one_v<T>;
    T sin_max = one_v<T>;

    T w           = width(x);
    T full_period = 2;
    T half_period = 1;

    if (w >= full_period) {
        // interval contains at least one full period -> return range of sin
        return { -1, 1 };
    }

    auto quadrant_lb = quadrant_pi(x.lb);
    auto quadrant_ub = quadrant_pi(x.ub);

    constexpr auto n = info::sinpi<T>::max_ulp_error;

    if (quadrant_lb == quadrant_ub) {
        if (w >= half_period) { // beyond single quadrant -> full range
            return { -1, 1 };
        } else if (quadrant_lb == 1 || quadrant_lb == 2) { // decreasing
            return { round_down<n>(sinpi(x.ub), sin_min),
                     round_up<n>(sinpi(x.lb), sin_max) };
        } else { // increasing
            return { round_down<n>(sinpi(x.lb), sin_min),
                     round_up<n>(sinpi(x.ub), sin_max) };
        }
    } else if (quadrant_lb == 3 && quadrant_ub == 0) { // increasing
        return { round_down<n>(sinpi(x.lb), sin_min),
                 round_up<n>(sinpi(x.ub), sin_max) };
    } else if (quadrant_lb == 1 && quadrant_ub == 2) { // decreasing
        return { round_down<n>(sinpi(x.ub), sin_min),
                 round_up<n>(sinpi(x.lb), sin_max) };
    } else if ((quadrant_lb == 3 || quadrant_lb == 0) && (quadrant_ub == 1 || quadrant_ub == 2)) {
        return { round_down<n>(min(sinpi(x.lb), sinpi(x.ub)), sin_min), 1 };
    } else if ((quadrant_lb == 1 || quadrant_lb == 2) && (quadrant_ub == 3 || quadrant_ub == 0)) {
        return { -1, round_up<n>(max(sinpi(x.lb), sinpi(x.ub)), sin_max) };
    } else {
        return { -1, 1 };
    }
}

// NOTE: Prefer cospi whenever possible to avoid immediate rounding error of pi during calculation.
template<typename T>
inline constexpr __device__ interval<T> cos(interval<T> x)
{
    using std::cos, std::max, std::min, intrinsic::round_down, intrinsic::round_up;

    if (empty(x)) {
        return x;
    }

    constexpr auto n   = info::cos<T>::max_ulp_error;
    constexpr auto pi  = pi_v<interval<T>>;
    constexpr auto tau = tau_v<interval<T>>;

    T cos_min = -one_v<T>;
    T cos_max = one_v<T>;

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
            return { round_down<n>(cos(x.lb), cos_min),
                     round_up<n>(cos(x.ub), cos_max) };
        } else { // decreasing
            return { round_down<n>(cos(x.ub), cos_min),
                     round_up<n>(cos(x.lb), cos_max) };
        }
    } else if (quadrant_lb == 2 && quadrant_ub == 3) { // increasing
        return { round_down<n>(cos(x.lb), cos_min),
                 round_up<n>(cos(x.ub), cos_max) };
    } else if (quadrant_lb == 0 && quadrant_ub == 1) { // decreasing
        return { round_down<n>(cos(x.ub), cos_min),
                 round_up<n>(cos(x.lb), cos_max) };
    } else if ((quadrant_lb == 2 || quadrant_lb == 3) && (quadrant_ub == 0 || quadrant_ub == 1)) {
        return { round_down<n>(min(cos(x.lb), cos(x.ub)), cos_min), 1 };
    } else if ((quadrant_lb == 0 || quadrant_lb == 1) && (quadrant_ub == 2 || quadrant_ub == 3)) {
        return { -1, round_up<n>(max(cos(x.lb), cos(x.ub)), cos_max) };
    } else {
        return { -1, 1 };
    }
}

template<typename T>
inline constexpr __device__ interval<T> cospi(interval<T> x)
{
    using ::cospi, std::max, std::min, intrinsic::round_down, intrinsic::round_up;

    if (empty(x)) {
        return x;
    }

    T cos_min = -one_v<T>;
    T cos_max = one_v<T>;

    T w           = width(x);
    T full_period = 2;
    T half_period = 1;

    if (w >= full_period) {
        // interval contains at least one full period -> return range of cos
        return { -1, 1 };
    }

    auto quadrant_lb = quadrant_pi(x.lb);
    auto quadrant_ub = quadrant_pi(x.ub);

    constexpr auto n = info::cospi<T>::max_ulp_error;

    if (quadrant_lb == quadrant_ub) {
        if (w >= half_period) { // beyond single quadrant -> full range
            return { -1, 1 };
        } else if (quadrant_lb == 2 || quadrant_lb == 3) { // increasing
            return { round_down<n>(cospi(x.lb), cos_min),
                     round_up<n>(cospi(x.ub), cos_max) };
        } else { // decreasing
            return { round_down<n>(cospi(x.ub), cos_min),
                     round_up<n>(cospi(x.lb), cos_max) };
        }
    } else if (quadrant_lb == 2 && quadrant_ub == 3) { // increasing
        return { round_down<n>(cospi(x.lb), cos_min),
                 round_up<n>(cospi(x.ub), cos_max) };
    } else if (quadrant_lb == 0 && quadrant_ub == 1) { // decreasing
        return { round_down<n>(cospi(x.ub), cos_min),
                 round_up<n>(cospi(x.lb), cos_max) };
    } else if ((quadrant_lb == 2 || quadrant_lb == 3) && (quadrant_ub == 0 || quadrant_ub == 1)) {
        return { round_down<n>(min(cospi(x.lb), cospi(x.ub)), cos_min), 1 };
    } else if ((quadrant_lb == 0 || quadrant_lb == 1) && (quadrant_ub == 2 || quadrant_ub == 3)) {
        return { -1, round_up<n>(max(cospi(x.lb), cospi(x.ub)), cos_max) };
    } else {
        return { -1, 1 };
    }
}

template<typename T>
inline constexpr __device__ interval<T> tan(interval<T> x)
{
    using std::tan, intrinsic::round_down, intrinsic::round_up;

    if (empty(x)) {
        return x;
    }

    constexpr auto pi = pi_v<interval<T>>;

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
        constexpr int n = info::tan<T>::max_ulp_error;
        return { round_down<n>(tan(x.lb)), round_up<n>(tan(x.ub)) };
    }
}

template<typename T>
inline constexpr __device__ interval<T> asin(interval<T> x)
{
    using std::asin, intrinsic::round_down, intrinsic::round_up;

    if (empty(x)) {
        return x;
    }

    constexpr auto pi_2_ub = pi_2_v<interval<T>>.ub;
    constexpr interval<T> domain { -one_v<T>, one_v<T> };
    constexpr int n = info::asin<T>::max_ulp_error;

    auto xx = intersection(x, domain);
    return { (xx.lb != 0) * round_down<n>(asin(xx.lb), -pi_2_ub),
             (xx.ub != 0) * round_up<n>(asin(xx.ub), pi_2_ub) };
}

template<typename T>
inline constexpr __device__ interval<T> acos(interval<T> x)
{
    using std::acos, intrinsic::round_down, intrinsic::round_up;

    if (empty(x)) {
        return x;
    }

    constexpr auto pi = pi_v<interval<T>>;
    constexpr interval<T> domain { -one_v<T>, one_v<T> };
    constexpr int n = info::acos<T>::max_ulp_error;

    auto xx = intersection(x, domain);
    return { round_down<n>(acos(xx.ub), zero_v<T>),
             round_up<n>(acos(xx.lb), pi.ub) };
}

template<typename T>
inline constexpr __device__ interval<T> atan(interval<T> x)
{
    using std::atan, intrinsic::round_down, intrinsic::round_up;

    if (empty(x)) {
        return x;
    }

    constexpr auto pi_2_ub = pi_2_v<interval<T>>.ub;

    constexpr int n = info::atan<T>::max_ulp_error;
    return { round_down<n>(atan(x.lb), -pi_2_ub),
             round_up<n>(atan(x.ub), pi_2_ub) };
}

template<typename T>
inline constexpr __device__ interval<T> atan2(interval<T> y, interval<T> x)
{
    using std::abs, std::atan2, intrinsic::round_down, intrinsic::round_up;

    if (empty(x) || empty(y)) {
        return empty<T>();
    }

    constexpr auto pi_2 = pi_2_v<interval<T>>;
    constexpr auto pi   = pi_v<interval<T>>;
    constexpr auto n    = info::atan2<T>::max_ulp_error;
    interval<T> range { -pi.ub, pi.ub };
    interval<T> half_range { -pi_2.ub, pi_2.ub };

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
            return { round_down<n>(atan2(y.lb, x.ub), range.lb),
                     round_up<n>(atan2(y.ub, x.lb), range.ub) };
        } else if (y.ub <= 0) {
            return { round_down<n>(atan2(y.lb, x.lb), range.lb),
                     round_up<n>(atan2(y.ub, x.ub), range.ub) };
        } else {
            return { round_down<n>(atan2(y.lb, x.lb), range.lb),
                     round_up<n>(atan2(y.ub, x.lb), range.ub) };
        }
    } else if (x.ub < 0) {
        if (just_zero(y)) {
            return pi;
        } else if (y.lb >= 0) {
            return { round_down<n>(atan2(y.ub, x.ub), range.lb),
                     round_up<n>(abs(atan2(y.lb, x.lb)), range.ub) };
        } else if (y.ub < 0) {
            return { round_down<n>(atan2(y.ub, x.lb), range.lb),
                     round_up<n>(atan2(y.lb, x.ub), range.ub) };
        } else {
            return range;
        }
    } else {
        if (x.lb == 0) {
            if (just_zero(y)) {
                return y;
            } else if (y.lb >= 0) {
                return { round_down<n>(atan2(y.lb, x.ub), range.lb),
                         pi_2.ub };
            } else if (y.ub <= 0) {
                return { -pi_2.ub, round_up<n>(atan2(y.ub, x.ub), range.ub) };
            } else {
                return half_range;
            }
        } else if (x.ub == 0) {
            if (just_zero(y)) {
                return pi;
            } else if (y.lb >= 0) {
                return { pi_2.lb, round_up<n>(abs(atan2(y.lb, x.lb)), range.ub) };
            } else if (y.ub < 0) {
                return { round_down<n>(atan2(y.ub, x.lb), range.lb), -pi_2.lb };
            } else {
                return range;
            }
        } else {
            if (y.lb >= 0) {
                return { round_down<n>(atan2(y.lb, x.ub), range.lb),
                         round_up<n>(abs(atan2(y.lb, x.lb)), range.ub) };
            } else if (y.ub < 0) {
                return { round_down<n>(atan2(y.ub, x.lb), range.lb),
                         round_up<n>(atan2(y.ub, x.ub), range.ub) };
            } else {
                return range;
            }
        }
    }
}

template<typename T>
inline constexpr __device__ interval<T> cot(interval<T> x)
{
    using intrinsic::neg_inf, intrinsic::round_down, intrinsic::round_up;

    auto cot = [](T x) -> T { using std::tan; return 1 / tan(x); };

    if (empty(x)) {
        return x;
    }

    constexpr auto pi = pi_v<interval<T>>;

    T w = width(x);

    if (w >= sup(pi)) {
        // interval contains at least one full period -> return range of cot
        return entire<T>();
    }

    auto quadrant_lb     = quadrant(x.lb);
    auto quadrant_ub     = quadrant(x.ub);
    auto quadrant_lb_mod = quadrant_lb % 2;
    auto quadrant_ub_mod = quadrant_ub % 2;

    constexpr int n = info::tan<T>::max_ulp_error;

    if ((quadrant_lb_mod == 1 && quadrant_ub_mod == 0)
        || (quadrant_lb_mod == quadrant_ub_mod && quadrant_lb != quadrant_ub)) {

        // NOTE: some test cases treat an interval [-1, 0] in such a way that 0 is only approached from the left and thus
        //       the output range should have -infinity as lower bound. This check covers this special case.
        //       For other similar scenarios with [x, k * pi] we do not have this issue because in floating point precision
        //       we never exactly reach k * pi, i.e. float64(k * pi) < k * pi.
        if (sup(x) == 0) {
            return { neg_inf<T>(), round_up<n>(cot(x.lb)) };
        }

        // crossing an asymptote -> return range of cot
        return entire<T>();
    } else {
        return { round_down<n>(cot(x.ub)), round_up<n>(cot(x.lb)) };
    }
}

//
// Hyperbolic functions
//

template<typename T>
inline constexpr __device__ interval<T> sinh(interval<T> x)
{
    using std::sinh, intrinsic::round_down, intrinsic::round_up;

    if (empty(x)) {
        return x;
    }

    constexpr int n = info::sinh<T>::max_ulp_error;
    return { round_down<n>(sinh(x.lb)), round_up<n>(sinh(x.ub)) };
}

template<typename T>
inline constexpr __device__ interval<T> cosh(interval<T> x)
{
    using std::cosh, intrinsic::pos_inf, intrinsic::round_down, intrinsic::round_up;

    if (empty(x)) {
        return x;
    }

    interval<T> range { one_v<T>, intrinsic::pos_inf<T>() };

    constexpr int n = info::cosh<T>::max_ulp_error;
    return { round_down<n>(cosh(mig(x)), range.lb), round_up<n>(cosh(mag(x)), range.ub) };
}

template<typename T>
inline constexpr __device__ interval<T> tanh(interval<T> x)
{
    using std::tanh, intrinsic::round_down, intrinsic::round_up;

    if (empty(x)) {
        return x;
    }

    constexpr int n = info::tanh<T>::max_ulp_error;
    return { round_down<n>(tanh(x.lb), -one_v<T>), round_up<n>(tanh(x.ub), one_v<T>) };
}

template<typename T>
inline constexpr __device__ interval<T> asinh(interval<T> x)
{
    using std::asinh, intrinsic::round_down, intrinsic::round_up;

    if (empty(x)) {
        return x;
    }

    constexpr int n = info::asinh<T>::max_ulp_error;
    return { round_down<n>(asinh(x.lb)), round_up<n>(asinh(x.ub)) };
}

template<typename T>
inline constexpr __device__ interval<T> acosh(interval<T> x)
{
    using std::acosh, intrinsic::pos_inf, intrinsic::round_down, intrinsic::round_up;

    if (empty(x)) {
        return x;
    }

    interval<T> range { zero_v<T>, pos_inf<T>() };
    interval<T> domain { one_v<T>, pos_inf<T>() };

    auto xx = intersection(x, domain);

    constexpr int n = info::acosh<T>::max_ulp_error;
    return { round_down<n>(acosh(inf(xx)), range.lb),
             round_up<n>(acosh(sup(xx)), range.ub) };
}

template<typename T>
inline constexpr __device__ interval<T> atanh(interval<T> x)
{
    using std::atanh, intrinsic::round_down, intrinsic::round_up;

    if (empty(x)) {
        return x;
    }

    interval<T> range = entire<T>();
    interval<T> domain { -one_v<T>, one_v<T> };

    auto xx = intersection(x, domain);

    // TODO: this should not be needed and is kind of a hack for now.
    if (xx.lb == xx.ub && (xx.lb == domain.lb || xx.lb == domain.ub)) {
        return empty<T>();
    }

    constexpr int n = info::atanh<T>::max_ulp_error;
    return { round_down<n>(atanh(inf(xx)), range.lb),
             round_up<n>(atanh(sup(xx)), range.ub) };
}

template<typename T>
inline constexpr __device__ interval<T> coth(interval<T> a)
{
    // same logic as recip. In fact could just be:
    //
    // return recip(tanh(a));
    //
    // but that is much less tight than the below implementation

    if (empty(a)) {
        return a;
    }

    using namespace intrinsic;
    using std::expm1;

    constexpr T zero = 0.;
    constexpr T one  = 1.;
    constexpr T inf  = std::numeric_limits<T>::infinity();

    auto coth_down = [](T x) {
        T exp2xm1 = expm1(2.0 * x);

        if (exp2xm1 == inf) {
            return one;
        }
        return div_down(add_down(exp2xm1, 2.0), exp2xm1);
    };

    auto coth_up = [](T x) {
        T exp2xm1 = expm1(2.0 * x);

        if (exp2xm1 == inf) {
            return one;
        }
        return div_up(add_up(exp2xm1, 2.0), exp2xm1);
    };

    if (contains(a, zero)) {
        if (a.lb < zero && zero == a.ub) {
            return { -inf, next_after(coth_up(a.lb), -one) };
        } else if (a.lb == zero && zero < a.ub) {
            return { next_after(coth_down(a.ub), one), inf };
        } else if (a.lb < zero && zero < a.ub) {
            return entire<T>();
        } else if (a.lb == zero && zero == a.ub) {
            return empty<T>();
        }
    }

    return { prev_floating(coth_down(a.ub)), next_floating(coth_up(a.lb)) };
}

//
// Special functions
//

template<typename T>
inline constexpr __device__ interval<T> hypot(interval<T> x, interval<T> y)
{
    if (empty(x) || empty(y)) {
        return empty<T>();
    }

    // not using builtin CUDA hypot functions as it has a maximum ulp error of 2,
    // whereas sqr and sqrt have intrinsic rounded operations with 0 ulp error.
    auto hypot = [](interval<T> a, interval<T> b) {
        return sqrt(sqr(a) + sqr(b));
    };

    return hypot(x, y);
}

template<typename T>
inline constexpr __device__ interval<T> erf(interval<T> x)
{
    using std::erf, intrinsic::round_down, intrinsic::round_up;

    if (empty(x)) {
        return x;
    }

    constexpr int n = info::erf<T>::max_ulp_error;
    return { round_down<n>(erf(x.lb), -one_v<T>), round_up<n>(erf(x.ub), one_v<T>) };
}

template<typename T>
inline constexpr __device__ interval<T> erfc(interval<T> x)
{
    using std::erfc, intrinsic::round_down, intrinsic::round_up;

    if (empty(x)) {
        return x;
    }

    constexpr int n = info::erfc<T>::max_ulp_error;
    return { round_down<n>(erfc(x.ub), zero_v<T>), round_up<n>(erfc(x.lb), two_v<T>) };
}

// split the interval in two at the given split_ratio
template<typename T>
inline constexpr __device__ split<T> bisect(interval<T> x, T split_ratio)
{
    assert(0 <= split_ratio && split_ratio <= 1);

    if (is_atomic(x)) {
        return { x, empty<T>() };
    }

    T split_point;
    T type_min = intrinsic::neg_inf<T>();
    T type_max = intrinsic::pos_inf<T>();

    using intrinsic::next_floating, intrinsic::prev_floating;

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

// split up the interval x into out_xs_size intervals xs of equal width
template<typename T>
inline constexpr __device__ void mince(interval<T> x, interval<T> *xs, std::size_t out_xs_size)
{
    if (is_atomic(x)) {
        xs[0] = x;
        for (std::size_t i = 1; i < out_xs_size; i++) {
            xs[i] = empty<T>();
        }
    } else {
        T lb   = x.lb;
        T ub   = x.ub;
        T step = (ub - lb) / static_cast<T>(out_xs_size);

        for (std::size_t i = 0; i < out_xs_size; i++) {
            xs[i] = { lb + i * step, lb + (i + 1) * step };
        }
    }
}

} // namespace cu

#endif // CUINTERVAL_ARITHMETIC_BASIC_CUH
