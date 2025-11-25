// SPDX-FileCopyrightText: 2025 Neil Kichler
// SPDX-License-Identifier: MIT
// See end of file for full license.

/* 
    CuInterval - A CUDA interval arithmetic library

    Single-Header version 0.2.1 from commit 03eedf6626fc76d3f000dd5cd33b74079c45cb27
    Generated: 2025-11-25T15:00
*/

#ifndef CUINTERVAL_CUH
#define CUINTERVAL_CUH
// cuinterval/arithmetic/intrinsic.cuh

namespace cu::intrinsic
{
// clang-format off
    #define ROUNDED_OP(OP) \
        template<typename T> inline constexpr __device__ T OP ## _down(const T &x, typename T::value_type y); \
        template<typename T> inline constexpr __device__ T OP ## _up  (const T &x, typename T::value_type y); \
        template<typename T> inline constexpr __device__ T OP ## _down(typename T::value_type x, const T &y); \
        template<typename T> inline constexpr __device__ T OP ## _up  (typename T::value_type x, const T &y); \

    ROUNDED_OP(add)
    ROUNDED_OP(sub)
    ROUNDED_OP(mul)

    #undef ROUNDED_OP

    template<typename T> inline __device__ T fma_down  (T x, T y, T z);
    template<typename T> inline __device__ T fma_up    (T x, T y, T z);
    template<typename T> inline __device__ T add_down  (T x, T y);
    template<typename T> inline __device__ T add_up    (T x, T y);
    template<typename T> inline __device__ T sub_down  (T x, T y);
    template<typename T> inline __device__ T sub_up    (T x, T y);
    template<typename T> inline __device__ T mul_down  (T x, T y);
    template<typename T> inline __device__ T mul_up    (T x, T y);
    template<typename T> inline __device__ T div_down  (T x, T y);
    template<typename T> inline __device__ T div_up    (T x, T y);
    template<typename T> inline __device__ T median    (T x, T y);
    template<typename T> inline __device__ T min       (T x, T y);
    template<typename T> inline __device__ T max       (T x, T y);
    template<typename T> inline __device__ T copy_sign (T x, T y);
    template<typename T> inline __device__ T next_after(T x, T y);
    template<typename T> inline __device__ T rcp_down  (T x);
    template<typename T> inline __device__ T rcp_up    (T x);
    template<typename T> inline __device__ T sqrt_down (T x);
    template<typename T> inline __device__ T sqrt_up   (T x);
    template<typename T> inline __device__ T int_down  (T x);
    template<typename T> inline __device__ T int_up    (T x);
    template<typename T> inline __device__ T trunc     (T x);
    template<typename T> inline __device__ T round_away(T x);
    template<typename T> inline __device__ T round_even(T x);
    template<typename T> inline __device__ T exp       (T x);
    template<typename T> inline __device__ T exp10     (T x);
    template<typename T> inline __device__ T exp2      (T x);
    template<typename T> inline __device__ __host__ T nan();
    template<typename T> inline __device__ T pos_inf();
    template<typename T> inline __device__ T neg_inf();
    template<typename T> inline __device__ T next_floating(T x);
    template<typename T> inline __device__ T prev_floating(T x);

    template<> inline __device__ double fma_down  (double x, double y, double z) { return __fma_rd(x, y, z); }    
    template<> inline __device__ double fma_up    (double x, double y, double z) { return __fma_ru(x, y, z); }    
    template<> inline __device__ double add_down  (double x, double y) { return __dadd_rd(x, y); }
    template<> inline __device__ double add_up    (double x, double y) { return __dadd_ru(x, y); }
    template<> inline __device__ double sub_down  (double x, double y) { return __dsub_rd(x, y); }
    template<> inline __device__ double sub_up    (double x, double y) { return __dsub_ru(x, y); }
    template<> inline __device__ double mul_down  (double x, double y) { return __dmul_rd(x, y); }
    template<> inline __device__ double mul_up    (double x, double y) { return __dmul_ru(x, y); }
    template<> inline __device__ double div_down  (double x, double y) { return __ddiv_rd(x, y); }
    template<> inline __device__ double div_up    (double x, double y) { return __ddiv_ru(x, y); }
    template<> inline __device__ double median    (double x, double y) { return (x + y) * .5; }
    template<> inline __device__ double min       (double x, double y) { return fmin(x, y); }
    template<> inline __device__ double max       (double x, double y) { return fmax(x, y); }
    template<> inline __device__ double copy_sign (double x, double y) { return copysign(x, y); }
    template<> inline __device__ double next_after(double x, double y) { return nextafter(x, y); }
    template<> inline __device__ double rcp_down  (double x)           { return __drcp_rd(x); }
    template<> inline __device__ double rcp_up    (double x)           { return __drcp_ru(x); }
    template<> inline __device__ double sqrt_down (double x)           { return __dsqrt_rd(x); }
    template<> inline __device__ double sqrt_up   (double x)           { return __dsqrt_ru(x); }
    template<> inline __device__ double int_down  (double x)           { return floor(x); }
    template<> inline __device__ double int_up    (double x)           { return ceil(x); }
    template<> inline __device__ double trunc     (double x)           { return ::trunc(x); }
    template<> inline __device__ double round_away(double x)           { return round(x); }
    template<> inline __device__ double round_even(double x)           { return nearbyint(x); }
    template<> inline __device__ double exp       (double x)           { return ::exp(x); }
    template<> inline __device__ double exp10     (double x)           { return ::exp10(x); }
    template<> inline __device__ double exp2      (double x)           { return ::exp2(x); }
    template<> inline __device__ __host__ double nan()                 { return ::nan(""); }
    template<> inline __device__ double neg_inf() { return __longlong_as_double(0xfff0000000000000ull); }
    template<> inline __device__ double pos_inf() { return __longlong_as_double(0x7ff0000000000000ull); }
    template<> inline __device__ double next_floating(double x)        { return nextafter(x, intrinsic::pos_inf<double>()); }
    template<> inline __device__ double prev_floating(double x)        { return nextafter(x, intrinsic::neg_inf<double>()); }

    template<> inline __device__ float fma_down   (float x, float y, float z) { return __fmaf_rd(x, y, z); }    
    template<> inline __device__ float fma_up     (float x, float y, float z) { return __fmaf_ru(x, y, z); } 
    template<> inline __device__ float add_down   (float x, float y)   { return __fadd_rd(x, y); } 
    template<> inline __device__ float add_up     (float x, float y)   { return __fadd_ru(x, y); }
    template<> inline __device__ float sub_down   (float x, float y)   { return __fsub_rd(x, y); }
    template<> inline __device__ float sub_up     (float x, float y)   { return __fsub_ru(x, y); }
    template<> inline __device__ float mul_down   (float x, float y)   { return __fmul_rd(x, y); }
    template<> inline __device__ float mul_up     (float x, float y)   { return __fmul_ru(x, y); }
    template<> inline __device__ float div_down   (float x, float y)   { return __fdiv_rd(x, y); }
    template<> inline __device__ float div_up     (float x, float y)   { return __fdiv_ru(x, y); }
    template<> inline __device__ float median     (float x, float y)   { return (x + y) * .5f; }
    template<> inline __device__ float min        (float x, float y)   { return fminf(x, y); }
    template<> inline __device__ float max        (float x, float y)   { return fmaxf(x, y); }
    template<> inline __device__ float copy_sign  (float x, float y)   { return copysignf(x, y); }
    template<> inline __device__ float next_after (float x, float y)   { return nextafterf(x, y); }
    template<> inline __device__ float rcp_down   (float x)            { return __frcp_rd(x); }
    template<> inline __device__ float rcp_up     (float x)            { return __frcp_ru(x); }
    template<> inline __device__ float sqrt_down  (float x)            { return __fsqrt_rd(x); }
    template<> inline __device__ float sqrt_up    (float x)            { return __fsqrt_ru(x); }
    template<> inline __device__ float int_down   (float x)            { return floorf(x); }
    template<> inline __device__ float int_up     (float x)            { return ceilf(x); }
    template<> inline __device__ float trunc      (float x)            { return truncf(x); }
    template<> inline __device__ float round_away (float x)            { return roundf(x); }
    template<> inline __device__ float round_even (float x)            { return nearbyintf(x); }
    template<> inline __device__ float exp        (float x)            { return ::expf(x); }
    template<> inline __device__ float exp10      (float x)            { return ::exp10f(x); }
    template<> inline __device__ float exp2       (float x)            { return ::exp2f(x); }
    template<> inline __device__ __host__ float nan()                  { return ::nanf(""); }
    template<> inline __device__ float neg_inf() { return __int_as_float(0xff800000); }
    template<> inline __device__ float pos_inf() { return __int_as_float(0x7f800000); }
    template<> inline __device__ float next_floating(float x)          { return nextafterf(x, intrinsic::pos_inf<float>()); }
    template<> inline __device__ float prev_floating(float x)          { return nextafterf(x, intrinsic::neg_inf<float>()); }

// clang-format on
} // namespace cu::intrinsic

// cuinterval/interval.h

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

// cuinterval/numbers.h


#include <numbers>

// Explicit specialization of math constants is allowed for custom types.
// See https://eel.is/c++draft/numbers#math.constants-2.
namespace std::numbers
{

// The enclosure is chosen to be the smallest representable floating point interval
// which still contains the real value.

template<>
inline constexpr cu::interval<double>
    e_v<cu::interval<double>> = { 0x1.5bf0a8b145769p+1, 0x1.5bf0a8b14576ap+1 };

template<>
inline constexpr cu::interval<float>
    e_v<cu::interval<float>> = { 0x1.5bf0a8p+1f, 0x1.5bf0aap+1f };

template<>
inline constexpr cu::interval<double>
    log2e_v<cu::interval<double>> = { 0x1.71547652b82fep+0, 0x1.71547652b82ffp+0 };

template<>
inline constexpr cu::interval<float>
    log2e_v<cu::interval<float>> = { 0x1.715476p+0f, 0x1.715478p+0f };

template<>
inline constexpr cu::interval<double>
    log10e_v<cu::interval<double>> = { 0x1.bcb7b1526e50ep-2, 0x1.bcb7b1526e50fp-2 };

template<>
inline constexpr cu::interval<float>
    log10e_v<cu::interval<float>> = { 0x1.bcb7b0p-2f, 0x1.bcb7b2p-2f };

template<>
inline constexpr cu::interval<double>
    pi_v<cu::interval<double>> = { 0x1.921fb54442d18p+1, 0x1.921fb54442d19p+1 };

template<>
inline constexpr cu::interval<float>
    pi_v<cu::interval<float>> = { 0x1.921fb4p+1f, 0x1.921fb6p+1f };

template<>
inline constexpr cu::interval<double>
    inv_pi_v<cu::interval<double>> = { 0x1.45f306dc9c882p-2, 0x1.45f306dc9c883p-2 };

template<>
inline constexpr cu::interval<float>
    inv_pi_v<cu::interval<float>> = { 0x1.45f306p-2f, 0x1.45f308p-2f };

template<>
inline constexpr cu::interval<double>
    inv_sqrtpi_v<cu::interval<double>> = { 0x1.20dd750429b6cp-1, 0x1.20dd750429b6dp-1 };

template<>
inline constexpr cu::interval<float>
    inv_sqrtpi_v<cu::interval<float>> = { 0x1.20dd74p-1f, 0x1.20dd76p-1f };

template<>
inline constexpr cu::interval<double>
    ln2_v<cu::interval<double>> = { 0x1.62e42fefa39efp-1, 0x1.62e42fefa39f0p-1 };

template<>
inline constexpr cu::interval<float>
    ln2_v<cu::interval<float>> = { 0x1.62e42ep-1f, 0x1.62e430p-1f };

template<>
inline constexpr cu::interval<double>
    ln10_v<cu::interval<double>> = { 0x1.26bb1bbb55515p+1, 0x1.26bb1bbb55516p+1 };

template<>
inline constexpr cu::interval<float>
    ln10_v<cu::interval<float>> = { 0x1.26bb1ap+1f, 0x1.26bb1cp+1f };

template<>
inline constexpr cu::interval<double>
    sqrt2_v<cu::interval<double>> = { 0x1.6a09e667f3bcdp+0, 0x1.6a09e667f3bcep+0 };

template<>
inline constexpr cu::interval<float>
    sqrt2_v<cu::interval<float>> = { 0x1.6a09e6p+0f, 0x1.6a09e8p+0f };

template<>
inline constexpr cu::interval<double>
    sqrt3_v<cu::interval<double>> = { 0x1.bb67ae8584caap+0, 0x1.bb67ae8584cabp+0 };

template<>
inline constexpr cu::interval<float>
    sqrt3_v<cu::interval<float>> = { 0x1.bb67aep+0f, 0x1.bb67b0p+0f };

template<>
inline constexpr cu::interval<double>
    inv_sqrt3_v<cu::interval<double>> = { 0x1.279a74590331cp-1, 0x1.279a74590331dp-1 };

template<>
inline constexpr cu::interval<float>
    inv_sqrt3_v<cu::interval<float>> = { 0x1.279a74p-1f, 0x1.279a76p-1f };

template<>
inline constexpr cu::interval<double>
    egamma_v<cu::interval<double>> = { 0x1.2788cfc6fb618p-1, 0x1.2788cfc6fb619p-1 };

template<>
inline constexpr cu::interval<float>
    egamma_v<cu::interval<float>> = { 0x1.2788cep-1f, 0x1.2788d0p-1f };

template<>
inline constexpr cu::interval<double>
    phi_v<cu::interval<double>> = { 0x1.9e3779b97f4a8p+0, 0x1.9e3779b97f4a9p+0 };

template<>
inline constexpr cu::interval<float>
    phi_v<cu::interval<float>> = { 0x1.9e3778p+0f, 0x1.9e377ap+0f };

} // namespace std::numbers

// In cu:: we provide access to all the standard math constants and some additional helpful ones.
namespace cu
{

using std::numbers::e_v;
using std::numbers::egamma_v;
using std::numbers::inv_pi_v;
using std::numbers::inv_sqrt3_v;
using std::numbers::inv_sqrtpi_v;
using std::numbers::ln10_v;
using std::numbers::ln2_v;
using std::numbers::log10e_v;
using std::numbers::log2e_v;
using std::numbers::phi_v;
using std::numbers::pi_v;
using std::numbers::sqrt2_v;
using std::numbers::sqrt3_v;

template<typename T>
inline constexpr T pi_2_v; // = pi / 2

template<>
inline constexpr interval<double>
    pi_2_v<interval<double>> = { 0x1.921fb54442d18p+0, 0x1.921fb54442d19p+0 };

template<>
inline constexpr interval<float>
    pi_2_v<interval<float>> = { 0x1.921fb4p+0f, 0x1.921fb6p+0f };

template<typename T>
inline constexpr T tau_v; // = 2 * pi

template<>
inline constexpr interval<double>
    tau_v<interval<double>> = { 0x1.921fb54442d18p+2, 0x1.921fb54442d19p+2 };

template<>
inline constexpr interval<float>
    tau_v<interval<float>> = { 0x1.921fb4p+2f, 0x1.921fb6p+2f };

template<typename T>
inline constexpr T zero_v = { 0 };
template<typename T>
inline constexpr T one_v = { 1 };
template<typename T>
inline constexpr T two_v = { 2 };

inline constexpr double zero = zero_v<double>;
inline constexpr double one  = one_v<double>;
inline constexpr double two  = two_v<double>;

} // namespace cu

// cuinterval/arithmetic/basic.cuh


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
    using std::cbrt, intrinsic::next_floating, intrinsic::prev_floating;

    if (empty(x)) {
        return x;
    }

    return { prev_floating(cbrt(x.lb)), next_floating(cbrt(x.ub)) };
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

    // TODO: we might want to split up the function into the bare interval operation and this part.
    //       we could perhaps use a monad for either result or empty using expected?
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
inline constexpr __device__ interval<T> operator+(interval<T> a, T b)
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
inline constexpr __device__ interval<T> operator-(interval<T> a, T b)
{
    using intrinsic::sub_down, intrinsic::sub_up;

    if (empty(a) || isnan(b)) {
        return empty<T>();
    }

    return { sub_down(a.lb, b), sub_up(a.ub, b) };
}

template<typename T>
inline constexpr __device__ interval<T> operator-(interval<T> a, auto b)
{
    using namespace intrinsic;
    if (empty(a) || isnan(b)) {
        return empty<T>();
    }

    return { sub_down(a.lb, b), sub_up(a.ub, b) };
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
inline constexpr __device__ interval<T> operator*(interval<T> a, T b)
{
    return b * a;
}

template<typename T>
inline constexpr __device__ interval<T> operator*(interval<T> a, std::integral auto b)
{
    return a * static_cast<T>(b);
}

template<typename T>
inline constexpr __device__ interval<T> operator*(std::integral auto a, interval<T> b)
{
    return static_cast<T>(a) * b;
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
        return mul_down(0.5, x.lb) + mul_up(0.5, x.ub);
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
    using intrinsic::copy_sign;

    if (empty(x)) {
        return x;
    }

    return { (x.lb != zero_v<T>)*copy_sign(one_v<T>, x.lb),
             (x.ub != zero_v<T>)*copy_sign(one_v<T>, x.ub) };
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
    using std::exp, intrinsic::next_after, intrinsic::next_floating;

    // NOTE: would not be needed if empty was using nan instead of inf
    if (empty(x)) {
        return x;
    }

    return { next_after(exp(x.lb), zero_v<T>), next_floating(exp(x.ub)) };
}

template<typename T>
inline constexpr __device__ interval<T> exp2(interval<T> x)
{
    using std::exp2, intrinsic::next_after, intrinsic::next_floating;

    if (empty(x)) {
        return x;
    }

    return { next_after(exp2(x.lb), zero_v<T>), next_floating(exp2(x.ub)) };
}

template<typename T>
inline constexpr __device__ interval<T> exp10(interval<T> x)
{
    using intrinsic::exp10, intrinsic::next_after, intrinsic::next_floating;

    if (empty(x)) {
        return x;
    }

    return { next_after(exp10(x.lb), zero_v<T>), next_floating(exp10(x.ub)) };
}

template<typename T>
inline constexpr __device__ interval<T> expm1(interval<T> x)
{
    using std::expm1, intrinsic::next_after, intrinsic::next_floating;

    if (empty(x)) {
        return x;
    }

    return { next_after(expm1(x.lb), -one_v<T>), next_floating(expm1(x.ub)) };
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
    using std::log, intrinsic::pos_inf, intrinsic::prev_floating, intrinsic::next_floating;

    if (empty(x) || sup(x) == 0) {
        return empty<T>();
    }

    auto z = intersection(x, { zero_v<T>, pos_inf<T>() });
    return { prev_floating(log(z.lb)), next_floating(log(z.ub)) };
}

// NOTE: The overestimation on the lower and upper bound is at most 2 ulps (unit in the last place)
//       (due to underlying function having error of at most 1 ulp).
template<typename T>
inline constexpr __device__ interval<T> log2(interval<T> x)
{
    using std::log2, intrinsic::prev_floating, intrinsic::next_floating;

    if (empty(x) || sup(x) == 0) {
        return empty<T>();
    }

    auto z = intersection(x, { zero_v<T>, intrinsic::pos_inf<T>() });
    return { (x.lb != 1) * prev_floating(prev_floating(log2(z.lb))),
             (x.ub != 1) * next_floating(next_floating(log2(z.ub))) };
}

template<typename T>
inline constexpr __device__ interval<T> log10(interval<T> x)
{
    using std::log10, intrinsic::prev_floating, intrinsic::next_floating;

    if (empty(x) || sup(x) == 0) {
        return empty<T>();
    }

    auto z = intersection(x, { zero_v<T>, intrinsic::pos_inf<T>() });
    return { (x.lb != 1) * prev_floating(prev_floating(log10(z.lb))),
             (x.ub != 1) * next_floating(next_floating(log10(z.ub))) };
}

template<typename T>
inline constexpr __device__ interval<T> log1p(interval<T> x)
{
    using std::log1p, intrinsic::pos_inf, intrinsic::prev_floating, intrinsic::next_floating;

    if (empty(x) || sup(x) == -1) {
        return empty<T>();
    }

    auto z = intersection(x, { -one_v<T>, pos_inf<T>() });
    return { prev_floating(log1p(z.lb)), next_floating(log1p(z.ub)) };
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
    using intrinsic::next_after, intrinsic::next_floating, intrinsic::prev_floating;

    auto pow = [](T x, std::integral auto n) -> T {
        // The default std::pow implementation returns a double for std::pow(float, int). We want a float.
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
                return { next_after(pow(inf(x), n), T { 0.0 }), next_floating(pow(sup(x), n)) };
            } else if (sup(x) <= 0) {
                return { next_after(pow(sup(x), n), T { 0.0 }), next_floating(pow(inf(x), n)) };
            } else {
                return { next_after(pow(mig(x), n), T { 0.0 }), next_floating(pow(mag(x), n)) };
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
inline constexpr __device__ interval<T> pow_(interval<T> x, T y)
{
    assert(inf(x) >= 0);

    using intrinsic::next_floating, intrinsic::prev_floating;
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
            interval<T> lb { prev_floating(pow(inf(x), y)), next_floating(pow(inf(x), y)) };
            interval<T> ub { prev_floating(pow(sup(x), y)), next_floating(pow(sup(x), y)) };
            return convex_hull(lb, ub);
        }
    }

    return {};
}

template<typename T>
inline constexpr __device__ interval<T> rootn(interval<T> x, std::integral auto n)
{
    using std::pow, intrinsic::neg_inf, intrinsic::pos_inf, intrinsic::next_after;

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

            return { next_after(pow(inf(y), 1.0 / m), domain.lb),
                     next_after(pow(sup(y), 1.0 / m), domain.ub) };
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
inline constexpr __device__ interval<T> pow(interval<T> x, auto y)
{
    return pown(x, y);
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
    using std::max, std::min, std::sin, intrinsic::next_after;

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

    if (quadrant_lb == quadrant_ub) {
        if (w >= half_period) { // beyond single quadrant -> full range
            return { -1, 1 };
        } else if (quadrant_lb == 1 || quadrant_lb == 2) { // decreasing
            return { next_after(sin(x.ub), sin_min),
                     next_after(sin(x.lb), sin_max) };
        } else { // increasing
            return { next_after(sin(x.lb), sin_min),
                     next_after(sin(x.ub), sin_max) };
        }
    } else if (quadrant_lb == 3 && quadrant_ub == 0) { // increasing
        return { next_after(sin(x.lb), sin_min),
                 next_after(sin(x.ub), sin_max) };
    } else if (quadrant_lb == 1 && quadrant_ub == 2) { // decreasing
        return { next_after(sin(x.ub), sin_min),
                 next_after(sin(x.lb), sin_max) };
    } else if ((quadrant_lb == 3 || quadrant_lb == 0) && (quadrant_ub == 1 || quadrant_ub == 2)) {
        return { next_after(min(sin(x.lb), sin(x.ub)), sin_min), 1 };
    } else if ((quadrant_lb == 1 || quadrant_lb == 2) && (quadrant_ub == 3 || quadrant_ub == 0)) {
        return { -1, next_after(max(sin(x.lb), sin(x.ub)), sin_max) };
    } else {
        return { -1, 1 };
    }
}

template<typename T>
inline constexpr __device__ interval<T> sinpi(interval<T> x)
{
    using ::sinpi, std::max, std::min, intrinsic::next_after;

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

    if (quadrant_lb == quadrant_ub) {
        if (w >= half_period) { // beyond single quadrant -> full range
            return { -1, 1 };
        } else if (quadrant_lb == 1 || quadrant_lb == 2) { // decreasing
            return { next_after(sinpi(x.ub), sin_min),
                     next_after(sinpi(x.lb), sin_max) };
        } else { // increasing
            return { next_after(sinpi(x.lb), sin_min),
                     next_after(sinpi(x.ub), sin_max) };
        }
    } else if (quadrant_lb == 3 && quadrant_ub == 0) { // increasing
        return { next_after(sinpi(x.lb), sin_min),
                 next_after(sinpi(x.ub), sin_max) };
    } else if (quadrant_lb == 1 && quadrant_ub == 2) { // decreasing
        return { next_after(sinpi(x.ub), sin_min),
                 next_after(sinpi(x.lb), sin_max) };
    } else if ((quadrant_lb == 3 || quadrant_lb == 0) && (quadrant_ub == 1 || quadrant_ub == 2)) {
        return { next_after(min(sinpi(x.lb), sinpi(x.ub)), sin_min), 1 };
    } else if ((quadrant_lb == 1 || quadrant_lb == 2) && (quadrant_ub == 3 || quadrant_ub == 0)) {
        return { -1, next_after(max(sinpi(x.lb), sinpi(x.ub)), sin_max) };
    } else {
        return { -1, 1 };
    }
}

// NOTE: Prefer cospi whenever possible to avoid immediate rounding error of pi during calculation.
template<typename T>
inline constexpr __device__ interval<T> cos(interval<T> x)
{
    using std::cos, std::max, std::min, intrinsic::next_after;

    if (empty(x)) {
        return x;
    }

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
            return { next_after(cos(x.lb), cos_min),
                     next_after(cos(x.ub), cos_max) };
        } else { // decreasing
            return { next_after(cos(x.ub), cos_min),
                     next_after(cos(x.lb), cos_max) };
        }
    } else if (quadrant_lb == 2 && quadrant_ub == 3) { // increasing
        return { next_after(cos(x.lb), cos_min),
                 next_after(cos(x.ub), cos_max) };
    } else if (quadrant_lb == 0 && quadrant_ub == 1) { // decreasing
        return { next_after(cos(x.ub), cos_min),
                 next_after(cos(x.lb), cos_max) };
    } else if ((quadrant_lb == 2 || quadrant_lb == 3) && (quadrant_ub == 0 || quadrant_ub == 1)) {
        return { next_after(min(cos(x.lb), cos(x.ub)), cos_min), 1 };
    } else if ((quadrant_lb == 0 || quadrant_lb == 1) && (quadrant_ub == 2 || quadrant_ub == 3)) {
        return { -1, next_after(max(cos(x.lb), cos(x.ub)), cos_max) };
    } else {
        return { -1, 1 };
    }
}

template<typename T>
inline constexpr __device__ interval<T> cospi(interval<T> x)
{
    using ::cospi, std::max, std::min, intrinsic::next_after;

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

    if (quadrant_lb == quadrant_ub) {
        if (w >= half_period) { // beyond single quadrant -> full range
            return { -1, 1 };
        } else if (quadrant_lb == 2 || quadrant_lb == 3) { // increasing
            return { next_after(cospi(x.lb), cos_min),
                     next_after(cospi(x.ub), cos_max) };
        } else { // decreasing
            return { next_after(cospi(x.ub), cos_min),
                     next_after(cospi(x.lb), cos_max) };
        }
    } else if (quadrant_lb == 2 && quadrant_ub == 3) { // increasing
        return { next_after(cospi(x.lb), cos_min),
                 next_after(cospi(x.ub), cos_max) };
    } else if (quadrant_lb == 0 && quadrant_ub == 1) { // decreasing
        return { next_after(cospi(x.ub), cos_min),
                 next_after(cospi(x.lb), cos_max) };
    } else if ((quadrant_lb == 2 || quadrant_lb == 3) && (quadrant_ub == 0 || quadrant_ub == 1)) {
        return { next_after(min(cospi(x.lb), cospi(x.ub)), cos_min), 1 };
    } else if ((quadrant_lb == 0 || quadrant_lb == 1) && (quadrant_ub == 2 || quadrant_ub == 3)) {
        return { -1, next_after(max(cospi(x.lb), cospi(x.ub)), cos_max) };
    } else {
        return { -1, 1 };
    }
}

template<typename T>
inline constexpr __device__ interval<T> tan(interval<T> x)
{
    using std::tan, intrinsic::prev_floating, intrinsic::next_floating;

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
        return { prev_floating(prev_floating(tan(x.lb))),
                 next_floating(next_floating(tan(x.ub))) };
    }
}

template<typename T>
inline constexpr __device__ interval<T> asin(interval<T> x)
{
    using std::asin, intrinsic::next_after;

    if (empty(x)) {
        return x;
    }

    constexpr auto pi_2_ub = pi_2_v<interval<T>>.ub;
    constexpr interval<T> domain { -one_v<T>, one_v<T> };

    auto xx = intersection(x, domain);
    return { (xx.lb != 0) * next_after(next_after(asin(xx.lb), -pi_2_ub), -pi_2_ub),
             (xx.ub != 0) * next_after(next_after(asin(xx.ub), pi_2_ub), pi_2_ub) };
}

template<typename T>
inline constexpr __device__ interval<T> acos(interval<T> x)
{
    using std::acos, intrinsic::next_after;

    if (empty(x)) {
        return x;
    }

    constexpr auto pi = pi_v<interval<T>>;
    constexpr interval<T> domain { -one_v<T>, one_v<T> };

    auto xx = intersection(x, domain);
    return { next_after(next_after(acos(xx.ub), zero_v<T>), zero_v<T>),
             next_after(next_after(acos(xx.lb), pi.ub), pi.ub) };
}

template<typename T>
inline constexpr __device__ interval<T> atan(interval<T> x)
{
    using std::atan, intrinsic::next_after;

    if (empty(x)) {
        return x;
    }

    constexpr auto pi_2_ub = pi_2_v<interval<T>>.ub;

    return { next_after(next_after(atan(x.lb), -pi_2_ub), -pi_2_ub),
             next_after(next_after(atan(x.ub), pi_2_ub), pi_2_ub) };
}

template<typename T>
inline constexpr __device__ interval<T> atan2(interval<T> y, interval<T> x)
{
    using std::abs, std::atan2, intrinsic::next_after;

    if (empty(x) || empty(y)) {
        return empty<T>();
    }

    constexpr auto pi_2 = pi_2_v<interval<T>>;
    constexpr auto pi   = pi_v<interval<T>>;
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

template<typename T>
inline constexpr __device__ interval<T> cot(interval<T> x)
{
    using intrinsic::neg_inf, intrinsic::next_floating, intrinsic::prev_floating;

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

    if ((quadrant_lb_mod == 1 && quadrant_ub_mod == 0)
        || (quadrant_lb_mod == quadrant_ub_mod && quadrant_lb != quadrant_ub)) {

        // NOTE: some test cases treat an interval [-1, 0] in such a way that 0 is only approached from the left and thus
        //       the output range should have -infinity as lower bound. This check covers this special case.
        //       For other similar scenarios with [x, k * pi] we do not have this issue because in floating point precision
        //       we never exactly reach k * pi, i.e. float64(k * pi) < k * pi.
        if (sup(x) == 0) {
            return { neg_inf<T>(), next_floating(next_floating(cot(x.lb))) };
        }

        // crossing an asymptote -> return range of cot
        return entire<T>();
    } else {
        return { prev_floating(prev_floating(cot(x.ub))),
                 next_floating(next_floating(cot(x.lb))) };
    }
}

//
// Hyperbolic functions
//

template<typename T>
inline constexpr __device__ interval<T> sinh(interval<T> x)
{
    using std::sinh, intrinsic::neg_inf, intrinsic::pos_inf, intrinsic::next_after;

    if (empty(x)) {
        return x;
    }

    return { next_after(next_after(sinh(x.lb), neg_inf<T>()), neg_inf<T>()),
             next_after(next_after(sinh(x.ub), pos_inf<T>()), pos_inf<T>()) };
}

template<typename T>
inline constexpr __device__ interval<T> cosh(interval<T> x)
{
    using std::cosh, intrinsic::next_after, intrinsic::pos_inf;

    if (empty(x)) {
        return x;
    }

    interval<T> range { one_v<T>, intrinsic::pos_inf<T>() };

    return { next_after(cosh(mig(x)), range.lb),
             next_after(cosh(mag(x)), range.ub) };
}

template<typename T>
inline constexpr __device__ interval<T> tanh(interval<T> x)
{
    using std::tanh, intrinsic::next_after;

    if (empty(x)) {
        return x;
    }

    return { next_after(tanh(x.lb), -one_v<T>), next_after(tanh(x.ub), one_v<T>) };
}

template<typename T>
inline constexpr __device__ interval<T> asinh(interval<T> x)
{
    using std::asinh, intrinsic::neg_inf, intrinsic::pos_inf, intrinsic::next_after;

    if (empty(x)) {
        return x;
    }

    return { next_after(next_after(asinh(x.lb), neg_inf<T>()), neg_inf<T>()),
             next_after(next_after(asinh(x.ub), pos_inf<T>()), pos_inf<T>()) };
}

template<typename T>
inline constexpr __device__ interval<T> acosh(interval<T> x)
{
    using std::acosh, intrinsic::neg_inf, intrinsic::pos_inf, intrinsic::next_after;

    if (empty(x)) {
        return x;
    }

    interval<T> range { zero_v<T>, pos_inf<T>() };
    interval<T> domain { one_v<T>, pos_inf<T>() };

    auto xx = intersection(x, domain);

    return { next_after(next_after(acosh(inf(xx)), range.lb), range.lb),
             next_after(next_after(acosh(sup(xx)), range.ub), range.ub) };
}

template<typename T>
inline constexpr __device__ interval<T> atanh(interval<T> x)
{
    using std::atanh, intrinsic::neg_inf, intrinsic::pos_inf, intrinsic::next_after;

    if (empty(x)) {
        return x;
    }

    interval<T> range { neg_inf<T>(), pos_inf<T>() };
    interval<T> domain { -one_v<T>, one_v<T> };

    auto xx = intersection(x, domain);

    // TODO: this should not be needed and is kind of a hack for now.
    if (xx.lb == xx.ub && (xx.lb == domain.lb || xx.lb == domain.ub)) {
        return empty<T>();
    }

    return { next_after(next_after(atanh(inf(xx)), range.lb), range.lb),
             next_after(next_after(atanh(sup(xx)), range.ub), range.ub) };
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
    using std::erf, intrinsic::next_after;

    if (empty(x)) {
        return x;
    }

    // TODO: account for 2 ulp error
    return { next_after(erf(x.lb), -one_v<T>), next_after(erf(x.ub), one_v<T>) };
}

template<typename T>
inline constexpr __device__ interval<T> erfc(interval<T> x)
{
    using std::erfc, intrinsic::next_after;

    if (empty(x)) {
        return x;
    }

    // TODO: account for 5 ulp error
    return { next_after(erfc(x.ub), zero_v<T>), next_after(erfc(x.lb), two_v<T>) };
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

// cuinterval/format.h


#include <format>
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

template<typename T>
struct std::formatter<cu::interval<T>> : std::formatter<T>
{
    auto format(const cu::interval<T> &x, std::format_context &ctx) const
    {
        auto out = ctx.out();

        out = std::format_to(out, "[");
        out = std::formatter<T>::format(x.lb, ctx);
        out = std::format_to(out, ", ");
        out = std::formatter<T>::format(x.ub, ctx);
        return std::format_to(out, "]");
    }
};

#endif // CUINTERVAL_CUH

/*
MIT License

Copyright (c) 2024 Neil Kichler

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
