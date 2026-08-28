#ifndef CUINTERVAL_ARITHMETIC_INTRINSICS_CUH
#define CUINTERVAL_ARITHMETIC_INTRINSICS_CUH

#include <cuinterval/arithmetic/intrinsics_api.h>
#include <cuinterval/interval.h>
#include <cuinterval/traits.h>

#include <bit>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>

namespace cu
{

namespace intrinsic
{
    template<std::floating_point T = double>
    inline constexpr __device__ T round_towards_(T x, T to, unsigned int n)
    {
        using std::bit_cast, std::isnan;
        using u32 = std::uint32_t;
        using u64 = std::uint64_t;
        using s32 = std::int32_t;
        using s64 = std::int64_t;

        using uint = std::conditional_t<sizeof(T) == 4, u32, u64>;
        using sint = std::conditional_t<sizeof(T) == 4, s32, s64>;

        auto y = to;

        uint ux = bit_cast<uint>(x);
        uint uy = bit_cast<uint>(y);

        if (isnan(x) || isnan(y))
            return x;

        if (x == y)
            return y; // prefer y for correct sign if x = +-0

        // set most-significant bit to 1 (sign bit)
        uint msb = uint(1) << (std::numeric_limits<uint>::digits - 1);

        // transform to monotonically increasing integers (for negative numbers)
        sint ox = (ux < msb) ? ux : (msb - ux);
        sint oy = (uy < msb) ? uy : (msb - uy);

        bool step_positive = ox < oy;
        uint abs_diff      = step_positive ? (uint)oy - (uint)ox
                                           : (uint)ox - (uint)oy;

        uint clamped_step = (abs_diff < (uint)n) ? abs_diff : n;

        ox = step_positive ? (uint)ox + clamped_step
                           : (uint)ox - clamped_step;

        uint fx = (ox < 0) ? (msb - ox) : ox;
        return bit_cast<T>(fx);
    }
} // namespace intrinsic

// clang-format off
template<>
struct intrinsics<double>
{
    static __device__ double fma_down(double x, double y, double z) { return __fma_rd(x, y, z); }
    static __device__ double fma_up(double x, double y, double z)   { return __fma_ru(x, y, z); }
    static __device__ double add_down(double x, double y)           { return __dadd_rd(x, y); }
    static __device__ double add_up(double x, double y)             { return __dadd_ru(x, y); }
    static __device__ double sub_down(double x, double y)           { return __dsub_rd(x, y); }
    static __device__ double sub_up(double x, double y)             { return __dsub_ru(x, y); }
    static __device__ double mul_down(double x, double y)           { return __dmul_rd(x, y); }
    static __device__ double mul_up(double x, double y)             { return __dmul_ru(x, y); }
    static __device__ double div_down(double x, double y)           { return __ddiv_rd(x, y); }
    static __device__ double div_up(double x, double y)             { return __ddiv_ru(x, y); }
    static __device__ double min(double x, double y)                { return ::fmin(x, y); }
    static __device__ double max(double x, double y)                { return ::fmax(x, y); }
    static __device__ double next_after(double x, double y)         { return ::nextafter(x, y); }
    static __device__ double round_towards(double x, double to, unsigned int n) { return intrinsic::round_towards_(x, to, n); }
    static __device__ double rcp_down(double x)                     { return __drcp_rd(x); }
    static __device__ double rcp_up(double x)                       { return __drcp_ru(x); }
    static __device__ double sqrt_down(double x)                    { return __dsqrt_rd(x); }
    static __device__ double sqrt_up(double x)                      { return __dsqrt_ru(x); }
    static __device__ double int_down(double x)                     { return ::floor(x); }
    static __device__ double int_up(double x)                       { return ::ceil(x); }
    static __device__ double trunc(double x)                        { return ::trunc(x); }
    static __device__ double round_away(double x)                   { return ::round(x); }
    static __device__ double round_even(double x)                   { return ::nearbyint(x); }
    static __device__ double exp(double x)                          { return ::exp(x); }
    static __device__ double exp10(double x)                        { return ::exp10(x); }
    static __device__ double exp2(double x)                         { return ::exp2(x); }
    static __device__ double next_floating(double x)                { return ::nextafter(x, +std::numeric_limits<double>::infinity()); }
    static __device__ double prev_floating(double x)                { return ::nextafter(x, -std::numeric_limits<double>::infinity()); }
};

template<>
struct intrinsics<float>
{
    static __device__ float fma_down   (float x, float y, float z) { return __fmaf_rd(x, y, z); }
    static __device__ float fma_up     (float x, float y, float z) { return __fmaf_ru(x, y, z); }
    static __device__ float add_down   (float x, float y)          { return __fadd_rd(x, y); }
    static __device__ float add_up     (float x, float y)          { return __fadd_ru(x, y); }
    static __device__ float sub_down   (float x, float y)          { return __fsub_rd(x, y); }
    static __device__ float sub_up     (float x, float y)          { return __fsub_ru(x, y); }
    static __device__ float mul_down   (float x, float y)          { return __fmul_rd(x, y); }
    static __device__ float mul_up     (float x, float y)          { return __fmul_ru(x, y); }
    static __device__ float div_down   (float x, float y)          { return __fdiv_rd(x, y); }
    static __device__ float div_up     (float x, float y)          { return __fdiv_ru(x, y); }
    static __device__ float min        (float x, float y)          { return ::fminf(x, y); }
    static __device__ float max        (float x, float y)          { return ::fmaxf(x, y); }
    static __device__ float next_after (float x, float y)          { return ::nextafterf(x, y); }
    static __device__ float round_towards(float x, float to, unsigned int n) { return intrinsic::round_towards_(x, to, n); }
    static __device__ float rcp_down   (float x)                   { return __frcp_rd(x); }
    static __device__ float rcp_up     (float x)                   { return __frcp_ru(x); }
    static __device__ float sqrt_down  (float x)                   { return __fsqrt_rd(x); }
    static __device__ float sqrt_up    (float x)                   { return __fsqrt_ru(x); }
    static __device__ float int_down   (float x)                   { return ::floorf(x); }
    static __device__ float int_up     (float x)                   { return ::ceilf(x); }
    static __device__ float trunc      (float x)                   { return ::truncf(x); }
    static __device__ float round_away (float x)                   { return ::roundf(x); }
    static __device__ float round_even (float x)                   { return ::nearbyintf(x); }
    static __device__ float exp        (float x)                   { return ::expf(x); }
    static __device__ float exp10      (float x)                   { return ::exp10f(x); }
    static __device__ float exp2       (float x)                   { return ::exp2f(x); }
    static __device__ float next_floating(float x)                 { return ::nextafterf(x, std::numeric_limits<float>::infinity()); }
    static __device__ float prev_floating(float x)                 { return ::nextafterf(x, -std::numeric_limits<float>::infinity()); }
};

template<numeric T>
struct intrinsics<interval<T>>
{
    static __device__ interval<T> fma_interval(interval<T> x, interval<T> y, interval<T> z)
    {
        using i = intrinsics<T>;

        return { { .lb = i::min(i::min(i::fma_down(x.lb, y.lb, z.lb), i::fma_down(x.lb, y.ub, z.lb)),
                                i::min(i::fma_down(x.ub, y.lb, z.lb), i::fma_down(x.ub, y.ub, z.lb))),

                   .ub = i::max(i::max(i::fma_up(x.lb, y.lb, z.ub), i::fma_up(x.lb, y.ub, z.ub)),
                                i::max(i::fma_up(x.ub, y.lb, z.ub), i::fma_up(x.ub, y.ub, z.ub))) } };
    }

    static __device__ interval<T> fma_down  (interval<T> x, interval<T> y, interval<T> z) { return fma_interval(x, y, z); }
    static __device__ interval<T> fma_up    (interval<T> x, interval<T> y, interval<T> z) { return fma_interval(x, y, z); }
    static __device__ interval<T> add_down  (interval<T> x, interval<T> y) { return x + y; }
    static __device__ interval<T> add_up    (interval<T> x, interval<T> y) { return x + y; }
    static __device__ interval<T> sub_down  (interval<T> x, interval<T> y) { return x - y; }
    static __device__ interval<T> sub_up    (interval<T> x, interval<T> y) { return x - y; }
    static __device__ interval<T> mul_down  (interval<T> x, interval<T> y) { return x * y; }
    static __device__ interval<T> mul_up    (interval<T> x, interval<T> y) { return x * y; }
    static __device__ interval<T> div_down  (interval<T> x, interval<T> y) { return x / y; }
    static __device__ interval<T> div_up    (interval<T> x, interval<T> y) { return x / y; }
    static __device__ interval<T> rcp_down  (interval<T> x)                { return recip(x); }
    static __device__ interval<T> rcp_up    (interval<T> x)                { return recip(x); }
    static __device__ interval<T> sqrt_down (interval<T> x)                { return sqrt(x); }
    static __device__ interval<T> sqrt_up   (interval<T> x)                { return sqrt(x); }
};

}; // namespace cu

namespace cu::intrinsic
{
    #define ROUNDED_OP(OP) \
        template<numeric T> inline __device__ T OP ## _down(const T &x, typename T::value_type y); \
        template<numeric T> inline __device__ T OP ## _up  (const T &x, typename T::value_type y); \
        template<numeric T> inline __device__ T OP ## _down(typename T::value_type x, const T &y); \
        template<numeric T> inline __device__ T OP ## _up  (typename T::value_type x, const T &y); \

    ROUNDED_OP(add)
    ROUNDED_OP(sub)
    ROUNDED_OP(mul)

    #undef ROUNDED_OP

    template<numeric T> inline __device__ T fma_down  (T x, T y, T z) { return cu::intrinsics<T>::fma_down(x, y, z); }
    template<numeric T> inline __device__ T fma_up    (T x, T y, T z) { return cu::intrinsics<T>::fma_up(x, y, z); }
    template<numeric T> inline __device__ T add_down  (T x, T y)      { return cu::intrinsics<T>::add_down(x, y); }
    template<numeric T> inline __device__ T add_up    (T x, T y)      { return cu::intrinsics<T>::add_up(x, y); }
    template<numeric T> inline __device__ T sub_down  (T x, T y)      { return cu::intrinsics<T>::sub_down(x, y); }
    template<numeric T> inline __device__ T sub_up    (T x, T y)      { return cu::intrinsics<T>::sub_up(x, y); }
    template<numeric T> inline __device__ T mul_down  (T x, T y)      { return cu::intrinsics<T>::mul_down(x, y); }
    template<numeric T> inline __device__ T mul_up    (T x, T y)      { return cu::intrinsics<T>::mul_up(x, y); }
    template<numeric T> inline __device__ T div_down  (T x, T y)      { return cu::intrinsics<T>::div_down(x, y); }
    template<numeric T> inline __device__ T div_up    (T x, T y)      { return cu::intrinsics<T>::div_up(x, y); }
    template<numeric T> inline __device__ T min       (T x, T y)      { return cu::intrinsics<T>::min(x, y); }
    template<numeric T> inline __device__ T max       (T x, T y)      { return cu::intrinsics<T>::max(x, y); }
    template<numeric T> inline __device__ T next_after(T x, T y)      { return cu::intrinsics<T>::next_after(x, y); }
    template<numeric T> inline __device__ T round_towards(T x, T to, unsigned int n) { return cu::intrinsics<T>::round_towards(x, to, n); }
    template<numeric T> inline __device__ T rcp_down  (T x)           { return cu::intrinsics<T>::rcp_down(x); }
    template<numeric T> inline __device__ T rcp_up    (T x)           { return cu::intrinsics<T>::rcp_up(x); }
    template<numeric T> inline __device__ T sqrt_down (T x)           { return cu::intrinsics<T>::sqrt_down(x); }
    template<numeric T> inline __device__ T sqrt_up   (T x)           { return cu::intrinsics<T>::sqrt_up(x); }
    template<numeric T> inline __device__ T int_down  (T x)           { return cu::intrinsics<T>::int_down(x); }
    template<numeric T> inline __device__ T int_up    (T x)           { return cu::intrinsics<T>::int_up(x); }
    template<numeric T> inline __device__ T trunc     (T x)           { return cu::intrinsics<T>::trunc(x); }
    template<numeric T> inline __device__ T round_away(T x)           { return cu::intrinsics<T>::round_away(x); }
    template<numeric T> inline __device__ T round_even(T x)           { return cu::intrinsics<T>::round_even(x); }
    template<numeric T> inline __device__ T exp       (T x)           { return cu::intrinsics<T>::exp(x); }
    template<numeric T> inline __device__ T exp10     (T x)           { return cu::intrinsics<T>::exp10(x); }
    template<numeric T> inline __device__ T exp2      (T x)           { return cu::intrinsics<T>::exp2(x); }
    template<numeric T> inline __device__ __host__ T nan()            { return std::numeric_limits<T>::quiet_NaN(); }
    template<numeric T> inline __device__ T next_floating(T x)        { return cu::intrinsics<T>::next_floating(x); }
    template<numeric T> inline __device__ T prev_floating(T x)        { return cu::intrinsics<T>::prev_floating(x); }

    template<numeric T> inline __device__ T neg_inf()                 { return -std::numeric_limits<T>::infinity(); }
    template<numeric T> inline __device__ T pos_inf()                 { return +std::numeric_limits<T>::infinity(); }

    template<int N = 1, numeric T = double>
    inline constexpr __device__ T round_down(T x, T to = -std::numeric_limits<T>::infinity())
    {
        return cu::intrinsic::round_towards_(x, to, N);
    }

    template<int N = 1, numeric T = double>
    inline constexpr __device__ T round_up(T x, T to = std::numeric_limits<T>::infinity())
    {
        return cu::intrinsic::round_towards_(x, to, N);
    }

} // namespace cu::intrinsic
// clang-format on

#endif // CUINTERVAL_ARITHMETIC_INTRINSICS_CUH
