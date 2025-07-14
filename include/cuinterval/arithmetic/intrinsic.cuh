#ifndef CUINTERVAL_ARITHMETIC_INTRINSIC_CUH
#define CUINTERVAL_ARITHMETIC_INTRINSIC_CUH

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

#endif // CUINTERVAL_ARITHMETIC_INTRINSIC_CUH
