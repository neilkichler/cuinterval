#ifndef CUINTERVAL_ARITHMETIC_ROUNDED_CUH
#define CUINTERVAL_ARITHMETIC_ROUNDED_CUH

namespace intrinsic
{
// clang-format off
    template<typename T> __device__ T fma_down  (T x, T y, T z);
    template<typename T> __device__ T fma_up    (T x, T y, T z);
    template<typename T> __device__ T add_down  (T x, T y);
    template<typename T> __device__ T add_up    (T x, T y);
    template<typename T> __device__ T sub_down  (T x, T y);
    template<typename T> __device__ T sub_up    (T x, T y);
    template<typename T> __device__ T mul_down  (T x, T y);
    template<typename T> __device__ T mul_up    (T x, T y);
    template<typename T> __device__ T div_down  (T x, T y);
    template<typename T> __device__ T div_up    (T x, T y);
    template<typename T> __device__ T median    (T x, T y);
    template<typename T> __device__ T min       (T x, T y);
    template<typename T> __device__ T max       (T x, T y);
    template<typename T> __device__ T copy_sign (T x, T y);
    template<typename T> __device__ T rcp_down  (T x);
    template<typename T> __device__ T rcp_up    (T x);
    template<typename T> __device__ T sqrt_down (T x);
    template<typename T> __device__ T sqrt_up   (T x);
    template<typename T> __device__ T int_down  (T x);
    template<typename T> __device__ T int_up    (T x);
    template<typename T> __device__ T trunc     (T x);
    template<typename T> __device__ __host__ T nan();
    template<typename T> __device__ T pos_inf();
    template<typename T> __device__ T neg_inf();

    template<> __device__ double fma_down (double x, double y, double z) { return __fma_rd(x, y, z); }    
    template<> __device__ double fma_up   (double x, double y, double z) { return __fma_ru(x, y, z); }    
    template<> __device__ double add_down (double x, double y) { return __dadd_rd(x, y); }
    template<> __device__ double add_up   (double x, double y) { return __dadd_ru(x, y); }
    template<> __device__ double sub_down (double x, double y) { return __dadd_rd(x, -y); }
    template<> __device__ double sub_up   (double x, double y) { return __dadd_ru(x, -y); }
    template<> __device__ double mul_down (double x, double y) { return __dmul_rd(x, y); }
    template<> __device__ double mul_up   (double x, double y) { return __dmul_ru(x, y); }
    template<> __device__ double div_down (double x, double y) { return __ddiv_rd(x, y); }
    template<> __device__ double div_up   (double x, double y) { return __ddiv_ru(x, y); }
    template<> __device__ double median   (double x, double y) { return (x + y) * .5; }
    template<> __device__ double min      (double x, double y) { return fmin(x, y); }
    template<> __device__ double max      (double x, double y) { return fmax(x, y); }
    template<> __device__ double copy_sign(double x, double y) { return copysign(x, y); }
    template<> __device__ double rcp_down (double x)           { return __drcp_rd(x); }
    template<> __device__ double rcp_up   (double x)           { return __drcp_ru(x); }
    template<> __device__ double sqrt_down(double x)           { return __dsqrt_rd(x); }
    template<> __device__ double sqrt_up  (double x)           { return __dsqrt_ru(x); }
    template<> __device__ double int_down (double x)           { return floor(x); }
    template<> __device__ double int_up   (double x)           { return ceil(x); }
    template<> __device__ double trunc    (double x)           { return ::trunc(x); }
    template<> __device__ __host__ double nan()                { return ::nan(""); }
    template<> __device__ double neg_inf() { return __longlong_as_double(0xfff0000000000000ull); }
    template<> __device__ double pos_inf() { return __longlong_as_double(0x7ff0000000000000ull); }

    template<> __device__ float fma_down   (float x, float y, float z) { return __fmaf_rd(x, y, z); }    
    template<> __device__ float fma_up     (float x, float y, float z) { return __fmaf_ru(x, y, z); } 
    template<> __device__ float add_down   (float x, float y)   { return __fadd_rd(x, y); } 
    template<> __device__ float add_up     (float x, float y)   { return __fadd_ru(x, y); }
    template<> __device__ float sub_down   (float x, float y)   { return __fadd_rd(x, -y); }
    template<> __device__ float sub_up     (float x, float y)   { return __fadd_ru(x, -y); }
    template<> __device__ float mul_down   (float x, float y)   { return __fmul_rd(x, y); }
    template<> __device__ float mul_up     (float x, float y)   { return __fmul_ru(x, y); }
    template<> __device__ float div_down   (float x, float y)   { return __fdiv_rd(x, y); }
    template<> __device__ float div_up     (float x, float y)   { return __fdiv_ru(x, y); }
    template<> __device__ float median     (float x, float y)   { return (x + y) * .5f; }
    template<> __device__ float min        (float x, float y)   { return fminf(x, y); }
    template<> __device__ float max        (float x, float y)   { return fmaxf(x, y); }
    template<> __device__ float copy_sign  (float x, float y)   { return copysignf(x, y); }
    template<> __device__ float rcp_down   (float x)            { return __frcp_rd(x); }
    template<> __device__ float rcp_up     (float x)            { return __frcp_ru(x); }
    template<> __device__ float sqrt_down  (float x)            { return __fsqrt_rd(x); }
    template<> __device__ float sqrt_up    (float x)            { return __fsqrt_ru(x); }
    template<> __device__ float int_down   (float x)            { return floorf(x); }
    template<> __device__ float int_up     (float x)            { return ceilf(x); }
    template<> __device__ float trunc      (float x)            { return truncf(x); }
    template<> __device__ __host__ float nan()                  { return nanf(""); }
    template<> __device__ float neg_inf() { return __int_as_float(0xff800000); }
    template<> __device__ float pos_inf() { return __int_as_float(0x7f800000); }
// clang-format on
} // namespace intrinsic

#endif // CUINTERVAL_ARITHMETIC_ROUNDED_CUH
