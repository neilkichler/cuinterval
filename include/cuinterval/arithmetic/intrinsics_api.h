#ifndef CU_INTRINSICS_API_H
#define CU_INTRINSICS_API_H

#include <cuinterval/interval.h>
#include <cuinterval/traits.h>

#include <cmath>
#include <limits>

namespace cu
{
// clang-format off

template<numeric X, numeric Y = X, numeric Z = X>
struct intrinsics
{
    static __host__ __device__ auto fma_down  (X x, Y y, Z z) { return fma(x, y, z); }
    static __host__ __device__ auto fma_up    (X x, Y y, Z z) { return fma(x, y, z); }
    static __host__ __device__ auto add_down  (X x, Y y)      { return x + y; }
    static __host__ __device__ auto add_up    (X x, Y y)      { return x + y; }
    static __host__ __device__ auto sub_down  (X x, Y y)      { return x - y; }
    static __host__ __device__ auto sub_up    (X x, Y y)      { return x - y; }
    static __host__ __device__ auto mul_down  (X x, Y y)      { return x * y; }
    static __host__ __device__ auto mul_up    (X x, Y y)      { return x * y; }
    static __host__ __device__ auto div_down  (X x, Y y)      { return x / y; }
    static __host__ __device__ auto div_up    (X x, Y y)      { return x / y; }
    static __host__ __device__ auto rcp_down  (X x)           { return recip(x); }
    static __host__ __device__ auto rcp_up    (X x)           { return recip(x); }
    static __host__ __device__ auto sqrt_down (X x)           { return sqrt(x); }
    static __host__ __device__ auto sqrt_up   (X x)           { return sqrt(x); }
    static __host__ __device__ auto min       (X x, Y y)      { return min(x, y); }
    static __host__ __device__ auto max       (X x, Y y)      { return max(x, y); }
    static __host__ __device__ X next_after   (X x, Y y);
    static __host__ __device__ X round_towards(X x, Y to, unsigned int n);
    static __host__ __device__ X int_down     (X x);
    static __host__ __device__ X int_up       (X x);
    static __host__ __device__ X trunc        (X x);
    static __host__ __device__ X round_away   (X x);
    static __host__ __device__ X round_even   (X x);
    static __host__ __device__ X exp          (X x);
    static __host__ __device__ X exp10        (X x);
    static __host__ __device__ X exp2         (X x);
    static __host__ __device__ X next_floating(X x) { using std::nextafter; return nextafter(x, +std::numeric_limits<X>::infinity()); }
    static __host__ __device__ X prev_floating(X x) { using std::nextafter; return nextafter(x, -std::numeric_limits<X>::infinity()); }
};

// clang-format on
} // namespace cu

#endif
