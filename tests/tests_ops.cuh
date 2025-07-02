#ifndef TESTS_OPS_CUH
#define TESTS_OPS_CUH

#include <cuinterval/cuinterval.h>

template<typename T>
__global__ void test_pos(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = +x[i];
    }
}

template<typename T>
__global__ void test_neg(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = -x[i];
    }
}

template<typename T>
__global__ void test_recip(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = recip(x[i]);
    }
}

template<typename T>
__global__ void test_sqr(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = sqr(x[i]);
    }
}

template<typename T>
__global__ void test_sqrt(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = sqrt(x[i]);
    }
}

template<typename T>
__global__ void test_cbrt(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = cbrt(x[i]);
    }
}

template<typename T>
__global__ void test_add(int n, cu::interval<T> *x, cu::interval<T> *y, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = x[i] + y[i];
    }
}

template<typename T>
__global__ void test_cancelPlus(int n, cu::interval<T> *x, cu::interval<T> *y, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = cancel_plus(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_cancelMinus(int n, cu::interval<T> *x, cu::interval<T> *y, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = cancel_minus(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_sub(int n, cu::interval<T> *x, cu::interval<T> *y, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = x[i] - y[i];
    }
}

template<typename T>
__global__ void test_mul(int n, cu::interval<T> *x, cu::interval<T> *y, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = x[i] * y[i];
    }
}

template<typename T>
__global__ void test_div(int n, cu::interval<T> *x, cu::interval<T> *y, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = x[i] / y[i];
    }
}

template<typename T>
__global__ void test_fma(int n, cu::interval<T> *x, cu::interval<T> *y, cu::interval<T> *z, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = fma(x[i], y[i], z[i]);
    }
}

template<typename T>
__global__ void test_inf(int n, cu::interval<T> *x, T *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = inf(x[i]);
    }
}

template<typename T>
__global__ void test_sup(int n, cu::interval<T> *x, T *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = sup(x[i]);
    }
}

template<typename T>
__global__ void test_mid(int n, cu::interval<T> *x, T *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = mid(x[i]);
    }
}

template<typename T>
__global__ void test_rad(int n, cu::interval<T> *x, T *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = rad(x[i]);
    }
}

template<typename T>
__global__ void test_mag(int n, cu::interval<T> *x, T *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = mag(x[i]);
    }
}

template<typename T>
__global__ void test_mig(int n, cu::interval<T> *x, T *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = mig(x[i]);
    }
}

template<typename T>
__global__ void test_wid(int n, cu::interval<T> *x, T *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = width(x[i]);
    }
}

template<typename T>
__global__ void test_floor(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = floor(x[i]);
    }
}

template<typename T>
__global__ void test_ceil(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = ceil(x[i]);
    }
}

template<typename T>
__global__ void test_trunc(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = trunc(x[i]);
    }
}

template<typename T>
__global__ void test_sign(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = sign(x[i]);
    }
}

template<typename T>
__global__ void test_abs(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = abs(x[i]);
    }
}

template<typename T>
__global__ void test_min(int n, cu::interval<T> *x, cu::interval<T> *y, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = min(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_max(int n, cu::interval<T> *x, cu::interval<T> *y, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = max(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_intersection(int n, cu::interval<T> *x, cu::interval<T> *y, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = intersection(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_convexHull(int n, cu::interval<T> *x, cu::interval<T> *y, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = convex_hull(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_equal(int n, cu::interval<T> *x, cu::interval<T> *y, bool *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = equal(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_subset(int n, cu::interval<T> *x, cu::interval<T> *y, bool *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = subset(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_interior(int n, cu::interval<T> *x, cu::interval<T> *y, bool *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = interior(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_disjoint(int n, cu::interval<T> *x, cu::interval<T> *y, bool *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = disjoint(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_isEmpty(int n, cu::interval<T> *x, bool *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = empty(x[i]);
    }
}

template<typename T>
__global__ void test_isEntire(int n, cu::interval<T> *x, bool *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = entire(x[i]);
    }
}

template<typename T>
__global__ void test_less(int n, cu::interval<T> *x, cu::interval<T> *y, bool *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = less(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_strictLess(int n, cu::interval<T> *x, cu::interval<T> *y, bool *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = strict_less(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_precedes(int n, cu::interval<T> *x, cu::interval<T> *y, bool *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = precedes(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_strictPrecedes(int n, cu::interval<T> *x, cu::interval<T> *y, bool *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = strict_precedes(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_isMember(int n, T *x, cu::interval<T> *y, bool *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = is_member(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_isSingleton(int n, cu::interval<T> *x, bool *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = is_singleton(x[i]);
    }
}

template<typename T>
__global__ void test_isCommonInterval(int n, cu::interval<T> *x, bool *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = is_common_interval(x[i]);
    }
}

template<typename T>
__global__ void test_roundTiesToEven(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = round_to_nearest_even(x[i]);
    }
}

template<typename T>
__global__ void test_roundTiesToAway(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = round_ties_to_away(x[i]);
    }
}

template<typename T>
__global__ void test_exp(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = exp(x[i]);
    }
}

template<typename T>
__global__ void test_exp2(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = exp2(x[i]);
    }
}

template<typename T>
__global__ void test_exp10(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = exp10(x[i]);
    }
}

template<typename T>
__global__ void test_expm1(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = expm1(x[i]);
    }
}

template<typename T>
__global__ void test_log(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = log(x[i]);
    }
}

template<typename T>
__global__ void test_log2(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = log2(x[i]);
    }
}

template<typename T>
__global__ void test_log10(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = log10(x[i]);
    }
}

template<typename T>
__global__ void test_log1p(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = log1p(x[i]);
    }
}

template<typename T>
__global__ void test_pown(int n, cu::interval<T> *x, int p, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = pown(x[i], p);
    }
}

template<typename T>
__global__ void test_sin(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = sin(x[i]);
    }
}

template<typename T>
__global__ void test_cos(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = cos(x[i]);
    }
}

template<typename T>
__global__ void test_tan(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = tan(x[i]);
    }
}

template<typename T>
__global__ void test_asin(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = asin(x[i]);
    }
}

template<typename T>
__global__ void test_acos(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = acos(x[i]);
    }
}

template<typename T>
__global__ void test_atan(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = atan(x[i]);
    }
}

template<typename T>
__global__ void test_sinh(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = sinh(x[i]);
    }
}

template<typename T>
__global__ void test_cosh(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = cosh(x[i]);
    }
}

template<typename T>
__global__ void test_tanh(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = tanh(x[i]);
    }
}

template<typename T>
__global__ void test_asinh(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = asinh(x[i]);
    }
}

template<typename T>
__global__ void test_acosh(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = acosh(x[i]);
    }
}

template<typename T>
__global__ void test_atanh(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = atanh(x[i]);
    }
}

template<typename T>
__global__ void test_atan2(int n, cu::interval<T> *y, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = atan2(y[i], x[i]);
    }
}

template<typename T>
__global__ void test_sinpi(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = sinpi(x[i]);
    }
}

template<typename T>
__global__ void test_cospi(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = cospi(x[i]);
    }
}

template<typename T>
__global__ void test_cot(int n, cu::interval<T> *x, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = cot(x[i]);
    }
}

template<typename T>
__global__ void test_pown(int n, cu::interval<T> *x, int *n_pow, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = pown(x[i], n_pow[i]);
    }
}

template<typename T>
__global__ void test_rootn(int n, cu::interval<T> *x, int *n_pow, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = rootn(x[i], n_pow[i]);
    }
}

template<typename T>
__global__ void test_pow(int n, cu::interval<T> *x, cu::interval<T> *y, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = pow(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_hypot(int n, cu::interval<T> *x, cu::interval<T> *y, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = hypot(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_bisect(int n, cu::interval<T> *x, T *y, cu::split<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = bisect(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_mince(int n, cu::interval<T> *x, int *d_offsets, cu::interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        mince(x[i], &res[d_offsets[i]], d_offsets[i + 1] - d_offsets[i]);
    }
}

#endif // TESTS_OPS_CUH
