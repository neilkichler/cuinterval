#ifndef TEST_OPS_CUH
#define TEST_OPS_CUH

#include <cuinterval/cuinterval.h>

template<typename T>
__global__ void test_pos(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = +x[i];
    }
}

template<typename T>
__global__ void test_neg(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = -x[i];
    }
}

template<typename T>
__global__ void test_recip(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = recip(x[i]);
    }
}

template<typename T>
__global__ void test_sqr(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = sqr(x[i]);
    }
}

template<typename T>
__global__ void test_sqrt(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = sqrt(x[i]);
    }
}

template<typename T>
__global__ void test_add(int n, interval<T> *x, interval<T> *y, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = x[i] + y[i];
    }
}

template<typename T>
__global__ void test_cancelPlus(int n, interval<T> *x, interval<T> *y, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = cancel_plus(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_cancelMinus(int n, interval<T> *x, interval<T> *y, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = cancel_minus(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_sub(int n, interval<T> *x, interval<T> *y, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = x[i] - y[i];
    }
}

template<typename T>
__global__ void test_mul(int n, interval<T> *x, interval<T> *y, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = x[i] * y[i];
    }
}

template<typename T>
__global__ void test_div(int n, interval<T> *x, interval<T> *y, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = x[i] / y[i];
    }
}

template<typename T>
__global__ void test_fma(int n, interval<T> *x, interval<T> *y, interval<T> *z, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = fma(x[i], y[i], z[i]);
    }
}

template<typename T>
__global__ void test_inf(int n, interval<T> *x, T *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = inf(x[i]);
    }
}

template<typename T>
__global__ void test_sup(int n, interval<T> *x, T *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = sup(x[i]);
    }
}

template<typename T>
__global__ void test_mid(int n, interval<T> *x, T *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = mid(x[i]);
    }
}

template<typename T>
__global__ void test_rad(int n, interval<T> *x, T *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = rad(x[i]);
    }
}

template<typename T>
__global__ void test_mag(int n, interval<T> *x, T *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = mag(x[i]);
    }
}

template<typename T>
__global__ void test_mig(int n, interval<T> *x, T *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = mig(x[i]);
    }
}

template<typename T>
__global__ void test_wid(int n, interval<T> *x, T *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = width(x[i]);
    }
}

template<typename T>
__global__ void test_floor(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = floor(x[i]);
    }
}

template<typename T>
__global__ void test_ceil(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = ceil(x[i]);
    }
}

template<typename T>
__global__ void test_trunc(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = trunc(x[i]);
    }
}

template<typename T>
__global__ void test_sign(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = sign(x[i]);
    }
}

template<typename T>
__global__ void test_abs(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = abs(x[i]);
    }
}

template<typename T>
__global__ void test_min(int n, interval<T> *x, interval<T> *y, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = min(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_max(int n, interval<T> *x, interval<T> *y, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = max(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_intersection(int n, interval<T> *x, interval<T> *y, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = intersection(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_convexHull(int n, interval<T> *x, interval<T> *y, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = convex_hull(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_equal(int n, interval<T> *x, interval<T> *y, bool *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = equal(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_subset(int n, interval<T> *x, interval<T> *y, bool *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = subset(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_interior(int n, interval<T> *x, interval<T> *y, bool *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = interior(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_disjoint(int n, interval<T> *x, interval<T> *y, bool *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = disjoint(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_isEmpty(int n, interval<T> *x, bool *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = empty(x[i]);
    }
}

template<typename T>
__global__ void test_isEntire(int n, interval<T> *x, bool *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = entire(x[i]);
    }
}

template<typename T>
__global__ void test_less(int n, interval<T> *x, interval<T> *y, bool *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = less(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_strictLess(int n, interval<T> *x, interval<T> *y, bool *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = strict_less(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_precedes(int n, interval<T> *x, interval<T> *y, bool *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = precedes(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_strictPrecedes(int n, interval<T> *x, interval<T> *y, bool *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = strict_precedes(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_isMember(int n, T *x, interval<T> *y, bool *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = is_member(x[i], y[i]);
    }
}

template<typename T>
__global__ void test_isSingleton(int n, interval<T> *x, bool *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = is_singleton(x[i]);
    }
}

template<typename T>
__global__ void test_isCommonInterval(int n, interval<T> *x, bool *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = is_common_interval(x[i]);
    }
}

template<typename T>
__global__ void test_roundTiesToEven(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = round_to_nearest_even(x[i]);
    }
}

template<typename T>
__global__ void test_roundTiesToAway(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = round_ties_to_away(x[i]);
    }
}

template<typename T>
__global__ void test_exp(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = exp(x[i]);
    }
}

template<typename T>
__global__ void test_exp2(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = exp2(x[i]);
    }
}

template<typename T>
__global__ void test_exp10(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = exp10(x[i]);
    }
}

template<typename T>
__global__ void test_expm1(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = expm1(x[i]);
    }
}

template<typename T>
__global__ void test_log(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = log(x[i]);
    }
}

template<typename T>
__global__ void test_log2(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = log2(x[i]);
    }
}

template<typename T>
__global__ void test_log10(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = log10(x[i]);
    }
}


template<typename T>
__global__ void test_log1p(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = log1p(x[i]);
    }
}

template<typename T>
__global__ void test_pown(int n, interval<T> *x, int p, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = pown(x[i], p);
    }
}

template<typename T>
__global__ void test_sin(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = sin(x[i]);
    }
}

template<typename T>
__global__ void test_cos(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = cos(x[i]);
    }
}

template<typename T>
__global__ void test_tan(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = tan(x[i]);
    }
}

template<typename T>
__global__ void test_asin(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = asin(x[i]);
    }
}

template<typename T>
__global__ void test_acos(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = acos(x[i]);
    }
}

template<typename T>
__global__ void test_atan(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = atan(x[i]);
    }
}

template<typename T>
__global__ void test_sinh(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = sinh(x[i]);
    }
}

template<typename T>
__global__ void test_cosh(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = cosh(x[i]);
    }
}

template<typename T>
__global__ void test_tanh(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = tanh(x[i]);
    }
}

template<typename T>
__global__ void test_asinh(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = asinh(x[i]);
    }
}

template<typename T>
__global__ void test_acosh(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = acosh(x[i]);
    }
}

template<typename T>
__global__ void test_atanh(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = atanh(x[i]);
    }
}

template<typename T>
__global__ void test_atan2(int n, interval<T> *y, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = atan2(y[i], x[i]);
    }
}

template<typename T>
__global__ void test_sinpi(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = sinpi(x[i]);
    }
}

template<typename T>
__global__ void test_cospi(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = cospi(x[i]);
    }
}

#endif // TEST_OPS_CUH
