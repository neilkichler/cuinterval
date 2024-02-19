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

#endif // TEST_OPS_CUH
