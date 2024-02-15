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
__global__ void test_sup(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = sup(x[i]);
    }
}

template<typename T>
__global__ void test_mid(int n, interval<T> *x, interval<T> *res)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = mid(x[i]);
    }
}

template<typename T>
__global__ void test_rad(int n, interval<T> *x, interval<T> *res)
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

#endif // TEST_OPS_CUH
