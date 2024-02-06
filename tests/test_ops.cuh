#ifndef TEST_OPS_CUH
#define TEST_OPS_CUH

#include <cuinterval/cuinterval.h>

template<typename T>
__global__ void test_pos(int n, interval<T> *x)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        x[i] = +x[i];
    }
}

template<typename T>
__global__ void test_neg(int n, interval<T> *x)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        x[i] = -x[i];
    }
}

template<typename T>
__global__ void test_recip(int n, interval<T> *x)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        x[i] = recip(x[i]);
    }
}

template<typename T>
__global__ void test_sqr(int n, interval<T> *x)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        x[i] = sqr(x[i]);
    }
}

template<typename T>
__global__ void test_sqrt(int n, interval<T> *x)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        x[i] = sqrt(x[i]);
    }
}

template<typename T>
__global__ void test_add(int n, interval<T> *x, interval<T> *y)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        x[i] = x[i] + y[i];
    }
}

template<typename T>
__global__ void test_sub(int n, interval<T> *x, interval<T> *y)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        x[i] = x[i] - y[i];
    }
}

template<typename T>
__global__ void test_mul(int n, interval<T> *x, interval<T> *y)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        x[i] = x[i] * y[i];
    }
}

template<typename T>
__global__ void test_div(int n, interval<T> *x, interval<T> *y)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        x[i] = x[i] / y[i];
    }
}

template<typename T>
__global__ void test_fma(int n, interval<T> *x, interval<T> *y, interval<T> *z)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        x[i] = fma(x[i], y[i], z[i]);
    }
}
#endif // TEST_OPS_CUH
