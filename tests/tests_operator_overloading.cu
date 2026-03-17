#include "tests.h"
#include "tests_operator_overloading.h"
#include "tests_utils.h"

#include <cuinterval/cuinterval.h>

template<typename T>
__global__ void test_overload(auto &&f, cu::interval<T> *x,
                              cu::interval<T> *y, cu::interval<T> *res, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = f(x[i], y[i]);
    }
}

template<typename T>
void test_compare_fns(auto &&fn_a, auto &&fn_b, const char *name, cuda_buffer buffer, cudaStream_t stream, cudaEvent_t event)
{
    using I = cu::interval<T>;

    T infinity = std::numeric_limits<T>::infinity();
    I empty    = { infinity, -infinity };
    I entire   = { -infinity, infinity };
    T NaN      = ::nan("");

    char *d_buffer = buffer.device;
    char *h_buffer = buffer.host;

    constexpr int n     = 10;
    const int n_bytes   = n * sizeof(I);
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;

    I *d_xs_  = (I *)d_buffer;
    I *d_ys_  = (I *)d_buffer + n_bytes;
    I *d_res_ = (I *)d_buffer + 2 * n_bytes;

    I *h_xs = new (h_buffer) I[n] {
        { -0.0, 0.0 },
        empty,
        entire,
        empty,
        entire,
        { -0.0, -0.0 },
        { -0.0, -0.0 },
        { -0.0, -0.0 },
        { 0.0, 4.0 },
        { -1.0, 1.0 },
    };

    h_buffer += align_to(n_bytes, alignof(I));
    I *h_ys = new (h_buffer) I[n] {
        { -0.0, 0.0 },
        empty,
        entire,
        entire,
        empty,
        empty,
        entire,
        { -4.0, -2.0 },
        { 4.0, 2.0 },
        { -1.0, 10.0 },
    };

    h_buffer += align_to(n_bytes, alignof(I));

    I *h_res = new (h_buffer) I[n] {};

    std::array<I, n> h_ref {};

    I *d_xs  = (I *)d_xs_;
    I *d_ys  = (I *)d_ys_;
    I *d_res = (I *)d_res_;
    CUDA_CHECK(cudaMemcpyAsync(d_xs, h_xs, n * sizeof(I), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_ys, h_ys, n * sizeof(I), cudaMemcpyHostToDevice, stream));

    test_overload<<<numBlocks, blockSize, 0, stream>>>(fn_a, d_xs, d_ys, d_res, n);
    CUDA_CHECK(cudaMemcpyAsync(h_res, d_res, n * sizeof(I), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(event, stream));
    CUDA_CHECK(cudaEventSynchronize(event));

    test_overload<<<numBlocks, blockSize, 0, stream>>>(fn_b, d_xs, d_ys, d_res, n);
    CUDA_CHECK(cudaMemcpyAsync(h_ref.data(), d_res, n * sizeof(I), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(event, stream));
    CUDA_CHECK(cudaEventSynchronize(event));

    int max_ulp_diff = 0;
    check_all_equal<I, n>(h_res, h_ref, max_ulp_diff, name, std::source_location::current(), h_xs);
};

void tests_operator_overloading(cuda_buffer buffer, cudaStream_t stream, cudaEvent_t event)
{
    using namespace boost::ut;

    auto compare_fns = [&](auto &&fn_a, auto &&fn_b, const char *name) {
        test_compare_fns<float>(fn_a, fn_b, name, buffer, stream, event);
        test_compare_fns<double>(fn_a, fn_b, name, buffer, stream, event);
    };

    auto add_assign = [] __device__(auto x, auto y) { auto z = x; z += y; return z; };
    auto sub_assign = [] __device__(auto x, auto y) { auto z = x; z -= y; return z; };
    auto mul_assign = [] __device__(auto x, auto y) { auto z = x; z *= y; return z; };
    auto div_assign = [] __device__(auto x, auto y) { auto z = x; z /= y; return z; };

    auto add = [] __device__(auto x, auto y) { return x + y; };
    auto sub = [] __device__(auto x, auto y) { return x - y; };
    auto mul = [] __device__(auto x, auto y) { return x * y; };
    auto div = [] __device__(auto x, auto y) { return x / y; };

    compare_fns(add_assign, add, "add");
    compare_fns(sub_assign, sub, "sub");
    compare_fns(mul_assign, mul, "mul");
    compare_fns(div_assign, div, "div");

    auto add_int = [] __device__(auto x, auto y) { return x + 1; };
    auto sub_int = [] __device__(auto x, auto y) { return x - 1; };
    auto mul_int = [] __device__(auto x, auto y) { return x * 1; };
    auto div_int = [] __device__(auto x, auto y) { return x / 1; };

    auto add_float = [] __device__(auto x, auto y) { return x + 1.0f; };
    auto sub_float = [] __device__(auto x, auto y) { return x - 1.0f; };
    auto mul_float = [] __device__(auto x, auto y) { return x * 1.0f; };
    auto div_float = [] __device__(auto x, auto y) { return x / 1.0f; };

    compare_fns(add_int, add_float, "add_constant");
    compare_fns(sub_int, sub_float, "sub_constant");
    compare_fns(mul_int, mul_float, "mul_constant");
    compare_fns(div_int, div_float, "div_constant");

    auto int_add = [] __device__(auto x, auto y) { return 1 + x; };
    auto int_sub = [] __device__(auto x, auto y) { return 1 - x; };
    auto int_mul = [] __device__(auto x, auto y) { return 1 * x; };
    auto int_div = [] __device__(auto x, auto y) { return 1 / x; };

    auto float_add = [] __device__(auto x, auto y) { return 1.0f + x; };
    auto float_sub = [] __device__(auto x, auto y) { return 1.0f - x; };
    auto float_mul = [] __device__(auto x, auto y) { return 1.0f * x; };
    auto float_div = [] __device__(auto x, auto y) { return 1.0f / x; };

    compare_fns(int_add, float_add, "constant_add");
    compare_fns(int_sub, float_sub, "constant_sub");
    compare_fns(int_mul, float_mul, "constant_mul");
    compare_fns(int_div, float_div, "constant_div");
}
