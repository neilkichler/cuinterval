#include <cuinterval/cuinterval.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

#include <vector>

using cu::interval;

struct rounding_fn
{
    template<typename I, int N = 2>
    __device__ I operator()(const I &x, int n) const
    {
        using cu::intrinsic::round_down, cu::intrinsic::round_up;
        return { round_down<N>(x.lb), round_up<N>(x.ub) };
    }
};

struct rounding_ref_fn
{
    template<typename I, int N = 2>
    __device__ I operator()(const I &x, int n) const
    {
        using cu::intrinsic::next_after;
        using T = I::value_type;

        constexpr T inf = std::numeric_limits<T>::infinity();
        I y             = x;
        for (int i = 0; i < N; i++) {
            y = { next_after(y.lb, -inf), next_after(y.ub, inf) };
        }
        return y;
    }
};

std::vector<interval<float>> compute_rounding(cudaStream_t stream, std::vector<interval<float>> xs, std::vector<int> exponents)
{
    using T = float;
    using I = interval<T>;

    using ET = int;

    thrust::host_vector<I> h_xs         = xs;
    thrust::host_vector<ET> h_exponents = exponents;

    auto n = xs.size();
    thrust::device_vector<I> d_res(n);
    thrust::device_vector<I> d_xs         = h_xs;
    thrust::device_vector<ET> d_exponents = h_exponents;
    thrust::transform(d_xs.begin(), d_xs.end(), d_exponents.begin(), d_res.begin(), rounding_fn());
    std::vector<I> h_res(n);
    thrust::copy(d_res.begin(), d_res.end(), h_res.begin());

    return h_res;
}

std::vector<interval<float>> compute_rounding_ref(cudaStream_t stream, std::vector<interval<float>> xs, std::vector<int> exponents)
{
    using T = float;
    using I = interval<T>;

    using ET = int;

    thrust::host_vector<I> h_xs         = xs;
    thrust::host_vector<ET> h_exponents = exponents;

    auto n = xs.size();
    thrust::device_vector<I> d_res(n);
    thrust::device_vector<I> d_xs         = h_xs;
    thrust::device_vector<ET> d_exponents = h_exponents;
    thrust::transform(d_xs.begin(), d_xs.end(), d_exponents.begin(), d_res.begin(), rounding_ref_fn());
    std::vector<I> h_res(n);
    thrust::copy(d_res.begin(), d_res.end(), h_res.begin());

    return h_res;
}
