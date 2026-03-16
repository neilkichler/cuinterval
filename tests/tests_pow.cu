#include <cuinterval/cuinterval.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

#include <vector>

using cu::interval;

template<typename T>
struct pow_fn
{
    template<typename I>
    __device__ I operator()(const I &x, T n) const
    {
        using cu::pow;
        return pow(x, n);
    }
};

template<typename ET>
std::vector<interval<float>> compute_pow(cudaStream_t stream, std::vector<interval<float>> xs, std::vector<ET> exponents)
{
    using T = float;
    using I = interval<T>;

    thrust::host_vector<I> h_xs         = xs;
    thrust::host_vector<ET> h_exponents = exponents;

    auto n = xs.size();
    thrust::device_vector<I> d_res(n);
    thrust::device_vector<I> d_xs         = h_xs;
    thrust::device_vector<ET> d_exponents = h_exponents;
    thrust::transform(d_xs.begin(), d_xs.end(), d_exponents.begin(), d_res.begin(), pow_fn<ET>());
    std::vector<I> h_res(n);
    thrust::copy(d_res.begin(), d_res.end(), h_res.begin());

    return h_res;
}

template std::vector<interval<float>> compute_pow<int>(cudaStream_t stream, std::vector<interval<float>> xs, std::vector<int> exponents);
template std::vector<interval<float>> compute_pow<float>(cudaStream_t stream, std::vector<interval<float>> xs, std::vector<float> exponents);
