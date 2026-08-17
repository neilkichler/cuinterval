#include <cuinterval/cuinterval.h>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

#include <vector>

using cu::interval;

struct add_fn
{
    template<typename T, typename U>
    __device__ T operator()(const T &x, const U &y) const
    {
        return x + y;
    }
};

template<typename T, typename U = T>
std::vector<T> compute_nested(cudaStream_t stream, std::vector<T> xs, std::vector<U> ys)
{
    thrust::host_vector<T> h_xs = xs;
    thrust::host_vector<U> h_ys = ys;

    auto n = xs.size();
    thrust::device_vector<T> d_res(n);
    thrust::device_vector<T> d_xs = h_xs;
    thrust::device_vector<U> d_ys = h_ys;
    thrust::transform(d_xs.begin(), d_xs.end(), d_ys.begin(), d_res.begin(), add_fn());
    std::vector<T> h_res(n);
    thrust::copy(d_res.begin(), d_res.end(), h_res.begin());

    return h_res;
}

template std::vector<interval<interval<float>>> compute_nested<interval<interval<float>>>(cudaStream_t stream, std::vector<interval<interval<float>>> xs, std::vector<interval<interval<float>>> ys);
template std::vector<interval<interval<float>>> compute_nested<interval<interval<float>>, interval<float>>(cudaStream_t stream, std::vector<interval<interval<float>>> xs, std::vector<interval<float>> ys);
