#include <cuinterval/cuinterval.h>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <numbers>

struct to_interval_fn
{
    template<typename T>
    __host__ __device__ interval<T> operator()(const T &x) const
    {
        return interval<T> { x, x };
    }
};

struct pi_recip_fn
{
    template<typename I>
    __device__ I operator()(const I &x) const
    {
        return recip(sqr(x));
    }
};

struct pi_pow_fn
{
    template<typename I>
    __device__ I operator()(const I &x) const
    {
        return recip(pow(x, 2));
    }
};

struct pi_inv_fn
{
    template<typename I>
    __device__ I operator()(const I &x) const
    {
        return I{1.0, 1.0}/(sqr(x));
    }
};

struct scale_fn
{
    template<typename I>
    __device__ I operator()(I x) const
    {
        return sqrt(x * I { 6.0, 6.0 });
    }
};

struct final_decrement_fn
{
    int n;

    final_decrement_fn(int _n)
        : n(_n)
    { }

    template<typename I>
    __device__ I operator()(I x) const
    {
        I n_lb = I { 0.0 + n, 0.0 + n };
        I n_ub = I { 1.0 + n, 1.0 + n };

        I inv_lb = recip(n_lb);
        I inv_ub = recip(n_ub);

        return x + I { inv_ub.lb, inv_lb.ub };
    }
};

template<typename T>
void tests_loop()
{
    int n   = 100'000;
    using I = interval<T>;
    using namespace boost::ut;
    using namespace thrust::placeholders;

    thrust::counting_iterator<T> seq_first(1);
    thrust::counting_iterator<T> seq_last = seq_first + n;

    auto tr_first     = thrust::make_transform_iterator(seq_first, to_interval_fn());
    auto tr_last      = thrust::make_transform_iterator(seq_last, to_interval_fn());
    auto pi_rcp_first = thrust::make_transform_iterator(tr_first, pi_recip_fn());
    auto pi_rcp_last  = thrust::make_transform_iterator(tr_last, pi_recip_fn());
    auto pi_pow_first = thrust::make_transform_iterator(tr_first, pi_pow_fn());
    auto pi_pow_last  = thrust::make_transform_iterator(tr_last, pi_pow_fn());
    auto pi_inv_first = thrust::make_transform_iterator(tr_first, pi_inv_fn());
    auto pi_inv_last  = thrust::make_transform_iterator(tr_last, pi_inv_fn());

    I sum_rcp = thrust::reduce(thrust::device, pi_rcp_first, pi_rcp_last, I {});
    I sum_pow = thrust::reduce(thrust::device, pi_rcp_first, pi_rcp_last, I {});
    I sum_inv = thrust::reduce(thrust::device, pi_inv_first, pi_inv_last, I {});

    // NOTE: The rest could (and normally should) be done on the CPU
    //       but for testing purposes we use the GPU.
    thrust::device_vector<I> d_pi { sum_rcp, sum_pow, sum_inv };

    thrust::transform(d_pi.begin(), d_pi.end(), d_pi.begin(), final_decrement_fn(n));
    thrust::transform(d_pi.begin(), d_pi.end(), d_pi.begin(), scale_fn());

    thrust::host_vector<I> h_pi = d_pi;

    for (I pi_approx : h_pi) {
        expect(contains(pi_approx, std::numbers::pi));
        expect(le(pi_approx.lb, std::numbers::pi));
        expect(ge(pi_approx.ub, std::numbers::pi));
    }
}
