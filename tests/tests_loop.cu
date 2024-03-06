#include <cuinterval/cuinterval.h>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>

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

struct scale_fn
{
    template<typename I>
    __device__ void operator()(I &x) const
    {
        x = sqrt(x * I { 6.0, 6.0 });
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
        I n_ub = I { 1.0 + n, 1.0 + n};

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

    thrust::counting_iterator<T> seq_first(1);
    thrust::counting_iterator<T> seq_last = seq_first + n;

    auto tr_first = thrust::make_transform_iterator(seq_first, to_interval_fn());
    auto tr_last  = thrust::make_transform_iterator(seq_last, to_interval_fn());
    auto pi_first = thrust::make_transform_iterator(tr_first, pi_recip_fn());
    auto pi_last  = thrust::make_transform_iterator(tr_last, pi_recip_fn());

    I sum = thrust::reduce(thrust::device, pi_first, pi_last, I {});

    // NOTE: The rest could (and normally should) be done on the CPU
    //       but for testing purposes we use the GPU.
    thrust::device_vector<I> d_pi { sum };

    thrust::transform(d_pi.begin(), d_pi.end(), d_pi.begin(), final_decrement_fn(n));
    thrust::for_each(d_pi.begin(), d_pi.end(), scale_fn());

    thrust::host_vector<I> h_pi = d_pi;
    I pi_approx                 = h_pi[0];

    expect(contains(pi_approx, std::numbers::pi));
    expect(le(pi_approx.lb, std::numbers::pi));
    expect(ge(pi_approx.ub, std::numbers::pi));
}
