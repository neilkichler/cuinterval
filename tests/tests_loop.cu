#include "tests.h"

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
        return I { 1.0, 1.0 } / (sqr(x));
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
void tests_pi_approximation(cudaStream_t stream)
{
    using I = interval<T>;
    using namespace boost::ut;

    constexpr int n = 100'000;
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

    I sum_rcp = thrust::reduce(thrust::cuda::par.on(stream), pi_rcp_first, pi_rcp_last, I {});
    I sum_pow = thrust::reduce(thrust::cuda::par.on(stream), pi_rcp_first, pi_rcp_last, I {});
    I sum_inv = thrust::reduce(thrust::cuda::par.on(stream), pi_inv_first, pi_inv_last, I {});

    // NOTE: The rest could (and normally should) be done on the CPU
    //       but for testing purposes we use the GPU.
    thrust::device_vector<I> d_pi { sum_rcp, sum_pow, sum_inv };

    thrust::transform(thrust::cuda::par.on(stream), d_pi.begin(), d_pi.end(), d_pi.begin(), final_decrement_fn(n));
    thrust::transform(thrust::cuda::par.on(stream), d_pi.begin(), d_pi.end(), d_pi.begin(), scale_fn());

    thrust::host_vector<I> h_pi = d_pi;

    for (I pi_approx : h_pi) {
        expect(contains(pi_approx, std::numbers::pi));
        expect(le(pi_approx.lb, std::numbers::pi));
        expect(ge(pi_approx.ub, std::numbers::pi));
    }
}

#include <cstdio>

struct coeff_fn
{
    template<typename T>
    __device__ interval<T> operator()(T x) const
    {
        using I = interval<T>;
        return I { 1.0, 1.0 } / I { x, x };
    }
};

template<typename I>
struct horner_fn
{
    I x;

    horner_fn(I _x)
        : x(_x)
    { }

    __device__ I operator()(I res, I coeff) const
    {
        return res * x + coeff;
    }
};

template<typename T>
void tests_horner(cudaStream_t stream)
{
    using I = interval<T>;
    using namespace boost::ut;

    // Approximate exp with Horner's scheme.
    constexpr int n_coefficients = 16;
    thrust::host_vector<T> ps(n_coefficients);

    thrust::counting_iterator<T> seq_first(1);
    thrust::counting_iterator<T> seq_last = seq_first + n_coefficients;

    thrust::inclusive_scan(seq_first, seq_last, ps.begin(), thrust::multiplies<T>());

    thrust::device_vector<T> d_ps = ps;
    thrust::device_vector<I> d_coefficients(n_coefficients);
    thrust::transform(d_ps.begin(), d_ps.end() - 1, d_coefficients.begin() + 1, coeff_fn());
    d_coefficients[0] = I { 1.0, 1.0 };

    // example input
    T eps = 1.0e-12;
    I x{ 1.0 - eps, 1.0 + eps };
    thrust::device_vector<I> d_res(n_coefficients);
    thrust::inclusive_scan(d_coefficients.rbegin(), d_coefficients.rend(), d_res.begin(), horner_fn<I>(x));

    thrust::host_vector<I> coefficients = d_coefficients;
    thrust::host_vector<I> res = d_res;

    I exp_approx = res[res.size() - 1];
    T exp_true = std::numbers::e;

    expect(contains(exp_approx, exp_true));
}
