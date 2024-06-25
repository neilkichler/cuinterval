
#include <cstdlib>

#include "cuinterval/examples/bisection.cuh"
#include <cuinterval/cuinterval.h>

#include "utils.h"

using cu::interval;

template<typename I>
__device__ I f(I x)
{
//     // return exp(I { -3.0, -3.0 } * x) - sin(x) * sin(x) * sin(x);
//     // return I{1.0, 1.0};
//     // return x*sqr(x) - (I{2.0, 2.0} * sqr(x)) + x;
//     // return sqr(sin(x)) - (I{1.0, 1.0} - cos(I{2.0, 2.0} * x)) / I{2.0, 2.0};
    return pow(x, 3) - pow(x, 2) - 17.0 * x - 15.0;
};

__device__ fn_t d_f = f<interval<double>>;

void example_bisection()
{
    using I = interval<double>;

    I x = { -5.0, 10.0 };

    constexpr double tolerance      = 1e-12;
    constexpr std::size_t max_depth = 512;
    std::size_t max_roots           = 16;

    std::size_t *d_max_roots;
    CUDA_CHECK(cudaMalloc(&d_max_roots, sizeof(*d_max_roots)));
    CUDA_CHECK(cudaMemcpy(d_max_roots, &max_roots, sizeof(*d_max_roots), cudaMemcpyHostToDevice));

    I *d_roots;
    CUDA_CHECK(cudaMalloc(&d_roots, max_roots * sizeof(*d_roots)));

    fn_t h_f;
    cudaMemcpyFromSymbol(&h_f, d_f, sizeof(h_f));

    bisection<double, max_depth><<<1, 1>>>(h_f, x, tolerance, d_roots, d_max_roots);
    CUDA_CHECK(cudaMemcpy(&max_roots, d_max_roots, sizeof(*d_max_roots), cudaMemcpyDeviceToHost));

    I *h_roots = (I *)std::malloc(max_roots * sizeof(*h_roots));
    CUDA_CHECK(cudaMemcpy(h_roots, d_roots, max_roots * sizeof(I), cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i < max_roots; i++) {
        printf("Root %zu in [%.15f, %.15f]\n", i, h_roots[i].lb, h_roots[i].ub);
    }

    std::free(h_roots);
    CUDA_CHECK(cudaFree(d_roots));
    CUDA_CHECK(cudaFree(d_max_roots));
}

int main()
{
    example_bisection();
    return 0;
}
