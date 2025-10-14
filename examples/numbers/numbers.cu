#include <cuinterval/cuinterval.h>

#include <cuda_runtime.h>

#define CUDA_CHECK(x)                                                                \
    do {                                                                             \
        cudaError_t err = x;                                                         \
        if (err != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, \
                    __FILE__, __LINE__, cudaGetErrorString(err),                     \
                    cudaGetErrorName(err), err);                                     \
            abort();                                                                 \
        }                                                                            \
    } while (0)

template <typename T>
__device__ T area_of_circle(T r) {
    return std::numbers::pi_v<T> * r * r;
}

__global__ void kernel(auto *xs, auto *res, std::integral auto n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        res[i] = area_of_circle(xs[i]);
    }
}

int main()
{
    constexpr int n = 16;
    using T         = cu::interval<double>;
    T xs[n], res[n];

    // generate dummy data
    for (int i = 0; i < n; i++) {
        double v = i;
        xs[i]    = { v };
    }

    T *d_xs, *d_res;
    CUDA_CHECK(cudaMalloc(&d_xs, n * sizeof(*xs)));
    CUDA_CHECK(cudaMalloc(&d_res, n * sizeof(*res)));

    CUDA_CHECK(cudaMemcpy(d_xs, xs, n * sizeof(*xs), cudaMemcpyHostToDevice));

    kernel<<<n, 1>>>(d_xs, d_res, n);

    CUDA_CHECK(cudaMemcpy(res, d_res, n * sizeof(*res), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        auto r = res[i];
        printf("area_of_circle(%g) = [%.15f, %.15f]\n", xs[i].lb, r.lb, r.ub);
    }

    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_res));

    return 0;
}
