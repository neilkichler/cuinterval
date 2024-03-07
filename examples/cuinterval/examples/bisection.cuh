
#include <cuinterval/cuinterval.h>

// Stack in local memory. Managed independently for each thread.
template<class T, std::size_t N>
struct local_stack
{
    using size_type = std::size_t;

    __device__ T const &top() const { return buf[len - 1]; }
    __device__ T &top() { return buf[len - 1]; }
    __device__ void push(T const &v) { buf[len++] = v; }
    __device__ T pop() { return buf[--len]; }
    __device__ bool full() { return len == N; }
    __device__ bool empty() { return len == 0; }

    T buf[N];
    size_type len {};
};

// Example implementation of the bisection method for finding all roots in a given interval.
template<typename T, int max_depth>
__global__ void bisection(interval<T> x_init, double tol, interval<T> *roots, std::size_t *max_roots)
{
    using I = interval<T>;

    std::size_t n_roots = 0;
    local_stack<I, max_depth> intervals;
    intervals.push(x_init);

    for (int depth = 0; !intervals.empty() && depth < max_depth; depth++) {
        I x = intervals.pop();
        I y = f(x);

        if (!contains(y, 0.0)) {
            continue; // no roots in this interval -> no further splitting
        }

        T m = mid(x);
        if (width(x) < tol || m == inf(x) || m == sup(x)) {
            // found a root
            roots[n_roots] = x;
            n_roots++;
            if (n_roots == *max_roots) {
                break; // reached max roots we can store
            }
        } else {
            // interval could still contain a root -> bisect
            split<T> c = bisect(x, 0.5);
            intervals.push(c.upper_half);
            intervals.push(c.lower_half);
        }
    }

    *max_roots = n_roots;
}


