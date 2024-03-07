
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

#include <cstdio>

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

        // printf("x = [%f, %f], y = [%f, %f]\n", x.lb, x.ub, y.lb, y.ub);

        if (!contains(y, 0.0)) {
            // printf("no roots in [%f, %f]\n", x.lb, x.ub);
            continue; // no roots in this interval -> no further splitting
        }

        T m = mid(x);
        if (width(x) < tol || m == inf(x) || m == sup(x)) {
            // found a root

            // try to absorb the root if close to an existing root interval
            bool absorbed = false;
            for (std::size_t i = 0; i < n_roots; i++) {
                I root = roots[i];
                if (inf(root) <= sup(x) && sup(x) <= sup(root)
                ||  inf(root) <= inf(x) && inf(x) <= sup(root)
                ||  inf(x) <= sup(root) && sup(root) <= sup(x)
                ||  inf(x) <= inf(root) && inf(root) <= sup(x)) {
                    roots[i] = convex_hull(root, x);
                    absorbed = true;
                    // printf("absorbed root at = %f, %f with new root = %f, %f\n", root.lb, root.ub, x.lb, x.ub);
                    break;
                }
            }

            if (!absorbed) {
                roots[n_roots] = x;
                // printf("found root at = %f, %f\n", x.lb, x.ub);
                n_roots++;
            }


            if (n_roots == *max_roots) {
                // printf("Reached max_roots = %zu\n", *max_roots);
                break; // reached max roots we can store
            }
        } else {
            // interval could still contain a root -> bisect
            split<T> c = bisect(x, 0.5);
            // we do depth-first search which often will not be optimal
            intervals.push(c.upper_half);
            intervals.push(c.lower_half);
            // printf("bisect = [%f, %f] -> ([%f, %f], [%f, %f])\n", x.lb, x.ub, c.lower_half.lb, c.lower_half.ub, c.upper_half.lb, c.upper_half.ub);
        }
    }

    // printf("n_roots = %d\n", (int)n_roots);

    *max_roots = n_roots;
}


