#pragma once
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

typedef cu::interval<double> (*fn_t)(cu::interval<double>);

// Example implementation of the bisection method for finding all roots in a given interval.
template<typename T, int max_depth>
__global__ void bisection(fn_t f, cu::interval<T> x_init, double tol, cu::interval<T> *roots, std::size_t *max_roots)
{
    using I = cu::interval<T>;

    std::size_t n_roots = 0;
    local_stack<I, max_depth> intervals;
    intervals.push(x_init);

    for (int depth = 0; !intervals.empty() && depth < max_depth; depth++) {
        I x = intervals.pop();
        I y = (*f)(x);

        if (!contains(y, 0.0)) {
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
                    || inf(root) <= inf(x) && inf(x) <= sup(root)
                    || inf(x) <= sup(root) && sup(root) <= sup(x)
                    || inf(x) <= inf(root) && inf(root) <= sup(x)) {
                    roots[i] = convex_hull(root, x);
                    absorbed = true;
                    break;
                }
            }

            if (!absorbed) {
                roots[n_roots] = x;
                n_roots++;
            }

            if (n_roots == *max_roots) {
                break; // reached max roots we can store
            }
        } else {
            // interval could still contain a root -> bisect
            cu::split<T> c = bisect(x, 0.5);
            // we do depth-first search which often will not be optimal
            intervals.push(c.upper_half);
            intervals.push(c.lower_half);
        }
    }

    // check if all roots are indeed inside the tolerance, otherwise mince
    constexpr int n_splits = 16;
    for (std::size_t i = 0; i < n_roots; i++) {
        I splits[n_splits] {};
        if (width(roots[i]) > tol) {
            mince(roots[i], splits, n_splits);

            I new_root {};
            int j;
            for (j = 0; j < n_splits; j++) {
                if (contains(f(splits[j]), 0.0)) {
                    new_root = splits[j];
                    break;
                }
            }
            for (j = j + 1; j < n_splits; j++) {
                if (contains(f(splits[j]), 0.0)) {
                    new_root = convex_hull(new_root, splits[j]);
                }
            }

            roots[i] = new_root;
        }
    }

    for (std::size_t i = 0; i < n_roots; i++) {
        assert(contains(f(roots[i]), 0.0));
        assert(width(roots[i]) <= tol);
    }

    *max_roots = n_roots;
}
