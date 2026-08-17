#ifndef CUINTERVAL_H
#define CUINTERVAL_H

#ifdef __CUDACC__
#include <cuinterval/arithmetic/intrinsics.cuh>
#include <cuinterval/arithmetic/operations.cuh>
#endif

#include <cuinterval/compare.h>
#include <cuinterval/interval.h>
#include <cuinterval/numbers.h>
#include <cuinterval/traits.h>

#endif // CUINTERVAL_H
