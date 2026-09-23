#ifndef CUINTERVAL_ARITHMETIC_INFO_CUH
#define CUINTERVAL_ARITHMETIC_INFO_CUH

//
// Provides information about the arithmetic operations.
//

// This information is CUDA specific and can be obtained from:
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#cuda-c-mathematical-standard-library-functions
//
// max_ulp_error (from the documentation):
// ---------------------------------------
// The maximum ULP error is stated as the maximum observed absolute value
// of the difference in ULPs between the value returned by the function and
// a correctly rounded result of the corresponding precision obtained according
// to the round-to-nearest ties-to-even rounding mode.

namespace info
{

#define MAX_ULP_ERROR(op, n_float, n_double)           \
    template<typename T>                               \
    struct op                                          \
    {                                                  \
        static constexpr int max_ulp_error = n_float;  \
    };                                                 \
    template<>                                         \
    struct op<float>                                   \
    {                                                  \
        static constexpr int max_ulp_error = n_float;  \
    };                                                 \
    template<>                                         \
    struct op<double>                                  \
    {                                                  \
        static constexpr int max_ulp_error = n_double; \
    };

// 5.5.7.2. Exponential Functions
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#exponential-functions
MAX_ULP_ERROR(exp, 2, 1);
MAX_ULP_ERROR(exp2, 2, 1);
MAX_ULP_ERROR(expm1, 1, 1);
MAX_ULP_ERROR(exp10, 2, 1);
MAX_ULP_ERROR(log, 1, 1);
MAX_ULP_ERROR(log10, 2, 1);
MAX_ULP_ERROR(log2, 1, 1);
MAX_ULP_ERROR(log1p, 1, 1);

// 5.5.7.3. Power Functions
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#power-functions
MAX_ULP_ERROR(pow, 4, 2);
MAX_ULP_ERROR(sqrt, 0, 0);
MAX_ULP_ERROR(cbrt, 1, 1);
MAX_ULP_ERROR(hypot, 3, 2);

// 5.5.7.4. Trigonometric Functions
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#trigonometric-functions
MAX_ULP_ERROR(sin, 2, 2);
MAX_ULP_ERROR(cos, 2, 2);
MAX_ULP_ERROR(tan, 4, 2);
MAX_ULP_ERROR(asin, 2, 2);
MAX_ULP_ERROR(acos, 2, 2);
MAX_ULP_ERROR(atan, 2, 2);
MAX_ULP_ERROR(atan2, 3, 2);

// 5.5.7.5. Hyperbolic Functions
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#hyperbolic-functions
MAX_ULP_ERROR(sinh, 3, 2);
MAX_ULP_ERROR(cosh, 2, 1);
MAX_ULP_ERROR(tanh, 2, 1);
MAX_ULP_ERROR(asinh, 3, 3);
MAX_ULP_ERROR(acosh, 4, 3);
MAX_ULP_ERROR(atanh, 3, 2);

// 5.5.7.6. Error and Gamma Functions
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#error-and-gamma-functions
MAX_ULP_ERROR(erf, 2, 2);
MAX_ULP_ERROR(erfc, 4, 5);

// 5.5.8. Non-Standard CUDA Mathematical Functions
// https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html#non-standard-cuda-mathematical-functions
MAX_ULP_ERROR(cospi, 1, 2);
MAX_ULP_ERROR(sinpi, 1, 2);
MAX_ULP_ERROR(sincospi, 1, 2);

#undef MAX_ULP_ERROR

} // namespace info

#endif // CUINTERVAL_ARITHMETIC_INFO_CUH
