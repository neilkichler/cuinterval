#ifndef CUINTERVAL_LIMITS_H
#define CUINTERVAL_LIMITS_H

#include <cuinterval/interval.h>

#include <limits>

namespace std
{

// behaves just like the underlying type T would behave
template<typename T>
struct numeric_limits<cu::interval<T>>
{
    // clang-format off
    static constexpr cu::interval<T>           min() noexcept { return numeric_limits<T>::min(); }
    static constexpr cu::interval<T>           max() noexcept { return numeric_limits<T>::max(); }
    static constexpr cu::interval<T>        lowest() noexcept { return numeric_limits<T>::lowest(); }
    static constexpr cu::interval<T>       epsilon() noexcept { return numeric_limits<T>::epsilon(); }
    static constexpr cu::interval<T>   round_error() noexcept { return numeric_limits<T>::round_error(); }
    static constexpr cu::interval<T>      infinity() noexcept { return numeric_limits<T>::infinity(); }
    static constexpr cu::interval<T>     quiet_NaN() noexcept { return numeric_limits<T>::quiet_NaN(); }
    static constexpr cu::interval<T> signaling_NaN() noexcept { return numeric_limits<T>::signaling_NaN(); }
    static constexpr cu::interval<T>    denorm_min() noexcept { return numeric_limits<T>::denorm_min(); }
    static constexpr bool is_specialized           = true;
    static constexpr bool is_signed                = true;
    static constexpr bool is_integer               = false;
    static constexpr bool is_exact                 = false;
    static constexpr int digits                    = numeric_limits<T>::digits;
    static constexpr int digits10                  = numeric_limits<T>::digits10;
    static constexpr int max_digits10              = numeric_limits<T>::max_digits10;
    static constexpr int radix                     = numeric_limits<T>::radix;
    static constexpr int min_exponent              = numeric_limits<T>::min_exponent;
    static constexpr int min_exponent10            = numeric_limits<T>::min_exponent10;
    static constexpr int max_exponent              = numeric_limits<T>::max_exponent;
    static constexpr int max_exponent10            = numeric_limits<T>::max_exponent10;
    static constexpr bool has_infinity             = numeric_limits<T>::has_infinity;
    static constexpr bool has_quiet_NaN            = numeric_limits<T>::has_quiet_NaN;
    static constexpr bool has_signaling_NaN        = numeric_limits<T>::has_signaling_NaN;
    static constexpr float_denorm_style has_denorm = numeric_limits<T>::has_denorm;
    static constexpr bool has_denorm_loss          = numeric_limits<T>::has_denorm_loss;
    static constexpr bool is_iec559                = numeric_limits<T>::is_iec559;
    static constexpr bool is_bounded               = numeric_limits<T>::is_bounded;
    static constexpr bool is_modulo                = numeric_limits<T>::is_modulo;
    static constexpr bool traps                    = numeric_limits<T>::traps;
    static constexpr bool tinyness_before          = numeric_limits<T>::tinyness_before;
    static constexpr float_round_style round_style = numeric_limits<T>::round_style;
    // clang-format on
};

} // namespace std
#endif // CUINTERVAL_LIMITS_H
