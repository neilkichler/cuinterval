#ifndef CUINTERVAL_NUMBERS_H
#define CUINTERVAL_NUMBERS_H

#include <cuinterval/interval.h>

#include <numbers>

// Explicit specialization of math constants is allowed for custom types.
// See https://eel.is/c++draft/numbers#math.constants-2.
namespace std::numbers
{

// The enclosure is chosen to be the smallest representable floating point interval
// which still contains the real pi value.
template<>
inline constexpr cu::interval<double>
    pi_v<cu::interval<double>> = { 0x1.921fb54442d18p+1, 0x1.921fb54442d19p+1 };

template<>
inline constexpr cu::interval<float>
    pi_v<cu::interval<float>> = { 0x1.921fb5p+1, 0x1.921fb6p+1 };

template<>
inline constexpr cu::interval<double>
    e_v<cu::interval<double>> = { 0x1.5bf0a8b145769p+1, 0x1.5bf0a8b14576ap+1 };

template<>
inline constexpr cu::interval<float>
    e_v<cu::interval<float>> = { 0x1.5bf0a8p+1, 0x1.5bf0a9p+1 };

template<>
inline constexpr cu::interval<double>
    log2e_v<cu::interval<double>> = { 0x1.71547652b82fep+0, 0x1.71547652b82ffp+0 };

template<>
inline constexpr cu::interval<float>
    log2e_v<cu::interval<float>> = { 0x1.715476p+0, 0x1.715477p+0 };

} // namespace std::numbers

// In cu:: we provide access to all the standard math constants and some additional helpful ones.
namespace cu
{

using std::numbers::e_v;
using std::numbers::log2e_v;
using std::numbers::pi_v;

template<typename T>
inline constexpr T pi_2_v; // = pi / 2

template<>
inline constexpr interval<double>
    pi_2_v<interval<double>> = { 0x1.921fb54442d18p+0, 0x1.921fb54442d19p+0 };

template<>
inline constexpr interval<float>
    pi_2_v<cu::interval<float>> = { 0x1.921fb5p+0, 0x1.921fb6p+0 };

template<typename T>
inline constexpr T tau_v; // = 2 * pi

template<>
inline constexpr interval<double>
    tau_v<interval<double>> = { 0x1.921fb54442d18p+2, 0x1.921fb54442d19p+2 };

template<>
inline constexpr interval<float>
    tau_v<interval<float>> = { 0x1.921fb5p+2, 0x1.921fb6p+2 };

} // namespace cu

#endif // CUINTERVAL_NUMBERS_H
