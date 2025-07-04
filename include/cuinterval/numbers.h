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

// TODO: cu::interval<float> specialization

} // namespace std::numbers

// In cu:: we provide access to all the standard math constants and some additional helpful ones.
namespace cu
{

using std::numbers::pi_v;

template<typename T>
inline constexpr T pi_2_v; // = pi / 2

template<>
inline constexpr interval<double>
    pi_2_v<interval<double>> = { 0x1.921fb54442d18p+0, 0x1.921fb54442d19p+0 };

template<typename T>
inline constexpr T tau_v; // = 2 * pi

template<>
inline constexpr interval<double>
    tau_v<interval<double>> = { 0x1.921fb54442d18p+2, 0x1.921fb54442d19p+2 };

} // namespace cu

#endif // CUINTERVAL_NUMBERS_H
