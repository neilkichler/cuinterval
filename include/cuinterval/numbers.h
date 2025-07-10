#ifndef CUINTERVAL_NUMBERS_H
#define CUINTERVAL_NUMBERS_H

#include <cuinterval/interval.h>

#include <numbers>

// Explicit specialization of math constants is allowed for custom types.
// See https://eel.is/c++draft/numbers#math.constants-2.
namespace std::numbers
{

// The enclosure is chosen to be the smallest representable floating point interval
// which still contains the real value.

template<>
inline constexpr cu::interval<double>
    e_v<cu::interval<double>> = { 0x1.5bf0a8b145769p+1, 0x1.5bf0a8b14576ap+1 };

template<>
inline constexpr cu::interval<float>
    e_v<cu::interval<float>> = { 0x1.5bf0a8p+1f, 0x1.5bf0aap+1f };

template<>
inline constexpr cu::interval<double>
    log2e_v<cu::interval<double>> = { 0x1.71547652b82fep+0, 0x1.71547652b82ffp+0 };

template<>
inline constexpr cu::interval<float>
    log2e_v<cu::interval<float>> = { 0x1.715476p+0f, 0x1.715478p+0f };

template<>
inline constexpr cu::interval<double>
    log10e_v<cu::interval<double>> = { 0x1.bcb7b1526e50ep-2, 0x1.bcb7b1526e50fp-2 };

template<>
inline constexpr cu::interval<float>
    log10e_v<cu::interval<float>> = { 0x1.bcb7b0p-2f, 0x1.bcb7b2p-2f };

template<>
inline constexpr cu::interval<double>
    pi_v<cu::interval<double>> = { 0x1.921fb54442d18p+1, 0x1.921fb54442d19p+1 };

template<>
inline constexpr cu::interval<float>
    pi_v<cu::interval<float>> = { 0x1.921fb4p+1f, 0x1.921fb6p+1f };

template<>
inline constexpr cu::interval<double>
    inv_pi_v<cu::interval<double>> = { 0x1.45f306dc9c882p-2, 0x1.45f306dc9c883p-2 };

template<>
inline constexpr cu::interval<float>
    inv_pi_v<cu::interval<float>> = { 0x1.45f306p-2f, 0x1.45f308p-2f };

template<>
inline constexpr cu::interval<double>
    inv_sqrtpi_v<cu::interval<double>> = { 0x1.20dd750429b6cp-1, 0x1.20dd750429b6dp-1 };

template<>
inline constexpr cu::interval<float>
    inv_sqrtpi_v<cu::interval<float>> = { 0x1.20dd74p-1f, 0x1.20dd76p-1f };

template<>
inline constexpr cu::interval<double>
    ln2_v<cu::interval<double>> = { 0x1.62e42fefa39efp-1, 0x1.62e42fefa39f0p-1 };

template<>
inline constexpr cu::interval<float>
    ln2_v<cu::interval<float>> = { 0x1.62e42ep-1f, 0x1.62e430p-1f };

template<>
inline constexpr cu::interval<double>
    ln10_v<cu::interval<double>> = { 0x1.26bb1bbb55515p+1, 0x1.26bb1bbb55516p+1 };

template<>
inline constexpr cu::interval<float>
    ln10_v<cu::interval<float>> = { 0x1.26bb1ap+1f, 0x1.26bb1cp+1f };

template<>
inline constexpr cu::interval<double>
    sqrt2_v<cu::interval<double>> = { 0x1.6a09e667f3bcdp+0, 0x1.6a09e667f3bcep+0 };

template<>
inline constexpr cu::interval<float>
    sqrt2_v<cu::interval<float>> = { 0x1.6a09e6p+0f, 0x1.6a09e8p+0f };

template<>
inline constexpr cu::interval<double>
    sqrt3_v<cu::interval<double>> = { 0x1.bb67ae8584caap+0, 0x1.bb67ae8584cabp+0 };

template<>
inline constexpr cu::interval<float>
    sqrt3_v<cu::interval<float>> = { 0x1.bb67aep+0f, 0x1.bb67b0p+0f };

template<>
inline constexpr cu::interval<double>
    inv_sqrt3_v<cu::interval<double>> = { 0x1.279a74590331cp-1, 0x1.279a74590331dp-1 };

template<>
inline constexpr cu::interval<float>
    inv_sqrt3_v<cu::interval<float>> = { 0x1.279a74p-1f, 0x1.279a76p-1f };

template<>
inline constexpr cu::interval<double>
    egamma_v<cu::interval<double>> = { 0x1.2788cfc6fb618p-1, 0x1.2788cfc6fb619p-1 };

template<>
inline constexpr cu::interval<float>
    egamma_v<cu::interval<float>> = { 0x1.2788cep-1f, 0x1.2788d0p-1f };

template<>
inline constexpr cu::interval<double>
    phi_v<cu::interval<double>> = { 0x1.9e3779b97f4a8p+0, 0x1.9e3779b97f4a9p+0 };

template<>
inline constexpr cu::interval<float>
    phi_v<cu::interval<float>> = { 0x1.9e3778p+0f, 0x1.9e377ap+0f };

} // namespace std::numbers

// In cu:: we provide access to all the standard math constants and some additional helpful ones.
namespace cu
{

using std::numbers::e_v;
using std::numbers::egamma_v;
using std::numbers::inv_pi_v;
using std::numbers::inv_sqrt3_v;
using std::numbers::inv_sqrtpi_v;
using std::numbers::ln10_v;
using std::numbers::ln2_v;
using std::numbers::log10e_v;
using std::numbers::log2e_v;
using std::numbers::phi_v;
using std::numbers::pi_v;
using std::numbers::sqrt2_v;
using std::numbers::sqrt3_v;

template<typename T>
inline constexpr T pi_2_v; // = pi / 2

template<>
inline constexpr interval<double>
    pi_2_v<interval<double>> = { 0x1.921fb54442d18p+0, 0x1.921fb54442d19p+0 };

template<>
inline constexpr interval<float>
    pi_2_v<cu::interval<float>> = { 0x1.921fb4p+0f, 0x1.921fb6p+0f };

template<typename T>
inline constexpr T tau_v; // = 2 * pi

template<>
inline constexpr interval<double>
    tau_v<interval<double>> = { 0x1.921fb54442d18p+2, 0x1.921fb54442d19p+2 };

template<>
inline constexpr interval<float>
    tau_v<interval<float>> = { 0x1.921fb4p+2f, 0x1.921fb6p+2f };

} // namespace cu

#endif // CUINTERVAL_NUMBERS_H
