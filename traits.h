#ifndef CU_TRAITS_H
#define CU_TRAITS_H

#include <type_traits>
#include <utility>

namespace cu
{

template<typename T>
concept numeric = requires(std::remove_cvref_t<T> a, std::remove_cvref_t<T> b) {
    { a + b } -> std::same_as<std::remove_cvref_t<T>>;
    { a - b } -> std::same_as<std::remove_cvref_t<T>>;
    { a * b } -> std::same_as<std::remove_cvref_t<T>>;
    { a / b } -> std::same_as<std::remove_cvref_t<T>>;
};

template<numeric T>
constexpr decltype(auto) value(T &&x) noexcept { return std::forward<T>(x); }

} // namespace cu

#endif // CU_TRAITS_H
