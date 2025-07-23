#ifndef CUINTERVAL_FORMAT_H
#define CUINTERVAL_FORMAT_H

#include <cuinterval/interval.h>

#include <format>
#include <ostream>

namespace cu
{

template<typename T>
std::ostream &operator<<(std::ostream &os, interval<T> x)
{
    return os << "[" << x.lb << ", " << x.ub << "]";
}

template<typename T>
std::ostream &operator<<(std::ostream &os, split<T> x)
{
    return os << "[" << x.lower_half << ", " << x.upper_half << "]";
}

} // namespace cu

template<typename T>
struct std::formatter<cu::interval<T>> : std::formatter<T>
{
    auto format(const cu::interval<T> &x, std::format_context &ctx) const
    {
        auto out = ctx.out();

        out = std::format_to(out, "[");
        out = std::formatter<T>::format(x.lb, ctx);
        out = std::format_to(out, ", ");
        out = std::formatter<T>::format(x.ub, ctx);
        return std::format_to(out, "]");
    }
};

#endif // CUINTERVAL_FORMAT_H
