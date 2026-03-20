#include <cuinterval/cuinterval.h>

#include <cuinterval/limits.h>
#include <cuinterval/format.h>

#include <format>
#include <iostream>
#include <limits>

int main()
{
    using interval_limits = std::numeric_limits<cu::interval<double>>;

    std::cout << std::format("interval<double> min: {}\n", interval_limits::min());
    std::cout << std::format("interval<double> max: {}\n", interval_limits::max());
    std::cout << std::format("interval<double> eps: {}\n", interval_limits::epsilon());

    return 0;
}
