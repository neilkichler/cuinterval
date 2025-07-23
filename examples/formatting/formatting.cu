#include <cuinterval/cuinterval.h>

// We do not put the formatting functionality in the default
// header because some might choose to use a different formatting.
// It does support ostream and std::format based output.
// The std::format accepts a format specifier that supports
// all the specifiers that exist for the underlying type T
// of the cu::interval<T>.
#include <cuinterval/format.h>

#include <iostream>

int main()
{
    cu::interval<double> interval { 1.0, 2.0 };
    std::cout << interval << '\n';
    std::cout << std::format("{}\n", interval);
    // in c++23 you may use std::print, std::println
    // std::println("{}", interval);

    // type specifiers like for double (applies to both lower and upper bound)
    std::cout << std::format("{:8.4f}\n", interval);

    return 0;
}
