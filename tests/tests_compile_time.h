#pragma once

#include <cuinterval/interval.h>

// #include <boost/ut.hpp>

#include <concepts>

// using namespace boost::ut;

static_assert(std::regular<cu::interval<int>>);
static_assert(std::regular<cu::interval<float>>);
static_assert(std::regular<cu::interval<double>>);

// void tests_compile_time()
// {
//     "regular"_test = [] consteval {
//         static_assert(std::regular<cu::interval<int>>);
//         static_assert(std::regular<cu::interval<float>>);
//         static_assert(std::regular<cu::interval<double>>);
//     };
// }
