
#include <cuinterval/arithmetic/basic.cuh>

// compiler bug fix; TODO: remove when fixed
#ifdef __CUDACC__
#pragma push_macro("__cpp_consteval")
#define consteval constexpr
#include <boost/ut.hpp>
#undef consteval
#pragma pop_macro("__cpp_consteval")
#else
#include <boost/ut.hpp>
#endif

#include <stdio.h>

constexpr auto sum(auto... values) { return (values + ...); }

int main() {
  using namespace boost::ut;

  printf("%d\n", _GLIBCXX_RELEASE);

  "sum"_test = [] {
    expect(0 == 0_i);
    expect(sum(1, 2) == 3_i);
    // expect(sum(1, 2) > 0_i and 41_i == sum(40, 2));
  };
}
