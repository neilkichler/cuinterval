#include "tests_bisect.cu"
#include "tests_loop.cu"

// #include "tests_example.cu"

template<typename T>
void tests_additional()
{
    tests_bisect<T>();
    tests_loop<T>();
    // tests_loop();
    // tests_example_reduce();
}
