#include "tests_bisect.cu"
#include "tests_loop.cu"

template<typename T>
void tests_additional() {
    tests_bisect<T>();
    // tests_loop<T>();
}
