#include "tests_bisect.cu"
#include "tests_loop.cu"

template<typename T>
void tests_additional(char *buffer)
{
    tests_bisect<T>(buffer);
    tests_bisection<T>(buffer);
    tests_pi_approximation<T>();
    tests_horner<T>();
}
