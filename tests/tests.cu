#include "tests_additional.cu"
#include "generated/tests_generated.cu"

int main()
{
    tests_additional<double>();
    tests_generated<double>();

    return 0;
}
