#include "generated/tests_generated.cu"
#include "tests_additional.cu"

int main()
{
    tests_additional<double>();
    tests_generated<double>();

    return 0;
}
