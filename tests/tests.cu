#include "tests_c_xsc.cu"
#include "tests_libieeep1788_elem.cu"
#include "tests_filib.cu"

int main()
{
    tests_c_xsc<double>();
    tests_libieeep1788_elem<double>();
    tests_filib<double>();

    return 0;
}
