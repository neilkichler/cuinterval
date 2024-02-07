#include "tests_pow_rev.cu"
#include "tests_c_xsc.cu"
#include "tests_libieeep1788_elem.cu"
#include "tests_mpfi.cu"
#include "tests_filib.cu"

int main()
{
    tests_pow_rev<double>();
    tests_c_xsc<double>();
    tests_libieeep1788_elem<double>();
    tests_mpfi<double>();
    tests_filib<double>();

    return 0;
}
