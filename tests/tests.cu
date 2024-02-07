#include "tests_libieeep1788_class.cu"
#include "tests_pow_rev.cu"
#include "tests_libieeep1788_mul_rev.cu"
#include "tests_ieee1788_constructors.cu"
#include "tests_abs_rev.cu"
#include "tests_c_xsc.cu"
#include "tests_libieeep1788_rev.cu"
#include "tests_libieeep1788_elem.cu"
#include "tests_libieeep1788_reduction.cu"
#include "tests_libieeep1788_rec_bool.cu"
#include "tests_libieeep1788_cancel.cu"
#include "tests_libieeep1788_bool.cu"
#include "tests_atan2.cu"
#include "tests_mpfi.cu"
#include "tests_libieeep1788_set.cu"
#include "tests_filib.cu"
#include "tests_ieee1788_exceptions.cu"
#include "tests_libieeep1788_num.cu"
#include "tests_libieeep1788_overlap.cu"

int main()
{
    tests_libieeep1788_class<double>();
    tests_pow_rev<double>();
    tests_libieeep1788_mul_rev<double>();
    tests_ieee1788_constructors<double>();
    tests_abs_rev<double>();
    tests_c_xsc<double>();
    tests_libieeep1788_rev<double>();
    tests_libieeep1788_elem<double>();
    tests_libieeep1788_reduction<double>();
    tests_libieeep1788_rec_bool<double>();
    tests_libieeep1788_cancel<double>();
    tests_libieeep1788_bool<double>();
    tests_atan2<double>();
    tests_mpfi<double>();
    tests_libieeep1788_set<double>();
    tests_filib<double>();
    tests_ieee1788_exceptions<double>();
    tests_libieeep1788_num<double>();
    tests_libieeep1788_overlap<double>();

    return 0;
}
