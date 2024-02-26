
#include <cmath>
#include <cstdio>

int main() {
    int quotient;
    double rem;

    for (int i = -10; i <= 10; i++) {
        double v = i * M_PI_2 - 0.5;
        rem = remquo(v - M_PI_4, M_PI_2, &quotient);
        unsigned int quadrant = static_cast<unsigned>(quotient) % 4;
        // int quadrant = quotient % 4;
        printf("remquo example i=%d; v=%f; rem: %e, quo: %d, quadrant: %d\n", i, v, rem, quotient, quadrant);
    }


    {
    double v = 0.0;
    rem = remquo(v, M_PI_2, &quotient);
    unsigned int quadrant = static_cast<unsigned>(quotient) % 4;
    printf("remquo example v=%e; rem: %e, quadrant: %d\n", v, rem, quadrant);
    }
    {
    double v = M_PI_2;
    rem = remquo(v, M_PI_2, &quotient);
    unsigned int quadrant = static_cast<unsigned>(quotient) % 4;
    printf("remquo example v=%e; rem: %e, quadrant: %d\n", v, rem, quadrant);
    }
    {
    double v = M_PI;
    rem = remquo(v, M_PI_2, &quotient);
    unsigned int quadrant = static_cast<unsigned>(quotient) % 4;
    printf("remquo example v=%e; rem: %e, quadrant: %d\n", v, rem, quadrant);
    }
    {
    double v = 3 * M_PI_2;
    rem = remquo(v, M_PI_2, &quotient);
    unsigned int quadrant = static_cast<unsigned>(quotient) % 4;
    printf("remquo example v=%e; rem: %e, quadrant: %d\n", v, rem, quadrant);
    }
    {
    double v = 2 * M_PI;
    rem = remquo(v, M_PI_2, &quotient);
    unsigned int quadrant = static_cast<unsigned>(quotient) % 4;
    printf("remquo example v=%e; rem: %e, quadrant: %d\n", v, rem, quadrant);
    }
    {
    double v = - M_PI_2;
    rem = remquo(v, M_PI_2, &quotient);
    unsigned int quadrant = static_cast<unsigned>(quotient) % 4;
    printf("remquo example v=%e; rem: %e, quadrant: %d\n", v, rem, quadrant);
    }
    {
    double v = - M_PI;
    rem = remquo(v, M_PI_2, &quotient);
    unsigned int quadrant = static_cast<unsigned>(quotient) % 4;
    printf("remquo example v=%e; rem: %e, quadrant: %d\n", v, rem, quadrant);
    }
    {
    double v = -3 * M_PI_2;
    rem = remquo(v, M_PI_2, &quotient);
    unsigned int quadrant = static_cast<unsigned>(quotient) % 4;
    printf("remquo example v=%e; rem: %e, quadrant: %d\n", v, rem, quadrant);
    }
    {
    double v = -2 * M_PI;
    rem = remquo(v, M_PI_2, &quotient);
    unsigned int quadrant = static_cast<unsigned>(quotient) % 4;
    printf("remquo example v=%e; rem: %e, quadrant: %d\n", v, rem, quadrant);
    }
    {
    double v = 0.8;
    rem = remquo(v, M_PI, &quotient);
    unsigned int quadrant = static_cast<unsigned>(quotient) % 4;
    printf("remquo example v=%e; rem: %e, quadrant: %d\n", v, rem, quadrant);
    }
    {
    printf("pi/2 is: %.16f\n", M_PI_2);
    double v = 1.0;
    rem = remquo(v, M_PI_2, &quotient);
    unsigned int quadrant = static_cast<unsigned>(quotient) % 4;
    printf("remquo example v=%e; rem: %e, quadrant: %d\n", v, rem, quadrant);
    }
    {
    double v = 2.0;
    rem = remquo(v, M_PI_2, &quotient);
    unsigned int quadrant = static_cast<unsigned>(quotient) % 4;
    printf("remquo example v=%e; rem: %e, quadrant: %d\n", v, rem, quadrant);
    }
    {
    double v = 3.0;
    rem = remquo(v, M_PI_2, &quotient);
    unsigned int quadrant = static_cast<unsigned>(quotient) % 4;
    printf("remquo example v=%e; rem: %e, quadrant: %d\n", v, rem, quadrant);
    }
    {
    double v = 4.0;
    rem = remquo(v, M_PI_2, &quotient);
    unsigned int quadrant = static_cast<unsigned>(quotient) % 4;
    printf("remquo example v=%e; rem: %e, quadrant: %d\n", v, rem, quadrant);
    }

    // printf("pi is: %.16f\n", M_PI);
    // printf("pi is: %a\n", M_PI);
    // // printf("pi is: %a\n", std::numbers::pi);
    // printf("pi is: %a\n", std::nextafter(M_PI, M_PI + 1.0));
    // printf("pi is: %a\n", std::nextafter(M_PI, M_PI - 1.0));
    //
    // printf("tau is: %a\n", 6.2831853071795864769252867665590057683943387987502116419498891846);
    // printf("tau is: %.16f\n", 0x1.921fb54442d18p+2);
    // printf("tau is: %.16f\n", 0x1.921fb54442d19p+2);
    //
    return 0;
}
