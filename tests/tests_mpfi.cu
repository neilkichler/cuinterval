
#include <cuinterval/cuinterval.h>

#include "tests.h"
#include "test_ops.cuh"

#include <stdio.h>

template<typename T>
void tests_mpfi() {
    using namespace boost::ut;

    using I = interval<T>;

    I empty         = ::empty<T>();
    I entire        = ::entire<T>();
    T infinity = std::numeric_limits<T>::infinity();

    const int n = 62; // count of largest test array
    const int n_bytes   = n * sizeof(I);
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;

    interval<T> *d_xs, *d_ys, *d_zs;
    CUDA_CHECK(cudaMalloc(&d_xs, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_ys, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_zs, n_bytes));

    "mpfi_add_add"_test = [&] {
        constexpr int n = 19;
        std::array<I, n> h_xs {{
            {+4.0,+8.0},
            {+4.0,+8.0},
            {-0.375,-0x10187p-256},
            {-0x1p-300,0x123456p+28},
            {-4.0,+7.0},
            {-infinity,+8.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {0.0,+8.0},
            {0.0,+8.0},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0x1000100010001p+8,0x1p+60},
            entire,
        }};

        std::array<I, n> h_ys {{
            {-4.0,-2.0},
            {-9.0,-8.0},
            {-0.125,0x1p-240},
            {-0x10000000000000p-93,0x789abcdp0},
            {-0x123456789abcdp-17,3e300},
            {0.0,+8.0},
            {-1.0,+8.0},
            {+8.0,+infinity},
            {-7.0,0.0},
            {0.0,+8.0},
            {-7.0,+8.0},
            {0.0,+8.0},
            {+8.0,+infinity},
            {-infinity,-7.0},
            {0.0,+8.0},
            {0.0,0.0},
            entire,
            {0x1000100010001p0,3.0e300},
            {0.0,+8.0},
        }};

        std::array<I, n> h_ref {{
            {0.0,+6.0},
            {-5.0,0.0},
            {-0x1p-1,-0x187p-256},
            {-0x10000000000001p-93,0x123456789abcdp0},
            {-0x123456791abcdp-17,0x8f596b3002c1bp+947},
            {-infinity,+16.0},
            {-infinity,+1.0},
            entire,
            {-7.0,+8.0},
            {0.0,+16.0},
            {-7.0,+infinity},
            {0.0,+infinity},
            {+8.0,+infinity},
            {-infinity,-7.0},
            {0.0,+8.0},
            {0.0,0.0},
            entire,
            {+0x1010101010101p+8,0x8f596b3002c1bp+947},
            entire,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        test_add<<<numBlocks, blockSize>>>(n, d_xs, d_ys);
        CUDA_CHECK(cudaMemcpy(h_xs.data(), d_xs, n_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<I, n>(h_xs, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("y = [%a, %a]\nr = [%a, %a]\n", h_ys[fail_id].lb, h_ys[fail_id].ub, h_ref[fail_id].lb, h_ref[fail_id].ub);
        }
    };

    "mpfi_add_d_add"_test = [&] {
        constexpr int n = 32;
        std::array<I, n> h_xs {{
            {-0x1fffffffffffffp-52,-0x1p-550},
            {-0x1fffffffffffffp-52,-0x1p-550},
            {-0xfb53d14aa9c2fp-47,-17.0},
            {-0xffp0,0x123456789abcdfp-52},
            {-0xffp0,0x123456789abcdfp-52},
            {-32.0,-0xfb53d14aa9c2fp-48},
            {-32.0,-17.0},
            {-infinity,-7.0},
            {-infinity,-7.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {-infinity,0.0},
            {-infinity,0.0},
            {-infinity,8.0},
            {-infinity,8.0},
            {-infinity,8.0},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,8.0},
            {0.0,8.0},
            {0.0,8.0},
            {0x123456789abcdfp-48,0x123456789abcdfp-4},
            {0x123456789abcdfp-48,0x123456789abcdfp-4},
            {0x123456789abcdfp-56,0x123456789abcdfp-4},
            {0x123456789abcdfp-56,0x123456789abcdfp-4},
            entire,
            entire,
            entire,
        }};

        std::array<I, n> h_ys {{
            {-4097.5,-4097.5},
            {4097.5,4097.5},
            {0xfb53d14aa9c2fp-47,0xfb53d14aa9c2fp-47},
            {-256.5,-256.5},
            {256.5,256.5},
            {0xfb53d14aa9c2fp-48,0xfb53d14aa9c2fp-48},
            {-0xfb53d14aa9c2fp-47,-0xfb53d14aa9c2fp-47},
            {-0x170ef54646d497p-107,-0x170ef54646d497p-107},
            {0.0,0.0},
            {0x170ef54646d497p-107,0x170ef54646d497p-107},
            {-0x170ef54646d497p-106,-0x170ef54646d497p-106},
            {0.0,0.0},
            {0x170ef54646d497p-106,0x170ef54646d497p-106},
            {-0x16345785d8a00000p0,-0x16345785d8a00000p0},
            {0.0,0.0},
            {0x16345785d8a00000p0,0x16345785d8a00000p0},
            {-0x50b45a75f7e81p-104,-0x50b45a75f7e81p-104},
            {0.0,0.0},
            {0x142d169d7dfa03p-106,0x142d169d7dfa03p-106},
            {-0x170ef54646d497p-109,-0x170ef54646d497p-109},
            {0.0,0.0},
            {0x170ef54646d497p-109,0x170ef54646d497p-109},
            {-0x114b37f4b51f71p-107,-0x114b37f4b51f71p-107},
            {0.0,0.0},
            {0x114b37f4b51f7p-103,0x114b37f4b51f7p-103},
            {-3.5,-3.5},
            {3.5,3.5},
            {-3.5,-3.5},
            {3.5,3.5},
            {+0x170ef54646d497p-105,+0x170ef54646d497p-105},
            {-0x170ef54646d497p-105,-0x170ef54646d497p-105},
            {0.0e-17,0.0e-17},
        }};

        std::array<I, n> h_ref {{
            {-0x10038p-4,-0x10018p-4},
            {0xfff8p-4,0x10018p-4},
            {0.0,0x7353d14aa9c2fp-47},
            {-0x1ff8p-4,-0xff5cba9876543p-44},
            {0x18p-4,0x101a3456789abdp-44},
            {-0x104ac2eb5563d1p-48,0.0},
            {-0x1fb53d14aa9c2fp-47,-0x18353d14aa9c2fp-47},
            {-infinity,-7.0},
            {-infinity,-7.0},
            {-infinity,-0x1bffffffffffffp-50},
            {-infinity,-8.0e-17},
            {-infinity,0.0},
            {-infinity,0x170ef54646d497p-106},
            {-infinity,-0x16345785d89fff00p0},
            {-infinity,8.0},
            {-infinity,0x16345785d8a00100p0},
            {-0x50b45a75f7e81p-104,+infinity},
            {0.0,+infinity},
            {0x142d169d7dfa03p-106,+infinity},
            {-0x170ef54646d497p-109,-0x170ef54646d497p-109},
            {0.0,0.0},
            {0x170ef54646d497p-109,0x170ef54646d497p-109},
            {-0x114b37f4b51f71p-107,8.0},
            {0.0,8.0},
            {0x114b37f4b51f7p-103,0x10000000000001p-49},
            {0xeb456789abcdfp-48,0x123456789abca7p-4},
            {0x15b456789abcdfp-48,0x123456789abd17p-4},
            {-0x36dcba98765434p-52,0x123456789abca7p-4},
            {0x3923456789abcdp-52,0x123456789abd17p-4},
            entire,
            entire,
            entire,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        test_add<<<numBlocks, blockSize>>>(n, d_xs, d_ys);
        CUDA_CHECK(cudaMemcpy(h_xs.data(), d_xs, n_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<I, n>(h_xs, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("y = [%a, %a]\nr = [%a, %a]\n", h_ys[fail_id].lb, h_ys[fail_id].ub, h_ref[fail_id].lb, h_ref[fail_id].ub);
        }
    };

    "mpfi_d_div_div"_test = [&] {
        constexpr int n = 30;
        std::array<I, n> h_xs {{
            {+0x170ef54646d497p-105,+0x170ef54646d497p-105},
            {-0x114b37f4b51f71p-107,-0x114b37f4b51f71p-107},
            {-0x16345785d8a00000p0,-0x16345785d8a00000p0},
            {-0x170ef54646d496p-107,-0x170ef54646d496p-107},
            {-0x170ef54646d497p-105,-0x170ef54646d497p-105},
            {-0x170ef54646d497p-106,-0x170ef54646d497p-106},
            {-0x170ef54646d497p-109,-0x170ef54646d497p-109},
            {-0x50b45a75f7e81p-104,-0x50b45a75f7e81p-104},
            {-2.5,-2.5},
            {-2.5,-2.5},
            {-2.5,-2.5},
            {-2.5,-2.5},
            {-2.5,-2.5},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0e-17,0.0e-17},
            {0x114b37f4b51f71p-107,0x114b37f4b51f71p-107},
            {0x142d169d7dfa03p-106,0x142d169d7dfa03p-106},
            {0x16345785d8a00000p0,0x16345785d8a00000p0},
            {0x170ef54646d496p-107,0x170ef54646d496p-107},
            {0x170ef54646d497p-106,0x170ef54646d497p-106},
            {0x170ef54646d497p-109,0x170ef54646d497p-109},
            {33.125,33.125},
            {33.125,33.125},
            {33.125,33.125},
            {33.125,33.125},
        }};

        std::array<I, n> h_ys {{
            entire,
            {0.0,7.0},
            {-infinity,8.0},
            {-infinity,-7.0},
            entire,
            {-infinity,0.0},
            {0.0,0.0},
            {0.0,+infinity},
            {-16.0,-7.0},
            {-8.0,-5.0},
            {-8.0,8.0},
            {11.0,143.0},
            {25.0,40.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {-infinity,8.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,7.0},
            entire,
            {0.0,7.0},
            {0.0,+infinity},
            {-infinity,8.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {0.0,0.0},
            {-530.0,-496.875},
            {52.0,54.0},
            {54.0,265.0},
            {8.28125,530.0},
        }};

        std::array<I, n> h_ref {{
            entire,
            {-infinity,-0x13c3ada9f391a5p-110},
            entire,
            {0.0,0x1a5a3ce29a1787p-110},
            entire,
            {0.0,+infinity},
            empty,
            {-infinity,0.0},
            {0x5p-5,0x16db6db6db6db7p-54},
            {0x5p-4,0.5},
            entire,
            {-0x1d1745d1745d18p-55,-0x11e6efe35b4cfap-58},
            {-0x1999999999999ap-56,-0x1p-4},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            empty,
            {0.0,0.0},
            {0.0,0.0},
            {0x13c3ada9f391a5p-110,+infinity},
            {0.0,+infinity},
            entire,
            {-0x1a5a3ce29a1787p-110,0.0},
            {-infinity,0.0},
            empty,
            {-0x11111111111112p-56,-0x1p-4},
            {0x13a12f684bda12p-53,0x14627627627628p-53},
            {0.125,0x13a12f684bda13p-53},
            {0x1p-4,4.0},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        test_div<<<numBlocks, blockSize>>>(n, d_xs, d_ys);
        CUDA_CHECK(cudaMemcpy(h_xs.data(), d_xs, n_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<I, n>(h_xs, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("y = [%a, %a]\nr = [%a, %a]\n", h_ys[fail_id].lb, h_ys[fail_id].ub, h_ref[fail_id].lb, h_ref[fail_id].ub);
        }
    };

    "mpfi_div_div"_test = [&] {
        constexpr int n = 62;
        std::array<I, n> h_xs {{
            {-0x1.02f0415f9f596p+0,-0x1.489c07caba163p-4},
            {-0x1.02f0415f9f596p+0,-0x754ep-16},
            {-0x1.18333622af827p+0,0x2.14b836907297p+0},
            {-0x1.25f2d73472753p+0,+0x9.9a19fd3c1fc18p-4},
            {-0x1.25f2d73472753p+0,-0x9.9a19fd3c1fc18p-4},
            {-0x1.25f2d73472753p+0,0.0},
            {-0x1.4298b2138f2a7p-4,0.0},
            {-0x1.4298b2138f2a7p-4,0.0},
            {-0x100p0,-0xe.bb80d0a0824ep-4},
            {-0x10p0,0xd0e9dc4p+12},
            {-0x123456789p0,-0x1.b0a62934c76e9p+0},
            {-0x123456789p0,-0x754ep+4},
            {-0x12p0,0x10p0},
            {-0x1p0,0x754ep-16},
            {-0x754ep0,0x1p+10},
            {-0x754ep0,0xd0e9dc4p+12},
            {-0x75bcd15p0,-0x1.489c07caba163p-4},
            {-0x75bcd15p0,-0x754ep0},
            {-0x75bcd15p0,0.0},
            {-0x75bcd15p0,0.0},
            {-0x75bcd15p0,0xa680p0},
            {-0xacbp+256,-0x6f9p0},
            {-0xacbp+256,0x6f9p0},
            {-0xb.5b90b4d32136p-4,0x6.e694ac6767394p+0},
            {-0xd.67775e4b8588p+0,-0x1.b0a62934c76e9p+0},
            {-0xd.67775e4b8588p-4,-0x754ep-53},
            {-0xeeeeeeeeep0,0.0},
            {-0xeeeeeeeeep0,0.0},
            {-100.0,-15.0},
            {-2.0,-0x1.25f2d73472753p+0},
            {-infinity,+8.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {0.0,+15.0},
            {0.0,+8.0},
            {0.0,+8.0},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0x1.5f6b03dc8c66fp+0},
            {0.0,0x1.acbf1702af6edp+0},
            {0.0,0x75bcd15p0},
            {0.0,0x75bcd15p0},
            {0.0,0xap0},
            {0.0,0xap0},
            {0x1.7f03f2a978865p+0,0xeeeeep0},
            {0x1.a9016514490e6p-4,0xeeeep0},
            {0x1.d7c06f9ff0706p-8,0x1ba2dc763p0},
            {0x5.efc1492p-4,0x1.008a3accc766dp+0},
            {0x5efc1492p0,0x1ba2dc763p0},
            {0x754ep-16,0x1.008a3accc766dp+4},
            {0x754ep0,+0xeeeeep0},
            {0x754ep0,0x75bcd15p0},
            {0x754ep0,0xeeeep0},
            {0x8.440e7d65be6bp-8,0x3.99982e9eae09ep+0},
            {0x9.ac412ff1f1478p-4,0x75bcd15p0},
            {0xe.1552a314d629p-4,0x1.064c5adfd0042p+0},
            {5.0,6.0},
            entire,
        }};

        std::array<I, n> h_ys {{
            {-0x2.e8e36e560704ap+0,-0x7.62ce64fbacd2cp-8},
            {-0x11ep0,-0x7.62ce64fbacd2cp-8},
            {0x1.263147d1f4bcbp+0,0x111p0},
            {-0x9.3b0c8074ccc18p-4,+0x4.788df5d72af78p-4},
            {-0x9.3b0c8074ccc18p-4,+0x4.788df5d72af78p-4},
            {-0x9.3b0c8074ccc18p-4,+0x4.788df5d72af78p-4},
            {-0x1p-8,-0xf.5e4900c9c19fp-12},
            {0xf.5e4900c9c19fp-12,0x9p0},
            {-0x1.7c6d760a831fap+0,0.0},
            {0x11ep0,0xbbbp0},
            {0x40bp-17,0x2.761ec797697a4p-4},
            {0x40bp0,0x11ep+4},
            {-0xbbbbbbbbbbp0,-0x9p0},
            {-0xccccccccccp0,-0x11ep0},
            {0x11ep0,0xbbbp0},
            {0x11ep0,0xbbbp0},
            {-0x2.e8e36e560704ap+4,-0x9p0},
            {-0x11ep0,-0x9p0},
            {-0x90p0,-0x9p0},
            {0x9p0,0x90p0},
            {-0xaf6p0,-0x9p0},
            {-0x7p0,0.0},
            {-0x7p0,0.0},
            {-0xdddddddddddp0,-0xc.f459be9e80108p-4},
            {0x4.887091874ffc8p-4,0x2.761ec797697a4p+4},
            {0x4.887091874ffc8p+0,0x11ep+201},
            {-0xaaaaaaaaap0,0.0},
            {0.0,+0x3p0},
            {0.0,+3.0},
            {0.0,+0x9.3b0c8074ccc18p-4},
            {0.0,+8.0},
            {-1.0,+8.0},
            {+8.0,+infinity},
            {-3.0,+3.0},
            {-7.0,+8.0},
            {-7.0,0.0},
            {0.0,+8.0},
            {0.0,+8.0},
            {+8.0,+infinity},
            {-infinity,-7.0},
            {0.0,+8.0},
            entire,
            {+0x2.39ad24e812dcep+0,0xap0},
            {-0x0.fp0,-0xe.3d7a59e2bdacp-4},
            {+0x9p0,+0xap0},
            {-0xap0,-0x9p0},
            {-0x9p0,0.0},
            {-1.0,+1.0},
            {0.0,0x1.48b08624606b9p+0},
            {-0xe.316e87be0b24p-4,0.0},
            {0x2fdd1fp-20,0xe.3d7a59e2bdacp+0},
            {0x2.497403b31d32ap+0,0x8.89b71p+0},
            {0x2fdd1fp0,0x889b71p0},
            {-0x11ep0,-0x2.497403b31d32ap+0},
            {0.0,+0x11ep0},
            {-0x11ep0,-0x9p0},
            {-0x11ep0,0.0},
            {0x8.29fa8d0659e48p-4,0xc.13d2fd762e4a8p-4},
            {-0x1.5232c83a0e726p+4,-0x9p0},
            {-0x5.0d4d319a50b04p-4,-0x2.d8f51df1e322ep-4},
            {-0x5.0d4d319a50b04p-4,0x2.d8f51df1e322ep-4},
            {0.0,+8.0},
        }};

        std::array<I, n> h_ref {{
            {0x7.0ef61537b1704p-8,0x2.30ee5eef9c36cp+4},
            {0x69p-16,0x2.30ee5eef9c36cp+4},
            {-0xf.3d2f5db8ec728p-4,0x1.cf8fa732de129p+0},
            entire,
            entire,
            entire,
            {0.0,0x1.4fdb41a33d6cep+4},
            {-0x1.4fdb41a33d6cep+4,0.0},
            {0x9.e9f24790445fp-4,+infinity},
            {-0xe.525982af70c9p-8,0xbaffep+12},
            {-0x480b3bp+17,-0xa.fc5e7338f3e4p+0},
            {-0x480b3bp0,-0x69p0},
            {-0x1.c71c71c71c71dp0,2.0},
            {-0x69p-16,0xe.525982af70c9p-12},
            {-0x69p0,0xe.525982af70c9p-2},
            {-0x69p0,0xbaffep+12},
            {0x7.0ef61537b1704p-12,0xd14fadp0},
            {0x69p0,0xd14fadp0},
            {0.0,0xd14fadp0},
            {-0xd14fadp0,0.0},
            {-0x1280p0,0xd14fadp0},
            {0xffp0,+infinity},
            entire,
            {-0x8.85e40b3c3f63p+0,0xe.071cbfa1de788p-4},
            {-0x2.f5008d2df94ccp+4,-0xa.fc5e7338f3e4p-8},
            {-0x2.f5008d2df94ccp-4,-0x69p-254},
            {0.0,+infinity},
            {-infinity,0.0},
            {-infinity,-5.0},
            {-infinity,-0x1.fd8457415f917p+0},
            entire,
            entire,
            {-infinity,0.0},
            entire,
            entire,
            {-infinity,0.0},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0x9.deb65b02baep-4},
            {-0x1.e1bb896bfda07p+0,0.0},
            {0.0,0xd14fadp0},
            {-0xd14fadp0,0.0},
            {-infinity,0.0},
            entire,
            {0x1.2a4fcda56843p+0,+infinity},
            {-infinity,-0x1.df1cc82e6a583p-4},
            {0x2.120d75be74b54p-12,0x93dp+20},
            {0xb.2p-8,0x7.02d3edfbc8b6p-4},
            {0xb2p0,0x93dp0},
            {-0x7.02d3edfbc8b6p+0,-0x69p-16},
            {0x69p0,+infinity},
            {-0xd14fadp0,-0x69p0},
            {-infinity,-0x69p0},
            {0xa.f3518768b206p-8,0x7.0e2acad54859cp+0},
            {-0xd14fadp0,-0x7.52680a49e5d68p-8},
            {-0x5.c1d97d57d81ccp+0,-0x2.c9a600c455f5ap+0},
            entire,
            entire,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        test_div<<<numBlocks, blockSize>>>(n, d_xs, d_ys);
        CUDA_CHECK(cudaMemcpy(h_xs.data(), d_xs, n_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<I, n>(h_xs, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("y = [%a, %a]\nr = [%a, %a]\n", h_ys[fail_id].lb, h_ys[fail_id].ub, h_ref[fail_id].lb, h_ref[fail_id].ub);
        }
    };

    "mpfi_div_d_div"_test = [&] {
        constexpr int n = 25;
        std::array<I, n> h_xs {{
            {-0x10000000000001p-20,-0x10000000000001p-53},
            {-0x10000000000001p-20,-0x10000020000001p-53},
            {-0x10000000000002p-20,-0x10000000000001p-53},
            {-0x10000000000002p-20,-0x10000020000001p-53},
            {-0x123456789abcdfp-53,0x10000000000001p-53},
            {-0x123456789abcdfp-53,0x123456789abcdfp-7},
            {-1.0,0x10000000000001p-53},
            {-1.0,0x123456789abcdfp-7},
            {-infinity,-7.0},
            {-infinity,-7.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {-infinity,0.0},
            {-infinity,8.0},
            {-infinity,8.0},
            {-infinity,8.0},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,8.0},
            {0.0,8.0},
            entire,
            entire,
            entire,
        }};

        std::array<I, n> h_ys {{
            {-1.0,-1.0},
            {0x10000000000001p-53,0x10000000000001p-53},
            {0x10000000000001p-53,0x10000000000001p-53},
            {0x10000000000001p-53,0x10000000000001p-53},
            {-0x123456789abcdfp0,-0x123456789abcdfp0},
            {-0x123456789abcdfp0,-0x123456789abcdfp0},
            {-0x123456789abcdfp0,-0x123456789abcdfp0},
            {-0x123456789abcdfp0,-0x123456789abcdfp0},
            {-7.0,-7.0},
            {0.0,0.0},
            {7.0,7.0},
            {-0x170ef54646d497p-106,-0x170ef54646d497p-106},
            {0x170ef54646d497p-106,0x170ef54646d497p-106},
            {-3.0,-3.0},
            {0.0,0.0},
            {3.0,3.0},
            {-0x50b45a75f7e81p-104,-0x50b45a75f7e81p-104},
            {0x142d169d7dfa03p-106,0x142d169d7dfa03p-106},
            {-0x170ef54646d497p-109,-0x170ef54646d497p-109},
            {0x170ef54646d497p-109,0x170ef54646d497p-109},
            {-0x114b37f4b51f71p-107,-0x114b37f4b51f71p-107},
            {0x114b37f4b51f71p-107,0x114b37f4b51f71p-107},
            {+0x170ef54646d497p-105,+0x170ef54646d497p-105},
            {-0x170ef54646d497p-105,-0x170ef54646d497p-105},
            {0.0e-17,0.0e-17},
        }};

        std::array<I, n> h_ref {{
            {0x10000000000001p-53,0x10000000000001p-20},
            {-0x1p+33,-0x1000001fffffffp-52},
            {-0x10000000000001p-19,-1.0},
            {-0x10000000000001p-19,-0x1000001fffffffp-52},
            {-0x1c200000000002p-106,0x1p-53},
            {-0x1p-7,0x1p-53},
            {-0x1c200000000002p-106,0x1c200000000001p-105},
            {-0x1p-7,0x1c200000000001p-105},
            {1.0,+infinity},
            empty,
            {-infinity,-1.0},
            {0.0,+infinity},
            {-infinity,0.0},
            {-0x15555555555556p-51,+infinity},
            empty,
            {-infinity,0x15555555555556p-51},
            {-infinity,0.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.0},
            {-0x1d9b1f5d20d556p+5,0.0},
            {0.0,0x1d9b1f5d20d556p+5},
            entire,
            entire,
            empty,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        test_div<<<numBlocks, blockSize>>>(n, d_xs, d_ys);
        CUDA_CHECK(cudaMemcpy(h_xs.data(), d_xs, n_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<I, n>(h_xs, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("y = [%a, %a]\nr = [%a, %a]\n", h_ys[fail_id].lb, h_ys[fail_id].ub, h_ref[fail_id].lb, h_ref[fail_id].ub);
        }
    };

    "mpfi_d_sub_sub"_test = [&] {
        constexpr int n = 32;
        std::array<I, n> h_xs {{
            {-0x114b37f4b51f71p-107,-0x114b37f4b51f71p-107},
            {-0x142d169d7dfa03p-106,-0x142d169d7dfa03p-106},
            {-0x16345785d8a00000p0,-0x16345785d8a00000p0},
            {-0x170ef54646d497p-105,-0x170ef54646d497p-105},
            {-0x170ef54646d497p-107,-0x170ef54646d497p-107},
            {-0x170ef54646d497p-109,-0x170ef54646d497p-109},
            {-0x170ef54646d497p-96,-0x170ef54646d497p-96},
            {-0x50b45a75f7e81p-104,-0x50b45a75f7e81p-104},
            {-0xfb53d14aa9c2fp-47,-0xfb53d14aa9c2fp-47},
            {-256.5,-256.5},
            {-3.5,-3.5},
            {-3.5,-3.5},
            {-4097.5,-4097.5},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0e-17,0.0e-17},
            {0x114b37f4b51f71p-107,0x114b37f4b51f71p-107},
            {0x16345785d8a00000p0,0x16345785d8a00000p0},
            {0x170ef54646d497p-105,0x170ef54646d497p-105},
            {0x170ef54646d497p-107,0x170ef54646d497p-107},
            {0x170ef54646d497p-109,0x170ef54646d497p-109},
            {0x170ef54646d497p-96,0x170ef54646d497p-96},
            {0xfb53d14aa9c2fp-47,0xfb53d14aa9c2fp-47},
            {0xfb53d14aa9c2fp-48,0xfb53d14aa9c2fp-48},
            {256.5,256.5},
            {3.5,3.5},
            {3.5,3.5},
            {4097.5,4097.5},
        }};

        std::array<I, n> h_ys {{
            {0.0,8.0},
            {0.0,+infinity},
            {-infinity,8.0},
            entire,
            {-infinity,-7.0},
            {0.0,0.0},
            {-infinity,0.0},
            {0.0,+infinity},
            {17.0,32.0},
            {-0x123456789abcdfp-52,0xffp0},
            {-0x123456789abcdfp-4,-0x123456789abcdfp-48},
            {-0x123456789abcdfp-4,-0x123456789abcdfp-56},
            {0x1p-550,0x1fffffffffffffp-52},
            {-infinity,-7.0},
            {-infinity,0.0},
            {-infinity,8.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,8.0},
            entire,
            {0.0,8.0},
            {-infinity,8.0},
            entire,
            {-infinity,-7.0},
            {0.0,0.0},
            {-infinity,0.0},
            {17.0,0xfb53d14aa9c2fp-47},
            {0xfb53d14aa9c2fp-48,32.0},
            {-0x123456789abcdfp-52,0xffp0},
            {-0x123456789abcdfp-4,-0x123456789abcdfp-48},
            {-0x123456789abcdfp-4,-0x123456789abcdfp-56},
            {0x1p-550,0x1fffffffffffffp-52},
        }};

        std::array<I, n> h_ref {{
            {-0x10000000000001p-49,-0x114b37f4b51f71p-107},
            {-infinity,-0x142d169d7dfa03p-106},
            {-0x16345785d8a00100p0,+infinity},
            entire,
            {0x1bffffffffffffp-50,+infinity},
            {-0x170ef54646d497p-109,-0x170ef54646d497p-109},
            {-0x170ef54646d497p-96,+infinity},
            {-infinity,-0x50b45a75f7e81p-104},
            {-0x1fb53d14aa9c2fp-47,-0x18353d14aa9c2fp-47},
            {-0x1ff8p-4,-0xff5cba9876543p-44},
            {0xeb456789abcdfp-48,0x123456789abca7p-4},
            {-0x36dcba98765434p-52,0x123456789abca7p-4},
            {-0x10038p-4,-0x10018p-4},
            {7.0,+infinity},
            {0.0,+infinity},
            {-8.0,+infinity},
            {-infinity,0.0},
            {0.0,0.0},
            {-8.0,0.0},
            entire,
            {-8.0,0x114b37f4b51f71p-107},
            {0x16345785d89fff00p0,+infinity},
            entire,
            {7.0,+infinity},
            {0x170ef54646d497p-109,0x170ef54646d497p-109},
            {0x170ef54646d497p-96,+infinity},
            {0.0,0x7353d14aa9c2fp-47},
            {-0x104ac2eb5563d1p-48,0.0},
            {0x18p-4,0x101a3456789abdp-44},
            {0x15b456789abcdfp-48,0x123456789abd17p-4},
            {0x3923456789abcdp-52,0x123456789abd17p-4},
            {0xfff8p-4,0x10018p-4},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        test_sub<<<numBlocks, blockSize>>>(n, d_xs, d_ys);
        CUDA_CHECK(cudaMemcpy(h_xs.data(), d_xs, n_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<I, n>(h_xs, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("y = [%a, %a]\nr = [%a, %a]\n", h_ys[fail_id].lb, h_ys[fail_id].ub, h_ref[fail_id].lb, h_ref[fail_id].ub);
        }
    };

    "mpfi_mul_mul"_test = [&] {
        constexpr int n = 50;
        std::array<I, n> h_xs {{
            {-0x01p0,0x1.90aa487ecf153p+0},
            {-0x01p0,0x10p0},
            {-0x01p0,0x11p0},
            {-0x01p0,0x2.db091cea593fap-4},
            {-0x01p0,0x6.e211fefc216ap-4},
            {-0x01p0,0xe.ca7ddfdb8572p-4},
            {-0x04p0,-0xa.497d533c3b2ep-8},
            {-0x07p0,0x07p0},
            {-0x0dp0,-0x09p0},
            {-0x0dp0,-0xd.f0e7927d247cp-4},
            {-0x1.15e079e49a0ddp+0,0x1p-8},
            {-0x1.1d069e75e8741p+8,0x01p0},
            {-0x1.c40db77f2f6fcp+0,0x1.8eb70bbd94478p+0},
            {-0x1.cb540b71699a8p+4,-0x0.33p0},
            {-0x1.cb540b71699a8p+4,-0x0.33p0},
            {-0x123456789ap0,-0x01p0},
            {-0x37p0,-0x07p0},
            {-0xa.8071f870126cp-4,0x10p0},
            {-0xb.6c67d3a37d54p-4,-0x0.8p0},
            {-0xb.6c67d3a37d54p-4,-0xa.497d533c3b2ep-8},
            {-0xe.063f267ed51ap-4,-0x0.33p0},
            {-0xe.26c9e9eb67b48p-4,-0x8.237d2eb8b1178p-4},
            {-0xe.ca7ddfdb8572p-4,0x1.1d069e75e8741p+8},
            {-3.0,+7.0},
            {-infinity,+8.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {0.0,+8.0},
            {0.0,+8.0},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0x0.3p0,0xa.a97267f56a9b8p-4},
            {0x01p0,0xcp0},
            {0x03p0,0x7.2bea531ef4098p+0},
            {0x123p-52,0x1.ec24910ac6aecp+0},
            {0x2.48380232f6c16p+0,0x7p0},
            {0x3.10e8a605572p-4,0x2.48380232f6c16p+0},
            {0x3p0,0x3.71cb6c53e68eep+0},
            {0x3p0,0x7p0},
            {0xb.38f1fb0ef4308p+0,0x2dp0},
            {0xcp0,0x1.1833fdcab4c4ap+10},
            {0xcp0,0x2dp0},
            {0xf.08367984ca1cp-4,0xa.bcf6c6cbe341p+0},
            entire,
            entire,
        }};

        std::array<I, n> h_ys {{
            {0x01p-53,0x1.442e2695ac81ap+0},
            {-0x02p0,0x03p0},
            {-0x07p0,-0x04p0},
            {-0x2.6bff2625fb71cp-4,0x1p-8},
            {-0x1p-4,0x1.8e3fe93a4ea52p+0},
            {-0x2.3b46226145234p+0,-0x0.1p0},
            {0xb.d248df3373e68p-4,0x04p0},
            {0x13p0,0x24p0},
            {-0x04p0,-0x02p0},
            {-0x04p0,-0xa.41084aff48f8p-8},
            {-0x2.77fc84629a602p+0,0x8.3885932f13fp-4},
            {-0x2.3b46226145234p+0,-0x0.1p0},
            {0x02p0,0x3.45118635235c6p+0},
            {-0x1.64dcaaa101f18p+0,0x01p0},
            {-0x1.64dcaaa101f18p+0,0x1.eb67a1a6ef725p+4},
            {0x01p0,0x10p0},
            {-0x01p0,0x22p0},
            {0x02p0,0x2.3381083e7d3b4p+0},
            {0x02p0,0x2.0bee4e8bb3dfp+0},
            {0xb.d248df3373e68p-4,0x2.0bee4e8bb3dfp+0},
            {-0x01p0,0x1.777ab178b4a1ep+0},
            {-0x5.8c899a0706d5p-4,-0x3.344e57a37b5e8p-4},
            {-0x2.3b46226145234p+0,-0x0.1p0},
            {0.0,0.0},
            {0.0,+8.0},
            {-1.0,+8.0},
            {+8.0,+infinity},
            {-7.0,+8.0},
            {-7.0,0.0},
            {0.0,+8.0},
            {0.0,+8.0},
            {+8.0,+infinity},
            {-infinity,-7.0},
            {0.0,+8.0},
            {0.0,0.0},
            entire,
            {-0x1.ec24910ac6aecp+0,0x7.2bea531ef4098p+0},
            {-0xe5p0,0x01p0},
            {-0x01p0,0xa.a97267f56a9b8p-4},
            {-0xa.a97267f56a9b8p-4,0x1p+32},
            {0x3.71cb6c53e68eep+0,0xbp0},
            {0xc.3d8e305214ecp-4,0x2.9e7db05203c88p+0},
            {0x5p-25,0x2.48380232f6c16p+0},
            {0x5p0,0xbp0},
            {-0x679p0,-0xa.4771d7d0c604p+0},
            {-0x2.4c0afc50522ccp+40,-0xe5p0},
            {-0x679p0,-0xe5p0},
            {-0x5.cbc445e9952c4p+0,-0x2.8ad05a7b988fep-8},
            {0.0,+8.0},
            {0.0,0.0},
        }};

        std::array<I, n> h_ref {{
            {-0x1.442e2695ac81ap+0,0x1.fb5fbebd0cbc6p+0},
            {-0x20p0,0x30p0},
            {-0x77p0,0x07p0},
            {-0x6.ea77a3ee43de8p-8,0x2.6bff2625fb71cp-4},
            {-0x1.8e3fe93a4ea52p+0,0xa.b52fe22d72788p-4},
            {-0x2.101b41d3d48b8p+0,0x2.3b46226145234p+0},
            {-0x10p0,-0x7.99b990532d434p-8},
            {-0xfcp0,0xfcp0},
            {0x12p0,0x34p0},
            {0x8.ef3aa21dba748p-8,0x34p0},
            {-0x8.ec5de73125be8p-4,0x2.adfe651d3b19ap+0},
            {-0x2.3b46226145234p+0,0x2.7c0bd9877f404p+8},
            {-0x5.c61fcad908df4p+0,0x5.17b7c49130824p+0},
            {-0x1.cb540b71699a8p+4,0x2.804cce4a3f42ep+4},
            {-0x3.71b422ce817f4p+8,0x2.804cce4a3f42ep+4},
            {-0x123456789a0p0,-0x01p0},
            {-0x74ep0,0x37p0},
            {-0x1.71dc5b5607781p+0,0x2.3381083e7d3b4p+4},
            {-0x1.7611a672948a5p+0,-0x01p0},
            {-0x1.7611a672948a5p+0,-0x7.99b990532d434p-8},
            {-0x1.491df346a9f15p+0,0xe.063f267ed51ap-4},
            {0x1.a142a930de328p-4,0x4.e86c3434cd924p-4},
            {-0x2.7c0bd9877f404p+8,0x2.101b41d3d48b8p+0},
            {0.0,0.0},
            {-infinity,+64.0},
            entire,
            {-infinity,0.0},
            {-56.0,+64.0},
            {-56.0,0.0},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {-0x1.47f2dbe4ef916p+0,0x4.c765967f9468p+0},
            {-0xabcp0,0xcp0},
            {-0x7.2bea531ef4098p+0,0x4.c765967f9468p+0},
            {-0x1.47f2dbe4ef916p+0,0x1.ec24910ac6aecp+32},
            {0x7.dc58fb323ad78p+0,0x4dp0},
            {0x2.587a32d02bc04p-4,0x5.fa216b7c20c6cp+0},
            {0xfp-25,0x7.dc58fb323ad7cp+0},
            {0xfp0,0x4dp0},
            {-0x12345p0,-0x7.35b3c8400ade4p+4},
            {-0x2.83a3712099234p+50,-0xabcp0},
            {-0x12345p0,-0xabcp0},
            {-0x3.e3ce52d4a139cp+4,-0x2.637164cf2f346p-8},
            entire,
            {0.0,0.0},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        test_mul<<<numBlocks, blockSize>>>(n, d_xs, d_ys);
        CUDA_CHECK(cudaMemcpy(h_xs.data(), d_xs, n_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<I, n>(h_xs, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("y = [%a, %a]\nr = [%a, %a]\n", h_ys[fail_id].lb, h_ys[fail_id].ub, h_ref[fail_id].lb, h_ref[fail_id].ub);
        }
    };

    "mpfi_mul_d_mul"_test = [&] {
        constexpr int n = 45;
        std::array<I, n> h_xs {{
            {-0x10000000000001p0,-0x1aaaaaaaaaaaaap-123},
            {-0x10000000000001p0,-0xaaaaaaaaaaaaap-123},
            {-0x10000000000001p0,0x10000000000001p0},
            {-0x10000000000001p0,0x1717170p+401},
            {-0x11717171717171p0,-0x10000000000001p0},
            {-0x11717171717171p0,-0xaaaaaaaaaaaaap-123},
            {-0x1717170p+36,-0x10000000000001p0},
            {-0x1717170p0,-0x1aaaaaaaaaaaaap-123},
            {-0x1717170p0,-0xaaaaaaaaaaaaap-123},
            {-0x1717170p0,-0xaaaaaaaaaaaaap-123},
            {-0xaaaaaaaaaaaaap0,0x10000000000001p0},
            {-0xaaaaaaaaaaaaap0,0x11717171717171p0},
            {-0xaaaaaaaaaaaaap0,0x1717170p+401},
            {-0xaaaaaaaaaaaaap0,0x1717170p+401},
            {-0xaaaaaaaaaaaabp0,0x11717171717171p0},
            {-0xaaaaaaaaaaaabp0,0x1717170p+401},
            {-infinity,-7.0},
            {-infinity,-7.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {-infinity,0.0},
            {-infinity,0.0},
            {-infinity,8.0},
            {-infinity,8.0},
            {-infinity,8.0},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,7.0},
            {0.0,8.0},
            {0.0,9.0},
            {0x10000000000001p0,0x11111111111111p0},
            {0x10000000000001p0,0x18888888888889p0},
            {0x10000000000001p0,0x888888888888p+654},
            {0x10000000000001p0,0x888888888888p+654},
            {0x10000000000010p0,0x11111111111111p0},
            {0x10000000000010p0,0x18888888888889p0},
            {0x10000000000010p0,0x888888888888p+654},
            {0x10000000000010p0,0x888888888888p+654},
            entire,
            entire,
            entire,
        }};

        std::array<I, n> h_ys {{
            {1.5,1.5},
            {1.5,1.5},
            {-1.5,-1.5},
            {-1.5,-1.5},
            {-1.5,-1.5},
            {-1.5,-1.5},
            {-1.5,-1.5},
            {1.5,1.5},
            {-1.5,-1.5},
            {1.5,1.5},
            {-1.5,-1.5},
            {1.5,1.5},
            {-1.5,-1.5},
            {1.5,1.5},
            {1.5,1.5},
            {1.5,1.5},
            {-0x17p0,-0x17p0},
            {0.0,0.0},
            {0x170ef54646d497p-107,0x170ef54646d497p-107},
            {-0x170ef54646d497p-106,-0x170ef54646d497p-106},
            {0.0,0.0},
            {0x170ef54646d497p-106,0x170ef54646d497p-106},
            {-0x16345785d8a00000p0,-0x16345785d8a00000p0},
            {0.0,0.0},
            {0x16345785d8a00000p0,0x16345785d8a00000p0},
            {-0x50b45a75f7e81p-104,-0x50b45a75f7e81p-104},
            {0.0,0.0},
            {0x142d169d7dfa03p-106,0x142d169d7dfa03p-106},
            {-0x170ef54646d497p-109,-0x170ef54646d497p-109},
            {0.0,0.0},
            {0x170ef54646d497p-109,0x170ef54646d497p-109},
            {-0x114b37f4b51f71p-107,-0x114b37f4b51f71p-107},
            {0.0,0.0},
            {0x114b37f4b51f71p-103,0x114b37f4b51f71p-103},
            {-2.125,-2.125},
            {2.125,2.125},
            {-2.125,-2.125},
            {2.125,2.125},
            {-2.125,-2.125},
            {2.125,2.125},
            {-2.125,-2.125},
            {2.125,2.125},
            {+0x170ef54646d497p-105,+0x170ef54646d497p-105},
            {-0x170ef54646d497p-105,-0x170ef54646d497p-105},
            {0.0e-17,0.0e-17},
        }};

        std::array<I, n> h_ref {{
            {-0x18000000000002p0,-0x27fffffffffffep-123},
            {-0x18000000000002p0,-0xfffffffffffffp-123},
            {-0x18000000000002p0,0x18000000000002p0},
            {-0x22a2a28p+401,0x18000000000002p0},
            {0x18000000000001p0,0x1a2a2a2a2a2a2ap0},
            {0xfffffffffffffp-123,0x1a2a2a2a2a2a2ap0},
            {0x18000000000001p0,0x22a2a28p+36},
            {-0x22a2a28p0,-0x27fffffffffffep-123},
            {0xfffffffffffffp-123,0x22a2a28p0},
            {-0x22a2a28p0,-0xfffffffffffffp-123},
            {-0x18000000000002p0,0xfffffffffffffp0},
            {-0xfffffffffffffp0,0x1a2a2a2a2a2a2ap0},
            {-0x22a2a28p+401,0xfffffffffffffp0},
            {-0xfffffffffffffp0,0x22a2a28p+401},
            {-0x10000000000001p0,0x1a2a2a2a2a2a2ap0},
            {-0x10000000000001p0,0x22a2a28p+401},
            {+0xa1p0,+infinity},
            {0.0,0.0},
            {-infinity,-0xa168b4ebefd020p-107},
            {0.0,+infinity},
            {0.0,0.0},
            {-infinity,0.0},
            {-0xb1a2bc2ec5000000p0,+infinity},
            {0.0,0.0},
            {-infinity,0xb1a2bc2ec5000000p0},
            {-infinity,0.0},
            {0.0,0.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {-0x790e87b0f3dc18p-107,0.0},
            {0.0,0.0},
            {0.0,0x9ba4f79a5e1b00p-103},
            {-0x12222222222223p+1,-0x22000000000002p0},
            {0x22000000000002p0,0x34222222222224p0},
            {-0x1222222222221p+654,-0x22000000000002p0},
            {0x22000000000002p0,0x1222222222221p+654},
            {-0x12222222222223p+1,-0x22000000000022p0},
            {0x22000000000022p0,0x34222222222224p0},
            {-0x1222222222221p+654,-0x22000000000022p0},
            {0x22000000000022p0,0x1222222222221p+654},
            entire,
            entire,
            {0.0,0.0},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        test_mul<<<numBlocks, blockSize>>>(n, d_xs, d_ys);
        CUDA_CHECK(cudaMemcpy(h_xs.data(), d_xs, n_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<I, n>(h_xs, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("y = [%a, %a]\nr = [%a, %a]\n", h_ys[fail_id].lb, h_ys[fail_id].ub, h_ref[fail_id].lb, h_ref[fail_id].ub);
        }
    };

    "mpfi_neg_neg"_test = [&] {
        constexpr int n = 8;
        std::array<I, n> h_xs {{
            {-infinity,+8.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {0.0,+8.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0x123456789p-16,0x123456799p-16},
            entire,
        }};

        std::array<I, n> h_ref {{
            {-8.0,+infinity},
            {+7.0,+infinity},
            {0.0,+infinity},
            {-8.0,0.0},
            {-infinity,0.0},
            {0.0,0.0},
            {-0x123456799p-16,-0x123456789p-16},
            entire,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        test_neg<<<numBlocks, blockSize>>>(n, d_xs);
        CUDA_CHECK(cudaMemcpy(h_xs.data(), d_xs, n_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<I, n>(h_xs, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("r = [%a, %a]\n", h_ref[fail_id].lb, h_ref[fail_id].ub);
        }
    };

    "mpfi_sqr_sqr"_test = [&] {
        constexpr int n = 11;
        std::array<I, n> h_xs {{
            {-0x1.64722ad2480c9p+0,0x1p0},
            {-infinity,+8.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {0.0,+8.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0x1.6b079248747a2p+0,0x2.b041176d263f6p+0},
            {0x6.61485c33c0b14p+4,0x123456p0},
            {0x8.6374d8p-4,0x3.f1d929p+8},
            entire,
        }};

        std::array<I, n> h_ref {{
            {0.0,0x1.f04dba0302d4dp+0},
            {0.0,+infinity},
            {+49.0,+infinity},
            {0.0,+infinity},
            {0.0,+64.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0x2.02ce7912cddf6p+0,0x7.3a5dee779527p+0},
            {0x2.8b45c3cc03ea6p+12,0x14b66cb0ce4p0},
            {0x4.65df11464764p-4,0xf.8f918d688891p+16},
            {0.0,+infinity},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        test_sqr<<<numBlocks, blockSize>>>(n, d_xs);
        CUDA_CHECK(cudaMemcpy(h_xs.data(), d_xs, n_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<I, n>(h_xs, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("r = [%a, %a]\n", h_ref[fail_id].lb, h_ref[fail_id].ub);
        }
    };

    "mpfi_sqr_sqrt"_test = [&] {
        constexpr int n = 7;
        std::array<I, n> h_xs {{
            {0.0,+9.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0xa.aa1p-4,0x1.0c348f804c7a9p+0},
            {0xaaa1p0,0x14b66cb0ce4p0},
            {0xe.49ae7969e41bp-4,0x1.0c348f804c7a9p+0},
            {0xe.49ae7969e41bp-4,0xaaa1p0},
        }};

        std::array<I, n> h_ref {{
            {0.0,+3.0},
            {0.0,+infinity},
            {0.0,0.0},
            {0xd.1p-4,0x1.06081714eef1dp+0},
            {0xd1p0,0x123456p0},
            {0xf.1ea42821b27a8p-4,0x1.06081714eef1dp+0},
            {0xf.1ea42821b27a8p-4,0xd1p0},
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        test_sqrt<<<numBlocks, blockSize>>>(n, d_xs);
        CUDA_CHECK(cudaMemcpy(h_xs.data(), d_xs, n_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<I, n>(h_xs, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("r = [%a, %a]\n", h_ref[fail_id].lb, h_ref[fail_id].ub);
        }
    };

    "mpfi_sub_sub"_test = [&] {
        constexpr int n = 19;
        std::array<I, n> h_xs {{
            {-0x1000100010001p+8,0x1p+60},
            {-0x1p-300,0x123456p+28},
            {-4.0,7.0},
            {-5.0,1.0},
            {-5.0,59.0},
            {-infinity,+8.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {0.0,+8.0},
            {0.0,+8.0},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {5.0,0x1p+70},
            entire,
        }};

        std::array<I, n> h_ys {{
            {-3e300,0x1000100010001p0},
            {-0x789abcdp0,0x10000000000000p-93},
            {-3e300,0x123456789abcdp-17},
            {1.0,0x1p+70},
            {17.0,81.0},
            {0.0,+8.0},
            {-1.0,+8.0},
            {+8.0,+infinity},
            {-7.0,+8.0},
            {-7.0,0.0},
            {0.0,+8.0},
            {0.0,+8.0},
            {+8.0,+infinity},
            {-infinity,-7.0},
            {0.0,+8.0},
            {0.0,0.0},
            entire,
            {3.0,5.0},
            {0.0,+8.0},
        }};

        std::array<I, n> h_ref {{
            {-0x10101010101011p+4,0x8f596b3002c1bp+947},
            {-0x10000000000001p-93,0x123456789abcdp0},
            {-0x123456791abcdp-17,0x8f596b3002c1bp+947},
            {-0x10000000000001p+18,0.0},
            {-86.0,42.0},
            {-infinity,+8.0},
            {-infinity,-6.0},
            {-infinity,-8.0},
            {-8.0,+15.0},
            {0.0,+15.0},
            {-8.0,+infinity},
            {-8.0,+infinity},
            {-infinity,-8.0},
            {+7.0,+infinity},
            {-8.0,0.0},
            {0.0,0.0},
            entire,
            {0.0,0x1p+70},
            entire,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        test_sub<<<numBlocks, blockSize>>>(n, d_xs, d_ys);
        CUDA_CHECK(cudaMemcpy(h_xs.data(), d_xs, n_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<I, n>(h_xs, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("y = [%a, %a]\nr = [%a, %a]\n", h_ys[fail_id].lb, h_ys[fail_id].ub, h_ref[fail_id].lb, h_ref[fail_id].ub);
        }
    };

    "mpfi_sub_d_sub"_test = [&] {
        constexpr int n = 32;
        std::array<I, n> h_xs {{
            {-0x1fffffffffffffp-52,-0x1p-550},
            {-0x1fffffffffffffp-52,-0x1p-550},
            {-0xfb53d14aa9c2fp-47,-17.0},
            {-0xffp0,0x123456789abcdfp-52},
            {-0xffp0,0x123456789abcdfp-52},
            {-32.0,-0xfb53d14aa9c2fp-48},
            {-32.0,-17.0},
            {-infinity,-7.0},
            {-infinity,-7.0},
            {-infinity,-7.0},
            {-infinity,0.0},
            {-infinity,0.0},
            {-infinity,0.0},
            {-infinity,8.0},
            {-infinity,8.0},
            {-infinity,8.0},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,+infinity},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,0.0},
            {0.0,8.0},
            {0.0,8.0},
            {0.0,8.0},
            {0x123456789abcdfp-48,0x123456789abcdfp-4},
            {0x123456789abcdfp-48,0x123456789abcdfp-4},
            {0x123456789abcdfp-56,0x123456789abcdfp-4},
            {0x123456789abcdfp-56,0x123456789abcdfp-4},
            entire,
            entire,
            entire,
        }};

        std::array<I, n> h_ys {{
            {-4097.5,-4097.5},
            {4097.5,4097.5},
            {-0xfb53d14aa9c2fp-47,-0xfb53d14aa9c2fp-47},
            {-256.5,-256.5},
            {256.5,256.5},
            {-0xfb53d14aa9c2fp-48,-0xfb53d14aa9c2fp-48},
            {0xfb53d14aa9c2fp-47,0xfb53d14aa9c2fp-47},
            {-0x170ef54646d497p-107,-0x170ef54646d497p-107},
            {0.0,0.0},
            {0x170ef54646d497p-107,0x170ef54646d497p-107},
            {-0x170ef54646d497p-106,-0x170ef54646d497p-106},
            {0.0,0.0},
            {0x170ef54646d497p-106,0x170ef54646d497p-106},
            {-0x16345785d8a00000p0,-0x16345785d8a00000p0},
            {0.0,0.0},
            {0x16345785d8a00000p0,0x16345785d8a00000p0},
            {-0x50b45a75f7e81p-104,-0x50b45a75f7e81p-104},
            {0.0,0.0},
            {0x142d169d7dfa03p-106,0x142d169d7dfa03p-106},
            {-0x170ef54646d497p-109,-0x170ef54646d497p-109},
            {0.0,0.0},
            {0x170ef54646d497p-109,0x170ef54646d497p-109},
            {-0x114b37f4b51f71p-107,-0x114b37f4b51f71p-107},
            {0.0,0.0},
            {0x114b37f4b51f71p-107,0x114b37f4b51f71p-107},
            {-3.5,-3.5},
            {3.5,3.5},
            {-3.5,-3.5},
            {3.5,3.5},
            {+0x170ef54646d497p-105,+0x170ef54646d497p-105},
            {-0x170ef54646d497p-105,-0x170ef54646d497p-105},
            {0.0e-17,0.0e-17},
        }};

        std::array<I, n> h_ref {{
            {0xfff8p-4,0x10018p-4},
            {-0x10038p-4,-0x10018p-4},
            {0.0,0x7353d14aa9c2fp-47},
            {0x18p-4,0x101a3456789abdp-44},
            {-0x1ff8p-4,-0xff5cba9876543p-44},
            {-0x104ac2eb5563d1p-48,0.0},
            {-0x1fb53d14aa9c2fp-47,-0x18353d14aa9c2fp-47},
            {-infinity,-0x1bffffffffffffp-50},
            {-infinity,-7.0},
            {-infinity,-7.0},
            {-infinity,0x170ef54646d497p-106},
            {-infinity,0.0},
            {-infinity,-8.0e-17},
            {-infinity,0x16345785d8a00100p0},
            {-infinity,8.0},
            {-infinity,-0x16345785d89fff00p0},
            {0x50b45a75f7e81p-104,+infinity},
            {0.0,+infinity},
            {-0x142d169d7dfa03p-106,+infinity},
            {+0x170ef54646d497p-109,+0x170ef54646d497p-109},
            {0.0,0.0},
            {-0x170ef54646d497p-109,-0x170ef54646d497p-109},
            {0x114b37f4b51f71p-107,0x10000000000001p-49},
            {0.0,8.0},
            {-0x114b37f4b51f71p-107,8.0},
            {0x15b456789abcdfp-48,0x123456789abd17p-4},
            {0xeb456789abcdfp-48,0x123456789abca7p-4},
            {0x3923456789abcdp-52,0x123456789abd17p-4},
            {-0x36dcba98765434p-52,0x123456789abca7p-4},
            entire,
            entire,
            entire,
        }};

        CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));
        test_sub<<<numBlocks, blockSize>>>(n, d_xs, d_ys);
        CUDA_CHECK(cudaMemcpy(h_xs.data(), d_xs, n_bytes, cudaMemcpyDeviceToHost));
        auto failed = check_all_equal<I, n>(h_xs, h_ref);
        for (auto fail_id : failed) {
            printf("failed at case %zu:\n", fail_id);
            printf("y = [%a, %a]\nr = [%a, %a]\n", h_ys[fail_id].lb, h_ys[fail_id].ub, h_ref[fail_id].lb, h_ref[fail_id].ub);
        }
    };


    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_ys));
    CUDA_CHECK(cudaFree(d_zs));
}
