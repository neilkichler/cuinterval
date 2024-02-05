import sys

def convert_to_test(file_path):
    try:
        with open(file_path, 'r') as file:
            tests = file.read().split('testcase')[1:] # ignore comment with 1:

            code = ''
            code_preamble = r'''
#include <cuinterval/cuinterval.h>

#include "test_ops.cuh"

#include <span>
#include <ostream>

#include <stdio.h>
#include <stdlib.h>

// compiler bug fix; TODO: remove when fixed
#ifdef __CUDACC__
#pragma push_macro("__cpp_consteval")
#define consteval constexpr
#include <boost/ut.hpp>
#undef consteval
#pragma pop_macro("__cpp_consteval")
#else
#include <boost/ut.hpp>
#endif

#define CUDA_CHECK(x)                                                                \
    do {                                                                             \
        cudaError_t err = x;                                                         \
        if (err != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error in %s at %s:%d: %s (%s=%d)\n", __FUNCTION__, \
                    __FILE__, __LINE__, cudaGetErrorString(err),                     \
                    cudaGetErrorName(err), err);                                     \
            abort();                                                                 \
        }                                                                            \
    } while (0)

template<typename T, int N>
void check_all_equal(std::span<T, N> h_xs, std::span<T, N> h_ref, const std::source_location location = std::source_location::current())
{
    using namespace boost::ut;

    for (size_t i = 0; i < h_xs.size(); ++i) {
        expect(eq(h_xs[i], h_ref[i]), location);
    }
}

template<typename T>
auto &operator<<(std::ostream &os, const interval<T> &x)
{
    return (os << '[' << std::hexfloat << x.lb << ',' << x.ub << ']');
}

template<typename T>
void tests() {
    using namespace boost::ut;

    using I = interval<T>;

    I empty         = ::empty<T>();
    I entire        = ::entire<T>();
    T infinity = std::numeric_limits<T>::infinity();
'''

            code_postamble ='''
    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_ys));
}

int main()
{
    tests<double>();
    return 0;
}
'''
            indent = ' ' * 4
            largest_n = 0

            for test in tests:
                # get the first word as the name of the test
                name, body = test.split(maxsplit=1)
                name = name.strip("test").strip('_')

                if name.endswith('dec'): # ignore decorated interval tests
                    continue

                body = body.replace('[', '{').replace(']', '}').replace(', ', ',')
                ops = body.splitlines()[1:-2]

                instr = ops[0].split()[0]

                instr_len  = len(ops[0].replace('=', ' ')[:-1].split())
                binary_op = instr_len == 4

                test_code = indent + f'"{instr}"_test = [&] {{\n'
                xs_code   = indent + '    std::array<I, n> h_xs {{\n'
                ys_code   = indent + '    std::array<I, n> h_ys {{\n' if binary_op else ''
                ref_code  = indent + '    std::array<I, n> h_ref {{\n'
                empty = '{empty}'
                entire = '{entire}'
                float_max = '0x1.FFFFFFFFFFFFFp1023'
                float_min = '-0x1.FFFFFFFFFFFFFp1023'

                # for op in sorted(ops, reverse=True):
                n = len(ops)
                for op in ops:
                    if op == '' or '//' in op:
                        n -= 1
                        continue
                    
                    el = op.replace('=', ' ')[:-1].split()
                    _, *el = el

                    # print(el)
                    for i, e in enumerate(el):
                        # check for double max
                        if e == empty or e == entire:
                            continue

                        vals = e[1:-1].split(',')
                        vals = ['std::numeric_limits<T>::max()' if v == float_max else 'std::numeric_limits<T>::lowest()' if v == float_min else v for v in vals]
                        el[i] = f'{{{vals[0]},{vals[1]}}}'

                    if not binary_op: # unary op
                        x, ref = el

                        if x == empty:
                            xs_code += indent + '        empty,\n'
                        elif x == entire:
                            xs_code += indent + '        entire,\n'
                        else:
                            xs_code += indent + f'        {x},\n'
                        if ref == empty:
                            ref_code += indent + '        empty,\n'
                        elif ref == entire:
                            ref_code += indent + '        entire,\n'
                        else:
                            ref_code += indent + f'        {ref},\n'

                    if binary_op: # binary op
                        x, y, ref = el

                        if x == empty:
                            xs_code += indent + '        empty,\n'

                        elif x == entire:
                            xs_code += indent + '        entire,\n'
                        else:
                            xs_code += indent + f'        {x},\n'
                        if ref == empty:
                            ref_code += indent + '        empty,\n'
                        elif ref == entire:
                            ref_code += indent + '        entire,\n'
                        else:
                            ref_code += indent + f'        {ref},\n'

                        if y == empty:
                            ys_code += indent + '        empty,\n'
                        elif y == entire:
                            ys_code += indent + '        entire,\n'
                        else:
                            ys_code += indent + f'        {y},\n'


                xs_code += indent + '    }};\n\n'
                ref_code += indent + '    }};\n\n'

                if binary_op:
                    ys_code += indent + '    }};\n\n'

                cuda_code =  indent + '    CUDA_CHECK(cudaMemcpy(d_xs, h_xs.data(), n_bytes, cudaMemcpyHostToDevice));\n'
                if binary_op:
                    cuda_code += indent + '    CUDA_CHECK(cudaMemcpy(d_ys, h_ys.data(), n_bytes, cudaMemcpyHostToDevice));\n'
                addon = ', d_ys' if binary_op else ''
                cuda_code += indent + f'    test_{instr}<<<numBlocks, blockSize>>>(n, d_xs{addon});\n'
                cuda_code += indent + '    CUDA_CHECK(cudaMemcpy(h_xs.data(), d_xs, n_bytes, cudaMemcpyDeviceToHost));\n'
                cuda_code += indent + '    check_all_equal<I, n>(h_xs, h_ref);\n'
                cuda_code += indent + '};\n\n'

                largest_n = max(n, largest_n)
                size_code = indent + f'    constexpr int n = {n};\n'
                test_code += size_code + xs_code + ys_code + ref_code + cuda_code 
                code += test_code

            code_constants = f'''
    const int n = {largest_n}; // count of largest test array
    const int n_bytes   = n * sizeof(I);
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;

    interval<T> *d_xs, *d_ys;
    CUDA_CHECK(cudaMalloc(&d_xs, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_ys, n_bytes));\n\n'''

            return code_preamble + code_constants + code + code_postamble

    except FileNotFoundError:
        return f"File '{file_path}' not found."


if __name__ == '__main__':

    # get file path from command line
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = 'libieeep1788_elem.itl'

    test_code = convert_to_test(file_path)

    # write to file
    with open('tests.cu', 'w') as f:
        f.write(test_code)

