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
std::vector<size_t> check_all_equal(std::span<T, N> h_xs, std::span<T, N> h_ref, const std::source_location location = std::source_location::current())
{
    using namespace boost::ut;

    std::vector<size_t> failed_ids;

    for (size_t i = 0; i < h_xs.size(); ++i) {
        if (h_xs[i] != h_ref[i])
            failed_ids.push_back(i);

        expect(eq(h_xs[i], h_ref[i]), location);
    }

    return failed_ids;
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
    CUDA_CHECK(cudaFree(d_zs));
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

                instr_len  = len(ops[0].replace('=', ' ')[:-1].split()) - 1
                vars = ['ref', 'xs', 'ys', 'zs'][:instr_len]
                n_vars = len(vars)
                n_ops = len(ops)
                var_codes = [''] * n_ops

                test_code = indent + f'"{instr}"_test = [&] {{\n'

                for i in range(n_vars):
                    ii = (i+1) % n_vars
                    var_codes[i] = indent + f'    std::array<I, n> h_{vars[ii]} {{{{\n'

                empty = '{empty}'
                entire = '{entire}'
                float_max = '0x1.FFFFFFFFFFFFFp1023'
                float_min = '-0x1.FFFFFFFFFFFFFp1023'

                for op in ops:
                    if op == '' or '//' in op:
                        n_ops -= 1
                        continue
                    
                    el = op.replace('=', ' ')[:-1].split()
                    _, *el = el

                    for i, e in enumerate(el):
                        # check for double max
                        if e == empty or e == entire:
                            continue

                        vals = e[1:-1].split(',')
                        vals = ['std::numeric_limits<T>::max()' if v == float_max else 'std::numeric_limits<T>::lowest()' if v == float_min else v for v in vals]
                        el[i] = f'{{{vals[0]},{vals[1]}}}'

                    # for i in [(i+1) % n_vars for i in range(n_vars)]:
                    for i in range(n_vars):
                        e = el[i]
                        if e == empty:
                            var_codes[i] += indent + '        empty,\n'
                        elif e == entire:
                            var_codes[i] += indent + '        entire,\n'
                        else:
                            var_codes[i] += indent + f'        {e},\n'

                cuda_code = ''
                for i in range(n_vars):
                    var_codes[i] += indent + '    }};\n\n'

                for i in range(1, n_vars):
                    cuda_code += indent + f'    CUDA_CHECK(cudaMemcpy(d_{vars[i]}, h_{vars[i]}.data(), n_bytes, cudaMemcpyHostToDevice));\n'

                device_vars = ''
                for v in vars[1:]:
                    device_vars += f', d_{v}'

                cuda_code += indent + f'    test_{instr}<<<numBlocks, blockSize>>>(n{device_vars});\n'
                cuda_code += indent + f'    CUDA_CHECK(cudaMemcpy(h_{vars[1]}.data(), d_{vars[1]}, n_bytes, cudaMemcpyDeviceToHost));\n'
                cuda_code += indent + f'    auto failed = check_all_equal<I, n>(h_{vars[1]}, h_ref);\n'
                cuda_code += indent + '    for (auto fail_id : failed) {\n'
                cuda_code += indent + '        printf("failed at case %zu:\\n", fail_id);\n'
                cuda_code += indent + '        printf("'
                
                params_code = ''
                for i in range(1, n_vars):
                    cuda_code += f'{vars[i][0]} = [%a, %a]\\n'
                    params_code += f', h_{vars[i]}[fail_id].lb, h_{vars[i]}[fail_id].ub'

                cuda_code += '"'
                cuda_code += params_code
                cuda_code += ');\n'
                cuda_code += indent + '    }\n'
                cuda_code += indent + '};\n\n'

                largest_n = max(n_ops, largest_n)
                size_code = indent + f'    constexpr int n = {n_ops};\n'
                test_code += size_code
                for var_code in var_codes:
                    test_code += var_code

                test_code += cuda_code 
                code += test_code

            code_constants = f'''
    const int n = {largest_n}; // count of largest test array
    const int n_bytes   = n * sizeof(I);
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;

    interval<T> *d_xs, *d_ys, *d_zs;
    CUDA_CHECK(cudaMalloc(&d_xs, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_ys, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_zs, n_bytes));\n\n'''

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

