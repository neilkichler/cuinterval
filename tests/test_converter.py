import sys
from collections import defaultdict

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
            supported = ['pos', 'neg', 'add', 'sub', 'mul', 'div', 'sqr', 'sqrt', 'fma']
            empty = '{empty}'
            entire = '{entire}'
            float_max = '0x1.FFFFFFFFFFFFFp1023'
            float_min = '-0x1.FFFFFFFFFFFFFp1023'

            for test in tests:
                # get the first word as the name prefix of the test
                name, body = test.split(maxsplit=1)
                if name.endswith('dec'): # ignore decorated interval tests
                    continue

                name = name.strip("test").strip('_')
                body = body.replace('[', '{').replace(']', '}').replace(', ', ',').replace('=', ' ')
                ops = body.splitlines()[1:-2]
                ops = [op.lstrip() for op in sorted(ops)]

                subtests = defaultdict(list)
                for op in ops:
                    if op in ['', '}'] or '//' in op:
                        continue
                    new_op, body = op.split(maxsplit=1)
                    subtests[new_op].append(body[:-1].split())
                
                for instr, ops in subtests.items():
                    instr_len = len(ops[0])
                    vars = ['ref', 'xs', 'ys', 'zs'][:instr_len]
                    vars = vars[1:] + vars[:1] # rotate ref to last place
                    n_vars = len(vars)
                    n_ops = len(ops)
                    var_codes = [''] * n_vars

                    if instr not in supported:
                        print(f'Skipping unsupported instruction: {instr}', file=sys.stderr)
                        continue

                    test_code = indent + f'"{name}_{instr}"_test = [&] {{\n'

                    for i in range(n_vars):
                        var_codes[i] = indent + f'    std::array<I, n> h_{vars[i]} {{{{\n'

                    for op in ops:
                        intervals = op

                        for i, interval in enumerate(intervals):
                            # check for double max
                            if interval == empty or interval == entire:
                                continue

                            vals = interval[1:-1].split(',')
                            vals = ['std::numeric_limits<T>::max()' if v == float_max else 'std::numeric_limits<T>::lowest()' if v == float_min else v for v in vals]
                            intervals[i] = f'{{{vals[0]},{vals[1]}}}'

                        for i in range(n_vars):
                            interval = intervals[i]
                            if interval == empty:
                                var_codes[i] += indent*3 + 'empty,\n'
                            elif interval == entire:
                                var_codes[i] += indent*3 + 'entire,\n'
                            else:
                                var_codes[i] += indent*3 + f'{interval},\n'

                    cuda_code = ''
                    for i in range(n_vars):
                        var_codes[i] += indent*2 + '}};\n\n'

                    for i in range(n_vars - 1):
                        cuda_code += indent*2 + f'CUDA_CHECK(cudaMemcpy(d_{vars[i]}, h_{vars[i]}.data(), n_bytes, cudaMemcpyHostToDevice));\n'

                    device_vars = ''
                    for v in vars[:-1]:
                        device_vars += f', d_{v}'

                    cuda_code += indent*2 + f'test_{instr}<<<numBlocks, blockSize>>>(n{device_vars});\n'
                    cuda_code += indent*2 + f'CUDA_CHECK(cudaMemcpy(h_{vars[0]}.data(), d_{vars[0]}, n_bytes, cudaMemcpyDeviceToHost));\n'
                    cuda_code += indent*2 + f'auto failed = check_all_equal<I, n>(h_{vars[0]}, h_ref);\n'
                    cuda_code += indent*2 + 'for (auto fail_id : failed) {\n'
                    cuda_code += indent*3 + 'printf("failed at case %zu:\\n", fail_id);\n'
                    cuda_code += indent*3 + 'printf("'
                    
                    params_code = ''
                    for i in range(1, n_vars):
                        cuda_code += f'{vars[i][0]} = [%a, %a]\\n'
                        params_code += f', h_{vars[i]}[fail_id].lb, h_{vars[i]}[fail_id].ub'

                    cuda_code += '"'
                    cuda_code += params_code
                    cuda_code += ');\n'
                    cuda_code += indent*2 + '}\n'
                    cuda_code += indent + '};\n\n'

                    largest_n = max(n_ops, largest_n)
                    size_code = indent*2 + f'constexpr int n = {n_ops};\n'
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

    # get input file path from command line
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = 'libieeep1788_elem.itl'

    # get output file path from command line (default: stdout)
    if len(sys.argv) > 2:
        out_file = sys.argv[2]
    else:
        out_file = None

    test_code = convert_to_test(file_path)

    if out_file:
        with open(out_file, 'w') as f:
            f.write(test_code)
    else:
        f = sys.stdout
        f.write(test_code)


