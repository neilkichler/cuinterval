import glob
import re
import sys
import os

from collections import defaultdict
from enum import Enum

ParamType = Enum('ParamType', ['I','B','T','N']) # interval | boolean | type | number

indent_one = ' ' * 4
indent_two = ' ' * 8
indent_three = ' ' * 12

def convert_to_test(file_path):
    try:
        with open(file_path, 'r') as file:
            test_name = file_path.rsplit('.', 1)[0].replace('-', '_')
            tests = file.read()

            # remove C++ style block comments
            comments = re.compile(r'(/\*.*?\*/)', re.DOTALL)
            tests = comments.sub('', tests).split('testcase')

            code = ''
            code_preamble = r'''
#include <cuinterval/cuinterval.h>

#include "tests.h"
#include "test_ops.cuh"

#include <stdio.h>

template<typename T>
void tests_''' + test_name + '''() {
    using namespace boost::ut;

    using I = interval<T>;
    using B = bool;
    using N = int;

    I empty         = ::empty<T>();
    I entire        = ::entire<T>();
    T infinity = std::numeric_limits<T>::infinity();
    T NaN = ::nan("");
'''

            code_postamble ='''
    CUDA_CHECK(cudaFree(d_xs_));
    CUDA_CHECK(cudaFree(d_ys_));
    CUDA_CHECK(cudaFree(d_zs_));
    CUDA_CHECK(cudaFree(d_res_));
}
'''
            largest_n = 0
            I = ParamType.I
            B = ParamType.B
            T = ParamType.T
            N = ParamType.N
            supported = {
                "pos": {"args": [I], "ret": I, "ulp_error": 0},
                "neg": {"args": [I], "ret": I, "ulp_error": 0},
                "add": {"args": [I, I], "ret": I, "ulp_error": 0},
                "sub": {"args": [I, I], "ret": I, "ulp_error": 0},
                "mul": {"args": [I, I], "ret": I, "ulp_error": 0},
                "div": {"args": [I, I], "ret": I, "ulp_error": 0},
                "recip": {"args": [I], "ret": I, "ulp_error": 0},
                "sqr": {"args": [I], "ret": I, "ulp_error": 0},
                "sqrt": {"args": [I], "ret": I, "ulp_error": 0},
                "fma": {"args": [I, I, I], "ret": I, "ulp_error": 0},
                "mig": {"args": [I, I], "ret": T, "ulp_error": 0},
                "mag": {"args": [I, I], "ret": T, "ulp_error": 0},
                "wid": {"args": [I, I], "ret": T, "ulp_error": 0},
                "inf": {"args": [I], "ret": T, "ulp_error": 0},
                "sup": {"args": [I], "ret": T, "ulp_error": 0},
                "mid": {"args": [I, I], "ret": T, "ulp_error": 0},
                "rad": {"args": [I], "ret": T, "ulp_error": 0},
                "floor": {"args": [I], "ret": I, "ulp_error": 0},
                "ceil": {"args": [I], "ret": I, "ulp_error": 0},
                "abs": {"args": [I], "ret": I, "ulp_error": 0},
                "min": {"args": [I, I], "ret": I, "ulp_error": 0},
                "max": {"args": [I, I], "ret": I, "ulp_error": 0},
                "trunc": {"args": [I], "ret": I, "ulp_error": 0},
                "sign": {"args": [I], "ret": I, "ulp_error": 0},
                "intersection": {"args": [I, I], "ret": I, "ulp_error": 0},
                "convexHull": {"args": [I, I], "ret": I, "ulp_error": 0},
                "equal": {"args": [I, I], "ret": B, "ulp_error": 0},
                "subset": {"args": [I, I], "ret": B, "ulp_error": 0},
                "interior": {"args": [I, I], "ret": B, "ulp_error": 0},
                "disjoint": {"args": [I, I], "ret": B, "ulp_error": 0},
                "isEmpty": {"args": [I], "ret": B, "ulp_error": 0},
                "isEntire": {"args": [I], "ret": B, "ulp_error": 0},
                "less": {"args": [I, I], "ret": B, "ulp_error": 0},
                "strictLess": {"args": [I, I], "ret": B, "ulp_error": 0},
                "precedes": {"args": [I, I], "ret": B, "ulp_error": 0},
                "strictPrecedes": {"args": [I, I], "ret": B, "ulp_error": 0},
                "isMember": {"args": [T, I], "ret": B, "ulp_error": 0},
                "isSingleton": {"args": [I], "ret": B, "ulp_error": 0},
                "isCommonInterval": {"args": [I, I], "ret": B, "ulp_error": 0},
                "cancelMinus": {"args": [I, I], "ret": I, "ulp_error": 0},
                "cancelPlus": {"args": [I, I], "ret": I, "ulp_error": 0},
                "roundTiesToEven": {"args": [I], "ret": I, "ulp_error": 0},
                "roundTiesToAway": {"args": [I], "ret": I, "ulp_error": 0},
                "cbrt": {"args": [I], "ret": I, "ulp_error": 1},
                "exp": {"args": [I], "ret": I, "ulp_error": 3},
                "exp2": {"args": [I], "ret": I, "ulp_error": 3},
                "exp10": {"args": [I], "ret": I, "ulp_error": 3},
                "expm1": {"args": [I], "ret": I, "ulp_error": 3},
                "log": {"args": [I], "ret": I, "ulp_error": 3},
                "log2": {"args": [I], "ret": I, "ulp_error": 3},
                "log10": {"args": [I], "ret": I, "ulp_error": 3},
                "log1p": {"args": [I], "ret": I, "ulp_error": 3},
                "sin": {"args": [I], "ret": I, "ulp_error": 2},
                "cos": {"args": [I], "ret": I, "ulp_error": 2},
                "tan": {"args": [I], "ret": I, "ulp_error": 3},
                "asin": {"args": [I], "ret": I, "ulp_error": 3},
                "acos": {"args": [I], "ret": I, "ulp_error": 3},
                "atan": {"args": [I], "ret": I, "ulp_error": 3},
                "atan2": {"args": [I, I], "ret": I, "ulp_error": 3},
                "sinh": {"args": [I], "ret": I, "ulp_error": 3},
                "cosh": {"args": [I], "ret": I, "ulp_error": 2},
                "tanh": {"args": [I], "ret": I, "ulp_error": 2},
                "asinh": {"args": [I], "ret": I, "ulp_error": 3},
                "acosh": {"args": [I], "ret": I, "ulp_error": 3},
                "atanh": {"args": [I], "ret": I, "ulp_error": 3},
                "sinpi": {"args": [I], "ret": I, "ulp_error": 3},
                "cospi": {"args": [I], "ret": I, "ulp_error": 3},
                "pown": {"args": [I, N], "ret": I, "ulp_error": 1},
                "pow": {"args": [I, I], "ret": I, "ulp_error": 1},
                "rootn": {"args": [I, N], "ret": I, "ulp_error": 2},
                # "cot": {"args": [I], "ret": I, "ulp_error": 4},
            }

            empty = '{empty}'
            entire = '{entire}'
            float_max = '0x1.FFFFFFFFFFFFFp1023'
            float_min = '-0x1.FFFFFFFFFFFFFp1023'

            failed_code = {
                'params': {
                    T: 'h_{}[fail_id]',
                    N: 'h_{}[fail_id]',
                    I: 'h_{}[fail_id].lb, h_{}[fail_id].ub'
                },
                'cuda': {
                    T: '{} = %a\\n',
                    B: '{} = %d\\n',
                    I: '{} = [%a, %a]\\n',
                    N: '{} = %d\\n'
                }
            }
            
            def replace_min_and_max(v):
                return 'std::numeric_limits<T>::max()' if v == float_max else 'std::numeric_limits<T>::lowest()' if v == float_min else v

            for test in tests:
                if test[0] == '\n':
                    continue
                # get the first word as the name prefix of the test
                name, body = test.split(maxsplit=1)
                name = name.strip("test").strip('_')
                if name.endswith('dec'): # ignore decorated interval tests
                    continue

                body = body.replace('[ ', '[').replace('[', '{').replace(']', '}').replace(' ,', ',').replace(', ', ',').replace('=', ' ')
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

                    arg_types = supported[instr]['args']
                    result_type = supported[instr]['ret'].name
                    max_ulp_diff = supported[instr]['ulp_error']
                    test_code = indent_one + f'"{name}_{instr}"_test = [&] {{\n'

                    n_args = n_vars - 1
                    for i in range(n_args):
                        var_codes[i] = indent_two + f'std::array<{arg_types[i].name}, n> h_{vars[i]} {{{{\n'

                    var_codes[n_args] = indent_two + f'std::array<{result_type}, n> h_res{{}};\n'
                    var_codes[n_args] += indent_two + f'{result_type} *d_res = ({result_type} *)d_res_;\n'

                    for i in range(n_args):
                        var_codes[n_args] += indent_two + f'{arg_types[i].name} *d_{vars[i]} = ({arg_types[i].name} *)d_{vars[i]}_;\n'

                    var_codes[n_args] += indent_two + f'int n_result_bytes = n * sizeof({result_type});\n'
                    var_codes[n_args] += indent_two + f'std::array<{result_type}, n> h_ref {{{{\n'

                    for elements in ops:
                        for i, el in enumerate(elements):
                            var_codes[i] += indent_three
                            if el == empty:
                                var_codes[i] += 'empty,\n'
                            elif el == entire:
                                var_codes[i] += 'entire,\n'
                            elif el[0] != '{':
                                var_codes[i] += f'{el},\n'
                            else:
                                vals = el[1:-1].split(',')
                                vals = [replace_min_and_max(v) for v in vals]
                                elements[i] = f'{{{vals[0]},{vals[1]}}}'

                                var_codes[i] += f'{el},\n'

                    cuda_code = ''
                    for i in range(n_vars):
                        var_codes[i] += indent_two + '}};\n\n'

                    for i in range(n_args):
                        cuda_code += indent_two + f'CUDA_CHECK(cudaMemcpy(d_{vars[i]}, h_{vars[i]}.data(), n_bytes, cudaMemcpyHostToDevice));\n'
                    
                    cuda_code += indent_two + 'CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));\n'

                    device_vars = ''
                    for v in vars[:-1]:
                        device_vars += f', d_{v}'

                    device_vars += ', d_res'

                    cuda_code += indent_two + f'test_{instr}<<<numBlocks, blockSize>>>(n{device_vars});\n'
                    cuda_code += indent_two + 'CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));\n'
                    cuda_code += indent_two + f'int max_ulp_diff = {max_ulp_diff};\n'
                    cuda_code += indent_two + f'auto failed = check_all_equal<{result_type}, n>(h_res, h_ref, max_ulp_diff);\n'
                    cuda_code += indent_two + 'for (auto fail_id : failed) {\n'
                    cuda_code += indent_three + 'printf("failed at case %zu:\\n", fail_id);\n'
                    cuda_code += indent_three + 'printf("'
                    
                    params_code = ''

                    for i in range(n_args):
                        var = vars[i]
                        cuda_code += failed_code['cuda'][arg_types[i]].format(var[0])
                        params_code += ', ' + failed_code['params'][arg_types[i]].format(var, var)

                    cuda_code += '"'
                    cuda_code += params_code
                    cuda_code += ');\n'
                    cuda_code += indent_two + '}\n'
                    cuda_code += indent_one + '};\n\n'

                    largest_n = max(n_ops, largest_n)
                    size_code = indent_two + f'constexpr int n = {n_ops};\n'
                    test_code += size_code
                    for var_code in var_codes:
                        test_code += var_code

                    test_code += cuda_code 
                    code += test_code

            code_constants = f'''
    const int n = {largest_n}; // count of largest test array
    const int n_bytes   = n * sizeof(I);
    const int blockSize = 256;
    [[maybe_unused]] const int numBlocks = (n + blockSize - 1) / blockSize;

    I *d_xs_, *d_ys_, *d_zs_, *d_res_;

    CUDA_CHECK(cudaMalloc(&d_xs_, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_ys_, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_zs_, n_bytes));
    CUDA_CHECK(cudaMalloc(&d_res_, n_bytes));\n\n'''

            return code_preamble + code_constants + code + code_postamble

    except FileNotFoundError:
        return f"File '{file_path}' not found."


if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__) + '/itl')
    files = glob.glob('*.itl', recursive=True)
    main_includes = ''
    main_tests = ''

    for f in files:
        test_code = convert_to_test(f)
        f = f.replace('-', '_')
        tests_name = 'tests_' + f.rsplit('.', 1)[0]
        out_file = tests_name + '.cu'
        with open(out_file, 'w') as f:
            f.write(test_code)
        main_includes += f'#include "{out_file}"\n'
        main_tests += indent_one + tests_name + '<double>();\n'
        print('generated ' + out_file)

    for f in glob.glob('*.cu'):
        os.replace(f, "../" + f)

    os.chdir(os.path.dirname(__file__))

    with open('tests.cu', 'w') as f:
        main_body = f'\nint main()\n{{\n{main_tests}\n    return 0;\n}}\n'
        main_code = main_includes + main_body
        f.write(main_code)

    print('Done!')
