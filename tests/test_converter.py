import glob
import re
import sys
import os

from collections import defaultdict

indent = ' ' * 4

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
            # TODO: create enum for "I", "B", and "T"
            supported = {
                "pos": {"args": ["I"], "ret": "I"},
                "neg": {"args": ["I"], "ret": "I"},
                "add": {"args": ["I", "I"], "ret": "I"},
                "sub": {"args": ["I", "I"], "ret": "I"},
                "mul": {"args": ["I", "I"], "ret": "I"},
                "div": {"args": ["I", "I"], "ret": "I"},
                "recip": {"args": ["I"], "ret": "I"},
                "sqr": {"args": ["I"], "ret": "I"},
                "sqrt": {"args": ["I"], "ret": "I"},
                "fma": {"args": ["I", "I", "I"], "ret": "I"},
                "mig": {"args": ["I", "I"], "ret": "T"},
                "mag": {"args": ["I", "I"], "ret": "T"},
                "wid": {"args": ["I", "I"], "ret": "T"},
                "inf": {"args": ["I"], "ret": "T"},
                "sup": {"args": ["I"], "ret": "T"},
                "mid": {"args": ["I", "I"], "ret": "T"},
                "rad": {"args": ["I"], "ret": "T"},
                "floor": {"args": ["I"], "ret": "I"},
                "ceil": {"args": ["I"], "ret": "I"},
                "abs": {"args": ["I"], "ret": "I"},
                "min": {"args": ["I", "I"], "ret": "I"},
                "max": {"args": ["I", "I"], "ret": "I"},
                "trunc": {"args": ["I"], "ret": "I"},
                "sign": {"args": ["I"], "ret": "I"},
                "intersection": {"args": ["I", "I"], "ret": "I"},
                "convexHull": {"args": ["I", "I"], "ret": "I"},
                "equal": {"args": ["I", "I"], "ret": "B"},
                "subset": {"args": ["I", "I"], "ret": "B"},
                "interior": {"args": ["I", "I"], "ret": "B"},
                "disjoint": {"args": ["I", "I"], "ret": "B"},
                "isEmpty": {"args": ["I"], "ret": "B"},
                "isEntire": {"args": ["I"], "ret": "B"},
                "less": {"args": ["I", "I"], "ret": "B"},
                "strictLess": {"args": ["I", "I"], "ret": "B"},
                "precedes": {"args": ["I", "I"], "ret": "B"},
                "strictPrecedes": {"args": ["I", "I"], "ret": "B"},
                "isMember": {"args": ["T", "I"], "ret": "B"},
                "isSingleton": {"args": ["I"], "ret": "B"},
                "isCommonInterval": {"args": ["I", "I"], "ret": "B"},
                "cancelMinus": {"args": ["I", "I"], "ret": "I"},
                "cancelPlus": {"args": ["I", "I"], "ret": "I"},
                "roundTiesToEven": {"args": ["I"], "ret": "I"},
                "roundTiesToAway": {"args": ["I"], "ret": "I"},
            }

            empty = '{empty}'
            entire = '{entire}'
            float_max = '0x1.FFFFFFFFFFFFFp1023'
            float_min = '-0x1.FFFFFFFFFFFFFp1023'

            failed_code = {
                'params': {
                    'T': 'h_{}[fail_id]',
                    'I': 'h_{}[fail_id].lb, h_{}[fail_id].ub'
                },
                'cuda': {
                    'T': '{} = %a\\n',
                    'B': '{} = %d\\n',
                    'I': '{} = [%a, %a]\\n'
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

                    arg_types = supported[instr]['args']
                    result_type = supported[instr]['ret']
                    test_code = indent + f'"{name}_{instr}"_test = [&] {{\n'

                    for i in range(n_vars - 1):
                        var_codes[i] = indent*2 + f'std::array<{arg_types[i]}, n> h_{vars[i]} {{{{\n'

                    var_codes[n_vars-1] = indent*2 + f'std::array<{result_type}, n> h_res{{}};\n'
                    var_codes[n_vars-1] += indent*2 + f'{result_type} *d_res = ({result_type} *)d_res_;\n'

                    for i in range(n_vars - 1):
                        var_codes[n_vars-1] += indent*2 + f'{arg_types[i]} *d_{vars[i]} = ({arg_types[i]} *)d_{vars[i]}_;\n'

                    var_codes[n_vars-1] += indent*2 + f'int n_result_bytes = n * sizeof({result_type});\n'

                    var_codes[n_vars-1] += indent*2 + f'std::array<{result_type}, n> h_ref {{{{\n'

                    for elements in ops:
                        for i, el in enumerate(elements):
                            var_codes[i] += indent*3
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
                        var_codes[i] += indent*2 + '}};\n\n'

                    for i in range(n_vars - 1):
                        cuda_code += indent*2 + f'CUDA_CHECK(cudaMemcpy(d_{vars[i]}, h_{vars[i]}.data(), n_bytes, cudaMemcpyHostToDevice));\n'
                    
                    cuda_code += indent*2 + 'CUDA_CHECK(cudaMemcpy(d_res, h_res.data(), n_result_bytes, cudaMemcpyHostToDevice));\n'

                    device_vars = ''
                    for v in vars[:-1]:
                        device_vars += f', d_{v}'

                    device_vars += ', d_res'

                    cuda_code += indent*2 + f'test_{instr}<<<numBlocks, blockSize>>>(n{device_vars});\n'
                    cuda_code += indent*2 + 'CUDA_CHECK(cudaMemcpy(h_res.data(), d_res, n_result_bytes, cudaMemcpyDeviceToHost));\n'
                    cuda_code += indent*2 + f'auto failed = check_all_equal<{result_type}, n>(h_res, h_ref);\n'
                    cuda_code += indent*2 + 'for (auto fail_id : failed) {\n'
                    cuda_code += indent*3 + 'printf("failed at case %zu:\\n", fail_id);\n'
                    cuda_code += indent*3 + 'printf("'
                    
                    params_code = ''

                    for i in range(n_vars - 1):
                        var = vars[i]
                        cuda_code += failed_code['cuda'][arg_types[i]].format(var[0])
                        params_code += ', ' + failed_code['params'][arg_types[i]].format(var, var)

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
        # if f != 'mpfi.itl':
        # if f != 'libieeep1788_elem.itl':
        # if f != 'libieeep1788_num.itl':
        #     continue
        test_code = convert_to_test(f)
        f = f.replace('-', '_')
        tests_name = 'tests_' + f.rsplit('.', 1)[0]
        out_file = tests_name + '.cu'
        with open(out_file, 'w') as f:
            f.write(test_code)
        main_includes += f'#include "{out_file}"\n'
        main_tests += indent + tests_name + '<double>();\n'
        print('generated ' + out_file)

    for f in glob.glob('*.cu'):
        os.replace(f, "../" + f)

    os.chdir(os.path.dirname(__file__))

    with open('tests.cu', 'w') as f:
        main_body = f'\nint main()\n{{\n{main_tests}\n    return 0;\n}}\n'
        main_code = main_includes + main_body
        f.write(main_code)

    print('Done!')
