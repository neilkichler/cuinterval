import glob
import re
import sys
import os

from collections import defaultdict

indent = ' ' * 4

def convert_to_test(file_path):
    try:
        with open(file_path, 'r') as file:
            tests = file.read()

            # remove C++ style block comments
            comments = re.compile(r'(/\*.*?\*/)', re.DOTALL)

            tests = comments.sub('', tests).split('testcase')

            test_name = file_path.rsplit('.', 1)[0].replace('-', '_')

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

    I empty         = ::empty<T>();
    I entire        = ::entire<T>();
    T infinity = std::numeric_limits<T>::infinity();
'''

            code_postamble ='''
    CUDA_CHECK(cudaFree(d_xs));
    CUDA_CHECK(cudaFree(d_ys));
    CUDA_CHECK(cudaFree(d_zs));
}
'''
            largest_n = 0
            supported = ['pos', 'neg', 'add', 'sub', 'mul', 'div', 'sqr', 'sqrt', 'fma', 'inf']
            empty = '{empty}'
            entire = '{entire}'
            float_max = '0x1.FFFFFFFFFFFFFp1023'
            float_min = '-0x1.FFFFFFFFFFFFFp1023'
            
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

                    test_code = indent + f'"{name}_{instr}"_test = [&] {{\n'

                    for i in range(n_vars):
                        var_codes[i] = indent + f'    std::array<I, n> h_{vars[i]} {{{{\n'

                    for op in ops:
                        intervals = op

                        for i, interval in enumerate(intervals):
                            # check for double max
                            if interval == empty or interval == entire:
                                continue

                            if interval[0] == '{': # check if it actually is an interval
                                vals = interval[1:-1].split(',')
                                vals = [replace_min_and_max(v) for v in vals]
                                intervals[i] = f'{{{vals[0]},{vals[1]}}}'
                            else: # or scalar
                                intervals[i] = replace_min_and_max(interval)

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
