# SPDX-License-Identifier: MIT
# Document generator. Requires Python 3.11+

import glob
import re
import sys
import tomllib
import os

from collections import defaultdict

indent_one = ' ' * 4
indent_two = ' ' * 8
indent_three = ' ' * 12

I = 'I' # interval
B = 'B' # boolean
T = 'T' # type
N = 'N' # number

# Load supported mapping from supported.toml. The TOML file places functions under a top-level
# table named "functions". We normalise the loaded structure into the same shape the rest of
# the script expects: a dict mapping name -> dict with keys args, ret, ulp_error, code_name, latex_name.
supported = {}
toml_path = os.path.join(os.path.dirname(__file__), 'supported.toml')
if not os.path.exists(toml_path):
    raise FileNotFoundError(f"supported.toml not found at {toml_path}")

with open(toml_path, 'rb') as f:
    data = tomllib.load(f)

functions = data.get('functions', {})
for name, item in functions.items():
    # item is expected to be a table mapping (args, ret, ulp_error, code_name, latex_name)
    supported[name] = {
        'args': item.get('args', []),
        'ret': item.get('ret'),
        'ulp_error': item.get('ulp_error', 0),
    'code_name': item.get('code_name'),
    'latex_name': item.get('latex_name'),
    'arg_names': item.get('arg_names', []),
    }


latex_type = { T: r"\mathbb{R}", I: r"\mathbb{IR}", N: r"\mathbb{N}", B: r"\mathbb{B}"}
code_type = { T: "T", I: r"<IntervalRef />", B: "bool", N: "std::integral auto"}


if __name__ == '__main__':
    buffer = ""
    # api_buffer = ""
    for k, v in supported.items():
        # Unpack fields explicitly to avoid relying on dict ordering
        args = v.get('args', [])
        ret = v.get('ret')
        ulp_error = v.get('ulp_error', 0)
        code_name = v.get('code_name')
        latex_name = v.get('latex_name')
        arg_names = v.get('arg_names', [])

        header = f"## {k}\n"

        # Build code argument list. If arg_names are provided and non-empty, use them
        # to create typed parameters like "<IntervalRef /> x". Otherwise fall back to
        # just listing types as before.
        if arg_names and any(name for name in arg_names):
            parts = []
            for tcode, name in zip(args, arg_names):
                t = code_type.get(tcode, 'auto')
                if name:
                    parts.append(f"{t} {name}")
                else:
                    parts.append(t)
            code_inputs = ", ".join(parts)
        else:
            code_inputs = ", ".join(code_type[arg] for arg in args)

        declaration = f"""
<FunctionDeclaration
  sourceUrl="arithmetic/basic.cuh#L22" 
  nvidiaUrl="group__CUDA__MATH__DOUBLE.html#_CPPv43sind"
>
  {code_type[ret]} {code_name}({code_inputs})
</FunctionDeclaration>
"""
        latex_inputs = " ".join(latex_type[arg] for arg in args)
        signature = f"$${latex_name} " + latex_inputs + R" \rightarrow " + latex_type[ret] + "$$"
        note = ""
        implementation = ""
        extra = note + implementation

        details = f"""
<FunctionDetails>
  <FunctionBrief error="{ulp_error}" slot="brief">
    Computes the sine of x (in radians).
  </FunctionBrief>
  
  <FunctionSignature>
    {signature}
  </FunctionSignature>
  {extra}
</FunctionDetails>

---
"""
        full = header + declaration + details

        buffer += full

    print(buffer)
