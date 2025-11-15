#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Document generator. Requires Python 3.11+

"""
Generate documentation entries for functions using (generated) functions.toml.

Usage: python3 gen/gen_docs.py
"""

import tomllib
import os
from collections import defaultdict

I = 'I' # interval
B = 'B' # boolean
T = 'T' # type
N = 'N' # number >= 0
Z = 'Z' # number (no restriction)
S = 'S' # split of interval (into two)

def snake_to_camel(snake_str):
    parts = snake_str.split('_')
    return parts[0] + ''.join(word.capitalize() for word in parts[1:])

# Load supported functions from functions.toml. The TOML file places functions under a top-level
# table named "functions". We normalise the loaded structure into the same shape the rest of
# the script expects: a dict mapping name -> dict with keys args, ret, ulp_error, code_name, latex_name.
fname = 'functions.toml'
base_path = '/cuinterval'
supported = {}
toml_path = os.path.join(os.path.dirname(__file__), f'{fname}')
if not os.path.exists(toml_path):
    raise FileNotFoundError(f"{fname} not found at {toml_path}")

with open(toml_path, 'rb') as f:
    data = tomllib.load(f)

functions = data.get('functions', {})
for name, item in functions.items():
    supported[name] = {
        'args': item.get('args', []),
        'ret': item.get('ret'),
        'ulp_error': item.get('ulp_error', 0),
        'code_name': item.get('code_name', name),
        'latex_name': item.get('latex_name', r'\mathrm{' + snake_to_camel(name) + "}"),
        'arg_names': item.get('arg_names', []),
        'source': item.get('source', ''),
        'group': item.get('group'),
        'brief': item.get('brief', ''),
        'description': item.get('description', ''),
        'constraints': item.get('constraints', [''] * len(item.get('args', []))),
    }


latex_type = { T: r"\mathbb{R}", I: r"\mathbb{IR}", Z: r"\mathbb{Z}", N: r"\mathbb{N}", B: r"\mathbb{B}", S: r"\{\mathbb{IR}, \mathbb{IR}\}"}
code_type = { T: "T", I: r"<IntervalRef />", B: "bool", Z: "<Integral />", N: "<Integral />", S: r"<SplitRef />"}


if __name__ == '__main__':
    buffer = ""
    api_buffer = ""
    overview_buffer = ""

    imports = r"""import { 
  Centered,
  FunctionDetails, 
  FunctionDeclaration, 
  FunctionSignature, 
  FunctionBrief,
  IntervalRef,
  Integral
} from '@components';
"""
    groups = {v['group']: f"""---\ntitle: {v['group']}\ndescription: An overview of the {v['group'].lower()} operations.\n---\n\n{imports}\n""" for k, v in supported.items()}
    groups_table = defaultdict(dict)
    for k, v in supported.items():
        group = v['group']
        code_name = v['code_name']
        groups_table[group][code_name] = {'inputs': '', 'output': '', 'error': 0, 'link': ''}

    for k, v in supported.items():
        # Unpack fields explicitly to avoid relying on dict ordering
        args = v.get('args', [])
        ret = v.get('ret')
        ulp_error = v.get('ulp_error', 0)
        code_name = v.get('code_name')
        latex_name = v.get('latex_name')
        arg_names = v.get('arg_names', [])
        sourceUrl = v.get('source', [])
        group = v.get('group')
        brief = v.get('brief')
        constraints = v.get('constraints')
        description = v.get('description')

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
  sourceUrl="{sourceUrl}" 
  nvidiaUrl="group__CUDA__MATH__DOUBLE.html#_CPPv43sind"
>
  {code_type[ret]} {code_name}({code_inputs})
</FunctionDeclaration>
"""
        def add_constraint(type, constraint):
            return type + constraint if not constraint or constraint[0] == '_' else constraint

        link_to_details = f"{base_path}/operations/{group.lower()}/#{code_name}"
        latex_inputs = "\\times".join(add_constraint(latex_type[arg], constraints[i]) for i,arg in enumerate(args))
        latex_output = add_constraint(latex_type[ret], constraints[-1])
        groups_table[group][code_name] = {
            'inputs': f"{latex_type[args[0]]}^{len(args)}" if len(set(args)) == 1 and len(args) > 1 else latex_inputs,
            'output': latex_output,
            'error': ulp_error,
            'link': link_to_details
        }

        signature = f"$${latex_name}: " + latex_inputs + R" \rightarrow " + latex_output + "$$"
        note = ""
        extra = note + description

        details = f"""
<FunctionDetails id="{code_name}">
  <FunctionBrief error="{ulp_error}" slot="brief">
    {brief}
  </FunctionBrief>
  
  <FunctionSignature>
    {signature}
  </FunctionSignature>
  {extra}
</FunctionDetails>

---
"""
        full = header + declaration + details

        groups[group] += full
        buffer += full

        api_declaration = f"""
<FunctionDeclaration
  sourceUrl="{sourceUrl}"
>
  {code_type[ret]} <Link name="{code_name}" link="{link_to_details}" />({code_inputs})
</FunctionDeclaration>
"""

        api_buffer += api_declaration

    # accumulate overview page and api page
    
    # overview page
    overview_header = r"""---
title: All Operations
description: An overview of the supported operations.
---

import { LinkButton } from '@astrojs/starlight/components';

The following operations are implemented as CUDA kernels. All operations are correctly-rounded, given the limitations of the precision of the underlying [CUDA operation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#mathematical-functions-appendix). The tightest interval is always a subset
of the computed interval. For most operations, the lower and upper bounds of the basic operations are at most 3 [ulps](https://en.wikipedia.org/wiki/Unit_in_the_last_place) away from the lower and upper bounds of the tightest interval, respectively.
The error for a particular operation is given below.

"""
    for group_name, functions in groups_table.items():
        overview_buffer += f"## {group_name}\n\n"
        overview_buffer += "| Operation | Function Description | Error [ulps] |\n"
        overview_buffer += "|-----------|----------------------|--------------|\n"
        for func_name, func_info in functions.items():
            op = f"[{func_name}]({func_info['link']})"
            decl = f"${func_info['inputs']} \\rightarrow {func_info['output']}$"
            overview_buffer += f"| {op:<80} | {decl:<40} | {func_info['error']} |\n"
        overview_buffer += "\n"


    # api page
    api_header = r"""---
title: API
description: A list of all functions and structs in CuInterval.
---

import { 
  FunctionDeclaration,
  IntervalRef,
  Integral,
  Link
} from '@components';

:::note
Functions are templatized on `T` (omitted for visual clarity). 
:::
"""

    overview_mdx = overview_header + overview_buffer
    with open("all.mdx", "w", encoding="utf-8") as f:
        f.write(overview_mdx)

    api_mdx = api_header + api_buffer
    with open("api.mdx", "w", encoding="utf-8") as f:
        f.write(api_mdx)

    # Write each group to its own .mdx file
    for group_name, content in groups.items():
      if group_name is None:
        continue
      filename = f"{group_name.lower()}.mdx"
      with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
