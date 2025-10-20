#!/usr/bin/env python3
"""
Extract function information (argument types, ulp error, signature, etc.) from
include headers and update functions.toml with args, arg_names, and ret fields.

Usage: python3 gen/gen_functions.py
"""

from pathlib import Path
import re
import sys
import os

ROOT = Path(__file__).resolve().parents[2]
INCLUDE = ROOT / 'include' / 'cuinterval'
FNAME = 'functions.toml'
TOML = ROOT / 'docs' / 'gen' / FNAME

# Functions to skip from extraction
SKIP_FUNCTIONS = {'quadrant', 'quadrant_pi'}

try:
    import tomllib
except Exception:
    tomllib = None
try:
    import tomli
except Exception:
    tomli = None
try:
    import tomli_w
except Exception:
    tomli_w = None


FUNC_RE = re.compile(r'''(?x)
    ^\s*(?:template\s*<[^>]+>\s*)?                          # optional template
    (?:(?:__device__|__host__)\s+)?                         # optional CUDA qualifiers
    inline\s+constexpr\s+                                   # require inline constexpr
    (?P<ret>[\w:\s<>,]+?)\s+                                # return type
    (?P<name>[a-zA-Z_]\w*[a-zA-Z0-9])\s*                    # function name (cannot end with _)
    \( (?P<args>[^)]*) \)                                   # args
''', re.MULTILINE)


def map_type(t: str):
    t = t.strip()
    if re.search(r'interval\s*<', t):
        return 'I'
    if 'bool' in t:
        return 'B'
    if re.search(r'\b(unsigned|size_t)\b', t):
        return 'N'
    if re.search(r'\b(int|long|short)\b', t):
        return 'Z'
    if re.search(r'std::integral\s+auto', t):
        return 'Z'
    # treat template params and floats as T
    return 'T'


def clean_name(n: str):
    n = n.strip()
    # remove default values
    n = re.sub(r'=.*$', '', n).strip()
    # remove pointer/reference chars attached to name
    n = n.lstrip('&*')
    # remove stray symbols that may leak in
    n = re.sub(r'[{}<>\[\];]', '', n)
    # extract trailing valid identifier if present
    m = re.search(r'([A-Za-z_][A-Za-z0-9_]*)$', n)
    if m:
        ident = m.group(1)
    else:
        ident = ''

    # filter out common C/C++ keywords that are not parameter names
    if ident in {'if', 'for', 'while', 'return', 'case', 'else', 'switch', 'struct', 'template'}:
        return ''

    return ident


def parse_args(argtext: str):
    argtext = argtext.strip()
    if not argtext or argtext.lower() == 'void':
        return [], []
    parts = re.split(r',(?![^<]*>)', argtext)
    codes = []
    names = []
    for p in parts:
        p = p.strip()
        if not p or p == '...':
            continue
        # remove qualifiers at end like const
        p = re.sub(r'\bconst\b', '', p)
        p = p.strip()
        # split type and name — name is usually last token
        tokens = p.rsplit(None, 1)
        if len(tokens) == 1:
            typ = tokens[0]
            name = ''
        else:
            typ, name = tokens
        name = clean_name(name)
        codes.append(map_type(typ))
        names.append(name)
    return codes, names


def parse_file(path: Path):
    original = path.read_text(encoding='utf-8')
    # detect triple-line section headers in basic.cuh: three consecutive comment lines
    sections = []  # list of (line_no, section_word)
    if path.name == 'basic.cuh':
        lines = original.splitlines()
        for i in range(0, len(lines) - 4):
            empty1 = lines[i].strip()
            a = lines[i + 1].strip()
            b = lines[i + 2].strip()
            c = lines[i + 3].strip()
            empty2 = lines[i + 4].strip()
            if not empty1 and not empty2 and a.startswith('//') and b.startswith('//') and c.startswith('//'):
                mid = b[2:].strip()
                if mid:
                    first_word = mid.split()[0]
                    # store 1-based line number of middle comment
                    sections.append((i + 2, first_word))

    txt = original
    found = {}
    for m in FUNC_RE.finditer(txt):
        name = m.group('name')
        ret = m.group('ret').strip()
        argtext = m.group('args')
        codes, names = parse_args(argtext)
        # compute line number where the function name appears
        try:
            name_pos = m.start('name')
        except Exception:
            name_pos = m.start()
        line_no = txt.count('\n', 0, name_pos) + 1
        rel = str(path.relative_to(ROOT))
        entry = {
            'args': codes,
            'arg_names': names,
            'ret': map_type(ret),
            'source': f"{rel}#L{line_no}",
        }
        # determine section group if in basic.cuh
        if path.name == 'basic.cuh' and sections:
            sec_name = None
            for sec_line, sec_word in sections:
                if sec_line < line_no:
                    sec_name = sec_word
                else:
                    break
            if sec_name:
                entry['group'] = sec_name

        found[name] = entry

    return found


def load_toml(path: Path):
    with open(path, 'rb') as f:
        if tomllib is not None:
            return tomllib.load(f)
        if tomli is not None:
            return tomli.load(f)
        raise RuntimeError('tomllib (py3.11) or tomli required')


def write_toml(data, path: Path):
    if tomli_w is not None:
        with open(path, 'wb') as f:
            f.write(tomli_w.dumps(data).encode('utf-8'))
        return
    # basic writer
    lines = ['[functions]']
    funcs = data.get('functions', {})
    # Emit top-level function tables (non-grouped)
    for fname, fdata in funcs.items():
        if isinstance(fdata, dict) and all(isinstance(v, dict) for v in fdata.values()):
            # group/dict of functions — emit later
            continue
        lines.append('')
        lines.append(f'[functions.{fname}]')
        for k, v in fdata.items():
            if isinstance(v, list):
                arr = ', '.join(f"'{x}'" for x in v)
                lines.append(f'{k} = [{arr}]')
            elif isinstance(v, str):
                lines.append(f"{k} = '{v}'")
            else:
                lines.append(f'{k} = {v}')
    # Emit grouped functions, e.g., functions.SectionName.func
    for group_name, group in funcs.items():
        if not (isinstance(group, dict) and all(isinstance(v, dict) for v in group.values())):
            continue
        for fname, fdata in group.items():
            lines.append('')
            lines.append(f'[functions.{group_name}.{fname}]')
            for k, v in fdata.items():
                if isinstance(v, list):
                    arr = ', '.join(f"'{x}'" for x in v)
                    lines.append(f'{k} = [{arr}]')
                elif isinstance(v, str):
                    lines.append(f"{k} = '{v}'")
                else:
                    lines.append(f'{k} = {v}')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    if not INCLUDE.exists():
        print('include directory not found', INCLUDE, file=sys.stderr)
        sys.exit(1)
    if not TOML.exists():
        print(f'{FNAME} not found', TOML, file=sys.stderr)
        sys.exit(1)

    all_found = {}
    for p in INCLUDE.rglob('*'):
        if p.is_file() and p.suffix in ('.h', '.hpp', '.cuh'):
            try:
                res = parse_file(p)
                all_found.update(res)
            except Exception as e:
                print('error parsing', p, e, file=sys.stderr)

    if not all_found:
        print('no functions found')
        sys.exit(0)

    data = load_toml(TOML)
    funcs = data.setdefault('functions', {})
    for name, info in all_found.items():
        if name in SKIP_FUNCTIONS:
            continue
        entry = funcs.get(name, {})
        entry['args'] = info['args']
        entry['arg_names'] = info['arg_names']
        entry['ret'] = info['ret']
        entry['source'] = info['source']
        entry['group'] = info['group']
        funcs[name] = entry

    data['functions'] = funcs
    write_toml(data, TOML)
    print('updated', TOML)


if __name__ == '__main__':
    main()
