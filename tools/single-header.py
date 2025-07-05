#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

import datetime
import os
import re
import subprocess

root_path = os.path.join("include")
starting_header = os.path.join(root_path, "cuinterval", "cuinterval.h")
output_header = "cuinterval.cuh"

file_header = """\
// SPDX-License-Identifier: MIT
// See end of file for full license.

/* 
    CuInterval - A CUDA interval arithmetic library
    Copyright (c) {year} Neil Kichler

    Single-Header version {version} from commit {commit_hash}
    Generated: {generation_time}
*/

"""

combined_headers = set()

# Adjust patterns to your needs
internal_include_parser = re.compile(r"\s*#include <(cuinterval/.*)>.*")
include_guard_start_parser = re.compile(r"\s*#ifndef\s+([A-Za-z_][A-Za-z_0-9]*)")
include_guard_define_parser = re.compile(r"\s*#define\s+([A-Za-z_][A-Za-z_0-9]*)")
include_guard_end_parser = re.compile(r"\s*#endif\s+\/\/\s+[A-Za-z_][A-Za-z_0-9]*")


def combine_files(out, filename: str) -> int:
    # If we've already processed this file, skip to avoid duplicate expansions
    if filename in combined_headers:
        return 0
    combined_headers.add(filename)

    # Collect lines that are not internal #includes
    code_lines = []
    # Collect the list of internal includes for recursion
    includes = []

    n_combined = 1  # Counting this file

    with open(filename, mode="r", encoding="utf-8") as input_file:
        inside_guarded_begin_section = False
        inside_guarded_inside_section = False

        for line in input_file:
            # Skip the typical include-guard patterns
            if include_guard_start_parser.match(line):
                inside_guarded_begin_section = True
                continue
            if inside_guarded_begin_section and include_guard_define_parser.match(line):
                inside_guarded_begin_section = False
                inside_guarded_inside_section = True
                continue
            if inside_guarded_inside_section and include_guard_end_parser.match(line):
                inside_guarded_inside_section = False
                continue

            # Match internal #include
            m = internal_include_parser.match(line)
            if m:
                includes.append(m.group(1))
            else:
                # Non-include lines go into a buffer, to be written after recursion
                code_lines.append(line)

    # Recursively process included headers
    for inc in includes:
        inc_path = os.path.join(root_path, inc)
        n_combined += combine_files(out, inc_path)

    # Write a header line for this file, followed by the file’s code
    # Only do this if the codelines are not empty (to avoid root file comment and whitespace)
    if any(s.strip() for s in code_lines):
        out.write(f"// {os.path.relpath(filename, root_path)}\n")
        out.writelines(code_lines)

    return n_combined


def latest_commit_hash(repo_path="."):
    result = subprocess.run(
        ["git", "rev-parse", "origin/main"],
        cwd=repo_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def formatted_file_header():
    return file_header.format(
        generation_time=datetime.datetime.now().isoformat(timespec="minutes"),
        year=datetime.date.today().year,
        commit_hash=latest_commit_hash(),
        version="0.1.0",
    )


def add_license(out):
    out.write("\n/*\n")
    with open("LICENSE", "r") as file:
        contents = file.read()
        out.write(contents)
    out.write("*/\n")


def generate_single_header():
    with open(output_header, mode="w", encoding="utf-8") as header:
        header.write(formatted_file_header())
        header.write("#ifndef CUINTERVAL_CUH\n")
        header.write("#define CUINTERVAL_CUH\n")
        total = combine_files(header, starting_header)
        print(f"Combined {total} headers into {output_header}")
        header.write("#endif // CUINTERVAL_CUH\n")
        add_license(header)


if __name__ == "__main__":
    generate_single_header()
