#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Nvidia docs math function version extractor

import re
import json
import argparse
from urllib.request import urlopen

BASE = "https://docs.nvidia.com/cuda/archive/{}/cuda-math-api/cuda_math_api/group__CUDA__MATH__DOUBLE.html"

ANCHOR_RE = re.compile(r'id="(_CPPv4\d+)([A-Za-z0-9_]+)"')


def scrape_cuda_double_math_versions(cuda_version: str):
    """
    Return dictionary:
        function_name -> version_tag

    Example:
        {"exp10d": "_CPPv45"}
    """

    url = BASE.format(cuda_version)

    with urlopen(url) as f:
        html = f.read().decode("utf-8")

    result = {}

    for version_tag, func in ANCHOR_RE.findall(html):
        result[func] = version_tag

    return result


def write_versions_to_file(cuda_version: str, data: dict):
    """
    Write dictionary to file named:
        nvidia_docs_<cuda_version>.json
    """
    filename = f"nvidia_docs_{cuda_version}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)

    return filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape NVIDIA CUDA double-precision math function versions for docs.")
    parser.add_argument(
        "--cuda-version",
        nargs="?",
        default="12.9.1",
        help="CUDA documentation version to scrape (default: 12.9.1)"
    )
    args = parser.parse_args()

    cuda_version = args.cuda_version
    versions = scrape_cuda_double_math_versions(cuda_version)
    filename = write_versions_to_file(cuda_version, versions)

    print(f"Wrote {len(versions)} entries to {filename}")
