#!/bin/sh
base_path=$(CDPATH= cd -- "$(dirname -- "$0")/../" && pwd -P)
doc_path="$base_path/src/content/docs"
gen_path="$base_path/gen"
cd $gen_path
python3 gen_functions.py
python3 gen_docs.py
mv "api.mdx" "$doc_path/reference"
mv *.mdx "$doc_path/operations"
