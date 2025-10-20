#!/bin/sh
doc_ops_path="src/content/docs/operations"
base_path=$(CDPATH= cd -- "$(dirname -- "$0")/../" && pwd -P)
out_dir="$base_path/$doc_ops_path"
gen_path="$base_path/gen"
cd $gen_path
python3 gen_functions.py
python3 gen_docs.py
mv *.mdx $out_dir
