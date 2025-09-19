#!/usr/bin/env bash

# Convert markdown files to ipynb files.

set -euo pipefail

for md_file in "$@"; do
    # Skip if the file does not contain `jupyter:`
    if ! grep -qE '^jupyter:$' "$md_file"; then
        continue
    fi
    ipynb_file="${md_file/%.md/.ipynb}"
    if [[ ! -f "$ipynb_file" ]]; then
        uvx jupytext -q --to ipynb --update "$md_file"
    else
        uvx jupytext -q --set-formats ipynb,md "$ipynb_file"
        uvx jupytext -q --sync "$ipynb_file"
    fi
done
