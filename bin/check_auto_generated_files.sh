#!/usr/bin/env bash

set -euo pipefail

echo "ERROR: Do not commit auto-generated files. Please gitignore them." >&2
echo "The following auto-generated files are staged for commit:" >&2
for FILE in "$@"; do
    echo "  - $FILE" >&2
done
echo "" >&2
exit 1
