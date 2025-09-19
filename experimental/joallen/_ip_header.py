#!/usr/bin/env -S uv run --script
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
# ]
# ///

"""Cosmos IP header checker/fixer."""

import argparse
import functools
import os
import sys

HEADER = """SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

HEADER_EARLY_ACCESS = """The early-access software is governed by the NVIDIA Evaluation License Agreement – EA Cosmos Code (v. Feb 2025).
The license reference will be the finalized version of the license linked above.
"""


def _format_header(s: str, ext: str) -> list[str] | None:
    header = s.splitlines()
    # Apply formatting according to file extension.
    if ext in (".py", ".yaml"):
        header = [f"# {line}".rstrip() for line in header]
    elif ext in (".c", ".cpp", ".cu", ".h", ".cuh"):
        header = ["/*"] + [f" * {line}".rstrip() for line in header] + [" */"]
    else:
        return None
    return header


@functools.cache
def _get_header(ext: str) -> tuple[list[str] | None, list[str] | None]:
    return _format_header(HEADER, ext), _format_header(HEADER_EARLY_ACCESS, ext)


def _apply_file(file: str, fix: bool = False, verbose: bool = False) -> bool:
    ext = os.path.splitext(file)[1]
    header, header_early_access = _get_header(ext)
    if header is None:
        # Skip unsupported file extensions.
        if verbose:
            print(f"skipping: {file}")
        return False
    assert header_early_access is not None

    content = open(file).read().splitlines()
    if content and content[0].startswith("#!"):
        # TODO: Handle shebang lines.
        if verbose:
            print(f"skipping: {file}")
        return False

    if _check_header(content, header) or _check_header(content, header_early_access):
        return False

    if fix:
        print(f"fixing: {file}")
        # Clean up leading blank lines.
        i = 0
        while i < len(content) and not content[i]:
            i += 1
        content = content[i:]

        open(file, "w").write("\n".join(header + [""] + content))
        return True
    else:
        print(f"failed: {file}")
        return False


def _check_header(content: list[str], header: list[str]) -> bool:
    if len(content) <= len(header):
        return False
    return content[: len(header)] == header


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix", action="store_true", help="Apply the fixes instead of checking"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Print verbose output"
    )
    args, files_to_check = parser.parse_known_args()

    results: dict[str, bool] = {}
    results = {
        file: _apply_file(file, fix=args.fix, verbose=args.verbose)
        for file in files_to_check
    }
    if any(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
