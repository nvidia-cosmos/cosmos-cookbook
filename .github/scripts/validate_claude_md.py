# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#!/usr/bin/env python3
"""
Validates CLAUDE.md files added or modified in a PR.

Rules:
- If Data Source Access is "Public", verify the dataset/model/URL is reachable
  and not gated, without downloading it.
- If Data Source Access is "Gated" or "Restricted", emit a warning and exit
  successfully (human reviewers must approve these).
- If the Access field is missing entirely, fail — contributors must declare it.

Usage:
    python validate_claude_md.py path/to/CLAUDE.md [path/to/another/CLAUDE.md ...]
"""

import re
import sys
import urllib.request
import urllib.error

# Optional — only needed for HuggingFace validation
try:
    from huggingface_hub import dataset_info, model_info
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_data_source_block(text):
    """
    Extract the ## Data Source section from CLAUDE.md content.
    Returns a dict with keys: access, size, license, command.
    """
    # Find the Data Source section
    section_match = re.search(
        r"^## Data Source\s*\n(.*?)(?=^## |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    if not section_match:
        return None

    section = section_match.group(1)

    # Strip HTML comments (the contributor guidance block)
    section = re.sub(r"<!--.*?-->", "", section, flags=re.DOTALL)

    access = _field(section, "Access")
    size = _field(section, "Size")
    license_ = _field(section, "License")
    command = _code_block(section)

    return {"access": access, "size": size, "license": license_, "command": command}


def _field(text, name):
    m = re.search(rf"\*\*{name}:\*\*\s*(.+)", text)
    return m.group(1).strip() if m else None


def _code_block(text):
    m = re.search(r"```(?:bash)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_hf_repo(repo_id, repo_type="dataset"):
    """
    Check that a HuggingFace repo exists and is not gated.
    Returns (ok: bool, message: str).
    """
    if not HF_HUB_AVAILABLE:
        return True, "huggingface_hub not installed — skipping HF validation"

    try:
        if repo_type == "dataset":
            dataset_info(repo_id)
        else:
            model_info(repo_id)
        return True, f"HuggingFace repo '{repo_id}' is public and accessible"
    except GatedRepoError:
        return False, (
            f"HuggingFace repo '{repo_id}' is GATED. "
            "Set Access to 'Gated' in the Data Source section."
        )
    except RepositoryNotFoundError:
        return False, f"HuggingFace repo '{repo_id}' was not found."
    except Exception as e:
        return False, f"Unexpected error checking '{repo_id}': {e}"


def validate_url(url):
    """HEAD request to confirm a URL is reachable. Returns (ok, message)."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        req.add_header("User-Agent", "cosmos-cookbook-ci/1.0")
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status < 400:
                return True, f"URL reachable ({resp.status}): {url}"
            return False, f"URL returned HTTP {resp.status}: {url}"
    except urllib.error.HTTPError as e:
        return False, f"URL returned HTTP {e.code}: {url}"
    except Exception as e:
        return False, f"Could not reach URL '{url}': {e}"


def validate_command(command):
    """
    Dispatch to the appropriate validator based on the command shape.
    Returns (ok, message).
    """
    if not command:
        return False, "No download command found in Data Source section."

    # huggingface-cli download <repo-id> [--repo-type dataset|model]
    hf_match = re.search(
        r"huggingface-cli\s+download\s+([\w\-./]+)(.*)", command
    )
    if hf_match:
        repo_id = hf_match.group(1)
        rest = hf_match.group(2)
        repo_type = "dataset" if "--repo-type dataset" in rest else "model"
        return validate_hf_repo(repo_id, repo_type)

    # wget or curl with a URL
    url_match = re.search(r"(?:wget|curl[^\"']*)\s+[\"']?(https?://[^\s\"']+)", command)
    if url_match:
        return validate_url(url_match.group(1))

    # Plain URL on its own line
    bare_url = re.search(r"https?://\S+", command)
    if bare_url:
        return validate_url(bare_url.group(0))

    return (
        False,
        "Could not parse a HuggingFace repo ID or URL from the Data Source command. "
        "Ensure it uses 'huggingface-cli download', 'wget', or 'curl'.",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def check_file(path):
    print(f"\n=== Checking: {path} ===")
    with open(path, encoding="utf-8") as f:
        content = f.read()

    data = parse_data_source_block(content)

    if data is None:
        print("ERROR: No '## Data Source' section found.")
        return False

    access = data["access"]
    if not access:
        print(
            "ERROR: 'Access' field is missing from Data Source. "
            "Add '**Access:** Public | Gated | Restricted' above the download command."
        )
        return False

    access_lower = access.lower()

    if access_lower.startswith("gated") or access_lower.startswith("restricted"):
        print(
            f"WARNING: Access is '{access}'. "
            "Skipping automated validation — a human reviewer must confirm "
            "that contributors can obtain this data."
        )
        return True  # Pass CI; flag for human review

    if access_lower.startswith("public"):
        ok, msg = validate_command(data["command"])
        if ok:
            print(f"OK: {msg}")
        else:
            print(f"ERROR: {msg}")
        return ok

    print(
        f"ERROR: Unrecognised Access value '{access}'. "
        "Must be 'Public', 'Gated', or 'Restricted'."
    )
    return False


def main():
    if len(sys.argv) < 2:
        print("Usage: validate_claude_md.py <CLAUDE.md> [...]")
        sys.exit(1)

    results = [check_file(p) for p in sys.argv[1:]]

    if not all(results):
        print("\nOne or more CLAUDE.md files failed validation.")
        sys.exit(1)

    print("\nAll CLAUDE.md files passed validation.")


if __name__ == "__main__":
    main()
