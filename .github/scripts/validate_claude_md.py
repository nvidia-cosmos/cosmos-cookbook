# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
import urllib.error
import urllib.request

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
    Extract the ## Data Source section from a CLAUDE.md file and parse its fields.

    Locates the section by its markdown heading, strips HTML comment blocks (which
    contain contributor guidance and should not be treated as field values), then
    extracts the Access, Size, and License bold-field values and the first bash
    code block (the download command).

    Returns a dict with keys: access, size, license, command.
    Returns None if the ## Data Source section is not present at all.
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
    """
    Extract the value of a bold markdown field of the form **Name:** value.

    Used to pull Access, Size, and License out of the Data Source section.
    Returns the stripped value string, or None if the field is not found.
    """
    m = re.search(rf"\*\*{name}:\*\*\s*(.+)", text)
    return m.group(1).strip() if m else None


def _code_block(text):
    """
    Extract the content of the first fenced code block (``` or ```bash) in text.

    Used to isolate the download command from the Data Source section.
    Returns the stripped command string, or None if no code block is found.
    """
    m = re.search(r"```(?:bash)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else None


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_hf_repo(repo_id, repo_type="dataset"):
    """
    Verify that a HuggingFace repository exists and is freely accessible.

    Calls the HuggingFace Hub API (dataset_info or model_info) without downloading
    any files. This catches two failure modes that would block a contributor:
      - The repo does not exist (typo, wrong org, deleted)
      - The repo is gated (contributor marked it Public but it requires a license agreement)

    Falls back to passing with a warning if huggingface_hub is not installed,
    so the check does not hard-fail in environments without the package.

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
    """
    Confirm that a direct URL is reachable without downloading its content.

    Issues an HTTP HEAD request so only headers are returned, keeping CI fast
    even for large files. Treats any response below HTTP 400 as success.
    Used as a fallback for Data Source commands that use wget or curl rather
    than the HuggingFace CLI.

    Returns (ok: bool, message: str).
    """
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
    Parse the Data Source download command and dispatch to the right validator.

    Supports three command shapes, tried in order:
      1. huggingface-cli download <repo-id> [--repo-type dataset|model]
         → validated via the HuggingFace Hub API (no download)
      2. wget <url> or curl <url>
         → validated via an HTTP HEAD request
      3. A bare https:// URL anywhere in the command
         → validated via an HTTP HEAD request

    Fails if none of these patterns are recognised, prompting the contributor
    to use a supported download tool.

    Returns (ok: bool, message: str).
    """
    if not command:
        return False, "No download command found in Data Source section."

    # huggingface-cli download <repo-id> [--repo-type dataset|model]
    hf_match = re.search(r"huggingface-cli\s+download\s+([\w\-./]+)(.*)", command)
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
    """
    Run all validation checks against a single CLAUDE.md file.

    Enforces the following rules in order:
      1. The ## Data Source section must exist.
      2. The Access field must be present and set to Public, Gated, or Restricted.
      3. If Access is Public, the download command must be reachable and not gated.
      4. If Access is Gated or Restricted, CI passes but prints a warning so human
         reviewers know manual approval is required.

    Returns True if the file passes (or is flagged for human review), False if it
    should block the PR from merging.
    """
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
    """
    Entry point. Accepts one or more CLAUDE.md file paths as arguments,
    runs check_file on each, and exits non-zero if any file fails validation.
    """
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
