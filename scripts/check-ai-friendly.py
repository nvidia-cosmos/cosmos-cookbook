#!/usr/bin/env python3
"""
check-ai-friendly.py — Cosmos Cookbook AI-Friendly Audit
CBK-001

Scans docs/recipes/ and docs/core_concepts/ for .md files and validates:
  1. YAML frontmatter presence (only at file start — mid-file --- are horizontal rules)
  2. Author blockquote  (> **Authors:** or > **Author:**)
  3. Metadata table     (| **Model** |)
  4. ## Prerequisites section (exact heading)
  5. Unlabeled fenced code blocks (``` without a language tag, open/close-aware)
  6. Broken internal links (relative .md links — checked on disk)
  7. External links (collected, not validated)

Frontmatter notes:
  - "has_frontmatter"             — any valid YAML frontmatter at file start
  - "has_cosmos_schema_fm"        — frontmatter contains cosmos_-prefixed keys
  - MkDocs/gallery files may have generic frontmatter; that is noted, not flagged

SUMMARY.md files are checked for broken links but NOT flagged for missing frontmatter.

Outputs:
  - Human-readable report to stdout
  - JSON manifest to scripts/output/ai-friendly-manifest.json

Usage:
    python scripts/check-ai-friendly.py
"""

import datetime
import json
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_ROOT = REPO_ROOT / "docs"

# Primary scan targets per spec (recipes + core_concepts)
SCAN_DIRS = [
    DOCS_ROOT / "recipes",
    DOCS_ROOT / "core_concepts",
]

OUTPUT_DIR = REPO_ROOT / "scripts" / "output"
MANIFEST_PATH = OUTPUT_DIR / "ai-friendly-manifest.json"

# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Author blockquote — "> **Authors:**" or "> **Author:**"
RE_AUTHOR_BQ = re.compile(r">\s*\*\*Authors?:\*\*", re.IGNORECASE)

# Metadata table — header row must contain "| **Model** |"
RE_METADATA_TABLE = re.compile(r"\|\s*\*\*Model\*\*\s*\|")

# Exact ## Prerequisites heading
RE_PREREQUISITES = re.compile(r"^##\s+Prerequisites\s*$", re.MULTILINE)

# First H1
RE_H1 = re.compile(r"^#\s+(.+)$", re.MULTILINE)

# Author line extraction (first line of the blockquote)
RE_AUTHOR_LINE = re.compile(r">\s*\*\*Authors?:\*\*\s*(.*?)(?:\n|$)", re.IGNORECASE)

# Strip markdown link syntax [label](url) -> label
RE_STRIP_LINK = re.compile(r"\[([^\[\]]+)\]\([^)]*\)")

# All links: [label](target)  — capture target
RE_ANY_LINK = re.compile(r"\[(?:[^\[\]]*)\]\(([^)]+)\)")

# External links subset
RE_EXTERNAL_LINK = re.compile(r"\[(?:[^\[\]]*)\]\((https?://[^)]+)\)")


# ---------------------------------------------------------------------------
# Frontmatter detection
# ---------------------------------------------------------------------------

def parse_frontmatter(content: str) -> tuple[bool, dict, str]:
    """
    Detect and parse YAML frontmatter ONLY if the file starts with exactly
    '---' on line 1.  Mid-file '---' horizontal rules are NOT frontmatter.

    Returns:
        (has_frontmatter, fields_dict, body)

    Frontmatter is delimited by '---' or '...' on a line by itself.
    """
    if not content.startswith("---"):
        return False, {}, content

    # First line must be exactly '---' (ignoring trailing whitespace / CR)
    nl = content.find("\n")
    if nl == -1:
        return False, {}, content
    if content[:nl].rstrip() != "---":
        return False, {}, content

    rest = content[nl + 1:]  # everything after the opening ---\n

    # Find closing delimiter: a line that is exactly '---' or '...'
    closing = re.search(r"^(---|\.\.\.)\s*$", rest, re.MULTILINE)
    if not closing:
        # No closing delimiter — malformed, treat as no frontmatter
        return False, {}, content

    fm_text = rest[: closing.start()]
    body = rest[closing.end():]
    if body.startswith("\n"):
        body = body[1:]

    # Parse simple key: value pairs (no multi-line YAML support needed here)
    fields: dict = {}
    for line in fm_text.splitlines():
        m = re.match(r"^([\w][\w_\-.]*)\s*:\s*(.*)\s*$", line)
        if m:
            fields[m.group(1).strip()] = m.group(2).strip()

    return True, fields, body


def has_cosmos_schema(fields: dict) -> bool:
    """True if any frontmatter key starts with 'cosmos_'."""
    return any(k.startswith("cosmos_") for k in fields)


# ---------------------------------------------------------------------------
# Unlabeled code blocks  (open/close state machine)
# ---------------------------------------------------------------------------

def count_unlabeled_code_blocks(content: str) -> int:
    """
    Count fenced code block openings (``` or ~~~) that carry NO language tag.

    Uses an open/close state machine so that:
      - Closing fences are never counted as new openings
      - Nested fences of the same char with shorter length are body content
    """
    count = 0
    in_block = False
    open_char = ""
    open_len = 0

    for raw_line in content.splitlines():
        line = raw_line.strip()
        m = re.match(r"^(`{3,}|~{3,})([\w\-+#.]*)[ \t]*$", line)
        if not m:
            continue

        fence_str = m.group(1)
        lang_tag = m.group(2)
        char = fence_str[0]
        length = len(fence_str)

        if not in_block:
            # Opening fence
            in_block = True
            open_char = char
            open_len = length
            if not lang_tag:
                count += 1
        else:
            # Potential closing fence — must match same char and length >= opening
            if char == open_char and length >= open_len:
                in_block = False
                open_char = ""
                open_len = 0
            # Otherwise it's content inside the block — ignore

    return count


# ---------------------------------------------------------------------------
# Link analysis
# ---------------------------------------------------------------------------

def analyze_links(source: Path, content: str) -> tuple[list[dict], list[str]]:
    """
    Scan all [label](target) links in content.

    Returns:
        broken_links  — list of {link, resolved} for internal links whose
                        target does not exist on disk
        external_links — deduplicated list of https?:// URLs
    """
    broken: list[dict] = []
    external_seen: set[str] = set()
    external: list[str] = []

    for m in RE_ANY_LINK.finditer(content):
        raw_target = m.group(1).strip()

        # External
        if raw_target.startswith(("http://", "https://", "mailto:", "ftp://", "//")):
            if raw_target.startswith(("http://", "https://")):
                url = raw_target.split(")")[0]  # guard against trailing )
                if url not in external_seen:
                    external_seen.add(url)
                    external.append(url)
            continue

        # Pure anchor or site-root absolute (can't resolve without mkdocs config)
        if raw_target.startswith(("#", "/")):
            continue

        # Strip anchor fragment for file-existence check
        path_part = raw_target.split("#")[0].strip()
        if not path_part:
            continue

        resolved = (source.parent / path_part).resolve()
        if not resolved.exists():
            try:
                rel = str(resolved.relative_to(REPO_ROOT))
            except ValueError:
                rel = str(resolved)
            broken.append({"link": raw_target, "resolved": rel})

    return broken, external


# ---------------------------------------------------------------------------
# Field extraction helpers
# ---------------------------------------------------------------------------

def extract_authors(content: str) -> list[str]:
    """Extract author names from the > **Authors:** blockquote line."""
    m = RE_AUTHOR_LINE.search(content)
    if not m:
        return []
    raw = m.group(1).strip()
    # Replace [Name](url) with just Name
    raw = RE_STRIP_LINK.sub(r"\1", raw)
    # Split on commas, bullet •, or " and "
    parts = re.split(r"\s*[,•]\s*|\s+and\s+", raw)
    return [p.strip().strip("*").strip() for p in parts if p.strip()]


def extract_table_value(content: str, column_header: str) -> str:
    """
    Extract a cell value from the cosmos metadata table for the given column.

    Expected shape:
        | **Model** | **Workload** | **Use Case** |
        |-----------|--------------|--------------|
        | value     | value        | value        |
    """
    # Regex: header row, separator row, data row  (all within ~3 lines)
    pattern = re.compile(
        r"(\|[^\n]+\|)\s*\n"   # header row
        r"\|[-| :]+\|\s*\n"    # separator row
        r"(\|[^\n]+\|)",       # data row
        re.IGNORECASE,
    )
    for hm in pattern.finditer(content):
        header_line = hm.group(1)
        data_line = hm.group(2)

        # Split and normalise cells
        header_cells = [c.strip().strip("*").strip() for c in header_line.split("|") if c.strip()]
        data_cells = [c.strip() for c in data_line.split("|") if c.strip() != ""]

        try:
            idx = next(
                i for i, h in enumerate(header_cells)
                if column_header.lower() in h.lower()
            )
        except StopIteration:
            continue

        if idx < len(data_cells):
            val = data_cells[idx].strip()
            val = RE_STRIP_LINK.sub(r"\1", val)  # strip link markup
            return val.strip()

    return ""


# ---------------------------------------------------------------------------
# Per-file analysis
# ---------------------------------------------------------------------------

def analyze_file(path: Path) -> dict:
    """Run all CBK-001 checks on a single .md file."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return {
            "path": str(path),
            "relative_path": str(path.relative_to(REPO_ROOT)),
            "error": str(exc),
        }

    relative = str(path.relative_to(REPO_ROOT))
    is_nav = path.name == "SUMMARY.md"

    # 1. Frontmatter
    has_fm, fm_fields, _body = parse_frontmatter(content)
    cosmos_fm = has_cosmos_schema(fm_fields) if has_fm else False

    # 2–4. Pattern checks on full content
    has_author = bool(RE_AUTHOR_BQ.search(content))
    has_table = bool(RE_METADATA_TABLE.search(content))
    has_prereqs = bool(RE_PREREQUISITES.search(content))

    # 5. Unlabeled code blocks
    unlabeled = count_unlabeled_code_blocks(content)

    # 6 & 7. Links
    broken_links, external_links = analyze_links(path, content)

    # Extraction for manifest enrichment
    title_m = RE_H1.search(content)
    extracted_title = title_m.group(1).strip() if title_m else ""
    extracted_model = extract_table_value(content, "Model")
    extracted_workload = extract_table_value(content, "Workload")
    extracted_use_case = extract_table_value(content, "Use Case")
    extracted_authors = extract_authors(content)

    # safe_to_apply_frontmatter: we have enough source data AND no charset issues
    def ascii_safe(s: str) -> bool:
        try:
            s.encode("ascii")
            return True
        except UnicodeEncodeError:
            return False

    charset_ok = all(
        ascii_safe(f)
        for f in [extracted_title, extracted_model, extracted_workload, extracted_use_case]
        + extracted_authors
    )
    # Safe if: not a nav file, doesn't already have our schema, has at least one
    # source field (author blockquote or metadata table), and charset is clean
    safe_to_apply = (
        not is_nav
        and not cosmos_fm
        and (has_author or has_table)
        and charset_ok
    )

    return {
        "path": str(path),
        "relative_path": relative,
        "is_nav_file": is_nav,
        "has_frontmatter": has_fm,
        "has_cosmos_schema_frontmatter": cosmos_fm,
        "frontmatter_fields": list(fm_fields.keys()),
        "has_author_blockquote": has_author,
        "has_metadata_table": has_table,
        "has_prerequisites_section": has_prereqs,
        "extracted_title": extracted_title,
        "extracted_model": extracted_model,
        "extracted_workload": extracted_workload,
        "extracted_use_case": extracted_use_case,
        "extracted_authors": extracted_authors,
        "unlabeled_code_block_count": unlabeled,
        "broken_internal_links": broken_links,
        "external_links": external_links,
        "safe_to_apply_frontmatter": safe_to_apply,
    }


# ---------------------------------------------------------------------------
# File collection
# ---------------------------------------------------------------------------

def collect_md_files() -> list[Path]:
    """Walk SCAN_DIRS and return all .md files sorted by relative path."""
    found: list[Path] = []
    for scan_dir in SCAN_DIRS:
        if not scan_dir.exists():
            print(f"WARNING: scan dir not found: {scan_dir}", file=sys.stderr)
            continue
        for root, dirs, files in os.walk(scan_dir):
            # Sort dirs for deterministic traversal
            dirs.sort()
            root_path = Path(root)
            for fname in sorted(files):
                if fname.endswith(".md"):
                    found.append(root_path / fname)
    return found


# ---------------------------------------------------------------------------
# Manifest assembly
# ---------------------------------------------------------------------------

def build_manifest(results: list[dict]) -> dict:
    content_files = [r for r in results if not r.get("is_nav_file") and not r.get("error")]
    nav_files = [r for r in results if r.get("is_nav_file")]

    missing_fm = [r for r in content_files if not r.get("has_frontmatter")]
    missing_prereqs = [r for r in content_files if not r.get("has_prerequisites_section")]
    broken_total = sum(len(r.get("broken_internal_links", [])) for r in results)
    unlabeled_total = sum(r.get("unlabeled_code_block_count", 0) for r in results)

    return {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "summary": {
            "total_files": len(results),
            "content_files": len(content_files),
            "nav_files": len(nav_files),
            "missing_frontmatter": len(missing_fm),
            "missing_prerequisites": len(missing_prereqs),
            "broken_internal_links": broken_total,
            "unlabeled_code_blocks": unlabeled_total,
        },
        "files": results,
    }


# ---------------------------------------------------------------------------
# Human-readable report
# ---------------------------------------------------------------------------

def print_report(manifest: dict, results: list[dict]) -> None:
    s = manifest["summary"]

    content_files = [r for r in results if not r.get("is_nav_file") and not r.get("error")]
    nav_files = [r for r in results if r.get("is_nav_file")]

    has_any_fm = [r for r in content_files if r.get("has_frontmatter")]
    has_cosmos_fm = [r for r in content_files if r.get("has_cosmos_schema_frontmatter")]
    has_generic_fm = [r for r in has_any_fm if not r.get("has_cosmos_schema_frontmatter")]
    missing_fm = [r for r in content_files if not r.get("has_frontmatter")]
    missing_prereqs = [r for r in content_files if not r.get("has_prerequisites_section")]

    files_with_unlabeled = [r for r in results if r.get("unlabeled_code_block_count", 0) > 0]
    total_unlabeled = sum(r.get("unlabeled_code_block_count", 0) for r in results)

    broken_pairs = [
        (r["relative_path"], bl)
        for r in results
        for bl in r.get("broken_internal_links", [])
    ]

    safe_files = [r for r in missing_fm if r.get("safe_to_apply_frontmatter")]
    unsafe_files = [r for r in missing_fm if not r.get("safe_to_apply_frontmatter")]

    W = 57  # separator width

    print()
    print("COSMOS COOKBOOK — AI-FRIENDLY AUDIT")
    print("=" * W)
    print(f"Scan dirs      : {', '.join(str(d.relative_to(REPO_ROOT)) for d in SCAN_DIRS)}")
    print(f"Generated at   : {manifest['generated_at']}")
    print()
    print(f"Total files scanned          : {s['total_files']}")
    print(f"  Content files              : {s['content_files']}")
    print(f"  Nav / SUMMARY files        : {s['nav_files']}")
    print()
    print(f"Has cosmos_ schema frontmatter : {len(has_cosmos_fm)}")
    print(f"Has generic/MkDocs frontmatter : {len(has_generic_fm)}")
    print(f"Missing frontmatter (any)      : {s['missing_frontmatter']}")
    print(f"Missing ## Prerequisites       : {s['missing_prerequisites']}")
    print(f"Broken internal links          : {s['broken_internal_links']}")
    print(f"Unlabeled code blocks          : {total_unlabeled}"
          f"  (across {len(files_with_unlabeled)} files)")
    print()

    # ── Files missing frontmatter ──────────────────────────────────────────
    if missing_fm:
        print(f"FILES MISSING FRONTMATTER  ({len(missing_fm)}):")
        for r in missing_fm:
            markers = []
            if r.get("has_author_blockquote"):
                markers.append("has-author")
            if r.get("has_metadata_table"):
                markers.append("has-table")
            suffix = f"  [{', '.join(markers)}]" if markers else "  [no source data]"
            print(f"  {r['relative_path']}{suffix}")
        print()
    else:
        print("FILES MISSING FRONTMATTER  : (none)\n")

    # ── Files already carrying frontmatter ────────────────────────────────
    if has_any_fm:
        print(f"FILES WITH EXISTING FRONTMATTER  ({len(has_any_fm)}):")
        for r in has_any_fm:
            kind = "[cosmos_ schema]" if r.get("has_cosmos_schema_frontmatter") else "[generic/MkDocs]"
            fields_str = ", ".join(r.get("frontmatter_fields", [])) or "(no keys parsed)"
            print(f"  {r['relative_path']}  {kind}")
            if r.get("frontmatter_fields"):
                print(f"      fields: {fields_str}")
        print()

    # ── Broken internal links ─────────────────────────────────────────────
    if broken_pairs:
        print(f"BROKEN INTERNAL LINKS  ({len(broken_pairs)}):")
        for src_path, bl in broken_pairs:
            print(f"  {src_path}")
            print(f"    → {bl['link']}  (NOT FOUND: {bl['resolved']})")
        print()
    else:
        print("BROKEN INTERNAL LINKS  : (none)\n")

    # ── Missing Prerequisites ─────────────────────────────────────────────
    if missing_prereqs:
        print(f"FILES MISSING ## Prerequisites  ({len(missing_prereqs)}):")
        for r in missing_prereqs:
            print(f"  {r['relative_path']}")
        print()
    else:
        print("FILES MISSING ## Prerequisites : (none)\n")

    # ── Unlabeled code blocks ─────────────────────────────────────────────
    if files_with_unlabeled:
        print(f"FILES WITH UNLABELED CODE BLOCKS  ({len(files_with_unlabeled)}):")
        for r in files_with_unlabeled:
            n = r["unlabeled_code_block_count"]
            print(f"  {r['relative_path']}  ({n} unlabeled block{'s' if n != 1 else ''})")
        print()
    else:
        print("FILES WITH UNLABELED CODE BLOCKS : (none)\n")

    # ── Frontmatter applicability summary ─────────────────────────────────
    print("-" * W)
    print(f"SAFE TO BATCH-APPLY FRONTMATTER : {len(safe_files)} files")
    for r in safe_files:
        print(f"  {r['relative_path']}")
    print()

    print(f"NOT SAFE (manual review needed) : {len(unsafe_files)} files")
    for r in unsafe_files:
        reasons = []
        if not r.get("has_author_blockquote"):
            reasons.append("no author blockquote")
        if not r.get("has_metadata_table"):
            reasons.append("no metadata table")
        reason_str = ", ".join(reasons) if reasons else "charset issues"
        print(f"  {r['relative_path']}  (reason: {reason_str})")
    print()

    print("-" * W)
    print(f"Manifest written to: {MANIFEST_PATH.relative_to(REPO_ROOT)}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    md_files = collect_md_files()
    if not md_files:
        print("ERROR: No .md files found in scan directories.", file=sys.stderr)
        sys.exit(1)

    results: list[dict] = [analyze_file(p) for p in md_files]

    manifest = build_manifest(results)

    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print_report(manifest, results)


if __name__ == "__main__":
    main()
