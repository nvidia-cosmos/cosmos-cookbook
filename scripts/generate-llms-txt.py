#!/usr/bin/env python3
"""
generate-llms-txt.py
====================
Generates docs/llms.txt by walking every SUMMARY.md under docs/ and
extracting titles + one-line descriptions from the linked Markdown files.

Output format follows the llms.txt spec (Willison / Answer.AI):
  https://llmstxt.org/

Run from the repo root:
    python scripts/generate-llms-txt.py

The script is intentionally stdlib-only (no third-party deps) so it runs
in any Python 3.8+ environment without installing extras.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_ROOT = REPO_ROOT / "docs"
OUTPUT_FILE = DOCS_ROOT / "llms.txt"

HEADER = """\
# Cosmos Cookbook

> NVIDIA Cosmos World Foundation Models — recipes, concepts, and guides for Physical AI practitioners.
>
> Cosmos models accelerate synthetic data generation, post-training, and evaluation for robotics,
> autonomous vehicles, and physical AI. This cookbook provides end-to-end recipes and core concepts
> for working with the full Cosmos model family.
>
> Source: https://github.com/nvidia-cosmos/cosmos-cookbook
> License: Apache 2.0
"""

# Ordered sections: (section heading, docs-relative directory glob, SUMMARY.md path)
# Each entry drives one section in the output.  The order here is the output order.
SECTIONS: list[tuple[str, Path | None]] = [
    ("Getting Started",      DOCS_ROOT / "getting_started" / "SUMMARY.md"),
    ("Inference Recipes",    DOCS_ROOT / "recipes" / "inference" / "SUMMARY.md"),
    ("Post-Training Recipes",DOCS_ROOT / "recipes" / "post_training" / "SUMMARY.md"),
    ("Data Curation Recipes",DOCS_ROOT / "recipes" / "data_curation" / "SUMMARY.md"),
    ("End-to-End Workflows", DOCS_ROOT / "recipes" / "end2end" / "SUMMARY.md"),
    ("Core Concepts",        DOCS_ROOT / "core_concepts" / "SUMMARY.md"),
    ("Gallery",              DOCS_ROOT / "gallery" / "SUMMARY.md"),
]

# For core-concept sections the SUMMARY points to sub-directories; we resolve
# those recursively one level deep.
CORE_CONCEPT_SUBSECTIONS: list[tuple[str, Path]] = [
    ("Prompt Guide",       DOCS_ROOT / "core_concepts" / "prompt_guide"),
    ("Control Modalities", DOCS_ROOT / "core_concepts" / "control_modalities"),
    ("Data Curation",      DOCS_ROOT / "core_concepts" / "data_curation"),
    ("Post-Training",      DOCS_ROOT / "core_concepts" / "post_training"),
    ("Evaluation",         DOCS_ROOT / "core_concepts" / "evaluation"),
    ("Distillation",       DOCS_ROOT / "core_concepts" / "distillation"),
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class Entry(NamedTuple):
    title: str
    rel_path: str        # repo-relative path shown in llms.txt
    description: str


# ---------------------------------------------------------------------------
# Markdown parsing helpers
# ---------------------------------------------------------------------------

_SUMMARY_LINK_RE = re.compile(r"^\s*-\s+\[([^\]]+)\]\(([^)]+)\)", re.MULTILINE)


def parse_summary_links(summary_path: Path) -> list[tuple[str, Path]]:
    """Return [(link_text, absolute_target_path), ...] from a SUMMARY.md.

    Skips entries that point to directories (sub-SUMMARYs) rather than files,
    and skips entries whose target file does not exist.
    """
    if not summary_path.exists():
        return []

    base_dir = summary_path.parent
    text = summary_path.read_text(encoding="utf-8")
    results: list[tuple[str, Path]] = []

    for m in _SUMMARY_LINK_RE.finditer(text):
        link_text = m.group(1).strip()
        raw_target = m.group(2).strip()

        # Skip anchors and external URLs
        if raw_target.startswith("#") or raw_target.startswith("http"):
            continue

        # Resolve relative to the SUMMARY.md's directory
        target = (base_dir / raw_target).resolve()

        # If the target is a directory, look for a SUMMARY.md inside it —
        # we'll handle that in the caller, not here.
        if target.is_dir():
            continue

        # Must be a Markdown file that exists
        if not target.exists():
            # Try appending index.md
            candidate = target / "index.md"
            if candidate.exists():
                target = candidate
            else:
                continue

        results.append((link_text, target))

    return results


def extract_h1(text: str) -> str:
    """Return the text of the first H1 heading, or empty string."""
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def extract_description(text: str) -> str:
    """Extract a one-line description from a Markdown file.

    Strategy (in order of preference):
    1. First non-empty paragraph that is NOT a blockquote, table row, list item,
       or HTML block, appearing after the H1 heading.
    2. First sentence of an ## Overview (or numbered ## N. Overview) section.
    3. First sentence of any section body.
    4. Fallback: empty string.

    The result is stripped to a single sentence (first sentence up to "."),
    truncated to 140 characters.
    """
    # Remove YAML frontmatter if present
    text = re.sub(r"^---\n.*?\n---\n", "", text, flags=re.DOTALL)

    # Strip inline HTML tags and their content (e.g. <video>…</video>, <div>…</div>)
    text = re.sub(r"<[^>]+>.*?</[^>]+>", " ", text, flags=re.DOTALL)
    # Strip remaining self-closing or opening tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Strip the H1 line itself so we don't pick it up as a paragraph
    text = re.sub(r"^#\s+.+\n", "", text, count=1, flags=re.MULTILINE)

    lines = text.splitlines()

    # --- Strategy 1: first prose paragraph after H1 ---
    para_lines: list[str] = []
    in_para = False
    for line in lines:
        stripped = line.strip()

        # Skip blockquote lines (> ...) — including continuation lines that are
        # purely Markdown link fragments from a multi-line blockquote author list:
        #   e.g. "[DeLesley Hutchins](url) • ..." or "] (url)"
        is_blockquote_continuation = (
            stripped.startswith(">")
            or re.match(r"^\]\(https?://", stripped)
            # bare link-only lines: "[Text](url)" or "[Text](url) • ..."
            or re.match(r"^\[.+?\]\(https?://[^)]+\)", stripped)
        )
        if is_blockquote_continuation:
            if in_para:
                break
            continue

        # Skip table rows (| ... |)
        if stripped.startswith("|"):
            if in_para:
                break
            continue

        # Skip headings
        if stripped.startswith("#"):
            if in_para:
                break
            continue

        # Skip HTML tags / raw HTML blocks
        if stripped.startswith("<"):
            if in_para:
                break
            continue

        # Skip image-only lines
        if re.match(r"^!\[", stripped):
            if in_para:
                break
            continue

        # Skip list items (- item, * item, 1. item) — e.g. navigation link lists
        if re.match(r"^[-*+]\s", stripped) or re.match(r"^\d+\.\s", stripped):
            if in_para:
                break
            continue

        # Skip horizontal rules
        if re.match(r"^---+$", stripped):
            if in_para:
                break
            continue

        if stripped == "":
            if in_para:
                # End of paragraph
                break
            continue

        para_lines.append(stripped)
        in_para = True

    candidate = " ".join(para_lines).strip()

    # --- Strategy 2: first sentence of Overview section (handles "## Overview"
    #     and numbered variants like "## 1. Overview") ---
    if not candidate:
        m = re.search(
            r"##\s+(?:\d+\.\s+)?Overview\s*\n+((?:(?!^#).+\n?)+)",
            text,
            re.MULTILINE,
        )
        if m:
            block = m.group(1).strip()
            # Remove blockquote markers
            block = re.sub(r"^>\s*", "", block, flags=re.MULTILINE)
            # Keep only lines that are plain prose (skip tables, images, lists, HTML)
            clean_lines = []
            for ln in block.splitlines():
                s = ln.strip()
                if not s:
                    if clean_lines:
                        break   # stop at first blank line after content
                    continue
                if re.match(r"^[|!<\-*+]", s) or re.match(r"^\d+\.\s", s):
                    if clean_lines:
                        break
                    continue
                # skip bare link lines (author continuation fragments)
                if re.match(r"^\[.+?\]\(https?://", s):
                    if clean_lines:
                        break
                    continue
                clean_lines.append(s)
            candidate = " ".join(clean_lines).strip()

    # --- Strategy 3: first non-empty non-special line anywhere ---
    if not candidate:
        for line in lines:
            s = line.strip()
            if s and not s.startswith(("#", ">", "|", "<", "!", "-", "*")):
                candidate = s
                break

    if not candidate:
        return ""

    # Trim to first sentence
    first_sentence_end = candidate.find(". ")
    if first_sentence_end != -1:
        candidate = candidate[: first_sentence_end + 1]

    # Strip Markdown link syntax [text](url) -> text
    candidate = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", candidate)
    # Strip bold/italic markers
    candidate = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", candidate)
    candidate = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", candidate)
    # Collapse whitespace
    candidate = re.sub(r"\s+", " ", candidate).strip()

    # Truncate
    if len(candidate) > 140:
        candidate = candidate[:137].rstrip() + "..."

    return candidate


def make_entry(link_text: str, abs_path: Path) -> Entry:
    """Build an Entry for a single linked document."""
    try:
        text = abs_path.read_text(encoding="utf-8")
    except OSError:
        return Entry(link_text, str(abs_path.relative_to(REPO_ROOT)), "")

    h1 = extract_h1(text)
    title = h1 if h1 else link_text
    description = extract_description(text)
    rel_path = str(abs_path.relative_to(REPO_ROOT))

    return Entry(title, rel_path, description)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def build_section_from_summary(
    heading: str,
    summary_path: Path,
) -> str | None:
    """Build a single llms.txt section from a SUMMARY.md."""
    links = parse_summary_links(summary_path)
    if not links:
        return None

    lines = [f"## {heading}"]
    for link_text, abs_path in links:
        entry = make_entry(link_text, abs_path)
        if entry.description:
            lines.append(f"- [{entry.title}]({entry.rel_path}): {entry.description}")
        else:
            lines.append(f"- [{entry.title}]({entry.rel_path})")

    return "\n".join(lines)


def build_core_concepts_section() -> str:
    """Build the Core Concepts section by iterating subsection directories."""
    lines = ["## Core Concepts"]

    for sub_heading, sub_dir in CORE_CONCEPT_SUBSECTIONS:
        summary = sub_dir / "SUMMARY.md"
        links = parse_summary_links(summary)
        if not links:
            continue
        lines.append(f"\n### {sub_heading}")
        for link_text, abs_path in links:
            entry = make_entry(link_text, abs_path)
            if entry.description:
                lines.append(f"- [{entry.title}]({entry.rel_path}): {entry.description}")
            else:
                lines.append(f"- [{entry.title}]({entry.rel_path})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level reference entries (index, faq, glossary, contributing)
# ---------------------------------------------------------------------------

def build_top_level_section() -> str:
    """Add top-level docs that appear in the root SUMMARY but not in sub-sections."""
    entries = [
        ("Overview",      DOCS_ROOT / "index.md"),
        ("FAQ",           DOCS_ROOT / "faq.md"),
        ("Glossary",      DOCS_ROOT / "glossary.md"),
        ("Contributing",  DOCS_ROOT / "contributing_doc.md"),
    ]
    lines = ["## Reference"]
    for label, path in entries:
        if not path.exists():
            continue
        entry = make_entry(label, path)
        if entry.description:
            lines.append(f"- [{entry.title}]({entry.rel_path}): {entry.description}")
        else:
            lines.append(f"- [{entry.title}]({entry.rel_path})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate() -> None:
    sections: list[str] = [HEADER]

    # Getting Started
    gs_section = build_section_from_summary(
        "Getting Started",
        DOCS_ROOT / "getting_started" / "SUMMARY.md",
    )
    if gs_section:
        sections.append(gs_section)

    # Inference Recipes
    inf_section = build_section_from_summary(
        "Inference Recipes",
        DOCS_ROOT / "recipes" / "inference" / "SUMMARY.md",
    )
    if inf_section:
        sections.append(inf_section)

    # Post-Training Recipes
    pt_section = build_section_from_summary(
        "Post-Training Recipes",
        DOCS_ROOT / "recipes" / "post_training" / "SUMMARY.md",
    )
    if pt_section:
        sections.append(pt_section)

    # Data Curation Recipes
    dc_section = build_section_from_summary(
        "Data Curation Recipes",
        DOCS_ROOT / "recipes" / "data_curation" / "SUMMARY.md",
    )
    if dc_section:
        sections.append(dc_section)

    # End-to-End Workflows
    e2e_section = build_section_from_summary(
        "End-to-End Workflows",
        DOCS_ROOT / "recipes" / "end2end" / "SUMMARY.md",
    )
    if e2e_section:
        sections.append(e2e_section)

    # Core Concepts (multi-subsection)
    sections.append(build_core_concepts_section())

    # Gallery
    gallery_section = build_section_from_summary(
        "Gallery",
        DOCS_ROOT / "gallery" / "SUMMARY.md",
    )
    if gallery_section:
        sections.append(gallery_section)

    # Reference (top-level pages)
    sections.append(build_top_level_section())

    output = "\n\n".join(s for s in sections if s) + "\n"
    OUTPUT_FILE.write_text(output, encoding="utf-8")

    line_count = output.count("\n")
    print(f"Written: {OUTPUT_FILE.relative_to(REPO_ROOT)}  ({line_count} lines)")


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    generate()
