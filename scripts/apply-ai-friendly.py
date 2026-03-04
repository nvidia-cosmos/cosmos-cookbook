#!/usr/bin/env python3
"""
CBK-003 — Batch YAML frontmatter applicator for Cosmos Cookbook.

Reads scripts/output/ai-friendly-manifest.json (produced by check-ai-friendly.py),
processes each file flagged as safe_to_apply_frontmatter=true, synthesizes a
cosmos_* YAML frontmatter block from existing in-file metadata, and prepends it.

Usage:
    python scripts/apply-ai-friendly.py          # dry-run (no writes)
    python scripts/apply-ai-friendly.py --apply  # write files
    python scripts/apply-ai-friendly.py --apply --file docs/recipes/.../inference.md
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).parent.parent
MANIFEST_PATH = REPO_ROOT / "scripts/output/ai-friendly-manifest.json"
LOG_PATH = REPO_ROOT / "scripts/output/apply-log.json"

# ---------------------------------------------------------------------------
# Model name normalization
# ---------------------------------------------------------------------------

MODEL_PATTERNS = [
    (re.compile(r"cosmos[- ]?predict[- ]?2\.5", re.I), "Cosmos Predict 2.5"),
    (re.compile(r"cosmos[- ]?predict[- ]?2(?!\.)", re.I), "Cosmos Predict 2"),
    (re.compile(r"cosmos[- ]?transfer[- ]?2\.5", re.I), "Cosmos Transfer 2.5"),
    (re.compile(r"cosmos[- ]?transfer[- ]?1", re.I), "Cosmos Transfer 1"),
    (re.compile(r"cosmos[- ]?reason[- ]?2", re.I), "Cosmos Reason 2"),
    (re.compile(r"cosmos[- ]?reason[- ]?1", re.I), "Cosmos Reason 1"),
    (re.compile(r"cosmos[- ]?curat", re.I), "Cosmos Curator"),
]

def normalize_model(raw: str) -> Optional[str]:
    for pattern, canonical in MODEL_PATTERNS:
        if pattern.search(raw):
            return canonical
    return None

# Workload normalization: raw table value → schema enum
WORKLOAD_MAP = {
    "inference": "inference",
    "post-training": "post-training",
    "post training": "post-training",
    "posttraining": "post-training",
    "data-curation": "data-curation",
    "data curation": "data-curation",
    "datacuration": "data-curation",
    "end-to-end": "end-to-end",
    "end to end": "end-to-end",
    "endtoend": "end-to-end",
    "workflow": "end-to-end",
}

def normalize_workload(raw: str) -> Optional[str]:
    return WORKLOAD_MAP.get(raw.strip().lower())

# Path-based workload fallback
def workload_from_path(rel_path: str) -> Optional[str]:
    p = rel_path.lower()
    if "/post_training/" in p or "/post-training/" in p:
        return "post-training"
    if "/inference/" in p:
        return "inference"
    if "/data_curation/" in p or "/data-curation/" in p:
        return "data-curation"
    if "/end2end/" in p or "/end-to-end/" in p:
        return "end-to-end"
    return None

# Path-based model fallback for core_concepts (last resort)
def model_from_path(rel_path: str) -> Optional[str]:
    p = rel_path.lower()
    for pattern, canonical in MODEL_PATTERNS:
        if pattern.search(p):
            return canonical
    return None

# Default hardware by workload
HARDWARE_DEFAULTS = {
    "inference": "1x A100 80GB",
    "post-training": "4x H100 80GB",
    "data-curation": "1x A100 80GB",
    "end-to-end": "4x H100 80GB",
}

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_metadata_table(content: str) -> dict:
    """
    Extract Model, Workload, Use Case from the | Model | Workload | Use Case | table.
    Handles multi-row tables (multiple model entries).
    """
    result = {"models": [], "workload": None, "use_case": None}
    in_table = False
    header_seen = False
    separator_seen = False

    for line in content.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            if in_table:
                break
            continue

        cells = [c.strip() for c in stripped.split("|") if c.strip()]
        if not cells:
            continue

        # Detect header row: | **Model** | **Workload** | **Use Case** |
        if any(re.search(r"\*\*model\*\*", c, re.I) for c in cells):
            in_table = True
            header_seen = True
            # Record column positions
            header_cells = [re.sub(r"\*\*", "", c).strip().lower() for c in cells]
            try:
                model_idx = next(i for i, h in enumerate(header_cells) if "model" in h)
                workload_idx = next(i for i, h in enumerate(header_cells) if "workload" in h)
                usecase_idx = next((i for i, h in enumerate(header_cells) if "use case" in h or "usecase" in h), None)
            except StopIteration:
                in_table = False
            continue

        if in_table and not separator_seen:
            # Separator row: | --- | --- |
            if all(re.match(r"^-+$", c) for c in cells):
                separator_seen = True
                continue

        if in_table and separator_seen:
            # Data row
            if len(cells) <= max(model_idx, workload_idx):
                continue
            raw_model = cells[model_idx] if model_idx < len(cells) else ""
            raw_workload = cells[workload_idx] if workload_idx < len(cells) else ""

            # Strip markdown links from model cell
            model_text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", raw_model)
            canonical = normalize_model(model_text)
            if canonical and canonical not in result["models"]:
                result["models"].append(canonical)

            if result["workload"] is None:
                result["workload"] = normalize_workload(raw_workload)

            if usecase_idx is not None and result["use_case"] is None:
                raw_uc = cells[usecase_idx] if usecase_idx < len(cells) else ""
                uc = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", raw_uc).strip()
                if uc:
                    result["use_case"] = uc

    return result


def parse_authors(content: str) -> list[str]:
    """
    Extract author display names from > **Authors:** blockquote.
    Strips markdown link syntax, returns plain names.
    """
    authors = []
    in_author_block = False
    for line in content.splitlines():
        stripped = line.strip()
        if re.match(r"^>\s*\*\*Authors?:\*\*", stripped, re.I):
            # Pull names from the same line after the label
            after = re.sub(r"^>\s*\*\*Authors?:\*\*\s*", "", stripped, flags=re.I)
            names = extract_names(after)
            authors.extend(names)
            in_author_block = True
        elif in_author_block and stripped.startswith(">"):
            text = stripped.lstrip(">").strip()
            # Stop at Organization line or empty blockquote continuation
            if re.match(r"\*\*Org", text, re.I) or not text:
                in_author_block = False
                continue
            # Some authors are on continuation lines joined by •
            names = extract_names(text)
            authors.extend(names)
        else:
            if in_author_block:
                in_author_block = False
    return authors


def extract_names(text: str) -> list[str]:
    """Strip markdown links, split by •, clean up."""
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    parts = [p.strip().strip("•").strip() for p in text.split("•")]
    return [p for p in parts if p and not re.match(r"https?://", p)]


def hardware_from_setup(recipe_file: Path, workload: str) -> str:
    """Try to read hardware spec from sibling setup.md."""
    setup = recipe_file.parent / "setup.md"
    if not setup.exists():
        return HARDWARE_DEFAULTS.get(workload, "1x A100 80GB")

    try:
        text = setup.read_text(encoding="utf-8")
    except Exception:
        return HARDWARE_DEFAULTS.get(workload, "1x A100 80GB")

    # Look for explicit GPU mentions like "8x H100", "4x A100"
    m = re.search(r"(\d)\s*[xX×]\s*(A100|H100|H200|B100|B200|A30|L40)", text)
    if m:
        count, gpu = m.group(1), m.group(2)
        if gpu == "H100":
            return f"{count}x H100 80GB"
        if gpu == "A100":
            return f"{count}x A100 80GB"
        return f"{count}x {gpu}"

    # Check for GPU family hints without count
    if re.search(r"\bH100\b", text):
        gpu_default = "H100"
    elif re.search(r"\bA100\b", text):
        gpu_default = "A100"
    else:
        return HARDWARE_DEFAULTS.get(workload, "1x A100 80GB")

    # Infer count from workload
    if workload == "post-training":
        count = "4"
    elif workload == "end-to-end":
        count = "4"
    else:
        count = "1"
    vram = "80GB"
    return f"{count}x {gpu_default} {vram}"


def git_first_date(file_path: Path) -> str:
    """Return YYYY-MM-DD of first commit for this file, or today."""
    try:
        result = subprocess.run(
            ["git", "log", "--diff-filter=A", "--follow", "--format=%ad",
             "--date=format:%Y-%m-%d", "--", str(file_path)],
            capture_output=True, text=True, cwd=REPO_ROOT, timeout=10
        )
        date = result.stdout.strip().splitlines()
        if date:
            return date[-1]  # oldest date
    except Exception:
        pass
    return "2025-01-01"  # fallback


def derive_tags(workload: str, use_case: Optional[str], rel_path: str,
                models: list[str]) -> list[str]:
    """Generate a minimal but useful tag list."""
    tags = set()

    # Workload tag
    tags.add(workload)

    # Model slug tags
    model_slug_map = {
        "Cosmos Predict 2": "predict-2",
        "Cosmos Predict 2.5": "predict-2-5",
        "Cosmos Transfer 1": "transfer-1",
        "Cosmos Transfer 2.5": "transfer-2-5",
        "Cosmos Reason 1": "reason-1",
        "Cosmos Reason 2": "reason-2",
        "Cosmos Curator": "curator",
    }
    for m in models:
        slug = model_slug_map.get(m)
        if slug:
            tags.add(slug)

    # Use-case derived tags (lowercased, hyphenated, 2+ chars)
    if use_case:
        words = re.sub(r"[^a-z0-9 ]", "", use_case.lower()).split()
        stop = {"and", "the", "a", "an", "of", "for", "in", "on", "with", "to"}
        meaningful = [w for w in words if w not in stop and len(w) > 2]
        if meaningful:
            tags.add("-".join(meaningful[:4]))

    # Path-based domain hints
    path_lower = rel_path.lower()
    domain_hints = [
        ("its", "intelligent-transportation"),
        ("intelligent-transportation", "intelligent-transportation"),
        ("warehouse", "warehouse"),
        ("robotics", "robotics"),
        ("gr00t", "robotics"),
        ("av_", "autonomous-driving"),
        ("autonomous", "autonomous-driving"),
        ("smart_city", "smart-city"),
        ("sports", "sports"),
        ("wafer", "semiconductor"),
        ("biotrove", "biology"),
        ("worker_safety", "safety"),
        ("carla", "simulation"),
        ("vss", "video-search"),
        ("intbot", "robotics"),
        ("x_mobility", "autonomous-driving"),
        ("distill", "distillation"),
        ("evaluation", "evaluation"),
        ("control_modal", "control-modalities"),
        ("prompt_guide", "prompting"),
    ]
    for hint, tag in domain_hints:
        if hint in path_lower:
            tags.add(tag)
            break

    # Synthetic data is super common, tag it if present
    if "synthetic" in (use_case or "").lower() or "sdg" in path_lower:
        tags.add("synthetic-data-generation")

    return sorted(tags)


# ---------------------------------------------------------------------------
# Frontmatter builder
# ---------------------------------------------------------------------------

def build_frontmatter(
    models: list[str],
    workload: str,
    tags: list[str],
    hardware_min: str,
    published_date: str,
    authors: list[str],
    use_case: Optional[str],
) -> str:
    lines = ["---"]
    # Required
    if len(models) == 1:
        lines.append(f'cosmos_model: ["{models[0]}"]')
    else:
        lines.append("cosmos_model:")
        for m in models:
            lines.append(f'  - "{m}"')
    lines.append(f"cosmos_workload: {workload}")
    if len(tags) == 1:
        lines.append(f'cosmos_tags: ["{tags[0]}"]')
    else:
        lines.append("cosmos_tags:")
        for t in tags:
            lines.append(f'  - "{t}"')
    lines.append(f'cosmos_hardware_min: "{hardware_min}"')
    lines.append(f"cosmos_published_date: {published_date}")
    # Optional
    if authors:
        lines.append("cosmos_authors:")
        for a in authors:
            lines.append(f'  - "{a}"')
    if use_case:
        lines.append(f'cosmos_use_case: "{use_case}"')
    lines.append("---")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Process a single file
# ---------------------------------------------------------------------------

def process_file(file_path: Path, dry_run: bool) -> dict:
    rel = str(file_path.relative_to(REPO_ROOT))
    content = file_path.read_text(encoding="utf-8")

    # Guard: skip if cosmos frontmatter already present
    if content.startswith("---\n") and "cosmos_model" in content[:500]:
        return {"file": rel, "status": "skipped", "reason": "already has cosmos frontmatter"}

    # Parse metadata table
    table = parse_metadata_table(content)
    models = table["models"]
    workload = table["workload"]
    use_case = table["use_case"]

    # Fallbacks from path
    if not workload:
        workload = workload_from_path(rel)
    if not workload:
        workload = "inference"  # safe default

    if not models:
        m = model_from_path(rel)
        if m:
            models = [m]
        else:
            # Try to find model from content (title line)
            title_match = re.search(r"^#\s+(.+)$", content, re.M)
            if title_match:
                m = normalize_model(title_match.group(1))
                if m:
                    models = [m]

    if not models:
        # Last resort: mark unknown but still apply
        models = ["Cosmos Curator"] if "curator" in rel.lower() else []

    # Authors
    authors = parse_authors(content)

    # Hardware
    hardware_min = hardware_from_setup(file_path, workload)

    # Date from git
    published_date = git_first_date(file_path)

    # Tags
    tags = derive_tags(workload, use_case, rel, models)

    if not models:
        return {"file": rel, "status": "skipped", "reason": "could not determine model"}

    # Build frontmatter
    frontmatter = build_frontmatter(
        models=models,
        workload=workload,
        tags=tags,
        hardware_min=hardware_min,
        published_date=published_date,
        authors=authors,
        use_case=use_case,
    )

    if not dry_run:
        new_content = frontmatter + "\n" + content
        file_path.write_text(new_content, encoding="utf-8")
        status = "applied"
    else:
        status = "dry-run"

    return {
        "file": rel,
        "status": status,
        "models": models,
        "workload": workload,
        "tags": tags,
        "hardware_min": hardware_min,
        "published_date": published_date,
        "authors": authors,
        "use_case": use_case,
        "frontmatter_preview": frontmatter,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Apply AI-friendly YAML frontmatter to Cosmos Cookbook recipes.")
    parser.add_argument("--apply", action="store_true", help="Write files (default is dry-run)")
    parser.add_argument("--file", help="Process only this specific file (relative or absolute)")
    args = parser.parse_args()

    dry_run = not args.apply

    if dry_run:
        print("DRY RUN — no files will be modified. Pass --apply to write.\n")

    # Load manifest
    if not MANIFEST_PATH.exists():
        print(f"ERROR: manifest not found at {MANIFEST_PATH}")
        print("Run: python scripts/check-ai-friendly.py")
        sys.exit(1)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    if args.file:
        target = Path(args.file)
        if not target.is_absolute():
            target = REPO_ROOT / target
        target_files = [str(target)]
        safe_files = [
            f for f in manifest["files"]
            if Path(f["path"]) == target
        ]
        if not safe_files:
            # Not in manifest — check manually
            safe_files = [{"path": str(target), "safe_to_apply_frontmatter": True}]
    else:
        safe_files = [f for f in manifest["files"] if f.get("safe_to_apply_frontmatter")]

    print(f"Files to process: {len(safe_files)}")
    print()

    results = []
    applied = 0
    skipped = 0
    errors = 0

    for entry in safe_files:
        file_path = Path(entry["path"])
        rel = str(file_path.relative_to(REPO_ROOT))
        try:
            result = process_file(file_path, dry_run=dry_run)
            results.append(result)
            status = result["status"]
            if status in ("applied", "dry-run"):
                applied += 1
                icon = "✓" if status == "applied" else "~"
                print(f"  {icon} [{status.upper():<8}] {rel}")
                print(f"           model: {result['models']}")
                print(f"        workload: {result['workload']}")
                print(f"            tags: {result['tags']}")
                print(f"        hardware: {result['hardware_min']}")
                print(f"            date: {result['published_date']}")
                print()
            elif status == "skipped":
                skipped += 1
                print(f"  - [SKIP    ] {rel} — {result['reason']}")
        except Exception as e:
            errors += 1
            err = {"file": rel, "status": "error", "error": str(e)}
            results.append(err)
            print(f"  ! [ERROR   ] {rel} — {e}")

    print()
    print("=" * 60)
    verb = "applied" if not dry_run else "would apply"
    print(f"  {verb}: {applied}  |  skipped: {skipped}  |  errors: {errors}")
    print("=" * 60)

    # Write log
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_data = {
        "dry_run": dry_run,
        "total_processed": len(safe_files),
        "applied": applied,
        "skipped": skipped,
        "errors": errors,
        "results": results,
    }
    with open(LOG_PATH, "w") as f:
        json.dump(log_data, f, indent=2)
    print(f"\nLog written to {LOG_PATH.relative_to(REPO_ROOT)}")

    if dry_run:
        print("\nRun with --apply to write files.")

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
