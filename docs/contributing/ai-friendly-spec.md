# AI-Friendly Recipe Spec

This document describes the conventions that make Cosmos Cookbook recipes
machine-readable: YAML frontmatter for structured metadata, `llms.txt` for
LLM-optimized navigation, and consistent prose conventions for prerequisites
and code blocks.

---

## YAML Frontmatter

Every recipe `.md` file should open with a YAML frontmatter block. All fields
use the `cosmos_` prefix to avoid conflicts with MkDocs Material reserved keys
(`title`, `description`, `template`, `tags`).

### Schema location

```
scripts/schema/cosmos-recipe-schema.yaml
```

### Required fields

| Field | Type | Description |
|---|---|---|
| `cosmos_model` | list of strings | Canonical model name(s). See `allowed_values` in the schema. |
| `cosmos_workload` | string enum | One of: `inference`, `post-training`, `data-curation`, `end-to-end` |
| `cosmos_tags` | list of strings | Lowercase, hyphen-separated thematic tags. At least one domain tag required. |
| `cosmos_hardware_min` | string | Minimum GPU requirement. Format: `"<count>x <GPU model>"`. |
| `cosmos_published_date` | string | ISO 8601 date: `YYYY-MM-DD`. |

### Optional fields

| Field | Type | Description |
|---|---|---|
| `cosmos_authors` | list of strings | Author display names (mirrors the in-body blockquote). |
| `cosmos_brev_launchable` | boolean | `true` only when a Brev launch button is configured. |
| `cosmos_use_case` | string | Short noun phrase (title case, 2–6 words). |

### Minimal example

```yaml
---
cosmos_model:
  - "Cosmos Predict 2"
cosmos_workload: inference
cosmos_tags:
  - synthetic-data-generation
  - intelligent-transportation
cosmos_hardware_min: "1x A100 80GB"
cosmos_published_date: "2025-06-15"
---
```

### Full example

```yaml
---
cosmos_model:
  - "Cosmos Transfer 2.5"
  - "Cosmos Reason 1"
cosmos_workload: end-to-end
cosmos_tags:
  - synthetic-data-generation
  - autonomous-driving
  - smart-cities
cosmos_hardware_min: "8x H100 80GB"
cosmos_published_date: "2025-07-01"
cosmos_authors:
  - "Aidan Ladenburg"
  - "Adityan Jothi"
cosmos_brev_launchable: false
cosmos_use_case: "Photorealistic SDG for Traffic Scenarios"
---
```

A complete annotated example is at `scripts/schema/example-frontmatter.md`.

---

## llms.txt

`docs/llms.txt` is a machine-readable index of the cookbook following the
[llms.txt spec](https://llmstxt.org/) (Willison / Answer.AI). LLMs and AI
coding tools can fetch this single file to understand the full cookbook
structure without crawling every page.

### Format

```
# Cosmos Cookbook

> One-paragraph description of the project.

## Section Name
- [Recipe Title](docs/path/to/recipe.md): One-sentence description.
- [Recipe Title](docs/path/to/recipe.md): One-sentence description.

## Another Section
...
```

Sections in output order:
1. Getting Started
2. Inference Recipes
3. Post-Training Recipes
4. Data Curation Recipes
5. End-to-End Workflows
6. Core Concepts (with subsections per concept category)
7. Gallery
8. Reference

### How it is generated

```bash
python scripts/generate-llms-txt.py
```

The generator (`scripts/generate-llms-txt.py`) is stdlib-only (Python 3.8+,
no third-party dependencies). It:

1. Reads each top-level `SUMMARY.md` for Inference, Post-Training, Data
   Curation, End-to-End, and Getting Started sections.
2. For Core Concepts, reads each subsection `SUMMARY.md` under
   `docs/core_concepts/`.
3. For each linked `.md` file, extracts:
   - The H1 heading as the display title.
   - The first non-blockquote, non-table, non-list prose paragraph as the
     description; falls back to the first sentence of the `## Overview`
     section if no opening paragraph is found.
4. Writes `docs/llms.txt`.

### When to regenerate

Regenerate `docs/llms.txt` whenever:

- A new recipe or concept page is added to a `SUMMARY.md`.
- An existing recipe's H1 title or opening description paragraph changes
  substantially.
- A `SUMMARY.md` entry is removed or renamed.

The file is committed to the repo so that it is available at
`https://raw.githubusercontent.com/nvidia-cosmos/cosmos-cookbook/main/docs/llms.txt`
without requiring a live docs build.

---

## Prerequisites Section

Every recipe that requires runtime execution (inference, post-training,
data-curation, end-to-end) must include a `## Prerequisites` section before
any code blocks. It must cover three sub-areas:

### Hardware

State the minimum GPU requirement explicitly. Example:

```markdown
### Hardware

- 1x NVIDIA A100 80 GB (or H100 equivalent)
- At least 64 GB system RAM
- 200 GB free disk space for checkpoints and outputs
```

### Software

List required tools and versions as a table or bullet list. Example:

```markdown
### Software

| Package | Version |
|---|---|
| Python | 3.10+ |
| CUDA | 12.1+ |
| Docker | 24.0+ (optional) |
```

### Accounts and Access

Call out any service accounts, API keys, or NGC access required. Example:

```markdown
### Accounts and Access

- NVIDIA NGC account — required to pull model checkpoints
  ([sign up](https://ngc.nvidia.com/))
- Hugging Face account — required if the recipe uses HF model hub downloads
```

---

## Code Block Labels

Use the correct language identifier on every fenced code block. This enables
syntax highlighting in the docs site and correct parsing by AI tools.

| Content | Label |
|---|---|
| Shell commands, CLI invocations | ` ```bash ` |
| Python source code | ` ```python ` |
| YAML configuration files | ` ```yaml ` |
| JSON payloads or config | ` ```json ` |
| Plain output / logs (no highlighting) | ` ```text ` |
| Dockerfile | ` ```dockerfile ` |

Do not use ` ```shell ` or ` ```sh ` — prefer ` ```bash ` for consistency
across the repo.

Single-command examples that show a prompt character should still use ` ```bash `:

```bash
python scripts/generate-llms-txt.py
```

Multi-step sequences should be a single ` ```bash ` block with comments, not
separate blocks interleaved with prose, so that readers can copy-paste the
entire sequence:

```bash
# Step 1: clone and enter the repo
git clone https://github.com/nvidia-cosmos/cosmos-cookbook.git
cd cosmos-cookbook

# Step 2: install dependencies
just install

# Step 3: serve docs locally
just serve-external
```
