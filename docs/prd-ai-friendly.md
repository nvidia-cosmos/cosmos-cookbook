# PRD: Cosmos Cookbook — AI-Friendly Enhancement

**For:** TJ Galda
**From:** Alex Sotelo / Team Claude
**Date:** 2026-03-04
**Status:** Phase 1 in execution. Phases 2–3 scoped below.

---

## Why This Matters

The Cosmos Cookbook is NVIDIA's primary reference layer for practitioners building with Cosmos World Foundation Models. Right now, it is optimized for humans reading a browser — not for AI agents navigating it programmatically.

That distinction matters more than it might seem.

When a developer uses an AI coding assistant, a Claude-powered research agent, or an internal tool to ask "how do I fine-tune Cosmos Reason 2 for warehouse safety?"  — the agent does not browse the site like a human does. It looks for machine-readable structure. It parses metadata. It follows `llms.txt` files. It reads YAML frontmatter to understand what a document is before it reads the document itself. If none of that structure exists, the cookbook becomes invisible to the agent layer — and every agent-assisted Cosmos workflow becomes harder to discover, replicate, and share.

The cookbook is also a community flywheel. Partners and developers submit recipes because they want their work cited and used. If the submission process is manual and friction-heavy, fewer recipes get submitted. If submitted recipes can't be discovered by AI tools, the ones that exist have lower impact.

**Making the cookbook AI-friendly is not a documentation polish task. It is an accessibility upgrade for the next generation of users who navigate documentation through agents, not browsers.**

---

## Goals

| Goal | Success Metric |
|------|----------------|
| All existing recipes have machine-readable metadata | 100% of recipe files have valid `cosmos_*` YAML frontmatter |
| Cookbook is indexable by LLMs and AI agents | `llms.txt` index exists at repo root; all 30+ recipes indexed |
| Future submissions arrive AI-friendly by default | Templates include frontmatter schema; new recipes pass validator on first submission |
| Practitioners can go from "I want to run this" to "running" in under 5 minutes | Brev Launchables on Cosmos Reason 2 inference recipes |
| Submission pipeline reduces manual review burden by 80% | Auto-improvement pass on new PRs improves or validates frontmatter, structure, and links before human review |
| Cookbook is surfaceable and actionable from any MCP-capable AI tool | Functional MCP server with search, fetch, and launch capabilities |

---

## Phase 1 — Make the Existing Cookbook AI-Friendly

**Status: IN EXECUTION as of 2026-03-04. Estimated completion: 1–2 days.**

Team Claude has already begun executing Phase 1 on branch `recipe/ai-friendly-sprint1` in the `nv-asotelo/cosmos-cookbook` fork. This section documents the scope and current state.

### What "AI-Friendly" Means (Our Definition)

**Layer 1 — Structured Metadata (YAML Frontmatter)**

Every recipe `.md` file gets a YAML block at the top containing:

```yaml
---
cosmos_model: ["Cosmos Reason 2"]
cosmos_workload: inference
cosmos_tags: ["inference", "warehouse", "safety"]
cosmos_hardware_min: "1x A100 80GB"
cosmos_published_date: 2026-01-29
cosmos_authors:
  - "Jane Smith"
cosmos_use_case: "Worker Safety Compliance"
---
```

This makes every recipe machine-readable without opening the file. An AI agent or search index can parse the corpus in seconds and answer questions like "what inference recipes run on a single A100?" or "which Cosmos Transfer 2.5 recipes involve synthetic data?"

**Layer 2 — LLM Discoverability Index (`llms.txt`)**

A `llms.txt` file at the repo root — following the emerging [llms.txt spec](https://llmstxt.org/) — gives LLMs a hierarchical, flat-text map of all cookbook content with one-line descriptions. It is to the cookbook what `sitemap.xml` is to Google: a structured declaration of what exists.

**Layer 3 — Execution Readiness**

- All code blocks labeled with language (enables syntax highlighting and agent-parseable detection of runnable vs. conceptual code)
- `## Prerequisites` section on all recipes (Hardware / Software / Accounts)
- Templates updated so new submissions are AI-friendly on first draft

### Current Progress

| Deliverable | Status |
|-------------|--------|
| AI-friendly validator script (`check-ai-friendly.py`) | ✅ Done — scans all docs/, produces JSON manifest |
| YAML frontmatter schema (`cosmos-recipe-schema.yaml`) | ✅ Done |
| Batch frontmatter applicator (`apply-ai-friendly.py`) | ✅ Done |
| Frontmatter applied to existing recipes | ✅ Done — 35 of 37 eligible recipe files |
| `llms.txt` generator script | ✅ Done |
| `llms.txt` at repo root | ✅ Done — 30+ recipes indexed |
| Templates updated with frontmatter + Prerequisites | ✅ Done |
| Full validation pass — 0 broken internal links | ✅ Done |
| Brev Launchables on Cosmos Reason 2 inference recipes | 🔄 In progress (CBK-006, CBK-007) |
| Remaining 28 files (setup.md companions, generic overviews) | ⏳ Manual review — lower priority, no model determinable from content |

**Audit results (2026-03-04):**

- 63 recipe/concept files scanned
- 35 now have `cosmos_*` YAML frontmatter (was 0)
- 0 broken internal links
- All 30+ recipes indexed in `llms.txt`

### What Requires Human Action Before Merge

- Omar Laymoun's PR review process needs to be ready to receive the PR
- Alex to promote draft PR to open when Omar is set up

---

## Phase 2 — Automated Ingestion & AI-Improvement Pipeline

**Status: Not started. Ready to scope and execute immediately after Phase 1 ships.**

### The Problem

The current contribution flow is:
1. Contributor forks the repo
2. Contributor writes a recipe (no structured template, no automated checks)
3. Contributor opens a PR
4. NVIDIA reviewer manually checks structure, formatting, frontmatter, links
5. Reviewer requests changes → contributor fixes → repeat

This puts the quality burden entirely on the human reviewer. It also means every PR that arrives without proper structure takes reviewer time to remediate — time that scales linearly with submission volume.

### The Solution

A contributor-facing automation pipeline that:
1. Accepts a PR with a new recipe
2. Runs the AI-friendly validator automatically
3. If the recipe is missing frontmatter, auto-generates it using Claude (same approach as `apply-ai-friendly.py`, but for incoming PRs)
4. Checks for broken links, unlabeled code blocks, missing Prerequisites section
5. Opens a follow-up commit or comment on the PR with the improvements applied or suggestions given
6. Labels the PR `ai-friendly-auto-reviewed` so the human reviewer knows the mechanical checks are done

The human reviewer's job becomes content quality, not structure policing.

### Scope

| Component | Description |
|-----------|-------------|
| **GitHub Actions workflow** | Triggers on `pull_request` events targeting `docs/recipes/**` |
| **Frontmatter synthesizer** | Calls the existing `apply-ai-friendly.py` logic on new/modified recipe files |
| **Claude API integration** | For recipes with insufficient in-file metadata, prompt Claude to synthesize frontmatter from recipe content (title, body, code blocks) |
| **Improvement committer** | If auto-improvements are available, commits them back to the PR branch (or opens a companion PR) |
| **PR comment summary** | Posts a structured comment: what was found, what was auto-fixed, what still needs manual attention |
| **Validation gate** | Optional: require `ai-friendly-auto-reviewed` label to pass before merge |

### Open Questions (Need TJ Direction)

1. **Auto-commit or comment-only?** — Should the bot commit improvements directly to contributor branches, or post suggestions as a PR comment for the contributor to apply? Auto-commit is faster but may surprise contributors. Comment-only is safer but leaves work on the contributor.

2. **Mandatory gate or advisory?** — Should recipes without valid frontmatter be blocked from merge until fixed, or is the automation advisory-only for now?

3. **Scope of "improvement"** — Does Phase 2 improvement include suggesting better tags, improving the recipe description using Claude, or is it strictly structural (frontmatter + links + labels)?

### Estimated Effort

2–3 days with Team Claude. The validator and frontmatter synthesizer already exist (Phase 1). The new work is the GitHub Actions integration and the Claude API call for synthesis from raw content.

---

## Phase 3 — MCP Skills for Cookbook Utility

**Status: Not started. Scoped here for prioritization.**

### The Problem

Even with AI-friendly recipes and an llms.txt index, the cookbook is still a passive resource. Practitioners using AI agents have to navigate to it, find the right recipe, and manually translate what they find into commands, configs, and launch steps.

An MCP server for the cookbook turns it into an active tool — something an agent can query, invoke, and act on without leaving the workflow.

### Proposed MCP Skills

#### `cosmos-cookbook/search`
**What it does:** Semantic search over all recipes by model, workload, use case, or free-text query.
**Example:** *"Find me a recipe for fine-tuning Cosmos Reason 2 on warehouse safety data"* → returns matching recipes with summaries, paths, and hardware requirements.
**Why it matters:** Makes the cookbook queryable from any MCP-capable agent (Claude, Cursor, internal tools). Reduces "I didn't know this existed" attrition.
**Implementation:** BM25 or embedding search over `llms.txt` + parsed frontmatter. Stateless. No GPU required.

#### `cosmos-cookbook/get-recipe`
**What it does:** Fetch full recipe content (structured frontmatter + body text) by recipe path or ID.
**Example:** Agent finds a matching recipe via `search`, then calls `get-recipe` to read the full implementation guide.
**Why it matters:** Enables agents to read and execute cookbook content programmatically. Foundation for `run-recipe`.
**Implementation:** File read + frontmatter parse. Simple.

#### `cosmos-cookbook/run-recipe`
**What it does:** Given a recipe path, generates a ready-to-run launch configuration — a Brev Launchable config, Docker command, or `uv run` invocation — pre-filled with the correct model, hardware spec, and script entry points from the recipe.
**Example:** *"Give me the launch command for the Cosmos Reason 2 warehouse safety inference recipe"* → returns a Brev config or Docker compose that an agent or human can execute in one step.
**Why it matters:** Closes the gap between "I found a recipe" and "I'm running it." This is the highest-leverage skill for practitioners who want to move fast.
**Implementation:** Reads recipe frontmatter + `cosmos_brev_launchable` flag, generates config. Integrates with Brev API for live launch links.

#### `cosmos-cookbook/submit-recipe`
**What it does:** Guided recipe submission workflow. Takes a description of what the contributor built, returns a partially-filled recipe template with `cosmos_*` frontmatter pre-populated based on the description.
**Example:** Contributor describes their Cosmos Transfer 2.5 pipeline for urban scenario mapping → gets back a template with model, workload, tags, and hardware fields pre-filled for review.
**Why it matters:** Makes Phase 2 contributor-facing. Contributors using Claude or Cursor get a head start on submission quality before they open a PR.
**Implementation:** Claude API call with recipe description as input + schema-aware output generation.

#### `cosmos-cookbook/validate-recipe`
**What it does:** Runs the AI-friendly validator on a local recipe file or file content. Returns structured JSON of issues (missing frontmatter fields, broken links, unlabeled code blocks, missing Prerequisites).
**Example:** Before opening a PR, a contributor runs validation from their editor via MCP → gets a checklist of what to fix.
**Why it matters:** Moves quality checks left — into the contributor's workflow before the PR, not the reviewer's workflow after. Pairs with `submit-recipe` for a full pre-submission loop.
**Implementation:** Wraps `check-ai-friendly.py` logic as a callable MCP skill.

### Recommended Sequencing

| Priority | Skill | Effort | Impact |
|----------|-------|--------|--------|
| P1 | `cosmos-cookbook/search` | 1 day | High — makes cookbook discoverable from agents |
| P1 | `cosmos-cookbook/get-recipe` | 0.5 days | High — enables agent read/execute loop |
| P2 | `cosmos-cookbook/run-recipe` | 2 days | High — closes the "found it → running it" gap |
| P2 | `cosmos-cookbook/validate-recipe` | 0.5 days | Medium — reuses Phase 1 work |
| P3 | `cosmos-cookbook/submit-recipe` | 1.5 days | Medium — depends on Phase 2 automation being live |

### What's Needed to Start Phase 3

1. Phase 1 merged to upstream (so MCP server has a stable corpus to index)
2. Decision on hosting: MCP server as a standalone FastAPI service, or as a GitHub Action / cloud function? A self-hosted MCP server means NVIDIA controls the search index and can update it on recipe merge. Cloud function is lighter but requires API access.
3. NGC API key or Brev API access for `run-recipe` Launchable generation

---

## Dependencies and Risks

| Risk | Mitigation |
|------|------------|
| Phase 1 branch not merging before GTC | Phase 1 is already in execution. Branch is sandboxed in `nv-asotelo` fork ready to PR upstream when Omar's process is ready. If merge is blocked, the work still exists and can be demonstrated from the fork. |
| Contributor community surprised by auto-commit in Phase 2 | Default to comment-only in Phase 2 initial rollout. Add "accept improvements" workflow after community feedback. |
| MCP server hosting not provisioned before Phase 3 | Phase 3 can run as a local MCP server for internal use while hosting is arranged. Doesn't require production infrastructure to demo. |
| `polyfill.io` CDN dependency in `mkdocs.yml` | Known supply chain risk (was compromised in 2024). Flagged. Not blocking Phase 1. Needs fix in a follow-on PR — replace with self-hosted polyfill or jsdelivr CDN. |

---

## Summary: What We're Asking For

| Phase | Ask |
|-------|-----|
| **Phase 1** | Approve and merge the branch when Omar's process is ready. Work is done. |
| **Phase 2** | Direction on auto-commit vs. comment-only, and whether validation is a merge gate. Team can execute in ~2–3 days once scoped. |
| **Phase 3** | Decision on MCP server hosting model + which P1/P2 skills to prioritize first. Team can start with `search` + `get-recipe` immediately after Phase 1 ships. |

---

*This document lives in the repo at `docs/prd-ai-friendly.md` on branch `recipe/ai-friendly-sprint1`. It will be updated as phases progress.*
