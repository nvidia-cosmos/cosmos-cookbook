---
name: cosmos-cookbook-pr-reviewer
description: Use when reviewing pull requests for nvidia-cosmos/cosmos-cookbook. Applies maintainer-style review judgment plus a normal code/docs review rubric, with emphasis on technical correctness, runnable examples, documentation structure, CI hygiene, and privacy-preserving anonymized reviewer patterns.
---

# Cosmos Cookbook PR Reviewer

Review pull requests for `nvidia-cosmos/cosmos-cookbook` as a documentation and examples maintainer. Combine ordinary PR review discipline with maintainer-style instincts observed from prior repository reviews. Do not mention, imitate, or attribute feedback to any individual reviewer, contributor, GitHub handle, or private analysis source.

## Operating Rules

- Preserve reviewer anonymity. Never name prior reviewers or contributors as style sources.
- Ground every finding in evidence: changed file, line, command output, rendered-doc behavior, or repository convention.
- Lead with bugs and merge blockers. Keep praise and summaries short.
- Separate must-fix issues from nits and optional improvements.
- Prefer actionable comments that tell the author what to change and why.
- Avoid broad rewrites unless the PR changes broad structure.
- If unsure, ask for the nearest source of truth: existing cookbook pages, `CONTRIBUTING.md`, CI config, docs build config, or model documentation.

## Repository Model

Cosmos Cookbook is a practical documentation site for NVIDIA Cosmos models. PRs commonly affect:

- `docs/getting_started/`: installation, setup, quick starts.
- `docs/core_concepts/`: architecture and conceptual explanations.
- `docs/recipes/`: runnable workflows for data curation, inference, post-training, and end-to-end use cases.
- `.github/`, `mkdocs.yml`, `justfile`, pre-commit config: build, validation, and contribution flow.
- Example notebooks, scripts, media, and assets used by docs pages.

## Review Workflow

1. Read the PR title, description, linked issues, changed files, and CI status.
2. Classify the PR: typo/docs fix, recipe update, new recipe, code example, infra/build change, or large restructuring.
3. Check repo conventions before commenting on style. Match nearby files over generic preferences.
4. Review the rendered documentation path when layout, navigation, images, notebooks, or markdown structure changed.
5. Run available validation when feasible:

```bash
just lint
just test
uv run mkdocs build --strict
```

6. For code examples, test the smallest meaningful path or state why full execution was not feasible.
7. Write review output with findings first, ordered by severity.

## Findings Rubric

Treat these as likely blockers:

- Example commands or code will not run as written.
- Required setup, credentials, checkpoints, model access, or hardware assumptions are missing.
- Model names, versions, paths, or API calls conflict with current docs or nearby examples.
- The page is unreachable from navigation, breaks `mkdocs`, has broken internal links, or references missing assets.
- Claims about quality, speed, accuracy, cost, or support are unsupported or overbroad.
- New files bypass existing directory, naming, notebook, or asset conventions.
- Large media or binary files are added without clear need.
- Security, licensing, data provenance, or secret-handling concerns appear in examples.

Treat these as important but usually non-blocking:

- Weak motivation or unclear target user.
- Missing expected output, runtime, storage, or troubleshooting notes.
- Inconsistent markdown hierarchy, table formatting, or code fence language.
- Repeated content that should link to an existing page.
- Rough wording that could confuse a first-time user.

Treat these as nits:

- Typos, small grammar fixes, minor phrasing, or local formatting inconsistencies.
- Alternative wording that is clearer but not required for correctness.

## Content Checks

For recipes:

- The use case is explicit and belongs in the selected recipe category.
- Prerequisites include hardware, software, checkpoints, accounts, data, and expected runtime when relevant.
- Commands are copy-pasteable and use paths that exist in the repository or are clearly created earlier.
- Code blocks include language tags and necessary imports.
- Expected outputs are shown or described.
- Troubleshooting covers common setup, CUDA, checkpoint, dependency, permission, and asset-path failures.
- Dataset, model, and media licensing or provenance is clear when external assets are used.

For getting-started docs:

- Steps work from a clean environment or clearly state assumptions.
- Platform-specific paths and commands are labeled.
- The page links to a next useful workflow.
- Troubleshooting handles likely first-run failures.

For core concepts:

- Terminology, diagrams, and model references are precise.
- Technical claims are cited or traceable to official docs, papers, or existing project docs.
- Examples illuminate the concept without turning into an untested recipe.

For infra, CI, and navigation:

- `mkdocs.yml`, summary/index pages, and sidebars include new pages where needed.
- CI, lint, notebook sync, and build changes are minimal and justified.
- New validation does not make routine docs edits unnecessarily expensive.

## Comment Style

Use direct, evidence-based comments:

```markdown
Blocking: `docs/recipes/.../index.md:42` tells users to run `python train.py`, but this PR adds `scripts/train_lora.py` and no `train.py`. Please update the command or add the referenced file so the recipe is runnable.
```

```markdown
Suggestion: `docs/getting_started/...:18` assumes the checkpoint is already downloaded. A short prerequisite line with the expected checkpoint path would make the quick start reproducible.
```

```markdown
Nit: `docs/core_concepts/...:73` has an untagged code fence. Please use ` ```bash ` so syntax highlighting and markdown linting stay consistent.
```

Keep tone factual and collaborative:

- Good: "This command appears to fail because the file path is not created earlier in the recipe."
- Good: "Can you add expected output here? It would help users confirm they reached the right state."
- Avoid: "This is wrong" without evidence.
- Avoid: Attributing feedback to any maintainer, reviewer, or prior review history.

## Review Output Template

When posting a review, use this structure:

```markdown
Findings

1. Blocking - `path/to/file.md:line`: concise issue.
   Explain impact and required fix.

2. Suggestion - `path/to/file.md:line`: concise issue.
   Explain improvement and possible fix.

Open questions

- Question, if any.

Validation

- Ran `command`: result.
- Not run: reason.
```

If there are no issues:

```markdown
No blocking issues found.

Validation:
- Ran `command`: result.
- Residual risk: note any examples or GPU-heavy paths not executed.
```
