# Example: Cosmos Recipe YAML Frontmatter

Paste the block below at the very top of a recipe `.md` file, before any other content.
The three dashes (`---`) are required YAML delimiters — do not omit them.

## Minimal (required fields only)

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

## Full (all fields)

```yaml
---
cosmos_model:
  - "Cosmos Predict 2"
cosmos_workload: inference
cosmos_tags:
  - synthetic-data-generation
  - intelligent-transportation
  - object-detection
  - sdg
cosmos_hardware_min: "1x A100 80GB"
cosmos_published_date: "2025-06-15"
cosmos_authors:
  - "Charul Verma"
  - "Reihaneh Entezari"
  - "Arihant Jain"
  - "Dharshi Devendran"
  - "Ratnesh Kumar"
cosmos_brev_launchable: true
cosmos_use_case: "Synthetic Data Generation"
---
```

## Notes

- `cosmos_model` must use the exact canonical names defined in
  `scripts/schema/cosmos-recipe-schema.yaml` (`allowed_values` list).
- `cosmos_workload` is an enum: `inference`, `post-training`, `data-curation`, `end-to-end`.
- `cosmos_tags` — use lowercase, hyphen-separated terms. At least one domain tag is expected
  (e.g., `robotics`, `autonomous-driving`, `warehouse`, `intelligent-transportation`).
- `cosmos_published_date` — ISO 8601: `YYYY-MM-DD`. Update when the recipe content changes
  substantially (not for typo fixes).
- `cosmos_brev_launchable` — omit entirely when false; only set to `true` when a Brev
  launch button is wired up in the recipe.
- None of these keys conflict with MkDocs Material reserved keys (`title`, `description`,
  `template`, `tags`) because every field uses the `cosmos_` prefix.

## End-to-end example (multi-model recipe)

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
  - vlm-fine-tuning
cosmos_hardware_min: "8x H100 80GB"
cosmos_published_date: "2025-07-01"
cosmos_authors:
  - "Aidan Ladenburg"
  - "Adityan Jothi"
cosmos_use_case: "Photorealistic SDG for Traffic Scenarios"
---
```
