# post-training

## What This Recipe Does
<!-- One sentence: what problem it solves and how. -->

## Model
<!-- Exact model identifier, e.g. a HuggingFace model ID. -->

## Entry Points
<!-- Exact commands to launch the training job, in order. -->
```bash
[command to launch training]
```

## Data Source
<!--
  Access: [Public | Gated — requires HuggingFace agreement at <URL> | Restricted — requires separate application]
  Size: ~XGB
  License: [e.g. Apache-2.0 / CC-BY-4.0 / non-commercial]
-->
**Access:** [Public | Gated | Restricted]
**Size:** ~XGB
**License:** [license name]

```bash
huggingface-cli download [org/dataset-name] --repo-type dataset --local-dir ./data
```

## Compute Requirements
<!--
  Claude Agent: Before running any entry points, check whether the user has sufficient
  local GPU resources. If not, ask:
    "This recipe requires at least the compute listed here. You can provision
     a matching instance on NVIDIA Brev. Do you have a Brev API token? If so, please
     paste it and I will set up the environment for you."
  Note: post-training jobs are long-running. Claude should launch the job and inform
  the user how to monitor progress — not wait for completion.
-->
<!-- Describe the minimum and recommended compute needed to run this recipe. -->

## Dependencies

```
torch>=2.0
[additional packages]
```

## Required Environment Variables
<!--
  List variable names and descriptions only. Never write credential values here.
  All values must be set in the user's system environment before running the recipe.
-->
| Variable        | Description                                        |
|-----------------|----------------------------------------------------|
| `HF_TOKEN`      | HuggingFace token — set via `export HF_TOKEN=...` |

## Setup Prerequisites
<!-- Blocking steps that must complete before entry points will work. -->
- [ ] `huggingface-cli login` completed
- [ ] Training data downloaded and preprocessed

## Key Files

| File               | Role                             |
|--------------------|----------------------------------|
| `train.py`         | Main training entry point        |
| `config.yaml`      | Training configuration           |
| `setup.md`         | Full environment setup guide     |

## Code Structure
<!-- Name real functions/classes and describe how they connect. -->
- `load_data()` — description
- `build_model()` — description
- `train()` — description
- `save_checkpoint()` — description

## Expected Output

```
[describe what a successful training run produces, e.g. checkpoint location, final metrics]
```

## Monitoring
<!-- How to check training progress after the job is launched. -->
```bash
[command to monitor training, e.g. tail -f train.log]
```

## Gotchas
<!-- Things that will waste a contributor's time if they don't know them. -->
-
