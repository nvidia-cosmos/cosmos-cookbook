# Cosmos Reason — Deploy Layer

This directory contains deployment configs and scripts for Cosmos Reason recipes.
It sits alongside the existing recipe documentation without modifying it.

## Structure

```
deploy/
  reason2/
    worker_safety/
      brev.yaml     ← one-click Brev Launchable (Cosmos-Reason2-2B)
      demo.sh       ← headless execution for Horde or any Linux GPU machine
    vss/
      README.md     ← pointer to official VSS Brev Launchable
    intbot_showcase/
      brev.yaml     ← one-click Brev Launchable (Cosmos-Reason2-8B)
      demo.sh       ← headless single-inference demo
```

## Quick start

### Brev (external / customer-accessible)

Click the Brev badge in the recipe's `inference.md`, or launch directly:

- **worker_safety** — `deploy/reason2/worker_safety/brev.yaml`
- **intbot_showcase** — `deploy/reason2/intbot_showcase/brev.yaml`
- **vss** — see `deploy/reason2/vss/README.md` for the official VSS Brev path

### Horde / local GPU (internal rehearsal)

```bash
export HF_TOKEN=hf_...
bash deploy/reason2/worker_safety/demo.sh
```

```bash
export HF_TOKEN=hf_...
bash deploy/reason2/intbot_showcase/demo.sh
```

## Requirements

| Recipe | Model | Min VRAM | Driver |
|---|---|---|---|
| worker_safety | Cosmos-Reason2-2B | 40 GB | ≥ 555 (CUDA 12.8) |
| intbot_showcase | Cosmos-Reason2-8B | 80 GB | ≥ 555 (CUDA 12.8) |
| vss | Cosmos-Reason2-8B | 80 GB (8xH100 recommended) | per VSS docs |

## HuggingFace access

Both models are gated. Accept the NVIDIA Open Model License before running:
- `nvidia/Cosmos-Reason2-2B`: huggingface.co/nvidia/Cosmos-Reason2-2B
- `nvidia/Cosmos-Reason2-8B`: huggingface.co/nvidia/Cosmos-Reason2-8B
