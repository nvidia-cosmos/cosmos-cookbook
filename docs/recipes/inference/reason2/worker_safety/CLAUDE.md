# Worker Safety — Cosmos Reason 2 Inference

## Model

`nvidia/Cosmos-Reason2-2B`

## Data Source

<!--
  Access: Gated — requires accepting the NVIDIA Open Model License
  at https://huggingface.co/nvidia/Cosmos-Reason2-2B
  Size: ~4GB
  License: NVIDIA Open Model License
-->

**Access:** Gated — requires accepting the NVIDIA Open Model License at https://huggingface.co/nvidia/Cosmos-Reason2-2B
**Size:** ~4GB
**License:** NVIDIA Open Model License

```bash
huggingface-cli download nvidia/Cosmos-Reason2-2B --repo-type model --local-dir ./models/Cosmos-Reason2-2B
```

## Compute Requirements

1x NVIDIA GPU with 40GB+ VRAM (tested on A100 80GB / H100 80GB), CUDA 12.8+ (driver >= 555)

## Execution

Full setup and run sequence for Claude or any agent on a headless Linux machine (no browser required).

### Pre-flight checks

```bash
nvidia-smi
```

Confirm GPU is visible and CUDA version is 13.x. Confirm at least 40GB VRAM is free.

### Step 1 — System dependencies

```bash
sudo apt-get update -y
sudo apt-get install -y curl ffmpeg git git-lfs
git lfs install
```

### Step 2 — Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### Step 3 — Clone cosmos-reason2

```bash
git clone https://github.com/nvidia-cosmos/cosmos-reason2.git $HOME/cosmos-reason2
cd $HOME/cosmos-reason2
git lfs pull
```

### Step 4 — HuggingFace authentication (interactive — requires HF token)

```bash
uvx hf auth login
```

Enter your HF token. Token must have access to `nvidia/Cosmos-Reason2-2B` (accept the NVIDIA Open Model License at huggingface.co first).

### Step 5 — Create Python environment (CUDA 12.8)

```bash
cd $HOME/cosmos-reason2
uv sync --extra cu128
source .venv/bin/activate
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

Expected output: `CUDA available: True`

### Step 6 — Install recipe dependencies

```bash
pip install -U fiftyone jupyterlab ipykernel
python -m ipykernel install --user --name cosmos-reason2 --display-name "Python (cosmos-reason2)"
```

### Step 7 — Download model weights

```bash
huggingface-cli download nvidia/Cosmos-Reason2-2B \
  --repo-type model \
  --local-dir $HOME/cosmos-reason2/models/Cosmos-Reason2-2B
```

### Step 8 — Copy recipe files into cosmos-reason2 workspace

```bash
COOKBOOK=$HOME/cosmos-cookbook   # adjust if cloned elsewhere
cp $COOKBOOK/docs/recipes/inference/reason2/worker_safety/worker_safety.py \
   $HOME/cosmos-reason2/worker_safety.py
cp -r $COOKBOOK/docs/recipes/inference/reason2/worker_safety/assets \
   $HOME/cosmos-reason2/assets
```

### Step 9 — Run inference headlessly (no browser required)

The recipe ends with a FiftyOne visualization that blocks on a headless machine.
Run with a no-op patch to skip the UI and exit cleanly after inference completes:

```bash
cd $HOME/cosmos-reason2
source .venv/bin/activate

python - <<'EOF'
import fiftyone as fo

# Patch FiftyOne app launch to be a no-op in headless mode
_noop_session = type("Session", (), {"wait": lambda self: None})()
fo.launch_app = lambda *args, **kwargs: _noop_session

# Run the recipe
exec(open("worker_safety.py").read())

# Save results summary to JSON
import json
dataset = fo.load_dataset("safe-unsafe-worker-behavior")
results = []
for sample in dataset:
    results.append({
        "filepath": sample.filepath,
        "ground_truth": sample.get_field("ground_truth").label if sample.get_field("ground_truth") else None,
        "safety_label": sample.get_field("safety_label").label if sample.get_field("safety_label") else None,
        "cosmos_error": sample.get_field("cosmos_error"),
    })
with open("inference_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Done. {len(results)} samples. Results saved to inference_results.json")
EOF
```

### Step 10 — Verify results

```bash
cat inference_results.json | python -c "
import json, sys
data = json.load(sys.stdin)
errors = [r for r in data if r.get('cosmos_error')]
labeled = [r for r in data if r.get('safety_label')]
print(f'Total samples: {len(data)}')
print(f'Successfully classified: {len(labeled)}')
print(f'Errors: {len(errors)}')
if labeled:
    print('Sample result:', labeled[0])
"
```

### Success criteria

- `CUDA available: True` in Step 5
- `Processing complete. Launching App...` printed during Step 9
- `inference_results.json` contains entries with non-null `safety_label` values
- Error count is 0 (or low — occasional JSON parse failures are acceptable)

## Cosmos Metadata

| Field     | Value                                                                                                      |
|-----------|------------------------------------------------------------------------------------------------------------|
| Workload  | inference                                                                                                  |
| Domain    | domain:industrial                                                                                          |
| Technique | technique:reasoning                                                                                        |
| Tags      | inference, reason-2, safety                                                                                |
| Summary   | Zero-shot warehouse safety inspection using Cosmos Reason 2 to classify worker behaviors from video without custom model training. |
