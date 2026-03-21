#!/bin/bash
# demo.sh — Worker Safety Cosmos Reason 2
# Headless inference for Horde or any Linux GPU machine.
# No browser or JupyterLab required.
#
# Usage:
#   export HF_TOKEN=hf_...
#   bash deploy/reason2/worker_safety/demo.sh
#
# Output: inference_results.json in the current directory

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
COSMOS_REASON2="$HOME/cosmos-reason2"
COOKBOOK="$REPO_ROOT"

# ── Pre-flight ──────────────────────────────────────────────────────────────

echo "=== Pre-flight checks ==="

# GPU check
if ! command -v nvidia-smi &>/dev/null; then
  echo "ERROR: nvidia-smi not found. A CUDA-capable GPU is required."
  exit 1
fi
nvidia-smi --query-gpu=name,memory.free,driver_version --format=csv,noheader
echo ""

# VRAM check (require >= 40000 MiB free)
VRAM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
if [ "$VRAM_FREE" -lt 40000 ]; then
  echo "ERROR: Only ${VRAM_FREE} MiB VRAM free. Cosmos-Reason2-2B requires >= 40000 MiB."
  exit 1
fi
echo "VRAM: ${VRAM_FREE} MiB free ✓"

# HF token check
if [ -z "$HF_TOKEN" ]; then
  echo ""
  echo "HuggingFace token required. Enter your token (hf_...):"
  read -r -s HF_TOKEN
  export HF_TOKEN
fi
echo "HF_TOKEN: set ✓"
echo ""

# ── Step 1: System dependencies ─────────────────────────────────────────────

echo "=== Step 1: System dependencies ==="
if ! command -v ffmpeg &>/dev/null || ! command -v git-lfs &>/dev/null; then
  sudo apt-get update -q
  sudo apt-get install -y -q curl ffmpeg git git-lfs
fi
git lfs install --skip-repo 2>/dev/null || true
echo "System deps ✓"

# ── Step 2: uv ───────────────────────────────────────────────────────────────

echo "=== Step 2: uv ==="
if ! command -v uv &>/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
source "$HOME/.local/bin/env" 2>/dev/null || true
echo "uv $(uv --version) ✓"

# ── Step 3: Clone cosmos-reason2 ─────────────────────────────────────────────

echo "=== Step 3: cosmos-reason2 ==="
if [ ! -d "$COSMOS_REASON2" ]; then
  git clone https://github.com/nvidia-cosmos/cosmos-reason2.git "$COSMOS_REASON2"
  git -C "$COSMOS_REASON2" lfs pull
  echo "Cloned ✓"
else
  echo "Already present ✓"
fi

# ── Step 4: HF login ─────────────────────────────────────────────────────────

echo "=== Step 4: HuggingFace auth ==="
echo "$HF_TOKEN" | huggingface-cli login --token 2>/dev/null || \
  huggingface-cli login --token "$HF_TOKEN"
echo "Authenticated ✓"

# ── Step 5: Python environment ───────────────────────────────────────────────

echo "=== Step 5: Python environment (cu128) ==="
cd "$COSMOS_REASON2"
uv sync --extra cu128 2>&1 | tail -5
source .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
echo "CUDA available ✓"

# ── Step 6: Recipe dependencies ──────────────────────────────────────────────

echo "=== Step 6: Recipe dependencies ==="
pip install -q -U fiftyone jupyterlab ipykernel
python -m ipykernel install --user --name cosmos-reason2 \
  --display-name "Python (cosmos-reason2)" 2>/dev/null
echo "FiftyOne + JupyterLab ✓"

# ── Step 7: Model download ───────────────────────────────────────────────────

echo "=== Step 7: Model download (~4 GB) ==="
MODEL_DIR="$COSMOS_REASON2/models/Cosmos-Reason2-2B"
if [ ! -d "$MODEL_DIR" ]; then
  huggingface-cli download nvidia/Cosmos-Reason2-2B \
    --repo-type model \
    --local-dir "$MODEL_DIR"
  echo "Downloaded ✓"
else
  echo "Already present ✓"
fi

# ── Step 8: Copy recipe files ────────────────────────────────────────────────

echo "=== Step 8: Copy recipe files ==="
RECIPE="$COOKBOOK/docs/recipes/inference/reason2/worker_safety"
cp "$RECIPE/worker_safety.py" "$COSMOS_REASON2/worker_safety.py"
cp -r "$RECIPE/assets" "$COSMOS_REASON2/assets" 2>/dev/null || true
echo "Recipe files in place ✓"

# ── Step 9: Run inference (headless) ─────────────────────────────────────────

echo ""
echo "=== Step 9: Running inference (headless) ==="
echo "This may take 20-30 minutes on first run."
echo ""

cd "$COSMOS_REASON2"

python - <<'PYEOF'
import fiftyone as fo

# Skip FiftyOne app launch in headless mode
_noop = type("S", (), {"wait": lambda self: None})()
fo.launch_app = lambda *a, **kw: _noop

exec(open("worker_safety.py").read())

# Export results
import json
try:
    dataset = fo.load_dataset("safe-unsafe-worker-behavior")
    results = []
    for sample in dataset:
        gt = sample.get_field("ground_truth")
        sl = sample.get_field("safety_label")
        results.append({
            "filepath": sample.filepath,
            "ground_truth": gt.label if gt else None,
            "safety_label": sl.label if sl else None,
            "error": sample.get_field("cosmos_error"),
        })
    with open("inference_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: inference_results.json ({len(results)} samples)")
except Exception as e:
    print(f"WARNING: Could not export results: {e}")
PYEOF

# ── Step 10: Verify ──────────────────────────────────────────────────────────

echo ""
echo "=== Step 10: Results summary ==="
python - <<'PYEOF'
import json, sys
try:
    with open("inference_results.json") as f:
        data = json.load(f)
    errors = [r for r in data if r.get("error")]
    labeled = [r for r in data if r.get("safety_label")]
    print(f"Total samples:          {len(data)}")
    print(f"Successfully classified: {len(labeled)}")
    print(f"Errors:                 {len(errors)}")
    if labeled:
        print(f"\nSample result:")
        r = labeled[0]
        print(f"  File:         {r['filepath']}")
        print(f"  Ground truth: {r['ground_truth']}")
        print(f"  Prediction:   {r['safety_label']}")
    if len(labeled) == 0:
        print("\nWARNING: No samples were classified. Check inference_results.json for errors.")
        sys.exit(1)
except FileNotFoundError:
    print("ERROR: inference_results.json not found.")
    sys.exit(1)
PYEOF

echo ""
echo "=== Done ==="
echo "Full results: $COSMOS_REASON2/inference_results.json"
