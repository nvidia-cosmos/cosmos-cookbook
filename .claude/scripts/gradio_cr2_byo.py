#!/usr/bin/env python3
"""
Cosmos Reason BYO-Video Gradio Demo — Multi-Checkpoint Edition
Version: 2026-04-29
Canonical source: ~/.claude/scripts/gradio_cr2_byo.py

New in this version:
  - Checkpoint selector: any HF model ID, local path, or NIM via Advanced Settings
  - Sequential variant runner: FP8 → NVFP4 → NIM (Run All Variants button)
  - On-demand load/unload (del + gc.collect + cuda.empty_cache between variants)
  - Right-side status panel: Step N/5 WIP + live token metrics (replaces grey loading box)
  - NIM mode: NVCF API via NGC_API_KEY (no local weights needed)
"""
import os, sys, gc, json, time, threading, warnings, base64, io, atexit, signal, subprocess as _sp_cleanup
warnings.filterwarnings("ignore")

def _kill_frpc():
    """BUG-005: kill orphaned frpc tunnel processes when Gradio exits."""
    try:
        _sp_cleanup.run(["pkill", "-f", "frpc"], capture_output=True, timeout=5)
    except Exception:
        pass

atexit.register(_kill_frpc)
try:
    signal.signal(signal.SIGTERM, lambda *_: (_kill_frpc(), sys.exit(0)))
except Exception:
    pass

try:
    import torch
except ImportError:
    print("[ERROR] torch not installed."); sys.exit(1)

try:
    import gradio as gr
except ImportError:
    print("[ERROR] gradio not installed. Run: pip install gradio"); sys.exit(1)

try:
    import qwen_vl_utils
except ImportError:
    print("[ERROR] qwen_vl_utils not installed (should be in cosmos-reason2 venv)."); sys.exit(1)

try:
    from transformers import (
        Qwen3VLForConditionalGeneration,
        AutoModelForCausalLM,
        AutoProcessor,
        TextIteratorStreamer,
    )
except ImportError as e:
    print(f"[ERROR] transformers not installed: {e}"); sys.exit(1)

try:
    import av as _av_module
    _AV_OK = True
except ImportError:
    _av_module = None
    _AV_OK = False

try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _requests = None
    _REQUESTS_OK = False

try:
    import vllm as _vllm_module
    _VLLM_VERSION = _vllm_module.__version__
except ImportError:
    _VLLM_VERSION = None

# ── Config ─────────────────────────────────────────────────────────────────────
HOME        = os.path.expanduser("~")
MODEL_DIR   = os.environ.get("MODEL_DIR",  f"{HOME}/cosmos-reason2/models/Cosmos-Reason2-2B")
MODEL_NAME  = os.environ.get("MODEL_NAME", "nvidia/Cosmos-Reason2-2B")
OUT_FILE    = os.environ.get("OUT_FILE",   "/tmp/byo_video_reason2_results.json")
PORT        = int(os.environ.get("GRADIO_PORT", "7860"))
SHARE       = os.environ.get("GRADIO_SHARE", "true").lower() != "false"
LOW_VRAM    = os.environ.get("LOW_VRAM", "false").lower() == "true"
HF_TOKEN    = os.environ.get("HF_TOKEN", "")
NGC_API_KEY = os.environ.get("NGC_API_KEY", "")

DEFAULT_FPS        = int(os.environ.get("GRADIO_FPS", "4" if LOW_VRAM else "8"))
DEFAULT_MAX_PIXELS = int(os.environ.get("GRADIO_MAX_PIXELS",
                         str(128*(32**2) if LOW_VRAM else 4096*(32**2))))
DEFAULT_MAX_TOKENS = 512

TARGET_INFER_S = 55.0
TEXT_TOKENS    = 50
_EMPIRICAL_VPF = 128
_BASELINE_PX   = 524288
_PIXEL_TIERS   = [131072, 262144, 524288, 1048576, 2097152]
# IMAGE_AUTO_CAP_MAX bounds the auto-cap result for single images.
# _EMPIRICAL_VPF was calibrated for video tokenization (temporal merging halves
# tokens/frame); for Cosmos3 single-image inputs the actual vision token count
# is ~3-5x higher per pixel, so the est-time math is too lenient and snaps to
# the highest tier (2M px). Bound the image path to ≤512K, which empirically
# yields ~30-50s prefill on RTX PRO 6000 Blackwell + HF Transformers.
# User can disable auto-cap to override and use higher resolutions.
IMAGE_AUTO_CAP_MAX = 524288

NIM_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"

# vLLM server (OpenAI-compatible) — set INFERENCE_BACKEND=vllm to use
INFERENCE_BACKEND = os.environ.get("INFERENCE_BACKEND", "hf").lower()   # hf | vllm | nim_local
VLLM_BASE_URL     = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY      = os.environ.get("VLLM_API_KEY", "EMPTY")             # vLLM default

# Prefill TPS for TTFT estimation.
# HF mode: ~23 tok/s measured on RTX PRO 6000 Blackwell (empirical, local HTML report).
# vLLM mode on H100: ~12000 tok/s (GSheet CR2-8B PBR; 374×374 · 30s clips:
#   2K → 174ms, 4K → 341ms, 8K → 724ms → avg ~11,700 tok/s).
_prefill_default = "12000" if INFERENCE_BACKEND == "vllm" else "23"
PREFILL_TPS = float(os.environ.get("GRADIO_PREFILL_TPS", _prefill_default))

# For vllm/nim_local: query the running server to get its actual served model name.
# vLLM uses --served-model-name (e.g. "nvidia/Cosmos-Reason2-2B-FP8") which differs
# from local checkpoint paths. NIM serves a fixed ID (e.g. "nvidia/cosmos-reason2-2b").
# We must use this exact name in chat/completions requests or get 404.
_NIM_LOCAL_MODEL_ID = None  # kept for backward compat
_SERVER_MODEL_ID = None
if INFERENCE_BACKEND in ("vllm", "nim_local"):
    try:
        import urllib.request as _urlreq, json as _json
        with _urlreq.urlopen(f"{VLLM_BASE_URL}/models", timeout=5) as _r:
            _srv_data = _json.loads(_r.read())
            _SERVER_MODEL_ID = _srv_data["data"][0]["id"]
            _NIM_LOCAL_MODEL_ID = _SERVER_MODEL_ID  # backward compat
            print(f"[{INFERENCE_BACKEND}] Detected server model: {_SERVER_MODEL_ID}", flush=True)
    except Exception as _e:
        print(f"[{INFERENCE_BACKEND}] Could not detect model from {VLLM_BASE_URL}/models: {_e}", flush=True)

# ── Size-driven model configs ──────────────────────────────────────────────────
# Each variant: (ui_label, local_dirname, hf_model_id, expected_quant)
# expected_quant: "bf16" | "fp8" | "nvfp4" — used to detect HF runtime upcasting
MODEL_CONFIGS = {
    "2B": {
        "variants": [
            ("CR2-2B FP8",   "Cosmos-Reason2-2B-FP8",   "nvidia/Cosmos-Reason2-2B-FP8",   "fp8"),
            ("CR2-2B NVFP4", "Cosmos-Reason2-2B-NVFP4", "nvidia/Cosmos-Reason2-2B-NVFP4", "nvfp4"),
            ("CR2-2B BF16",  "Cosmos-Reason2-2B",       "nvidia/Cosmos-Reason2-2B",       "bf16"),
        ],
        "nim": None,  # container available (nvcr.io/nim/nvidia/cosmos-reason2-2b:latest) but NOT on NVCF hosted API
    },
    "8B": {
        "variants": [
            ("CR2-8B FP8",   "Cosmos-Reason2-8B-FP8",   "nvidia/Cosmos-Reason2-8B-FP8",   "fp8"),
            ("CR2-8B NVFP4", "Cosmos-Reason2-8B-NVFP4", "nvidia/Cosmos-Reason2-8B-NVFP4", "nvfp4"),
            ("CR2-8B BF16",  "Cosmos-Reason2-8B",       "nvidia/Cosmos-Reason2-8B",       "bf16"),
        ],
        # extended_variants: appended to Run All Variants only (not in checkpoint dropdown).
        # Runs against the loaded vLLM model — switch server to 32B first for true 32B benchmarks.
        # In HF mode, requires unloading 8B first (~64GB weights); OOMs if 8B stays loaded.
        "extended_variants": [
            ("CR2-32B BF16", "Cosmos-Reason2-32B",    "nvidia/Cosmos-Reason2-32B",    "bf16"),
            ("CR2-32B AV",   "Cosmos-Reason2-32B-AV", "nvidia/Cosmos-Reason2-32B-AV", "bf16"),
        ],
        "nim": "nvidia/cosmos-reason2-8b",
    },
    "32B": {
        # 32B BF16 requires ~64GB weights; fits on 1x H100 80GB only with small KV cache.
        # 32B BF16 and 32B-AV are separate checkpoints (AV = action-value variant).
        # Serve via vLLM with --max-model-len 4096 to leave room for KV cache.
        "variants": [
            ("CR2-32B BF16", "Cosmos-Reason2-32B",    "nvidia/Cosmos-Reason2-32B",    "bf16"),
            ("CR2-32B AV",   "Cosmos-Reason2-32B-AV", "nvidia/Cosmos-Reason2-32B-AV", "bf16"),
        ],
        "nim": None,  # TBD
    },
    # ── Cosmos3-Reasoner (C3-2B/C3-32B/C3-super gated; C3-8B = Cosmos3-Nano-Reasoner, public) ──
    "C3-2B": {
        "variants": [
            ("C3R-2B BF16", "Cosmos3-Reasoner-2B", "nvidia/Cosmos3-Reasoner-2B-Private", "bf16"),
        ],
        "nim": None,
    },
    "C3-8B": {
        "variants": [
            ("C3R-Nano BF16", "Cosmos3-Nano-Reasoner", "nvidia/Cosmos3-Nano-Reasoner", "bf16"),
        ],
        "nim": None,
    },
    "C3-32B": {
        "variants": [
            ("C3R-32B BF16", "Cosmos3-Reasoner-32B", "nvidia/Cosmos3-Reasoner-32B-Private", "bf16"),
        ],
        "nim": None,
        # disk_gb: 1024 (1TB minimum — Alex explicit requirement for 32B)
        # vLLM: --tensor-parallel-size 1 --gpu-memory-utilization 0.93 on single H100 80GB
        # Weight size unverified (25 files, est ~60-70GB BF16). Review if OOM occurs.
    },
    "C3-super": {
        "variants": [
            ("C3-Super BF16", "Cosmos3-Super-Reasoner", "nvidia/Cosmos3-Super-Reasoner", "bf16"),
        ],
        "nim": None,
        # 32B model on H200 SXM 141GB (confirmed 2026-05-05 live deployment).
        # Architecture: NemotronVLForConditionCausalLM. vLLM may raise "Unsupported architecture"
        # if arch is not registered. Use INFERENCE_BACKEND=hf as fallback (confirmed working).
    },
    # ── Qwen3-VL (public — no HF_TOKEN required) ────────────────────────────────
    # vLLM-only: uses video_url content type with file:// path (same pattern as Nemotron).
    # Frame control via extra_body mm_processor_kwargs at inference time (not server flags).
    # Three variants per size: Instruct (general), FP8 (quantized), Thinking (reasoning).
    "QW3-2B": {
        "variants": [
            ("Qwen3-VL-2B Instruct", "Qwen3-VL-2B-Instruct",     "Qwen/Qwen3-VL-2B-Instruct",     "bf16"),
            ("Qwen3-VL-2B FP8",      "Qwen3-VL-2B-Instruct-FP8", "Qwen/Qwen3-VL-2B-Instruct-FP8", "fp8"),
            ("Qwen3-VL-2B Thinking", "Qwen3-VL-2B-Thinking",     "Qwen/Qwen3-VL-2B-Thinking",     "bf16"),
        ],
        "nim": None,
        "vllm_swap_flags": ["--allowed-local-media-path", "/tmp"],
        "vllm_swap_env":   {},
    },
    "QW3-8B": {
        "variants": [
            ("Qwen3-VL-8B Instruct", "Qwen3-VL-8B-Instruct",     "Qwen/Qwen3-VL-8B-Instruct",     "bf16"),
            ("Qwen3-VL-8B FP8",      "Qwen3-VL-8B-Instruct-FP8", "Qwen/Qwen3-VL-8B-Instruct-FP8", "fp8"),
            ("Qwen3-VL-8B Thinking", "Qwen3-VL-8B-Thinking",     "Qwen/Qwen3-VL-8B-Thinking",     "bf16"),
        ],
        "nim": None,
        "vllm_swap_flags": ["--allowed-local-media-path", "/tmp"],
        "vllm_swap_env":   {},
    },
    "QW3-32B": {
        "variants": [
            ("Qwen3-VL-32B Instruct", "Qwen3-VL-32B-Instruct",     "Qwen/Qwen3-VL-32B-Instruct",     "bf16"),
            ("Qwen3-VL-32B FP8",      "Qwen3-VL-32B-Instruct-FP8", "Qwen/Qwen3-VL-32B-Instruct-FP8", "fp8"),
            ("Qwen3-VL-32B Thinking", "Qwen3-VL-32B-Thinking",     "Qwen/Qwen3-VL-32B-Thinking",     "bf16"),
        ],
        "nim": None,
        "vllm_swap_flags": ["--allowed-local-media-path", "/tmp"],
        "vllm_swap_env":   {},
    },
    # ── Nemotron-Nano-12B-v2-VL (gated — HF_TOKEN with nvidia org required) ──────
    # vLLM-only: uses opencv backend + file:// video URL (not base64 frames).
    # Requires vLLM nightly; PyPI vLLM ≤0.11.0 unsupported.
    "NEM-12B": {
        "variants": [
            ("Nem-12B BF16", "NVIDIA-Nemotron-Nano-12B-v2-VL-BF16", "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16", "bf16"),
            ("Nem-12B FP8",  "NVIDIA-Nemotron-Nano-12B-v2-VL-FP8",  "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8",  "fp8"),
        ],
        "nim": None,
        # Required for hot-swap: opencv video decoder + file:// media path + flashinfer bypass.
        "vllm_swap_flags": ["--allowed-local-media-path", "/tmp"],
        "vllm_swap_env":   {"VLLM_VIDEO_LOADER_BACKEND": "opencv", "FLASHINFER_DISABLE_VERSION_CHECK": "1"},
    },
}

# Per-model-size slider defaults shown in Advanced Settings when a checkpoint is selected.
# fps/max_pixels that are None inherit the GPU-tier value from DEFAULT_FPS/DEFAULT_MAX_PIXELS.
# Thinking variants get a max_tokens boost automatically in _ckpt_slider_defaults().
_MODEL_SIZE_DEFAULTS = {
    # file:// models (NEM-12B, QW3-*) — fps slider unused by vLLM; inherit GPU-tier fps
    "NEM-12B":  {"max_tokens": 512},
    "QW3-2B":   {"max_tokens": 512},
    "QW3-8B":   {"max_tokens": 512},
    "QW3-32B":  {"max_tokens": 1024},
    # base64-frame models — fps controls extraction; set sensible per-size values
    "2B":    {"fps": 2, "max_tokens": 512},
    "8B":    {"fps": 2, "max_tokens": 512},
    "32B":   {"fps": 1, "max_tokens": 512},
    "C3-2B": {"fps": 2, "max_tokens": 512},
    "C3-8B": {"fps": 2, "max_tokens": 512},
    "C3-32B":  {"fps": 1, "max_tokens": 1024},
    "C3-super":{"fps": 1, "max_tokens": 1024},
}
# Flat map: checkpoint UI label → MODEL_CONFIGS size key (built after MODEL_CONFIGS is complete)
_LABEL_TO_MODEL_SIZE = {
    label: size_key
    for size_key, cfg in MODEL_CONFIGS.items()
    for label, _, _, _ in cfg.get("variants", [])
}

MODEL_SIZE   = os.environ.get("MODEL_SIZE", "2B").upper()
# .upper() normalises input but breaks mixed-case keys. Remap known exceptions.
_MODEL_SIZE_FIX = {"C3-SUPER": "C3-super"}
MODEL_SIZE = _MODEL_SIZE_FIX.get(MODEL_SIZE, MODEL_SIZE)
if MODEL_SIZE not in MODEL_CONFIGS:
    print(f"[ERROR] MODEL_SIZE={MODEL_SIZE} not supported. Use 2B, 8B, 32B, C3-2B, C3-8B, C3-32B, C3-super, NEM-12B, QW3-2B, QW3-8B, or QW3-32B."); sys.exit(1)

# Model-size specific fps default for the UI slider (HF mode uses lower fps to bound prefill time)
_UI_DEFAULT_FPS = _MODEL_SIZE_DEFAULTS.get(MODEL_SIZE, {}).get("fps", DEFAULT_FPS)

# Hard cap on frames sampled in HF mode — prevents multi-minute prefills on long videos.
# qwen_vl_utils respects "nframes" in the conversation dict to uniformly subsample the clip.
_MAX_HF_FRAMES = 32

_cfg         = MODEL_CONFIGS[MODEL_SIZE]
_MODELS_BASE = os.path.join(HOME, "cosmos-reason2", "models")

# Override MODEL_DIR/MODEL_NAME from size if not explicitly set
if not os.environ.get("MODEL_DIR"):
    MODEL_DIR  = os.path.join(_MODELS_BASE, _cfg["variants"][0][1])
if not os.environ.get("MODEL_NAME"):
    MODEL_NAME = _cfg["variants"][0][2]

NIM_MODEL_API = _cfg["nim"] or ""

def _resolve(dirname, hf_id):
    local = os.path.join(_MODELS_BASE, dirname)
    return local if os.path.exists(local) else hf_id

def _ckpt_slider_defaults(label):
    """Return (fps, max_pixels, max_tokens) for a checkpoint label.
    Falls back to GPU-tier globals; Thinking variants get a max_tokens boost."""
    size = _LABEL_TO_MODEL_SIZE.get(label, MODEL_SIZE)
    ov   = _MODEL_SIZE_DEFAULTS.get(size, {})
    fps  = ov.get("fps",        DEFAULT_FPS)
    mpx  = ov.get("max_pixels", DEFAULT_MAX_PIXELS)
    mtok = ov.get("max_tokens", DEFAULT_MAX_TOKENS)
    if "Think" in label:
        mtok = max(mtok, 1024)
    return fps, mpx, mtok

CHECKPOINT_PRESETS = [
    (label, _resolve(dirname, hf_id))
    for label, dirname, hf_id, _ in _cfg["variants"]
]
# Always show NIM entry so the user gets an explicit result rather than silence
_nim_api_id = _cfg["nim"] or ""
CHECKPOINT_PRESETS.append((f"NIM {MODEL_SIZE}", f"nim://{_nim_api_id}"))

# When running in NIM mode, the HF quantization variants (FP8 / NVFP4 / BF16)
# are stale artifacts — the NIM container ships its own fixed quantization, so
# all four entries collapse to the running NIM. Replace the dropdown with the
# upstream VLM NIM catalog from
#   https://docs.nvidia.com/nim/vision-language-models/latest/introduction.html
# (cross-referenced against KNOWN_VLM_NIMS in ~/.claude/scripts/nim_catalog.py).
# We skip the docker-manifest probe at module-load to keep Gradio startup snappy
# — probing happens at switch time when the user actually wants to swap models.
_NIM_CATALOG = []
if INFERENCE_BACKEND == "nim_local":
    try:
        # Catalog helper sits next to this script (and is also scp'd to /tmp/).
        for _p in (os.path.dirname(os.path.abspath(__file__)), "/tmp"):
            if _p and _p not in sys.path:
                sys.path.insert(0, _p)
        from nim_catalog import list_available_nims, KNOWN_VLM_NIMS  # type: ignore
        _NIM_CATALOG = list_available_nims(
            ngc_api_key=NGC_API_KEY,
            vram_mb=None,           # don't VRAM-filter at startup; show everything supported
            use_upstream=True,      # fetch docs.nvidia.com — this is the agent's source of truth
            do_probe=False,         # probe lazily on switch (avoids 10-30s startup penalty)
        )
        if not _NIM_CATALOG:
            # Network unreachable or upstream removed every family we know — fall
            # back to the static slug map so the user is not stranded.
            _NIM_CATALOG = list(KNOWN_VLM_NIMS)
        # NO `nim://` prefix in nim_local mode — that prefix routes
        # run_inference() to _run_nim_inference (the NVCF hosted-API path)
        # which skips models like CR2-2B that are not in the public NVCF
        # catalog. In nim_local mode the user wants the LOCAL Docker
        # container, which is OpenAI-compatible and reached via
        # _run_vllm_inference. Pass the served_model_id directly so
        # _is_nim() returns False and routing falls through to the
        # vLLM-style path against VLLM_BASE_URL=http://localhost:8000/v1.
        CHECKPOINT_PRESETS = [(n.label, n.served_model_id) for n in _NIM_CATALOG]
        print(f"[nim_local] catalog: {len(_NIM_CATALOG)} VLM NIMs from {len([n for n in _NIM_CATALOG])} entries", flush=True)
    except Exception as _e:
        print(f"[nim_local] catalog fetch failed ({_e}); keeping default presets", flush=True)

# ── vLLM dropdown: all variants across all model sizes ─────────────────────────
# In vLLM mode the checkpoint dropdown exposes every quantization across 2B/8B/32B.
# _VLLM_DD_META maps label → (local_path, hf_id) so _on_checkpoint_change can
# locate the model and pass the right served-model-name to _launch_vllm_swap.
_ALL_VARIANTS_DD_RAW = [
    ("C3R-2B BF16",   "Cosmos3-Reasoner-2B",                  "nvidia/Cosmos3-Reasoner-2B-Private",             "bf16"),
    ("C3R-Nano BF16", "Cosmos3-Nano-Reasoner",                 "nvidia/Cosmos3-Nano-Reasoner",                   "bf16"),
    ("C3R-32B BF16",  "Cosmos3-Reasoner-32B",                 "nvidia/Cosmos3-Reasoner-32B-Private",            "bf16"),
    ("C3-Super BF16", "Cosmos3-Super-Reasoner",               "nvidia/Cosmos3-Super-Reasoner",                  "bf16"),
    ("CR2-2B BF16",   "Cosmos-Reason2-2B",                    "nvidia/Cosmos-Reason2-2B",                       "bf16"),
    ("CR2-2B FP8",    "Cosmos-Reason2-2B-FP8",                "nvidia/Cosmos-Reason2-2B-FP8",                   "fp8"),
    ("CR2-2B NVFP4",  "Cosmos-Reason2-2B-NVFP4",              "nvidia/Cosmos-Reason2-2B-NVFP4",                 "nvfp4"),
    ("CR2-8B BF16",   "Cosmos-Reason2-8B",                    "nvidia/Cosmos-Reason2-8B",                       "bf16"),
    ("CR2-8B FP8",    "Cosmos-Reason2-8B-FP8",                "nvidia/Cosmos-Reason2-8B-FP8",                   "fp8"),
    ("CR2-8B NVFP4",  "Cosmos-Reason2-8B-NVFP4",              "nvidia/Cosmos-Reason2-8B-NVFP4",                 "nvfp4"),
    ("CR2-32B BF16",  "Cosmos-Reason2-32B",                   "nvidia/Cosmos-Reason2-32B",                      "bf16"),
    ("CR2-32B AV",    "Cosmos-Reason2-32B-AV",                "nvidia/Cosmos-Reason2-32B-AV",                   "bf16"),
    ("Nem-12B BF16",      "NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",  "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16",    "bf16"),
    ("Nem-12B FP8",       "NVIDIA-Nemotron-Nano-12B-v2-VL-FP8",   "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8",     "fp8"),
    # Qwen3-VL — public, no HF_TOKEN required; uses file:// video_url like Nemotron
    ("Qwen3-VL-2B",       "Qwen3-VL-2B-Instruct",      "Qwen/Qwen3-VL-2B-Instruct",      "bf16"),
    ("Qwen3-VL-2B FP8",   "Qwen3-VL-2B-Instruct-FP8",  "Qwen/Qwen3-VL-2B-Instruct-FP8",  "fp8"),
    ("Qwen3-VL-2B Think", "Qwen3-VL-2B-Thinking",      "Qwen/Qwen3-VL-2B-Thinking",      "bf16"),
    ("Qwen3-VL-8B",       "Qwen3-VL-8B-Instruct",      "Qwen/Qwen3-VL-8B-Instruct",      "bf16"),
    ("Qwen3-VL-8B FP8",   "Qwen3-VL-8B-Instruct-FP8",  "Qwen/Qwen3-VL-8B-Instruct-FP8",  "fp8"),
    ("Qwen3-VL-8B Think", "Qwen3-VL-8B-Thinking",      "Qwen/Qwen3-VL-8B-Thinking",      "bf16"),
    ("Qwen3-VL-32B",      "Qwen3-VL-32B-Instruct",     "Qwen/Qwen3-VL-32B-Instruct",     "bf16"),
    ("Qwen3-VL-32B FP8",  "Qwen3-VL-32B-Instruct-FP8", "Qwen/Qwen3-VL-32B-Instruct-FP8", "fp8"),
    ("Qwen3-VL-32B Think","Qwen3-VL-32B-Thinking",     "Qwen/Qwen3-VL-32B-Thinking",     "bf16"),
]
_VLLM_DD_META = {
    label: (os.path.join(_MODELS_BASE, dirname), hf_id)
    for label, dirname, hf_id, _ in _ALL_VARIANTS_DD_RAW
}
if INFERENCE_BACKEND == "vllm":
    CHECKPOINT_PRESETS = [
        (label, _resolve(dirname, hf_id))
        for label, dirname, hf_id, _ in _ALL_VARIANTS_DD_RAW
    ]
    CHECKPOINT_PRESETS.append(("NIM 8B", "nim://nvidia/cosmos-reason2-8b"))
    # Auto-select the checkpoint that matches the loaded model so no swap fires on first use.
    _VLLM_DD_MAP = {
        "NEM-12B": "Nem-12B BF16",
        "QW3-2B":  "Qwen3-VL-2B",
        "QW3-8B":  "Qwen3-VL-8B",
        "QW3-32B": "Qwen3-VL-32B",
        "2B":      "CR2-2B BF16",
        "8B":      "CR2-8B BF16",
        "C3-2B":   "C3R-2B BF16",
        "C3-8B":   "C3R-Nano BF16",
        "C3-32B":  "C3R-32B BF16",
        "C3-super":"C3-Super BF16",
        "32B":     "CR2-32B BF16",
    }
    _VLLM_DD_DEFAULT = _VLLM_DD_MAP.get(MODEL_SIZE, "CR2-8B BF16")
elif INFERENCE_BACKEND == "nim_local" and _NIM_CATALOG:
    # Default to the currently-served NIM (whichever one the running container
    # reports via /v1/models). If we can't match, fall back to first entry.
    _matched = next(
        (n.label for n in _NIM_CATALOG if n.served_model_id == (_SERVER_MODEL_ID or "")),
        None,
    )
    _VLLM_DD_DEFAULT = _matched or CHECKPOINT_PRESETS[0][0]
else:
    _VLLM_DD_DEFAULT = CHECKPOINT_PRESETS[0][0]

DEFAULT_SYSTEM = "You are a helpful assistant that analyzes videos."
DEFAULT_PROMPT = "Describe what is happening in this video. What are the key actions, objects, and events?"

_GENERIC_SYSTEM = "You are a helpful assistant."
_WAREHOUSE_SYSTEM = "You are a helpful warehouse monitoring system."

# (label, user_prompt, system_prompt, reasoning_on)
# reasoning_on signals whether the prompt instructs the model to wrap its
# chain-of-thought in <think>...</think>. The Gradio "Reasoning: ON/OFF"
# badge reads this; the actual rendering of <think> blocks is handled by
# _render_with_think() regardless.
DEMO_PROMPTS = [
    ("General description",
     "Describe what is happening in this video. What are the key actions, objects, and events?",
     DEFAULT_SYSTEM, False),
    ("Safety analysis",
     "Identify any safety hazards, risks, or unsafe behaviors visible in this video. Be specific.",
     DEFAULT_SYSTEM, False),
    ("Non-expert summary",
     "Summarize this video in plain language for someone with no domain expertise.",
     DEFAULT_SYSTEM, False),
    ("Action recognition",
     "List every distinct action or motion performed in this video, in the order they occur.",
     DEFAULT_SYSTEM, False),
    ("Object inventory",
     "List all objects, equipment, and people visible. Note their state.",
     DEFAULT_SYSTEM, False),
    ("Anomaly detection",
     "Identify anything unusual, unexpected, or out of place in this video.",
     DEFAULT_SYSTEM, False),
    ("Temporal summary",
     "Break this video into time segments and describe what changes in each segment.",
     DEFAULT_SYSTEM, False),

    # ── Cosmos Reason2 reasoning showcases (mirror build.nvidia.com) ───────────
    ("Race car: timestamps",
     "Describe the video. Add timestamps in mm:ss format.\n\n"
     "Answer the question using the following format:\n\n"
     "<think>\nYour reasoning.\n</think>\n\n"
     "Write your final answer immediately after the </think> tag and "
     "include the timestamps.",
     _GENERIC_SYSTEM, True),
    ("Forklift: load weight (JSON)",
     "Locate the bounding box of the load and determine if its size and "
     "weight of load within the forklift's limits. Estimate weights. "
     "Return all as json. Include json location, estimated weight of the "
     "load, and if it's in the limit.",
     _GENERIC_SYSTEM, False),
    ("Mail package: pickup allowed?",
     "Is the person allowed to pick up the packages?\n"
     "Answer the question using the following format:\n\n"
     "<think>\nYour reasoning.\n</think>\n\n"
     "Write your final answer immediately after the </think> tag.",
     _GENERIC_SYSTEM, True),
    ("Warehouse: who picked up the box?",
     "Which worker picked up the dropped box?\n"
     "Answer the question using the following format:\n\n"
     "<think>\nYour reasoning.\n</think>\n\n"
     "Write your final answer immediately after the </think> tag.",
     _WAREHOUSE_SYSTEM, True),
    ("AV: next ego action",
     "What's the next immediate action for the Ego vehicle?\n\n"
     "Answer the question using the following format:\n\n"
     "<think>\nYour reasoning.\n</think>\n\n"
     "Write your final answer immediately after the </think> tag.",
     _GENERIC_SYSTEM, True),
    ("Robot arm: 2D trajectory (JSON)",
     "You are given the task \"Move the tape into the basket\". Specify "
     "the 2D trajectory your end effector should follow in pixel space. "
     "Return the trajectory coordinates in JSON format like this: "
     "{\"point_2d\": [x, y], \"label\": \"gripper trajectory\"}.\n\n"
     "Answer the question using the following format:\n\n"
     "<think>\nYour reasoning.\n</think>\n\n"
     "Write your final answer immediately after the </think> tag.",
     _GENERIC_SYSTEM, True),
    ("SDG critic: approve / reject",
     "Approve or reject this generated video for inclusion in a dataset "
     "for physical world model ai training. It must perfectly adhere to "
     "physics, object permanence, and have no anomalies. Any issue or "
     "concern causes rejection.\n"
     "Answer the question using the following format:\n\n"
     "<think>\nYour reasoning.\n</think>\n\n"
     "Write your final answer immediately after the </think> tag. "
     "Answer with Approve or Reject only.",
     _GENERIC_SYSTEM, True),
]


def _reasoning_badge_html(reasoning_on):
    """Inline pill rendered above the response area. NV-green for ON,
    slate-300 for OFF — both readable on Gradio's light theme."""
    if reasoning_on:
        return (
            '<div style="display:inline-flex;align-items:center;gap:6px;'
            'background:#76b900;color:#0f172a;padding:4px 12px;'
            'border-radius:14px;font-size:0.85em;font-weight:600;'
            'margin-bottom:6px">🧠 Reasoning: ON</div>'
        )
    return (
        '<div style="display:inline-flex;align-items:center;gap:6px;'
        'background:#cbd5e1;color:#0f172a;padding:4px 12px;'
        'border-radius:14px;font-size:0.85em;font-weight:600;'
        'margin-bottom:6px">○ Reasoning: OFF</div>'
    )

HF_STEPS = [
    "Resolve checkpoint & GPU",
    "Load model weights",
    "Preprocess video frames",
    "Prefill tokens",
    "Generate response",
]

NIM_STEPS = [
    "Resolve checkpoint & GPU",
    "Validate NGC API key",
    "Extract video frames",
    "Send to NVCF endpoint",
    "Stream response",
]

VLLM_STEPS = [
    "Resolve checkpoint & GPU",
    "Check vLLM server",
    "Extract video frames",
    "Send to vLLM endpoint",
    "Stream response",
]

# ── Global model state ─────────────────────────────────────────────────────────
_state_lock = threading.Lock()
_loaded = {
    "model":          None,
    "processor":      None,
    "model_id":       None,
    "load_time":      0.0,
    "upcast_warning": None,   # set when HF silently upcasts quantized weights to bf16
}

# ── Run log (persists across Gradio calls) ─────────────────────────────────────
_run_log: list = []
_log_lock = threading.Lock()


# ── Pure helpers ───────────────────────────────────────────────────────────────
def _est_tokens(n_frames, max_pixels):
    return int(n_frames * max_pixels / _BASELINE_PX * _EMPIRICAL_VPF) + TEXT_TOKENS


def _auto_cap(n_frames, current_max_pixels):
    for px in reversed(_PIXEL_TIERS):
        if px > current_max_pixels:
            continue
        est_s = _est_tokens(n_frames, px) / PREFILL_TPS
        if est_s <= TARGET_INFER_S:
            return px, est_s
    px = _PIXEL_TIERS[0]
    return px, _est_tokens(n_frames, px) / PREFILL_TPS


def get_video_meta(path):
    class _M:
        width = height = fps = duration_s = 0
    m = _M()
    if not _AV_OK or not path:
        return m
    try:
        container = _av_module.open(path)
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream:
            m.width      = stream.width
            m.height     = stream.height
            m.fps        = float(stream.average_rate) if stream.average_rate else 0.0
            m.duration_s = float(container.duration) / 1_000_000 if container.duration else 0.0
        container.close()
    except Exception:
        pass
    return m


def get_free_vram_mib():
    try:
        return torch.cuda.mem_get_info()[0] // (1024 * 1024)
    except Exception:
        return 0


def _is_nim(model_id):
    return model_id.startswith("nim://")

def _is_vllm(model_id):
    return INFERENCE_BACKEND in ("vllm", "nim_local") and not _is_nim(model_id)

def _is_nemotron(model_id):
    """True for Nemotron models — they use file:// video URL instead of base64 frames."""
    return "nemotron" in (model_id or "").lower()

def _is_qwen3vl(model_id):
    """True for Qwen3-VL models — they use file:// video URL (same as Nemotron)."""
    mid = (model_id or "").lower()
    return "qwen3-vl" in mid or "qwen3vl" in mid

def _uses_file_url(model_id):
    """True for any model that uses file:// video URL to vLLM (vs base64 JPEG frames)."""
    return _is_nemotron(model_id) or _is_qwen3vl(model_id)

def _expected_quant(model_id):
    """Infer expected quantization from model path/ID name."""
    mid = model_id.lower()
    if "nvfp4" in mid or ("-fp4" in mid and "nvidia" in mid):
        return "nvfp4"
    if "fp8" in mid:
        return "fp8"
    return "bf16"


# ── Reasoning panel (build.nvidia.com style) ─────────────────────────────────
# Cosmos Reason 2 emits its chain-of-thought between <think>...</think> tags
# inside the streamed `delta.content`. Mirrors build.nvidia.com's UI: a
# collapsible "Reasoning Complete ✓" header above the final answer; while
# tokens are still streaming inside <think>, the header reads "Reasoning…"
# and stays expanded so the user can watch the model think.
_REASONING_PANEL_CSS = """
/* Self-contained dark wrapper so contrast holds regardless of Gradio
   theme (light/dark). Mirrors build.nvidia.com's reasoning panel. */
.cr-output {
  padding: 14px 16px;
  font-size: 0.95em;
  line-height: 1.5;
  max-height: 540px;
  overflow-y: auto;
  background: #0f172a;          /* slate-900 — guarantees dark backdrop */
  color: #f1f5f9;               /* slate-100 — default text on the wrapper */
  border-radius: 8px;
  border: 1px solid #334155;    /* slate-700 — visible edge on light themes */
}
.cr-prelude {
  white-space: pre-wrap;
  color: #cbd5e1;               /* slate-300 */
  padding: 4px 0;
  font-style: italic;
  font-size: 0.92em;
}
.cr-reasoning {
  border: 1px solid #76b900;    /* NV green */
  border-radius: 6px;
  padding: 10px 12px;
  margin-bottom: 12px;
  background: rgba(118,185,0,0.10);
}
.cr-reasoning summary {
  cursor: pointer;
  color: #76b900;
  font-weight: 600;
  user-select: none;
  list-style: none;
  padding: 2px 0;
  font-size: 0.95em;
}
.cr-reasoning summary::-webkit-details-marker { display: none; }
.cr-reasoning summary::before { content: '▸ '; color: #76b900; font-size: 0.9em; }
.cr-reasoning[open] summary::before { content: '▾ '; }
.cr-reasoning .cr-reasoning-blurb {
  color: #94a3b8;               /* slate-400 */
  font-size: 0.85em;
  margin: 6px 0 8px 0;
  font-style: italic;
}
.cr-reasoning .cr-think-body {
  white-space: pre-wrap;
  color: #e2e8f0;               /* slate-200 — clearly readable on dark */
  padding: 6px 10px;
  border-left: 2px solid #76b900;
  font-size: 0.92em;
  background: rgba(15,23,42,0.5);
  border-radius: 0 4px 4px 0;
}
.cr-answer {
  white-space: pre-wrap;
  color: #f8fafc;               /* slate-50 — brightest, the headline output */
  padding: 8px 0 4px 0;
  font-size: 1.02em;
  font-weight: 500;
}
"""


def _render_with_think(text):
    """Split streamed model output on <think>...</think> and render the
    reasoning as a collapsible panel + the answer plainly below.
    Plain text (no <think>) renders as-is. Streaming mid-think shows a
    "Reasoning…" header (open by default); once </think> arrives the
    header switches to "Reasoning Complete ✓" and collapses by default.
    Always returns HTML safe for gr.HTML — escapes user-visible text."""
    import html as _html
    if not text:
        return "<div class='cr-output'></div>"
    open_tag, close_tag = "<think>", "</think>"
    i_open = text.find(open_tag)
    if i_open < 0:
        return f"<div class='cr-output'><div class='cr-answer'>{_html.escape(text)}</div></div>"
    pre = text[:i_open]
    body_start = i_open + len(open_tag)
    i_close = text.find(close_tag, body_start)
    if i_close < 0:
        reasoning = text[body_start:]
        summary = "⏳ Reasoning…"
        details_open = " open"
        answer_html = ""
        blurb = "<div class='cr-reasoning-blurb'>The model is thinking… stream continues below.</div>"
    else:
        reasoning = text[body_start:i_close]
        post = text[i_close + len(close_tag):].lstrip("\n")
        summary = "✓ Reasoning Complete"
        details_open = ""
        answer_html = f"<div class='cr-answer'>{_html.escape(post)}</div>" if post else ""
        blurb = "<div class='cr-reasoning-blurb'>Below is the entire thinking process the model went through to arrive at its response.</div>"
    pre_html = f"<div class='cr-prelude'>{_html.escape(pre)}</div>" if pre.strip() else ""
    reasoning_html = (
        f"<details{details_open} class='cr-reasoning'>"
        f"<summary>{summary}</summary>"
        f"{blurb}"
        f"<div class='cr-think-body'>{_html.escape(reasoning)}</div>"
        f"</details>"
    )
    return f"<div class='cr-output'>{pre_html}{reasoning_html}{answer_html}</div>"


def _status_html(statuses, metrics=None, steps=None):
    """
    statuses : list of 5 strings — "ok" | "run" | "wait"
    metrics  : optional dict with keys: model_id, load_s, tokens_in, tokens_out,
               ttft_s, infer_s, elapsed_s
    steps    : optional list of 5 step label strings (defaults to HF_STEPS)
    """
    if steps is None:
        steps = HF_STEPS

    ICON  = {"ok": "✅", "run": "⟳", "wait": "—"}
    COLOR = {"ok": "#4CAF50", "run": "#FFA500", "wait": "#fff"}

    rows = ""
    for i, (label, st) in enumerate(zip(steps, statuses)):
        icon  = ICON[st]
        color = COLOR[st]
        weight = "bold" if st == "run" else "normal"
        rows += (
            f'<div style="color:{color};font-weight:{weight};margin:3px 0">'
            f'<b>Step {i+1}/5:</b> {label} &nbsp;{icon}'
            f'</div>'
        )

    mhtml = ""
    if metrics:
        mhtml = '<hr style="margin:8px 0;border:none;border-top:1px solid #555"/>'
        # Backend badge
        _be = metrics.get("backend", INFERENCE_BACKEND.upper())
        _be_color = "#a78bfa" if _be == "VLLM" else ("#7dd3fc" if _be == "NIM" else "#86efac")
        mhtml += (f'<div style="margin:2px 0;color:#fff">🔧 <b>Backend:</b> '
                  f'<code style="font-size:11px;color:{_be_color}">{_be}</code></div>')
        if metrics.get("elapsed_s") is not None:
            mhtml += f'<div style="margin:2px 0;color:#fff">⏱ <b>Elapsed:</b> {metrics["elapsed_s"]:.1f}s</div>'
        if metrics.get("model_id"):
            mid   = metrics["model_id"]
            short = mid.split("/")[-1] if "/" in mid else mid
            mhtml += f'<div style="margin:2px 0;color:#fff">🤖 <code style="font-size:11px;color:#7dd3fc">{short}</code></div>'
        if metrics.get("upcast_warning"):
            mhtml += (f'<div style="margin:4px 0;color:#fbbf24;font-size:11px">'
                      f'⚠ {metrics["upcast_warning"]}</div>')
        if metrics.get("load_s") is not None:
            mhtml += f'<div style="margin:2px 0;color:#fff">🔄 <b>Load:</b> {metrics["load_s"]:.1f}s</div>'
        if metrics.get("tokens_in"):
            mhtml += f'<div style="margin:2px 0;color:#fff">📥 <b>Prefill:</b> {metrics["tokens_in"]:,} tokens</div>'
        if metrics.get("tokens_out") is not None:
            mhtml += f'<div style="margin:2px 0;color:#fff">📤 <b>Generated:</b> {metrics["tokens_out"]:,} tokens</div>'
        if metrics.get("ttft_s") is not None:
            mhtml += f'<div style="margin:2px 0;color:#fff">⚡ <b>TTFT:</b> {metrics["ttft_s"]:.2f}s</div>'
        if metrics.get("infer_s") is not None:
            mhtml += f'<div style="margin:2px 0;color:#fff">⏱ <b>Inference:</b> {metrics["infer_s"]:.1f}s</div>'

    return (
        '<div style="font-family:monospace;font-size:13px;line-height:1.6;'
        'padding:10px;background:#1a1a1a;border-radius:6px;color:#fff">'
        + rows + mhtml + '</div>'
    )


def _table_html():
    """Render _run_log as a dark HTML benchmark table."""
    TH = ('style="padding:6px 10px;text-align:left;border-bottom:1px solid #444;'
          'color:#7dd3fc;white-space:nowrap"')
    TD = 'style="padding:5px 10px;border-bottom:1px solid #333;color:#fff;white-space:nowrap"'
    with _log_lock:
        log = list(_run_log)
    if not log:
        return ('<div style="font-family:monospace;font-size:12px;color:#888;'
                'padding:10px;background:#1a1a1a;border-radius:6px">'
                'No runs logged yet.</div>')
    cols = ["Model", "Load (s)", "TTFT (s)", "Infer (s)", "Tokens Out", "Total (s)", "Status", "Notes"]
    hdr = "".join(f"<th {TH}>{c}</th>" for c in cols)
    rows_html = ""
    for r in log:
        def _f(v, fmt=".1f"):
            return f"{v:{fmt}}" if v is not None else "—"
        notes_val = r.get("notes", "")
        notes_html = (
            f'<span style="color:#fbbf24;font-size:11px">⚠ {notes_val}</span>'
            if notes_val else "—"
        )
        cells = [
            r.get("model", "—"),
            _f(r.get("load_s")),
            _f(r.get("ttft_s"), ".2f"),
            _f(r.get("infer_s")),
            str(r.get("tokens_out") or "—"),
            _f(r.get("total_s")),
            r.get("status", "—"),
            notes_html,
        ]
        _TD_WIDE = 'style="padding:5px 10px;border-bottom:1px solid #333;color:#fff;white-space:normal;max-width:260px"'
        row_tds = [f"<td {TD}>{c}</td>" for c in cells[:-1]]
        row_tds.append(f"<td {_TD_WIDE}>{cells[-1]}</td>")
        rows_html += "<tr>" + "".join(row_tds) + "</tr>"
    return (
        '<div style="overflow-x:auto;margin-top:10px">'
        '<table style="width:100%;border-collapse:collapse;font-family:monospace;'
        'font-size:12px;background:#1a1a1a;border-radius:6px">'
        f"<thead><tr>{hdr}</tr></thead><tbody>{rows_html}</tbody></table></div>"
    )


def _log_run(model_id, load_s=None, ttft_s=None, infer_s=None,
             tokens_out=None, total_s=None, status="ok", notes=None, display_label=None):
    if display_label:
        short = display_label
    elif model_id.startswith("nim://"):
        short = f"NIM-{MODEL_SIZE}"
    else:
        short = model_id.split("/")[-1] if "/" in model_id else model_id
    with _log_lock:
        _run_log.append({
            "model": short, "load_s": load_s, "ttft_s": ttft_s,
            "infer_s": infer_s, "tokens_out": tokens_out,
            "total_s": total_s, "status": status, "notes": notes or "",
        })


# ── Model load / unload ────────────────────────────────────────────────────────
def _unload():
    global _loaded
    with _state_lock:
        if _loaded["model"] is None:
            return
        mid = _loaded["model_id"]
        print(f"[model] Unloading {mid}", flush=True)
        del _loaded["model"]
        del _loaded["processor"]
        _loaded["model"]          = None
        _loaded["processor"]      = None
        _loaded["model_id"]       = None
        _loaded["load_time"]      = 0.0
        _loaded["upcast_warning"] = None
        gc.collect()
        torch.cuda.empty_cache()
        free = get_free_vram_mib()
        print(f"[model] Unloaded. VRAM free: {free:,} MiB", flush=True)


def _load(model_id):
    """Load an HF checkpoint. Unloads current if different. Returns (model, processor, load_time_s)."""
    global _loaded
    with _state_lock:
        if _loaded["model_id"] == model_id and _loaded["model"] is not None:
            return _loaded["model"], _loaded["processor"], _loaded["load_time"]

        if _loaded["model"] is not None:
            print(f"[model] Switching {_loaded['model_id']} → {model_id}", flush=True)
            del _loaded["model"]
            del _loaded["processor"]
            _loaded["model"]     = None
            _loaded["processor"] = None
            _loaded["model_id"]  = None
            gc.collect()
            torch.cuda.empty_cache()

        print(f"[model] Loading {model_id} ...", flush=True)
        t0 = time.time()

        # BUG-001 fix: detect model_type from config.json so NemotronVL (nemotron_siglip2)
        # and other non-qwen3_vl architectures route to AutoModelForCausalLM instead of
        # Qwen3VLForConditionalGeneration, which raises NotImplementedError for unknown archs.
        _model_type = "qwen3_vl"
        _config_path = os.path.join(model_id, "config.json") if os.path.isdir(model_id) else ""
        if _config_path and os.path.exists(_config_path):
            with open(_config_path) as _cf:
                _model_type = json.load(_cf).get("model_type", "qwen3_vl")

        kwargs = dict(dtype="auto", device_map="auto")
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN

        if _model_type == "qwen3_vl":
            kwargs["attn_implementation"] = "sdpa"
            model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
        else:
            kwargs["trust_remote_code"] = True
            print(f"[model] model_type={_model_type!r} — using AutoModelForCausalLM", flush=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

        proc_kwargs = {"trust_remote_code": True}
        if HF_TOKEN:
            proc_kwargs["token"] = HF_TOKEN
        processor = AutoProcessor.from_pretrained(model_id, **proc_kwargs)

        load_time = time.time() - t0

        dtype_str = str(next(model.parameters()).dtype)

        # Detect silent upcasting: quantized checkpoint loaded as bf16 by HF transformers
        upcast_warning = None
        expected = _expected_quant(model_id)
        if expected == "fp8" and "float8" not in dtype_str:
            upcast_warning = (f"FP8 weights upcast to {dtype_str} — "
                              f"HF has no FP8 compute kernels. Use vLLM for real FP8 speed.")
        elif expected == "nvfp4" and "float4" not in dtype_str:
            upcast_warning = (f"NVFP4 weights upcast to {dtype_str} — "
                              f"NVFP4 requires TRT-LLM or NIM; HF runs at BF16 speed.")
        elif expected == "bf16" and "float32" in dtype_str and _model_type != "qwen3_vl":
            # BUG-006: NemotronVL and other non-qwen3_vl models may load as float32
            # when transformers version doesn't honor dtype="auto" for the architecture.
            upcast_warning = (f"BF16 model loaded as float32 (arch={_model_type!r}) — "
                              f"2× memory usage. Set torch_dtype=torch.bfloat16 if OOM.")
        if upcast_warning:
            print(f"[upcast] {upcast_warning}", flush=True)

        _loaded["model"]          = model
        _loaded["processor"]      = processor
        _loaded["model_id"]       = model_id
        _loaded["load_time"]      = load_time
        _loaded["upcast_warning"] = upcast_warning

        print(
            f"[model] Ready in {load_time:.1f}s | dtype={dtype_str} | "
            f"VRAM free: {get_free_vram_mib():,} MiB",
            flush=True,
        )
        return model, processor, load_time


# ── Frame extraction for NIM ───────────────────────────────────────────────────
def _extract_frames_b64(video_path, fps=1, max_frames=8):
    """Return list of base64 JPEG strings sampled from `video_path`.

    Sampling strategy:
      - target_count = duration_s * fps (what fps would yield without a cap)
      - if target_count <= max_frames: pick every `frame_rate/fps`-th frame
        starting at 0. Behaviour identical to the prior version.
      - if target_count > max_frames: uniformly distribute `max_frames` picks
        across the FULL video duration (frame indices spaced by
        (total_frames-1)/(max_frames-1)). This is the critical fix for
        long clips where the cap previously truncated everything to the
        first ~max_frames/fps seconds, e.g. for a 30-fps 40-s clip with
        fps=8 max=5 the prior logic took only the first 0.4 s of footage.
    """
    if not _AV_OK:
        return []
    try:
        container = _av_module.open(video_path)
        stream = container.streams.video[0]
        frame_rate = float(stream.average_rate) or 25.0

        # PyAV's stream.frames is reliable for most container formats; fall
        # back to duration*rate when it's 0 (some streaming codecs).
        total_frames = int(stream.frames or 0)
        if total_frames <= 0:
            duration_s = float(stream.duration * stream.time_base) if stream.duration else 0.0
            total_frames = max(1, int(duration_s * frame_rate))

        # How many frames the requested sample fps would produce if no cap.
        target_count = max(1, int(round(total_frames * (fps / frame_rate))))

        if target_count <= max_frames:
            # Below the cap — take every Nth frame from the start (fps == sample fps).
            interval = max(1, int(round(frame_rate / fps)))
            target_indices = set(range(0, total_frames, interval))
            n_to_take = max_frames
        else:
            # Hit the cap — distribute uniformly across the full video.
            n_to_take = max_frames
            if n_to_take == 1:
                target_indices = {0}
            else:
                target_indices = {
                    int(round(j * (total_frames - 1) / (n_to_take - 1)))
                    for j in range(n_to_take)
                }

        frames = []
        for i, frame in enumerate(container.decode(stream)):
            if i in target_indices:
                img = frame.to_image().convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=80)
                frames.append(base64.b64encode(buf.getvalue()).decode())
                if len(frames) >= n_to_take:
                    break
        container.close()
        return frames
    except Exception as e:
        print(f"[warn] frame extraction failed: {e}", flush=True)
        return []


# ── vLLM inference (OpenAI-compatible local server) ───────────────────────────
def _run_vllm_inference(video_path, prompt, system, fps, max_tokens, model_id, t_run_start=None, display_label=None, extra_note=None, is_image=False, temperature=0.0, top_p=1.0, rep_penalty=1.0):
    """Generator: (response_text, status_html, table_html) via local vLLM/NIM server."""
    steps = VLLM_STEPS
    if t_run_start is None:
        t_run_start = time.time()
    _be_label = "NIM" if INFERENCE_BACKEND == "nim_local" else "VLLM"

    # NIM-8B-FP8-THINK-EOS safety net: at greedy decode the FP8 model emits
    # `<think>` then an EOS-like token, finishing in 2-3 tokens with no answer.
    # Clamp to a small positive temperature to break the deterministic trap.
    if INFERENCE_BACKEND == "nim_local" and temperature < 0.3:
        print(f"[nim_local] Clamping temperature {temperature} → 0.3 (NIM-8B-FP8 <think>+EOS bug at greedy decode)", flush=True)
        temperature = 0.3

    def _elapsed():
        return time.time() - t_run_start

    # Step 1
    yield "", _status_html(["run", "wait", "wait", "wait", "wait"],
                           {"elapsed_s": _elapsed(), "backend": _be_label}, steps=steps), gr.update()

    if not _REQUESTS_OK:
        msg = "[vLLM] Skipped — requests library not installed"
        _log_run(model_id, total_s=_elapsed(), status="skipped-no-requests", display_label=display_label)
        yield msg, _status_html(["ok", "wait", "wait", "wait", "wait"],
                                {"elapsed_s": _elapsed(), "backend": _be_label}, steps=steps), _table_html()
        return

    # Step 2: ping server
    endpoint = f"{VLLM_BASE_URL}/chat/completions"
    print(f"[vllm] endpoint={endpoint} model={model_id}", flush=True)
    yield "", _status_html(["ok", "ok", "run", "wait", "wait"],
                           {"model_id": model_id, "elapsed_s": _elapsed(), "backend": _be_label},
                           steps=steps), gr.update()

    # Step 3: prepare media content
    # Images: single image_url (file:// for Nemotron/Qwen, base64 for CR2/C3).
    # Videos: Nemotron/Qwen3-VL use file:// video_url; CR2/C3 use base64 JPEG frames.
    _nem = _uses_file_url(model_id) or _uses_file_url(_SERVER_MODEL_ID or "")
    if is_image:
        if _nem:
            content = [
                {"type": "image_url", "image_url": {"url": f"file://{video_path}"}},
                {"type": "text", "text": prompt},
            ]
            print(f"[vllm/image] file:// image: {video_path}", flush=True)
        else:
            try:
                from PIL import Image as _pil_img
                _img = _pil_img.open(video_path).convert("RGB")
                _buf = io.BytesIO()
                _img.save(_buf, format="JPEG", quality=85)
                _img_b64 = base64.b64encode(_buf.getvalue()).decode()
            except Exception as _img_err:
                msg = f"[vLLM ERROR] Could not read image: {_img_err}"
                _log_run(model_id, total_s=_elapsed(), status="image-error", display_label=display_label)
                yield msg, _status_html(["ok", "ok", "wait", "wait", "wait"],
                                        {"elapsed_s": _elapsed(), "backend": _be_label}, steps=steps), _table_html()
                return
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_img_b64}"}},
            ]
            print(f"[vllm/image] base64 image prepared", flush=True)
    elif _nem:
        import shutil as _shutil
        _nem_path = "/tmp/gradio_upload.mp4"
        try:
            if os.path.abspath(video_path) != os.path.abspath(_nem_path):
                _shutil.copy2(video_path, _nem_path)
        except Exception as _cp_err:
            _nem_path = video_path  # fall back to original path if copy fails
            print(f"[vllm/nemotron] copy to /tmp failed ({_cp_err}), using original path", flush=True)
        content = [
            {"type": "video_url", "video_url": {"url": f"file://{_nem_path}"}},
            {"type": "text", "text": prompt},
        ]
        print(f"[vllm/nemotron] video_url: file://{_nem_path}", flush=True)
    else:
        # NIM container hard-caps at 5 images/prompt; vLLM has no such cap.
        _max_frames = 5 if INFERENCE_BACKEND == "nim_local" else 8
        print(f"[vllm] Extracting frames fps={fps} max={_max_frames}", flush=True)
        frames_b64 = _extract_frames_b64(video_path, fps=fps, max_frames=_max_frames)
        if not frames_b64:
            msg = "[vLLM ERROR] Could not extract frames (PyAV missing or video unreadable)"
            _log_run(model_id, total_s=_elapsed(), status="frame-error", display_label=display_label)
            yield msg, _status_html(["ok", "ok", "wait", "wait", "wait"],
                                    {"elapsed_s": _elapsed(), "backend": _be_label}, steps=steps), _table_html()
            return
        print(f"[vllm] {len(frames_b64)} frames extracted", flush=True)
        content = [{"type": "text", "text": f"[Video — {len(frames_b64)} frames at {fps}fps]\n{prompt}"}]
        for fb64 in frames_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{fb64}"},
            })

    # Step 4: send to vLLM

    yield "", _status_html(["ok", "ok", "ok", "run", "wait"],
                            {"model_id": model_id, "elapsed_s": _elapsed(), "backend": _be_label},
                            steps=steps), gr.update()

    t_start = time.time()
    try:
        resp = _requests.post(
            endpoint,
            headers={"Authorization": f"Bearer {VLLM_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": content},
                ],
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": rep_penalty,
                "stream": True,
            },
            stream=True,
            timeout=180,
        )
        resp.raise_for_status()
    except Exception as e:
        e_str = str(e)
        if "Connection refused" in e_str or "Failed to establish" in e_str or "ConnectionError" in type(e).__name__:
            if INFERENCE_BACKEND == "nim_local":
                err = (
                    f"[NIM] Container not responding at {VLLM_BASE_URL}\n\n"
                    "Start the NIM container first:\n"
                    "  bash /tmp/nim_launch.sh <NGC_API_KEY> [HF_TOKEN]\n\n"
                    "Or manually:\n"
                    "  docker run -d --gpus all --shm-size=16GB -p 8000:8000 \\\n"
                    "    -e NGC_API_KEY=<key> nvcr.io/nim/nvidia/cosmos-reason2-2b:latest\n\n"
                    "Check status: docker logs cosmos-nim\n"
                    "Check ready: curl http://localhost:8000/v1/models"
                )
            else:
                err = (
                    f"[vLLM] Server not running at {VLLM_BASE_URL}\n\n"
                    "Restart vLLM on the instance, then retry:\n"
                    f"  vllm serve <model-path> --served-model-name <model-id> "
                    "--port 8000 --dtype auto --trust-remote-code --max-model-len 8192"
                )
        else:
            err = f"[{_be_label} ERROR] {e}"
        print(err, flush=True)
        _log_run(model_id, total_s=_elapsed(), status="api-error", display_label=display_label)
        yield err, _status_html(["ok", "ok", "ok", "wait", "wait"],
                                 {"model_id": model_id, "elapsed_s": _elapsed(), "backend": _be_label},
                                 steps=steps), _table_html()
        return

    # Step 5: stream
    parts          = []
    ttft_s         = None
    tokens_decoded = 0

    for line in resp.iter_lines():
        if not line:
            continue
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            delta = json.loads(data)["choices"][0]["delta"].get("content", "")
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
        if delta:
            if ttft_s is None:
                ttft_s = time.time() - t_start
            parts.append(delta)
            tokens_decoded += 1
            yield "".join(parts), _status_html(
                ["ok", "ok", "ok", "ok", "run"],
                {"model_id": model_id, "tokens_out": tokens_decoded,
                 "ttft_s": ttft_s, "elapsed_s": _elapsed(), "backend": _be_label},
                steps=steps,
            ), gr.update()

    infer_time = time.time() - t_start
    response   = "".join(parts)
    total_s    = _elapsed()
    print(f"[vllm done] {infer_time:.1f}s · {tokens_decoded} tok out", flush=True)
    _log_run(model_id, ttft_s=ttft_s, infer_s=infer_time,
             tokens_out=tokens_decoded, total_s=total_s, status="ok",
             display_label=display_label, notes=extra_note or "")
    yield response, _status_html(
        ["ok", "ok", "ok", "ok", "ok"],
        {"model_id": model_id, "tokens_out": tokens_decoded, "ttft_s": ttft_s,
         "infer_s": infer_time, "elapsed_s": total_s, "backend": _be_label},
        steps=steps,
    ), _table_html()


# ── NIM inference ──────────────────────────────────────────────────────────────
def _run_nim_inference(video_path, prompt, system, fps, max_tokens, model_id, t_run_start=None, display_label=None, is_image=False):
    """Generator: (response_text, status_html, table_html) via NVCF streaming API."""
    steps = NIM_STEPS
    if t_run_start is None:
        t_run_start = time.time()

    def _elapsed():
        return time.time() - t_run_start

    # Step 1
    yield "", _status_html(["run", "wait", "wait", "wait", "wait"],
                           {"elapsed_s": _elapsed()}, steps=steps), gr.update()

    if not _REQUESTS_OK:
        msg = "[NIM] Skipped — requests library not installed"
        _log_run(model_id, total_s=_elapsed(), status="skipped-no-requests", display_label=display_label)
        yield msg, _status_html(
            ["ok", "wait", "wait", "wait", "wait"],
            {"elapsed_s": _elapsed()}, steps=steps
        ), _table_html()
        return

    if not NIM_MODEL_API:
        msg = (f"[NIM] Skipped — {MODEL_SIZE} not in public NVCF catalog. "
               f"Ask DLAlgo team for internal NIM access.")
        _log_run(model_id, total_s=_elapsed(), status="skipped-no-catalog", display_label=display_label)
        yield msg, _status_html(
            ["ok", "wait", "wait", "wait", "wait"],
            {"elapsed_s": _elapsed()}, steps=steps
        ), _table_html()
        return

    if not NGC_API_KEY:
        msg = "[NIM] Skipped — NGC_API_KEY not set (export NGC_API_KEY=nvapi-...)"
        _log_run(model_id, total_s=_elapsed(), status="skipped-no-key", display_label=display_label)
        yield msg, _status_html(
            ["ok", "wait", "wait", "wait", "wait"],
            {"elapsed_s": _elapsed()}, steps=steps
        ), _table_html()
        return

    # Step 2: validate key (lightweight)
    print(f"[nim] NGC_API_KEY present ({len(NGC_API_KEY)} chars)", flush=True)
    yield "", _status_html(["ok", "ok", "run", "wait", "wait"],
                           {"elapsed_s": _elapsed()}, steps=steps), gr.update()

    # Step 3: prepare media content
    if is_image:
        try:
            from PIL import Image as _pil_img
            _img = _pil_img.open(video_path).convert("RGB")
            _buf = io.BytesIO()
            _img.save(_buf, format="JPEG", quality=85)
            _img_b64 = base64.b64encode(_buf.getvalue()).decode()
        except Exception as _img_err:
            msg = f"[NIM ERROR] Could not read image: {_img_err}"
            _log_run(model_id, total_s=_elapsed(), status="image-error", display_label=display_label)
            yield msg, _status_html(["ok", "ok", "wait", "wait", "wait"],
                               {"elapsed_s": _elapsed()}, steps=steps), _table_html()
            return
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_img_b64}"}},
        ]
        print(f"[nim/image] base64 image prepared", flush=True)
    else:
        print(f"[nim] Extracting frames fps={fps} max={8}", flush=True)
        frames_b64 = _extract_frames_b64(video_path, fps=fps, max_frames=8)
        if not frames_b64:
            msg = "[NIM ERROR] Could not extract frames from video (PyAV missing or video unreadable)"
            _log_run(model_id, total_s=_elapsed(), status="frame-error", display_label=display_label)
            yield msg, _status_html(["ok", "ok", "wait", "wait", "wait"],
                               {"elapsed_s": _elapsed()}, steps=steps), _table_html()
            return
        print(f"[nim] {len(frames_b64)} frames extracted", flush=True)
        content = [{"type": "text", "text": f"[Video — {len(frames_b64)} frames at {fps}fps]\n{prompt}"}]
        for fb64 in frames_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{fb64}"},
            })

    # Step 4: send to NVCF

    yield "", _status_html(["ok", "ok", "ok", "run", "wait"],
                            {"model_id": model_id, "elapsed_s": _elapsed()}, steps=steps), gr.update()

    t_start = time.time()
    try:
        resp = _requests.post(
            NIM_ENDPOINT,
            headers={"Authorization": f"Bearer {NGC_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": NIM_MODEL_API,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": content},
                ],
                "max_tokens": max_tokens,
                "stream": True,
            },
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()
    except Exception as e:
        err = f"[NIM ERROR] {e}"
        print(err, flush=True)
        _log_run(model_id, total_s=_elapsed(), status="api-error")
        yield err, _status_html(["ok", "ok", "ok", "wait", "wait"],
                                  {"model_id": model_id, "elapsed_s": _elapsed()}, steps=steps), _table_html()
        return

    # Step 5: stream
    parts  = []
    ttft_s = None
    tokens_decoded = 0

    for line in resp.iter_lines():
        if not line:
            continue
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            delta = json.loads(data)["choices"][0]["delta"].get("content", "")
        except (json.JSONDecodeError, KeyError, IndexError):
            continue
        if delta:
            if ttft_s is None:
                ttft_s = time.time() - t_start
            parts.append(delta)
            tokens_decoded += 1
            cur_metrics = {
                "model_id": model_id,
                "tokens_out": tokens_decoded,
                "ttft_s": ttft_s,
                "elapsed_s": _elapsed(),
            }
            yield "".join(parts), _status_html(
                ["ok", "ok", "ok", "ok", "run"], cur_metrics, steps=steps
            ), gr.update()

    infer_time = time.time() - t_start
    response   = "".join(parts)
    total_s    = _elapsed()

    result = {
        "model": model_id, "prompt": prompt, "response": response,
        "infer_time_s": round(infer_time, 1),
        "ttft_s": round(ttft_s, 2) if ttft_s else None,
        "tokens_out": tokens_decoded,
        "frames_sent": len(frames_b64), "fps": fps, "status": "success",
    }
    try:
        with open(OUT_FILE, "w") as f:
            json.dump(result, f, indent=2)
    except Exception:
        pass

    print(f"[nim done] {infer_time:.1f}s · {tokens_decoded} tok out", flush=True)
    _log_run(model_id, ttft_s=ttft_s, infer_s=infer_time,
             tokens_out=tokens_decoded, total_s=total_s, status="ok", display_label=display_label)
    yield response, _status_html(
        ["ok", "ok", "ok", "ok", "ok"],
        {"model_id": model_id, "tokens_out": tokens_decoded,
         "ttft_s": ttft_s, "infer_s": infer_time, "elapsed_s": total_s},
        steps=steps,
    ), _table_html()


# ── HF inference ───────────────────────────────────────────────────────────────
def run_inference(video_path, user_prompt, system_prompt, fps, max_pixels, max_new_tokens, model_id, disable_autocap=False, display_label=None, is_image=False, temperature=0.0, top_p=1.0, rep_penalty=1.0):
    """Generator: (response_text, status_html, table_html). Routes to NIM or HF path."""
    if video_path is None:
        yield "Upload a video or image first.", _status_html(["wait"] * 5), gr.update()
        return
    if not user_prompt.strip():
        user_prompt = DEFAULT_PROMPT

    fps            = int(fps)
    max_pixels     = int(max_pixels)
    max_new_tokens = int(max_new_tokens)
    t_run_start    = time.time()

    def _elapsed():
        return time.time() - t_run_start

    # Step 1: resolve checkpoint
    yield "", _status_html(["run", "wait", "wait", "wait", "wait"],
                           {"elapsed_s": _elapsed()}), gr.update()

    # NIM local Docker is served by `_run_vllm_inference` via the OpenAI-compatible
    # client at VLLM_BASE_URL. The NIM container enforces a 5-image-per-prompt cap;
    # frame clamping is applied below at extraction time when INFERENCE_BACKEND=nim_local.

    if _is_nim(model_id):
        yield from _run_nim_inference(
            video_path, user_prompt, system_prompt, fps, max_new_tokens, model_id,
            t_run_start=t_run_start,
            display_label=display_label,
            is_image=is_image,
        )
        return

    if _is_vllm(model_id):
        # Use the server's actual served model name (from --served-model-name),
        # not the local checkpoint path which the API doesn't recognise.
        _effective_mid = _SERVER_MODEL_ID or model_id
        # When serving a different model than requested, note it in the table.
        _req_short = model_id.split("/")[-1] if "/" in model_id else model_id
        _srv_short = _effective_mid.split("/")[-1] if "/" in _effective_mid else _effective_mid
        _extra_note = f"vLLM serves {_srv_short}" if _srv_short != _req_short else None
        yield from _run_vllm_inference(
            video_path, user_prompt, system_prompt, fps, max_new_tokens, _effective_mid,
            t_run_start=t_run_start,
            display_label=display_label,
            extra_note=_extra_note,
            is_image=is_image,
            temperature=temperature,
            top_p=top_p,
            rep_penalty=rep_penalty,
        )
        return

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[infer] model={model_id} gpu={gpu_name}", flush=True)

    # Step 2: load model
    yield "", _status_html(["ok", "run", "wait", "wait", "wait"],
                           {"model_id": model_id, "elapsed_s": _elapsed()}), gr.update()
    try:
        model, processor, load_t = _load(model_id)
    except Exception as e:
        err = f"[ERROR] Failed to load {model_id}: {e}"
        print(err, flush=True)
        _log_run(model_id, total_s=_elapsed(), status="load-error",
                 notes=f"Load failed: {str(e)[:80]}", display_label=display_label)
        yield err, _status_html(["ok", "wait", "wait", "wait", "wait"],
                                {"model_id": model_id, "elapsed_s": _elapsed(),
                                 "backend": "HF"}), _table_html()
        return

    _upcast = _loaded.get("upcast_warning")

    # Step 3: preprocess video
    yield "", _status_html(
        ["ok", "ok", "run", "wait", "wait"],
        {"model_id": model_id, "load_s": load_t, "elapsed_s": _elapsed(),
         "upcast_warning": _upcast, "backend": "HF"},
    ), gr.update()

    if is_image:
        # Image auto-cap: same _auto_cap math as video with n_frames=1, then
        # bound at IMAGE_AUTO_CAP_MAX. _EMPIRICAL_VPF is calibrated for video
        # token math; for Cosmos3 single-image inputs the per-pixel token rate
        # is ~3-5x higher, so unbounded auto-cap snaps to the 2M-px tier and
        # still produces 5+ min prefills on HF backend.
        if not disable_autocap:
            capped_px, est_s = _auto_cap(1, max_pixels)
            capped_px = min(capped_px, IMAGE_AUTO_CAP_MAX)
            if capped_px < max_pixels:
                print(f"[auto-cap image] {max_pixels:,} → {capped_px:,} px (~{est_s:.0f}s est, hard cap={IMAGE_AUTO_CAP_MAX:,})", flush=True)
                max_pixels = capped_px
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                {"type": "image", "image": video_path, "max_pixels": max_pixels},
                {"type": "text", "text": user_prompt},
            ]},
        ]
    else:
        m = get_video_meta(video_path)
        if m.duration_s > 0 and not disable_autocap:
            n_frames = max(1, int(m.duration_s * fps))
            capped_px, est_s = _auto_cap(n_frames, max_pixels)
            if capped_px < max_pixels:
                print(f"[auto-cap] {max_pixels:,} → {capped_px:,} px (~{est_s:.0f}s est)", flush=True)
                max_pixels = capped_px
        # BUG-FPS-NFRAMES: newer qwen_vl_utils raises "Only accept either fps or nframes"
        # if both keys are present in the video dict. Pick whichever yields fewer frames:
        # use nframes (uniform subsample to _MAX_HF_FRAMES) when fps would exceed the cap,
        # otherwise pass fps and let the backend sample naturally.
        n_at_fps = max(1, int((m.duration_s if m.duration_s > 0 else 1) * fps))
        video_part = {"type": "video", "video": video_path, "max_pixels": max_pixels}
        if n_at_fps > _MAX_HF_FRAMES:
            video_part["nframes"] = _MAX_HF_FRAMES
        else:
            video_part["fps"] = fps
        conversation = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [
                video_part,
                {"type": "text", "text": user_prompt},
            ]},
        ]

    t1 = time.time()
    text_prompt = processor.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = qwen_vl_utils.process_vision_info(conversation)
    inputs = processor(
        text=[text_prompt],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    tokens_in = inputs["input_ids"].shape[1]

    # Step 4: prefill
    print(f"[infer] Prefilling {tokens_in:,} tokens ...", flush=True)
    yield "", _status_html(
        ["ok", "ok", "ok", "run", "wait"],
        {"model_id": model_id, "load_s": load_t, "tokens_in": tokens_in,
         "elapsed_s": _elapsed(), "upcast_warning": _upcast, "backend": "HF"},
    ), gr.update()

    streamer = TextIteratorStreamer(
        processor.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
        timeout=180.0,
    )
    gen_thread = threading.Thread(
        target=model.generate,
        kwargs={
            **inputs,
            "max_new_tokens":     max_new_tokens,
            "do_sample":          False,
            "repetition_penalty": 1.05,
            "streamer":           streamer,
        },
        daemon=True,
    )
    gen_thread.start()

    # Step 5: generate
    parts          = []
    ttft_s         = None
    tokens_decoded = 0
    t_gen          = time.time()

    for chunk in streamer:
        if chunk:
            if ttft_s is None:
                ttft_s = time.time() - t_gen
            parts.append(chunk)
            tokens_decoded += 1  # TextIteratorStreamer emits ~1 token per chunk
            cur_metrics = {
                "model_id":       model_id,
                "load_s":         load_t,
                "tokens_in":      tokens_in,
                "tokens_out":     tokens_decoded,
                "ttft_s":         ttft_s,
                "elapsed_s":      _elapsed(),
                "upcast_warning": _upcast,
                "backend":        "HF",
            }
            yield "".join(parts), _status_html(
                ["ok", "ok", "ok", "ok", "run"], cur_metrics
            ), gr.update()

    gen_thread.join(timeout=10)
    infer_time = time.time() - t1
    response   = "".join(parts)
    tokens_out = len(processor.tokenizer.encode(response, add_special_tokens=False))

    result = {
        "model":        model_id,
        "prompt":       user_prompt,
        "response":     response,
        "load_time_s":  round(load_t, 1),
        "infer_time_s": round(infer_time, 1),
        "ttft_s":       round(ttft_s, 2) if ttft_s else None,
        "tokens_in":    tokens_in,
        "tokens_out":   tokens_out,
        "fps":          fps,
        "status":       "success",
    }
    try:
        with open(OUT_FILE, "w") as f:
            json.dump(result, f, indent=2)
    except Exception:
        pass

    total_s = _elapsed()
    print(f"[done] {infer_time:.1f}s · {tokens_out} tok · ttft={ttft_s:.2f}s", flush=True)
    _log_run(model_id, load_s=load_t, ttft_s=ttft_s, infer_s=infer_time,
             tokens_out=tokens_out, total_s=total_s, status="ok",
             notes=_upcast or "", display_label=display_label)
    yield response, _status_html(
        ["ok", "ok", "ok", "ok", "ok"],
        {"model_id": model_id, "load_s": load_t,
         "tokens_in": tokens_in, "tokens_out": tokens_out,
         "ttft_s": ttft_s, "infer_s": infer_time, "elapsed_s": total_s,
         "upcast_warning": _upcast, "backend": "HF"},
    ), _table_html()


# ── Sequential variant runner ──────────────────────────────────────────────────
def run_all_variants(video_path, user_prompt, system_prompt, fps, max_pixels, max_new_tokens, disable_autocap=False, reload_vllm=False):
    """Generator: (combined_text, status_html, table_html). Runs all MODEL_SIZE variants."""
    if video_path is None:
        yield "Upload a video first.", _status_html(["wait"] * 5), gr.update()
        return

    # Build full variant list including extended (e.g. 32B after 8B BF16)
    _hf_variants = [
        (label, dirname, hf_id, _resolve(dirname, hf_id))
        for label, dirname, hf_id, _ in _cfg["variants"]
    ]
    _extended = [
        (label, dirname, hf_id, _resolve(dirname, hf_id))
        for label, dirname, hf_id, _ in _cfg.get("extended_variants", [])
    ]
    _nim_entry = [(f"NIM-{MODEL_SIZE}", None, _nim_api_id, f"nim://{_nim_api_id}")]
    # Order: all but last HF, then NIM, then last HF (BF16 = most accurate), then 32B
    _variants_full = _hf_variants[:-1] + _nim_entry + _hf_variants[-1:] + _extended

    _do_reload = reload_vllm and INFERENCE_BACKEND == "vllm"

    if INFERENCE_BACKEND in ("vllm", "nim_local"):
        _be_hdr = "NIM (local Docker)" if INFERENCE_BACKEND == "nim_local" else "vLLM"
        if _do_reload:
            combined = (f"[{_be_hdr} mode — real benchmarks: reloading vLLM per variant]\n"
                        f"Each variant restarts the vLLM server. Gradio stays live. "
                        f"Allow ~60–90s between variants for model loading.\n\n")
        else:
            combined = (f"[{_be_hdr} mode — backend: {VLLM_BASE_URL}]\n"
                        f"Serves one model at a time. Each variant uses the loaded model; "
                        f"non-matching variants are noted in the table.\n\n")
    else:
        combined = ""

    all_results = {}
    bench_file  = "/tmp/byo_video_benchmark.json"

    for var_label, var_dirname, var_hf_id, var_model_id in _variants_full:
        header_line = f"\n{'='*52}\n▶  {var_label}\n{'='*52}\n"
        combined += header_line

        # Hot-swap vLLM if requested and this is a non-NIM HF variant
        if _do_reload and not _is_nim(var_model_id) and var_hf_id:
            if not os.path.exists(var_model_id):
                # Model not on disk — skip swap, preserve running server
                combined += (
                    f"[Skipping vLLM reload for {var_label} — model not on disk. "
                    f"Running server will be used.]\n"
                )
                yield combined, _status_html(
                    ["run", "wait", "wait", "wait", "wait"],
                    {"model_id": var_model_id, "backend": "VLLM"},
                    steps=VLLM_STEPS,
                ), gr.update()
            else:
                launched = _launch_vllm_swap(var_model_id, var_hf_id)
                if not launched:
                    err = f"[vLLM binary not found — skipping {var_label}]\n"
                    combined += err
                    _log_run(var_model_id, total_s=0, status="reload-error", display_label=var_label)
                    yield combined, _status_html(["ok", "wait", "wait", "wait", "wait"],
                                                 {}, steps=VLLM_STEPS), _table_html()
                    continue
                # Inline polling loop — yields progress every 5 s so Gradio stays responsive
                _swap_start = time.time()
                actual_id   = None
                while True:
                    _elapsed = time.time() - _swap_start
                    if _elapsed > _VLLM_SWAP_TIMEOUT:
                        break
                    _remain_lo = max(0, 60 - _elapsed)
                    _remain_hi = max(0, 90 - _elapsed)
                    _prog = (
                        f"[Loading {var_label} into vLLM — "
                        f"{_elapsed:.0f}s elapsed, "
                        f"~{_remain_lo:.0f}–{_remain_hi:.0f}s remaining…]\n"
                    )
                    yield combined + _prog, _status_html(
                        ["run", "wait", "wait", "wait", "wait"],
                        {"model_id": var_model_id, "backend": "VLLM"},
                        steps=VLLM_STEPS,
                    ), gr.update()
                    time.sleep(5)
                    try:
                        import urllib.request as _urlreq2
                        with _urlreq2.urlopen(f"{VLLM_BASE_URL}/models", timeout=3) as _r2:
                            _mdata = json.loads(_r2.read())
                        _ids = [m["id"] for m in _mdata.get("data", [])]
                        if _ids:
                            actual_id = _ids[0]
                            break
                    except Exception:
                        pass
                if actual_id is None:
                    err = f"[vLLM reload timed out for {var_label}. Skipping.]\n"
                    combined += err
                    _log_run(var_model_id, total_s=0, status="reload-timeout", display_label=var_label)
                    yield combined, _status_html(["ok", "wait", "wait", "wait", "wait"],
                                                 {}, steps=VLLM_STEPS), _table_html()
                    continue
                global _SERVER_MODEL_ID
                _SERVER_MODEL_ID = actual_id
                _swap_dur = time.time() - _swap_start
                combined += f"[vLLM ready — serving {actual_id} ({_swap_dur:.0f}s)]\n"

        yield combined, _status_html(
            ["run", "wait", "wait", "wait", "wait"],
            {"model_id": var_model_id},
        ), gr.update()

        last_text   = ""
        last_table  = gr.update()

        for text, status, tbl in run_inference(
            video_path, user_prompt, system_prompt,
            fps, max_pixels, max_new_tokens, var_model_id,
            disable_autocap=disable_autocap,
            display_label=var_label,
        ):
            last_text  = text
            last_table = tbl
            yield combined + text, status, tbl

        if not _is_nim(var_model_id) and not _is_vllm(var_model_id):
            _unload()

        all_results[var_label] = {"model": var_model_id, "response": last_text}
        combined += last_text + "\n"

    # Save comparison JSON
    try:
        with open(bench_file, "w") as f:
            json.dump(all_results, f, indent=2)
        combined += f"\n*All results → {bench_file}*"
    except Exception:
        pass

    yield combined, _status_html(["ok", "ok", "ok", "ok", "ok"]), _table_html()


# ── Startup: pre-load default model if available locally ───────────────────────
# PRELOAD-001: skip HF preload in vLLM mode — model is served by vLLM process, not Python.
# Loading here wastes ~9 GB VRAM (weights loaded twice) and risks OOM on 80 GB GPUs.
# Set SKIP_HF_PRELOAD=1 to force-skip even in HF mode (useful when VRAM is tight).
_SKIP_PRELOAD = (
    INFERENCE_BACKEND == "vllm"
    or os.environ.get("SKIP_HF_PRELOAD", "").lower() in ("1", "true", "yes")
)
_preloaded_ok = False
if os.path.exists(MODEL_DIR) and not _SKIP_PRELOAD:
    print(f"[demo] Pre-loading {MODEL_DIR} ...", flush=True)
    try:
        _load(MODEL_DIR)
        _preloaded_ok = True
    except Exception as e:
        print(f"[demo] Pre-load failed ({e}) — model loads on first inference", flush=True)
elif _SKIP_PRELOAD and INFERENCE_BACKEND == "vllm":
    print(f"[demo] vLLM mode — skipping HF preload (PRELOAD-001 fix)", flush=True)
else:
    print(f"[demo] MODEL_DIR not found — model loads on first inference", flush=True)

gpu_name  = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
free_vram = get_free_vram_mib()
load_time = _loaded["load_time"]

_header_model = _loaded["model_id"] or MODEL_NAME
_load_note    = f"Load: {load_time:.1f}s" if _preloaded_ok else "Load: on demand"


# ── vLLM hot-swap (restart server without killing Gradio) ─────────────────────
_VLLM_PROC = None  # track last vLLM subprocess so we can kill it cleanly

_VLLM_SWAP_TIMEOUT = 150  # seconds to wait for vLLM to become ready

def _launch_vllm_swap(local_path, served_name, gpu_util="0.85", max_model_len="32768"):
    """
    Kill the current vLLM server and launch a new one for `local_path`.
    Returns immediately — caller is responsible for polling /v1/models.
    Returns True on successful launch, False if the vLLM binary is not found.
    """
    global _VLLM_PROC
    import subprocess as _sp

    vllm_bin = os.path.join(HOME, "cosmos-reason2/.venv/bin/vllm")
    if not os.path.exists(vllm_bin):
        print(f"[vllm-swap] vLLM binary not found at {vllm_bin}", flush=True)
        return False

    _sp.run(["pkill", "-f", "vllm serve"], capture_output=True)
    if _VLLM_PROC is not None:
        try:
            _VLLM_PROC.terminate()
        except Exception:
            pass
        _VLLM_PROC = None
    time.sleep(4)  # allow GPU memory to release

    # Per-model extra flags (e.g. --allowed-local-media-path for Nemotron/Qwen).
    _model_cfg   = MODEL_CONFIGS.get(MODEL_SIZE, {})
    extra_flags  = _model_cfg.get("vllm_swap_flags", [])
    extra_env    = {**os.environ, **_model_cfg.get("vllm_swap_env", {})}

    print(f"[vllm-swap] Launching {served_name} (path={local_path}) flags={extra_flags}", flush=True)
    _VLLM_PROC = _sp.Popen(
        [
            vllm_bin, "serve", local_path,
            "--served-model-name", served_name,
            "--port", "8000",
            "--dtype", "auto",
            "--trust-remote-code",
            "--max-model-len", max_model_len,
            "--gpu-memory-utilization", gpu_util,
        ] + extra_flags,
        env=extra_env,
        stdout=_sp.DEVNULL,
        stderr=_sp.DEVNULL,
    )
    return True


def _swap_banner_html(model_short, state, elapsed=0, remain_lo=60, remain_hi=90):
    if state == "loading":
        return (
            f'<div style="background:#7c4d00;border:1px solid #f59e0b;border-radius:6px;'
            f'padding:10px 14px;font-size:13px;color:#fef3c7;margin:4px 0">'
            f'⚠ Restarting vLLM for <b>{model_short}</b>… '
            f'{elapsed:.0f}s elapsed, ~{remain_lo:.0f}–{remain_hi:.0f}s remaining. '
            f'Gradio stays live.</div>'
        )
    if state == "ready":
        return (
            f'<div style="background:#14532d;border:1px solid #22c55e;border-radius:6px;'
            f'padding:10px 14px;font-size:13px;color:#dcfce7;margin:4px 0">'
            f'✅ vLLM now serving <b>{model_short}</b> ({elapsed:.0f}s)</div>'
        )
    if state == "error":
        return (
            f'<div style="background:#450a0a;border:1px solid #f87171;border-radius:6px;'
            f'padding:10px 14px;font-size:13px;color:#fee2e2;margin:4px 0">'
            f'❌ vLLM reload failed for <b>{model_short}</b> (timed out after {elapsed:.0f}s). '
            f'Check server logs.</div>'
        )
    if state == "nodisk":
        return (
            f'<div style="background:#1c1917;border:1px solid #a8a29e;border-radius:6px;'
            f'padding:10px 14px;font-size:13px;color:#d6d3d1;margin:4px 0">'
            f'⚠ <b>{model_short}</b> not on disk — use HF token to download. '
            f'Running server unchanged.</div>'
        )
    return ""


def _vllm_swap_yields(label, local_path, hf_id):
    """Shared generator for the vLLM swap+poll loop.
    Yields (banner_html, btn_update, fps_update, maxpx_update)."""
    _fps, _mpx, _mtok = _ckpt_slider_defaults(label)
    _no_change = gr.update()
    yield _swap_banner_html(label, "loading", 0, 60, 90), gr.update(visible=False), _no_change, _no_change
    launched = _launch_vllm_swap(local_path, hf_id)
    if not launched:
        yield _swap_banner_html(label, "error", 0), gr.update(visible=True), _no_change, _no_change
        return
    _swap_start = time.time()
    actual_id = None
    while True:
        _elapsed = time.time() - _swap_start
        if _elapsed > _VLLM_SWAP_TIMEOUT:
            break
        _remain_lo = max(0, 60 - _elapsed)
        _remain_hi = max(0, 90 - _elapsed)
        yield _swap_banner_html(label, "loading", _elapsed, _remain_lo, _remain_hi), gr.update(visible=False), _no_change, _no_change
        time.sleep(5)
        try:
            import urllib.request as _urlreq3
            with _urlreq3.urlopen(f"{VLLM_BASE_URL}/models", timeout=3) as _r3:
                _mdata3 = json.loads(_r3.read())
            _ids3 = [m["id"] for m in _mdata3.get("data", [])]
            if _ids3:
                actual_id = _ids3[0]
                break
        except Exception:
            pass
    _elapsed_final = time.time() - _swap_start
    if actual_id is None:
        yield _swap_banner_html(label, "error", _elapsed_final), gr.update(visible=True), _no_change, _no_change
        return
    global _SERVER_MODEL_ID
    _SERVER_MODEL_ID = actual_id
    yield _swap_banner_html(label, "ready", _elapsed_final), gr.update(visible=False), \
          gr.update(value=_fps), gr.update(value=_mpx)


def _on_checkpoint_change(label):
    """Reload vLLM when the user selects a new checkpoint (vLLM mode only).
    Yields (banner_html, download_btn_update, fps_update, maxpx_update)."""
    _no_change = gr.update()
    if INFERENCE_BACKEND != "vllm":
        yield "", gr.update(visible=False), _no_change, _no_change
        return
    meta = _VLLM_DD_META.get(label)
    if meta is None:
        yield "", gr.update(visible=False), _no_change, _no_change
        return
    local_path, hf_id = meta
    _fps, _mpx, _mtok = _ckpt_slider_defaults(label)
    if not os.path.exists(local_path):
        yield _swap_banner_html(label, "nodisk"), gr.update(visible=True), \
              gr.update(value=_fps), gr.update(value=_mpx)
        return
    yield from _vllm_swap_yields(label, local_path, hf_id)


def _on_download_and_load(label):
    """Download model from HF Hub then load it into vLLM.
    Yields (banner_html, download_btn_update, fps_update, maxpx_update)."""
    _no_change = gr.update()
    meta = _VLLM_DD_META.get(label)
    if meta is None:
        yield _swap_banner_html(label, "nodisk"), gr.update(visible=True), _no_change, _no_change
        return
    local_path, hf_id = meta
    if os.path.exists(local_path):
        yield from _vllm_swap_yields(label, local_path, hf_id)
        return

    import threading as _thr

    _dl_state = {"done": False, "error": None}

    def _dl():
        try:
            from huggingface_hub import snapshot_download as _snap
            _snap(
                repo_id=hf_id,
                local_dir=local_path,
                token=HF_TOKEN or None,
                local_dir_use_symlinks=False,
            )
            # Clean HF cache blob for this model — files are now in local_dir
            try:
                import shutil as _sh
                _cache_repo = os.path.join(
                    _HF_CACHE_DIR, "models--" + hf_id.replace("/", "--")
                )
                if os.path.exists(_cache_repo):
                    _sh.rmtree(_cache_repo, ignore_errors=True)
            except Exception:
                pass
            _dl_state["done"] = True
        except Exception as e:
            _dl_state["error"] = str(e)
            _dl_state["done"] = True

    _thr.Thread(target=_dl, daemon=True).start()
    _dl_start = time.time()

    while not _dl_state["done"]:
        _elapsed = time.time() - _dl_start
        try:
            import subprocess as _sp2
            _sz = _sp2.check_output(
                ["du", "-sh", local_path], stderr=_sp2.DEVNULL, timeout=3
            ).decode().split()[0]
            _sz_note = f" · {_sz} downloaded"
        except Exception:
            _sz_note = ""
        yield (
            f'<div style="background:#1e3a5f;border:1px solid #60a5fa;border-radius:6px;'
            f'padding:10px 14px;font-size:13px;color:#dbeafe;margin:4px 0">'
            f'⬇ Downloading <b>{label}</b> from HF Hub{_sz_note} · '
            f'{_elapsed:.0f}s elapsed · large models take 10–30 min</div>'
        ), gr.update(visible=False), gr.update(), gr.update()
        time.sleep(10)

    if _dl_state["error"]:
        yield (
            f'<div style="background:#450a0a;border:1px solid #f87171;border-radius:6px;'
            f'padding:10px 14px;font-size:13px;color:#fee2e2;margin:4px 0">'
            f'❌ Download failed for <b>{label}</b>: {_dl_state["error"][:160]}</div>'
        ), gr.update(visible=True), gr.update(), gr.update()
        return

    yield from _vllm_swap_yields(label, local_path, hf_id)


# ── HF auth helpers ───────────────────────────────────────────────────────────
def _hf_apply_token(token_val):
    """Apply an HF token to the running process and HF token cache."""
    global HF_TOKEN
    token_val = token_val.strip()
    if not token_val:
        return '<div style="color:#fbbf24;font-size:12px">Enter a token first.</div>'
    try:
        from huggingface_hub import login as _hf_login, whoami as _hf_whoami
        _hf_login(token=token_val, add_to_git_credential=False)
        HF_TOKEN = token_val
        info = _hf_whoami(token=token_val)
        name = info.get("name", "unknown")
        return (
            f'<div style="color:#86efac;font-size:12px">'
            f'✅ Authenticated as <b>{name}</b>. Gated models now accessible in this session.</div>'
        )
    except Exception as e:
        return f'<div style="color:#f87171;font-size:12px">❌ Auth failed: {e}</div>'


def _hf_check_status():
    """Return current HF auth status."""
    try:
        from huggingface_hub import whoami as _hf_whoami
        info = _hf_whoami()
        return (
            f'<div style="color:#86efac;font-size:12px">'
            f'✅ Authenticated as <b>{info.get("name","?")}</b></div>'
        )
    except Exception:
        return '<div style="color:#fbbf24;font-size:12px">⚠ Not authenticated — gated models require a token.</div>'


# ── Disk / storage helpers ────────────────────────────────────────────────────
_HF_CACHE_DIR = os.path.join(HOME, ".cache", "huggingface", "hub")

def _disk_status_html():
    import subprocess as _sp
    try:
        df_out = _sp.check_output(["df", "-h", "/"], timeout=5).decode().strip().split("\n")
        parts = df_out[1].split()
        total, used, avail, pct = parts[1], parts[2], parts[3], parts[4]
        bar_color = "#f87171" if int(pct.rstrip("%")) >= 90 else "#fbbf24" if int(pct.rstrip("%")) >= 75 else "#86efac"

        rows = ""
        if os.path.exists(_MODELS_BASE):
            for d in sorted(os.listdir(_MODELS_BASE)):
                try:
                    sz = _sp.check_output(["du", "-sh", os.path.join(_MODELS_BASE, d)], stderr=_sp.DEVNULL, timeout=8).decode().split()[0]
                    rows += (
                        f'<tr><td style="padding:2px 8px;font-size:12px;color:#d1d5db">'
                        f'models/{d}</td>'
                        f'<td style="padding:2px 8px;font-size:12px;text-align:right;color:#d1d5db">{sz}</td></tr>'
                    )
                except Exception:
                    pass

        hf_sz = ""
        if os.path.exists(_HF_CACHE_DIR):
            try:
                hf_sz = _sp.check_output(["du", "-sh", _HF_CACHE_DIR], stderr=_sp.DEVNULL, timeout=10).decode().split()[0]
                rows += (
                    f'<tr style="border-top:1px solid #374151">'
                    f'<td style="padding:2px 8px;font-size:12px;color:#fbbf24">HF download cache</td>'
                    f'<td style="padding:2px 8px;font-size:12px;text-align:right;color:#fbbf24">{hf_sz} ← safe to clean</td></tr>'
                )
            except Exception:
                pass

        table = (
            f'<table style="width:100%;border-collapse:collapse;margin-top:6px">'
            f'<tr style="border-bottom:1px solid #374151">'
            f'<th style="text-align:left;font-size:11px;color:#6b7280;padding:2px 8px">Directory</th>'
            f'<th style="text-align:right;font-size:11px;color:#6b7280;padding:2px 8px">Size</th></tr>'
            f'{rows}</table>'
        ) if rows else ""

        return (
            f'<div style="font-size:13px;padding:4px 0">'
            f'<b>Disk:</b> <span style="color:{bar_color}">{used} used / {total} ({pct})</span>'
            f' &nbsp;·&nbsp; <span style="color:#86efac">{avail} free</span>'
            f'</div>{table}'
        )
    except Exception as e:
        return f'<div style="color:#f87171;font-size:12px">Could not read disk info: {e}</div>'


def _clean_hf_cache():
    if not os.path.exists(_HF_CACHE_DIR):
        return _disk_status_html() + '<div style="color:#86efac;font-size:12px;margin-top:4px">HF cache already empty.</div>'
    import subprocess as _sp, shutil as _sh
    try:
        sz = _sp.check_output(["du", "-sh", _HF_CACHE_DIR], stderr=_sp.DEVNULL, timeout=10).decode().split()[0]
        _sh.rmtree(_HF_CACHE_DIR, ignore_errors=True)
        msg = f'<div style="color:#86efac;font-size:12px;margin-top:4px">✅ Cleared {sz} from HF download cache.</div>'
    except Exception as e:
        msg = f'<div style="color:#f87171;font-size:12px;margin-top:4px">❌ {e}</div>'
    return _disk_status_html() + msg


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Cosmos Reason — BYO Video Demo",
    css=_REASONING_PANEL_CSS,
) as demo:

    _variant_labels = " → ".join(lbl for lbl, _, _, _ in _cfg["variants"])
    if _cfg["nim"]:
        _variant_labels += f" → NIM-{MODEL_SIZE}"
    _backend_note = (
        f"**Backend:** NIM local Docker (`{VLLM_BASE_URL}`)"
        if INFERENCE_BACKEND == "nim_local"
        else f"**Backend:** vLLM (`{VLLM_BASE_URL}`)"
        if INFERENCE_BACKEND == "vllm"
        else "**Backend:** HF transformers *(quantized models upcast to BF16)*"
    )
    gr.Markdown(
        f"# 🌌 Cosmos Reason — BYO Video Demo ({MODEL_SIZE})\n"
        f"**{_load_note}** &nbsp;·&nbsp; **GPU:** {gpu_name} &nbsp;·&nbsp; "
        f"**VRAM free:** {free_vram:,} MiB &nbsp;·&nbsp; {_backend_note}\n\n"
        f"Upload any MP4 or image (JPG/PNG/WebP) and ask the model a question. "
        f"Select a checkpoint from the dropdown to load it into vLLM."
    )

    # ── Input row ───────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Video"):
                    video_input = gr.Video(label="Upload Video (MP4)", sources=["upload"], height=250)
                with gr.TabItem("Image"):
                    image_input = gr.Image(
                        label="Upload Image (JPG / PNG / WebP)",
                        type="filepath",
                        sources=["upload"],
                        height=250,
                    )
            clip_info = gr.Markdown("*Upload a video or image to see info*")

        with gr.Column(scale=1):
            demo_picker = gr.Dropdown(
                label="Demo Prompt",
                choices=[p[0] for p in DEMO_PROMPTS],
                value=DEMO_PROMPTS[0][0],
                info="Quick preset prompts",
            )
            with gr.Row():
                run_btn = gr.Button("▶  Run Inference",    variant="primary",   size="lg")
                all_btn = gr.Button("⚡  Run All Variants", variant="secondary", size="lg",
                                    visible=False)

    # ── Advanced Settings ────────────────────────────────────────────────────
    with gr.Accordion("⚙️  Advanced Settings", open=False):

        # System + User prompt — first thing in Advanced Settings so users
        # don't have to scroll past backend / NIM panel to edit them.
        # lines=10, max_lines=10 → fixed-height textboxes with internal
        # scrollbar for prompts longer than ~10 visual lines (covers the
        # full <think>-template demo prompts without truncation).
        with gr.Row():
            system_box = gr.Textbox(label="System Prompt", value=DEFAULT_SYSTEM,
                                    lines=10, max_lines=10)
            user_box   = gr.Textbox(label="User Prompt",   value=DEFAULT_PROMPT,
                                    lines=10, max_lines=10)

        # ── Backend selector ────────────────────────────────────────────────
        _active_be = INFERENCE_BACKEND.upper()
        _vllm_label = f"vLLM ({_VLLM_VERSION})" if _VLLM_VERSION else "vLLM (not installed)"
        _be_choices = [
            "HF Transformers" + (" (active)" if _active_be == "HF" else ""),
            _vllm_label + (" (active)" if _active_be == "VLLM" else ""),
            "NIM (local Docker)" + (" (active)" if _active_be == "NIM_LOCAL" else ""),
            "TRT-LLM (not yet supported)",
        ]
        _be_map = {c: v for c, v in zip(
            _be_choices, ["hf", "vllm", "nim_local", "trtllm"]
        )}
        _current_be_choice = _be_choices[{"HF": 0, "VLLM": 1, "NIM_LOCAL": 2}.get(_active_be, 0)]

        with gr.Row():
            backend_radio = gr.Radio(
                label="Inference Backend",
                choices=_be_choices,
                value=_current_be_choice,
                info=f"Active backend (from env INFERENCE_BACKEND): {_active_be}",
                interactive=True,
            )
            vllm_url_box = gr.Textbox(
                label="vLLM / NIM Base URL",
                value=VLLM_BASE_URL,
                placeholder="http://localhost:8000/v1",
                info="Used when backend = vLLM or NIM (local Docker)",
                visible=_active_be in ("VLLM", "NIM_LOCAL"),
                interactive=True,
            )

        backend_warn = gr.HTML(value="", visible=False)

        def _on_backend_change(choice):
            selected = _be_map.get(choice, "hf")
            active   = INFERENCE_BACKEND
            if selected == "trtllm":
                warn_html = (
                    '<div style="background:#7f1d1d;color:#fca5a5;padding:10px 14px;'
                    'border-radius:6px;margin:4px 0;font-size:13px">'
                    '🚫 <b>TRT-LLM</b> is not yet supported in this demo. '
                    'Use HF Transformers or vLLM backend instead.</div>'
                )
                return gr.update(value=warn_html, visible=True), gr.update(visible=False)
            if selected == "nim_local" and active != "nim_local":
                _code_pill = ('background:#dbeafe;color:#1e293b;padding:2px 6px;'
                              'border-radius:4px;font-size:12px')
                nim_html = (
                    '<div style="background:#eff6ff;border:1px solid #3b82f6;color:#1e3a8a;'
                    'padding:10px 14px;border-radius:6px;margin:4px 0;font-size:13px">'
                    '<b style="color:#1e3a8a">NIM (local Docker)</b> — runs the NIM container on this instance.<br>'
                    '<b style="color:#1e3a8a">Step 1:</b> Start NIM container (one-time, ~10-20 min download):<br>'
                    f'<code style="{_code_pill}">bash /tmp/nim_launch.sh &lt;NGC_API_KEY&gt;</code><br>'
                    '<b style="color:#1e3a8a">Step 2:</b> Restart Gradio with NIM backend:<br>'
                    f'<code style="{_code_pill}">INFERENCE_BACKEND=nim_local '
                    'VLLM_BASE_URL=http://localhost:8000/v1 python /tmp/gradio_cr2_byo.py</code><br>'
                    '<b style="color:#1e3a8a">Model name for checkpoint:</b> use the name from '
                    f'<code style="{_code_pill}">curl http://localhost:8000/v1/models</code> '
                    f'or set Custom Checkpoint ID to <code style="{_code_pill}">nvidia/cosmos-reason2-2b</code><br>'
                    '<b style="color:#1e3a8a">Check NIM status:</b> '
                    f'<code style="{_code_pill}">docker logs cosmos-nim</code>'
                    '</div>'
                )
                return gr.update(value=nim_html, visible=True), gr.update(visible=True)
            if selected != active:
                env_cmd = f"INFERENCE_BACKEND={selected}"
                if selected == "vllm":
                    env_cmd += "  VLLM_BASE_URL=http://localhost:8000/v1"
                _cuda_note = ""
                if selected == "vllm":
                    _cuda_note = (
                        '<br><span style="color:#fca5a5">⚠ vLLM 0.12.0 requires CUDA 12.9+ '
                        '(driver 575.x+). This instance has CUDA 12.8 (driver 570.x). '
                        'vLLM server will fail to start unless you use a CUDA 12.9+ instance.</span>'
                    )
                warn_html = (
                    '<div style="background:#78350f;color:#fde68a;padding:10px 14px;'
                    'border-radius:6px;margin:4px 0;font-size:13px">'
                    f'⚠ <b>Backend change requires a Gradio restart.</b><br>'
                    f'Stop Gradio, then relaunch with:<br>'
                    f'<code style="background:#1c1917;padding:3px 6px;border-radius:4px;'
                    f'font-size:12px">{env_cmd} python /tmp/gradio_cr2_byo.py</code>'
                    f'{_cuda_note}'
                    '</div>'
                )
                show_url = selected in ("vllm", "nim_local")
                return gr.update(value=warn_html, visible=True), gr.update(visible=show_url)
            return gr.update(value="", visible=False), gr.update(visible=selected in ("vllm", "nim_local"))

        backend_radio.change(
            fn=_on_backend_change,
            inputs=[backend_radio],
            outputs=[backend_warn, vllm_url_box],
        )

        _is_nim_local = INFERENCE_BACKEND == "nim_local"
        _ckpt_label = "VLM NIM" if _is_nim_local else "Checkpoint"
        _ckpt_info = (
            "VLM NIMs supported on this target. Source: "
            "docs.nvidia.com/nim/vision-language-models/latest/introduction.html. "
            "Pick a different NIM and click 'Switch to selected NIM' to swap the "
            "running container — pull + load takes 5-15 min on first run."
            if _is_nim_local else
            "Preset checkpoints — changing this in vLLM mode reloads the server"
        )
        with gr.Row():
            checkpoint_dd = gr.Dropdown(
                label=_ckpt_label,
                choices=[p[0] for p in CHECKPOINT_PRESETS],
                value=_VLLM_DD_DEFAULT,
                info=_ckpt_info,
            )
            custom_ckpt = gr.Textbox(
                label="Custom Checkpoint ID (overrides dropdown)",
                placeholder="nvidia/Cosmos-Reason2-8B  or  /path/to/local",
                value="",
                visible=not _is_nim_local,
            )

        _vllm_mode = INFERENCE_BACKEND == "vllm"
        with gr.Row(visible=_vllm_mode):
            vllm_swap_banner = gr.HTML(value="", visible=_vllm_mode)
            download_load_btn = gr.Button(
                "⬇ Download & Load",
                variant="primary",
                visible=False,
                scale=0,
                min_width=180,
            )
        if not _vllm_mode:
            vllm_swap_banner = gr.HTML(value="", visible=False)
            download_load_btn = gr.Button(visible=False)

        # NIM runtime swap — handled out-of-band, not via a Gradio button. The
        # earlier in-Gradio swap button (commit 4ed5951) attempted to stream
        # `nim_launch.sh` output through the browser tunnel, but a 5-15 min
        # docker pull plus 1-3 min vLLM warmup easily blows past Gradio's
        # streaming heartbeat — the UI appeared to freeze even when the
        # backend was making progress. Two reliable paths instead:
        #   (a) Ask the runtime agent (Claude) to swap — it has SSH and can
        #       run the commands below, watch the logs, and report back.
        #   (b) Run the commands manually on the host (instructions below).
        # When the swap finishes, refresh this page — Gradio re-queries
        # /v1/models on every reload and picks up the new served model id.
        if _is_nim_local:
            gr.HTML(
                "<div style=\"background:#eff6ff;border:1px solid #3b82f6;border-radius:6px;"
                "padding:12px 14px;margin:8px 0;color:#1e3a8a;font-size:13px;line-height:1.5\">"
                "<b style=\"color:#1e3a8a\">↻ Switching the running NIM</b><br>"
                "<span style=\"color:#475569\">"
                "The dropdown above lists every VLM NIM in the upstream catalog "
                "(<a href=\"https://docs.nvidia.com/nim/vision-language-models/latest/introduction.html\" "
                "target=\"_blank\" style=\"color:#1d4ed8\">docs source</a>). To actually swap "
                "the running container, do one of:"
                "</span>"
                "<ol style=\"margin:8px 0 4px 18px;color:#1e3a8a\">"
                "<li><b>Ask the runtime agent</b> (e.g. Claude) — say <i>“switch the NIM to "
                "Cosmos Reason2 2B”</i> and it will SSH in, stop the container, run "
                "<code style=\"font-size:11px;background:#dbeafe;color:#1e293b;padding:1px 4px;"
                "border-radius:3px\">nim_launch.sh</code>, and confirm "
                "<code style=\"font-size:11px;background:#dbeafe;color:#1e293b;padding:1px 4px;"
                "border-radius:3px\">/v1/models</code> is back up.</li>"
                "<li><b>Run it yourself by SSH</b> (no Claude needed):"
                "<pre style=\"background:#0f1a2e;border:1px solid #334155;border-radius:4px;"
                "padding:8px 10px;margin:6px 0;color:#e5e7eb;font-size:11px;overflow-x:auto;"
                "white-space:pre-wrap\">"
                "ssh &lt;user@host&gt;\n"
                "docker rm -f cosmos-nim\n"
                "MODEL=&lt;short-id&gt; CONTAINER_NAME=cosmos-nim PORT=8000 \\\n"
                "  bash /tmp/nim_launch.sh &lt;NGC_API_KEY&gt;\n"
                "# Then reload this Gradio page."
                "</pre>"
                "<span style=\"color:#475569;font-size:12px\">"
                "Short-id values: <code style=\"background:#dbeafe;color:#1e293b;padding:1px 4px;"
                "border-radius:3px\">cosmos-reason2-2b</code> · "
                "<code style=\"background:#dbeafe;color:#1e293b;padding:1px 4px;border-radius:3px\">"
                "cosmos-reason2-8b</code> · "
                "<code style=\"background:#dbeafe;color:#1e293b;padding:1px 4px;border-radius:3px\">"
                "cosmos-reason1-7b</code> · "
                "<code style=\"background:#dbeafe;color:#1e293b;padding:1px 4px;border-radius:3px\">"
                "nemotron-nano-12b-v2-vl</code> · "
                "<code style=\"background:#dbeafe;color:#1e293b;padding:1px 4px;border-radius:3px\">"
                "llama-3.1-nemotron-nano-vl-8b-v1</code> · "
                "<code style=\"background:#dbeafe;color:#1e293b;padding:1px 4px;border-radius:3px\">"
                "llama-3.2-11b-vision-instruct</code> · "
                "<code style=\"background:#dbeafe;color:#1e293b;padding:1px 4px;border-radius:3px\">"
                "llama-3.2-90b-vision-instruct</code> · "
                "<code style=\"background:#dbeafe;color:#1e293b;padding:1px 4px;border-radius:3px\">"
                "llama-4-maverick-17b-128e-instruct</code> · "
                "<code style=\"background:#dbeafe;color:#1e293b;padding:1px 4px;border-radius:3px\">"
                "llama-4-scout-17b-16e-instruct</code> · "
                "<code style=\"background:#dbeafe;color:#1e293b;padding:1px 4px;border-radius:3px\">"
                "mistral-small-3.2-24b-instruct-2506</code>. "
                "First-time pulls take 5–15 min; subsequent restarts ~3 min for vLLM warmup."
                "</span>"
                "</li></ol>"
                "</div>",
                visible=True,
            )

        with gr.Row():
            fps_slider = gr.Slider(
                minimum=1, maximum=8, step=1, value=_UI_DEFAULT_FPS,
                label="Video sampling rate (fps) — ignored for images",
                info="Higher = more frames = more tokens = slower. HF mode caps at 32 frames total.",
            )
            maxpx_slider = gr.Slider(
                minimum=64*(32**2), maximum=4096*(32**2), step=64*(32**2),
                value=DEFAULT_MAX_PIXELS,
                label="Max pixels per frame",
            )
            maxtok_slider = gr.Slider(
                minimum=64, maximum=2048, step=64, value=DEFAULT_MAX_TOKENS,
                label="Max output tokens",
            )

        # NIM-8B-FP8-THINK-EOS: greedy decode (temp=0) on the FP8-quantized
        # cosmos-reason2-8b NIM emits `<think>` then an EOS-like token, terminating
        # before any reasoning or final answer is produced. Temperature ≥ ~0.3 breaks
        # the trap by avoiding the deterministic post-`<think>` EOS path.
        # build.nvidia.com NIM defaults for cosmos-reason2-8b: T=0.6, top_p=0.3, rep=1.2
        _NIM_DEFAULTS = (0.6, 0.3, 1.2)
        _nim_active = INFERENCE_BACKEND == "nim_local"
        _default_temp = _NIM_DEFAULTS[0] if _nim_active else 0.0
        _default_top_p = _NIM_DEFAULTS[1] if _nim_active else 1.0
        _default_rep   = _NIM_DEFAULTS[2] if _nim_active else 1.05
        nim_defaults_chk = gr.Checkbox(
            label="Use build.nvidia.com parameter settings",
            info="One-click apply: Temperature=0.6, Top P=0.3, Repetition Penalty=1.2 (cosmos-reason2-8b NIM defaults).",
            value=_nim_active,
        )
        with gr.Row():
            temp_slider = gr.Slider(
                minimum=0.0, maximum=1.0, step=0.05, value=_default_temp,
                label="Temperature",
                info=("0 = deterministic. Higher = more creative/varied output."
                      + (" NIM mode: keep ≥ 0.3 — greedy decode triggers a <think>+EOS"
                         " bug on the FP8-quantized 8B NIM."
                         if _nim_active else "")),
            )
            top_p_slider = gr.Slider(
                minimum=0.01, maximum=1.0, step=0.01, value=_default_top_p,
                label="Top P",
                info="Nucleus sampling threshold. 1.0 = disabled.",
            )
            rep_penalty_slider = gr.Slider(
                minimum=1.0, maximum=2.0, step=0.05, value=_default_rep,
                label="Repetition Penalty",
                info="1.0 = no penalty. Higher discourages repeated phrases.",
            )

        def _apply_nim_defaults(checked):
            # Toggle ON  → snap sliders to build.nvidia.com NIM defaults.
            # Toggle OFF → no-op; preserve whatever the user has dialed in.
            if checked:
                t, p, r = _NIM_DEFAULTS
                return gr.update(value=t), gr.update(value=p), gr.update(value=r)
            return gr.update(), gr.update(), gr.update()

        nim_defaults_chk.change(
            fn=_apply_nim_defaults,
            inputs=[nim_defaults_chk],
            outputs=[temp_slider, top_p_slider, rep_penalty_slider],
        )

        with gr.Row(visible=INFERENCE_BACKEND != "vllm"):
            disable_autocap_chk = gr.Checkbox(
                label="Disable resolution auto-cap",
                value=INFERENCE_BACKEND != "hf",
                info=(
                    "Auto-cap reduces max_pixels per frame so HF prefill finishes "
                    "in ~55s on H100. Off by default on fast backends (vLLM, NIM); "
                    "on by default on HF. Toggle to override."
                ),
                interactive=True,
            )
            if INFERENCE_BACKEND == "hf":
                gr.HTML(
                    '<div style="color:#f87171;font-size:12px;padding-top:4px">'
                    '⚠ <b>Auto-cap is on by default on HF.</b> Disabling sends full-resolution '
                    'frames; longer videos can run <b>3–5× slower</b> or OOM. Upload a video to '
                    'see the estimated time multiplier.'
                    '</div>'
                )
            else:
                gr.HTML(
                    '<div style="color:#475569;font-size:12px;padding-top:4px">'
                    'Auto-cap is <b>off</b> by default on this backend — full-resolution '
                    'frames go to the model. Auto-cap only matters for HF Transformers, '
                    'where it keeps prefill from ballooning to many minutes per clip.'
                    '</div>'
                )

        # Run All Variants disabled — reload checkbox hidden accordingly
        reload_vllm_chk = gr.Checkbox(value=False, visible=False, interactive=False)

        # HF auth is only relevant when the backend loads gated checkpoints from
        # the HF Hub (HF Transformers / vLLM modes). NIM ships its own weights,
        # so this whole block is hidden in nim_local mode to remove a confusing
        # surface area. The "Get Login URL (browser)" button was removed because
        # the HF device flow can't be reliably captured from inside this Gradio
        # process (no TTY, no browser handoff) — pasting a token directly from
        # huggingface.co/settings/tokens is the supported path.
        _hf_auth_visible = INFERENCE_BACKEND in ("hf", "vllm")
        with gr.Group(visible=_hf_auth_visible):
            gr.HTML('<hr style="margin:12px 0;border:none;border-top:1px solid #444"/>')
            gr.Markdown(
                "**HuggingFace Auth** — required for gated models (e.g. CR2-8B BF16, CR2-8B FP8, custom checkpoints).  \n"
                "Get a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and paste it below."
            )
            with gr.Row():
                hf_token_box = gr.Textbox(
                    label="HF Token",
                    placeholder="hf_...",
                    type="password",
                    info="Paste a token from huggingface.co/settings/tokens",
                    scale=3,
                    interactive=True,
                )
                with gr.Column(scale=1, min_width=160):
                    hf_apply_btn  = gr.Button("Apply Token",        size="sm", variant="primary")
                    hf_status_btn = gr.Button("Check Auth Status",  size="sm")
            hf_auth_html = gr.HTML(_hf_check_status())

            hf_apply_btn.click(fn=_hf_apply_token,    inputs=[hf_token_box], outputs=[hf_auth_html])
            hf_status_btn.click(fn=_hf_check_status,  inputs=[],             outputs=[hf_auth_html])

        gr.HTML('<hr style="margin:12px 0;border:none;border-top:1px solid #444"/>')
        with gr.Accordion("Storage & Disk Management", open=False):
            gr.HTML(
                '<div style="color:#9ca3af;font-size:12px;margin-bottom:6px">'
                'Note: Brev does not support in-place storage resize. '
                'To add disk space you must delete and recreate the instance with a larger storage tier. '
                'The <b>HF download cache</b> is safe to clean — model files in '
                '<code>cosmos-reason2/models/</code> are independent copies.'
                '</div>'
            )
            disk_status_out = gr.HTML(_disk_status_html())
            with gr.Row():
                disk_refresh_btn = gr.Button("Refresh", size="sm", scale=0, min_width=120)
                clean_cache_btn  = gr.Button("Clean HF Download Cache", size="sm",
                                             variant="stop", scale=0, min_width=220)
            disk_refresh_btn.click(fn=_disk_status_html,  inputs=[], outputs=[disk_status_out])
            clean_cache_btn.click( fn=_clean_hf_cache,    inputs=[], outputs=[disk_status_out])

    # ── Output row ───────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=2):
            # Initial badge state matches the first DEMO_PROMPT (which is the
            # default selection in the dropdown). Updated by on_demo().
            reasoning_badge = gr.HTML(
                value=_reasoning_badge_html(DEMO_PROMPTS[0][3]),
                show_label=False,
            )
            response_out = gr.HTML(label="Model Response", value="", show_label=True)
        with gr.Column(scale=1):
            status_panel = gr.HTML(_status_html(["wait"] * 5), show_progress="hidden")

    # ── Benchmark results table ───────────────────────────────────────────────
    results_table = gr.HTML(_table_html(), label="Benchmark Log", show_progress="hidden")

    # ── Event handlers ───────────────────────────────────────────────────────
    def on_upload(path, fps_val, disable_autocap):
        if not path:
            return "*Upload a video to see clip info*", gr.update()
        m = get_video_meta(path)
        if not m.width:
            return "*Clip info unavailable (PyAV not installed)*", gr.update()
        fps_val  = max(1, int(fps_val))
        n_frames = max(1, int(m.duration_s * fps_val))
        info_str = (f"**{m.width}×{m.height}** · {m.fps:.1f} fps · {m.duration_s:.1f}s · "
                    f"{n_frames} frames sampled")
        # Fast backends (vLLM, NIM) are 100-500x faster than HF — HF-based
        # timing estimates are meaningless and auto-cap is unnecessary.
        if INFERENCE_BACKEND in ("vllm", "nim_local"):
            return info_str, gr.update()
        est_s_full = _est_tokens(n_frames, DEFAULT_MAX_PIXELS) / PREFILL_TPS
        capped_px, est_s_capped = _auto_cap(n_frames, DEFAULT_MAX_PIXELS)
        if not disable_autocap:
            # Auto-cap ENABLED — the exception (HF and other unoptimized backends).
            if capped_px < DEFAULT_MAX_PIXELS:
                ratio = est_s_full / max(est_s_capped, 1.0)
                info_str += (
                    f"\n> ⚠ **Auto-cap ENABLED** · capping at {capped_px:,} px/frame "
                    f"→ ~{est_s_capped:.0f}s est "
                    f"(full-res would be ~{est_s_full:.0f}s, **~{ratio:.1f}× slower**)"
                )
                return info_str, gr.update(value=capped_px)
            info_str += f"\n> ✓ **Auto-cap ENABLED** · cap not needed for this clip · ~{est_s_full:.0f}s est"
            return info_str, gr.update()
        # Auto-cap DISABLED — the default. Generic + expressive.
        info_str += "\n> ✓ **Full resolution** — every frame delivered at the native pixel count above"
        return info_str, gr.update()

    video_input.change(on_upload, inputs=[video_input, fps_slider, disable_autocap_chk], outputs=[clip_info, maxpx_slider])
    fps_slider.change(on_upload, inputs=[video_input, fps_slider, disable_autocap_chk], outputs=[clip_info, maxpx_slider])
    disable_autocap_chk.change(on_upload, inputs=[video_input, fps_slider, disable_autocap_chk], outputs=[clip_info, maxpx_slider])

    def on_image_upload(path, disable_autocap):
        if not path:
            return "*Upload an image to see info*", gr.update()
        try:
            from PIL import Image as _pil
            img = _pil.open(path)
            w, h = img.size
            info_str = f"**{w}×{h}** · {img.mode} image"
        except Exception:
            return "*Image info unavailable*", gr.update()
        # Fast backends (vLLM, NIM) serve at high throughput; HF-backend
        # timing math doesn't apply and auto-cap is unnecessary.
        if INFERENCE_BACKEND in ("vllm", "nim_local"):
            return info_str, gr.update()
        est_s_full = _est_tokens(1, DEFAULT_MAX_PIXELS) / PREFILL_TPS
        capped_px, est_s_capped = _auto_cap(1, DEFAULT_MAX_PIXELS)
        capped_px = min(capped_px, IMAGE_AUTO_CAP_MAX)
        if not disable_autocap:
            # Auto-cap ENABLED — the exception (HF and other unoptimized backends).
            if capped_px < DEFAULT_MAX_PIXELS:
                ratio = est_s_full / max(est_s_capped, 1.0)
                info_str += (
                    f"\n> ⚠ **Auto-cap ENABLED** · capping at {capped_px:,} px "
                    f"→ ~{est_s_capped:.0f}s est "
                    f"(full-res would be ~{est_s_full:.0f}s, **~{ratio:.1f}× slower**)"
                )
                return info_str, gr.update(value=capped_px)
            info_str += f"\n> ✓ **Auto-cap ENABLED** · cap not needed for this image · ~{est_s_full:.0f}s est"
            return info_str, gr.update()
        # Auto-cap DISABLED — the default. Generic + expressive.
        info_str += "\n> ✓ **Full resolution** — image delivered at the native pixel count above"
        return info_str, gr.update()

    image_input.change(on_image_upload, inputs=[image_input, disable_autocap_chk], outputs=[clip_info, maxpx_slider])

    def on_demo(name):
        """Pick a demo: populate user prompt + system prompt + Reasoning badge."""
        for entry in DEMO_PROMPTS:
            if entry[0] == name:
                _, user_p, system_p, reasoning_on = entry
                return user_p, system_p, _reasoning_badge_html(reasoning_on)
        return DEFAULT_PROMPT, DEFAULT_SYSTEM, _reasoning_badge_html(False)

    demo_picker.change(
        on_demo,
        inputs=[demo_picker],
        outputs=[user_box, system_box, reasoning_badge],
    )

    if INFERENCE_BACKEND == "vllm":
        checkpoint_dd.change(
            fn=_on_checkpoint_change,
            inputs=[checkpoint_dd],
            outputs=[vllm_swap_banner, download_load_btn, fps_slider, maxpx_slider],
        )
        download_load_btn.click(
            fn=_on_download_and_load,
            inputs=[checkpoint_dd],
            outputs=[vllm_swap_banner, download_load_btn, fps_slider, maxpx_slider],
        )
    # nim_local: no click wiring — swap is done out-of-band (see info panel).

    def resolve_model_id(ckpt_name, custom_val):
        if custom_val.strip():
            return custom_val.strip()
        for name, mid in CHECKPOINT_PRESETS:
            if name == ckpt_name:
                return mid
        return CHECKPOINT_PRESETS[0][1]

    def _run(video_path, image_path, user_prompt, system_prompt, fps, max_pixels, max_new_tokens,
             ckpt_name, custom_val, disable_autocap, temperature, top_p, rep_penalty):
        model_id  = resolve_model_id(ckpt_name, custom_val)
        media     = video_path
        is_image  = False
        if media is None and image_path is not None:
            media    = image_path
            is_image = True
        for text, status, tbl in run_inference(
            media, user_prompt, system_prompt,
            fps, max_pixels, max_new_tokens, model_id,
            disable_autocap=disable_autocap,
            is_image=is_image,
            temperature=temperature,
            top_p=top_p,
            rep_penalty=rep_penalty,
        ):
            yield _render_with_think(text), status, tbl

    def _run_all(video_path, user_prompt, system_prompt, fps, max_pixels, max_new_tokens,
                 disable_autocap, reload_vllm):
        for combined, status, tbl in run_all_variants(
            video_path, user_prompt, system_prompt,
            fps, max_pixels, max_new_tokens,
            disable_autocap=disable_autocap,
            reload_vllm=reload_vllm,
        ):
            yield _render_with_think(combined), status, tbl

    run_btn.click(
        fn=_run,
        inputs=[video_input, image_input, user_box, system_box, fps_slider, maxpx_slider, maxtok_slider,
                checkpoint_dd, custom_ckpt, disable_autocap_chk,
                temp_slider, top_p_slider, rep_penalty_slider],
        outputs=[response_out, status_panel, results_table],
    )

    all_btn.click(
        fn=_run_all,
        inputs=[video_input, user_box, system_box, fps_slider, maxpx_slider, maxtok_slider,
                disable_autocap_chk, reload_vllm_chk],
        outputs=[response_out, status_panel, results_table],
    )

    SAMPLE_VIDEO = f"{HOME}/cosmos-reason2/assets/sample.mp4"
    if os.path.exists(SAMPLE_VIDEO):
        gr.Examples(
            examples=[[SAMPLE_VIDEO, DEFAULT_PROMPT, DEFAULT_SYSTEM,
                       DEFAULT_FPS, DEFAULT_MAX_PIXELS, DEFAULT_MAX_TOKENS]],
            inputs=[video_input, user_box, system_box, fps_slider, maxpx_slider, maxtok_slider],
            label="Sample video",
        )

_app, _local_url, _share_url = demo.launch(
    server_name="0.0.0.0", server_port=PORT, share=SHARE, prevent_thread_lock=True,
    theme=gr.themes.Base(primary_hue="green", font=gr.themes.GoogleFont("Inter")),
)
_pub = _share_url or _local_url or f"http://0.0.0.0:{PORT}"
print(f"[launch] {_pub}", flush=True)
try:
    with open("/tmp/gradio_url.txt", "w") as _f:
        _f.write(_pub)
except Exception:
    pass
demo.block_thread()
