#!/usr/bin/env python3
"""
Cosmos BYO Video Demo setup + launch.
Version: 2026-04-30
Canonical source: ~/.claude/scripts/byo_video_setup.py

Runs on the GPU instance. Prints live progress with ETAs.
MODEL_SIZE-driven: downloads all variants for the selected model size.
At the end, prints a clickable OSC 8 hyperlink to the Gradio URL.
URL is also written to /tmp/gradio_url.txt for agent capture.

Env vars:
  HF_TOKEN          — required for gated model download (checks ~/.cache/huggingface/token if not set)
  NGC_API_KEY       — required for NIM mode (nvapi-... prefix, 8B only; not needed for Cosmos3)
  MODEL_SIZE        — 2B | 8B | 32B | C3-2B | C3-8B | C3-32B | C3-super | NEM-12B | QW3-2B | QW3-8B | QW3-32B  (default: C3-2B)
  MODEL_DIR         — override local download path for primary model
  GRADIO_PORT       — port for Gradio (default: 7860)
  SKIP_HF_PRELOAD   — set to 1 to skip HF model preload at Gradio startup (auto in vLLM mode)
  VLLM_MAX_MODEL_LEN — max context length for vLLM (default: 32768; do not reduce below 32768 for video)
"""
import os, sys, time, subprocess, re, shutil, json, urllib.request, socket

# ── ANSI helpers ────────────────────────────────────────────────────────────
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def ok(msg):     print(f"  {GREEN}✓{RESET}  {msg}", flush=True)
def run(msg):    print(f"  {YELLOW}⟳{RESET}  {msg}", flush=True)
def info(msg):   print(f"  {CYAN}→{RESET}  {msg}", flush=True)
def warn(msg):   print(f"  {YELLOW}⚠{RESET}  {msg}", flush=True)
def header(msg, eta=None):
    eta_str = f"  {DIM}[est. {eta}]{RESET}" if eta else ""
    print(f"\n{BOLD}{msg}{RESET}{eta_str}", flush=True)

def hyperlink(url, label=None):
    label = label or url
    return f"\033]8;;{url}\033\\{BOLD}{CYAN}{label}{RESET}\033]8;;\033\\"

def run_cmd(args, cwd=None, env=None, timeout=None):
    try:
        result = subprocess.run(
            args, cwd=cwd, env=env, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        return result.returncode, result.stdout
    except FileNotFoundError:
        return 1, f"command not found: {args[0]}"
    except subprocess.TimeoutExpired:
        return 1, "timeout"

def stream_cmd(args, cwd=None, env=None, prefix=""):
    proc = subprocess.Popen(
        args, cwd=cwd, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            print(f"     {DIM}{prefix}{line}{RESET}", flush=True)
    proc.wait()
    return proc.returncode

# ── Size-driven model config (mirrors gradio_cr2_byo.py MODEL_CONFIGS) ───────
_MODEL_CONFIGS = {
    # ── Cosmos3-Reasoner (C3-2B/C3-32B/C3-super gated; C3-8B = Cosmos3-Nano-Reasoner, public) ──
    "C3-2B": {
        "variants": [
            ("C3R-2B BF16", "Cosmos3-Reasoner-2B", "nvidia/Cosmos3-Reasoner-2B-Private", "~TBD"),
        ],
        "nim": None,
    },
    "C3-8B": {
        "variants": [
            ("C3R-Nano BF16", "Cosmos3-Nano-Reasoner", "nvidia/Cosmos3-Nano-Reasoner", "~TBD"),
        ],
        "nim": None,
    },
    "C3-32B": {
        "variants": [
            ("C3R-32B BF16", "Cosmos3-Reasoner-32B", "nvidia/Cosmos3-Reasoner-32B-Private", "~TBD"),
        ],
        "nim": None,
        # disk_gb: 1024 (1TB minimum — Alex explicit requirement for 32B)
        "disk_gb": 1024,
        # vLLM flags: tensor-parallel-size 1 + high memory utilization for single H100 80GB.
        # NOTE: weight size unverified (25 safetensor files, estimate ~60-70GB BF16).
        # If weights exceed ~72GB, OOM will occur — flag for review before production deploy.
        "vllm_extra_flags": ["--tensor-parallel-size", "1", "--gpu-memory-utilization", "0.93"],
    },
    "C3-super": {
        "variants": [
            ("C3-Super BF16", "Cosmos3-Super-Reasoner", "nvidia/Cosmos3-Super-Reasoner", "~TBD"),
        ],
        "nim": None,
        # 32B model — requires H200 SXM 141GB (confirmed from live deployment 2026-05-05).
        "disk_gb": 1024,
        # Architecture: NemotronVLForConditionCausalLM. vLLM may raise "Unsupported architecture"
        # if this arch is not registered in the installed vLLM build. Use INFERENCE_BACKEND=hf
        # as a fallback — confirmed working on H200 at 2026-05-05 live run.
        "vllm_extra_flags": ["--tensor-parallel-size", "1", "--gpu-memory-utilization", "0.93"],
    },
    # ── Cosmos Reason2 ──
    "2B": {
        "variants": [
            ("CR2-2B BF16", "Cosmos-Reason2-2B",     "nvidia/Cosmos-Reason2-2B",     "~4 GB"),
            ("CR2-2B FP8",  "Cosmos-Reason2-2B-FP8", "nvidia/Cosmos-Reason2-2B-FP8", "~2 GB"),
        ],
        "nim": None,
    },
    "8B": {
        "variants": [
            ("CR2-8B BF16",  "Cosmos-Reason2-8B",       "nvidia/Cosmos-Reason2-8B",       "~16 GB"),
            ("CR2-8B NVFP4", "Cosmos-Reason2-8B-NVFP4", "nvidia/Cosmos-Reason2-8B-NVFP4", "~4 GB"),
        ],
        "nim": "nvidia/cosmos-reason2-8b",
    },
    "32B": {
        "variants": [
            ("CR2-32B BF16", "Cosmos-Reason2-32B",    "nvidia/Cosmos-Reason2-32B",    "~66 GB"),
            ("CR2-32B AV",   "Cosmos-Reason2-32B-AV", "nvidia/Cosmos-Reason2-32B-AV", "~66 GB"),
        ],
        "nim": None,
        # 33B params × 2 bytes BF16 = ~66GB weights. On H100 80GB: use 0.95 utilization.
        # KV cache budget is ~4GB at this utilization — reduce max-model-len accordingly.
        "vllm_extra_flags": ["--gpu-memory-utilization", "0.95"],
        "vllm_max_model_len": 4096,
    },
    # ── Qwen3-VL (public — no HF_TOKEN required) ────────────────────────────────
    # vLLM-only: uses video_url content type with file:// path (same as Nemotron).
    # Frame control via extra_body mm_processor_kwargs at inference time.
    # Variants: 2B fits 1x H100; 8B fits 1x H100 80GB; 32B needs TP=2 or FP8 on 1x H100.
    "QW3-2B": {
        "variants": [
            ("Qwen3-VL-2B Instruct",    "Qwen3-VL-2B-Instruct",     "Qwen/Qwen3-VL-2B-Instruct",     "~5 GB"),
            ("Qwen3-VL-2B FP8",         "Qwen3-VL-2B-Instruct-FP8", "Qwen/Qwen3-VL-2B-Instruct-FP8", "~3 GB"),
            ("Qwen3-VL-2B Thinking",    "Qwen3-VL-2B-Thinking",     "Qwen/Qwen3-VL-2B-Thinking",     "~5 GB"),
        ],
        "nim": None,
        "vllm_extra_flags": ["--gpu-memory-utilization", "0.85", "--allowed-local-media-path", "/tmp"],
        "vllm_max_model_len": 32768,
    },
    "QW3-8B": {
        "variants": [
            ("Qwen3-VL-8B Instruct",    "Qwen3-VL-8B-Instruct",     "Qwen/Qwen3-VL-8B-Instruct",     "~16 GB"),
            ("Qwen3-VL-8B FP8",         "Qwen3-VL-8B-Instruct-FP8", "Qwen/Qwen3-VL-8B-Instruct-FP8", "~9 GB"),
            ("Qwen3-VL-8B Thinking",    "Qwen3-VL-8B-Thinking",     "Qwen/Qwen3-VL-8B-Thinking",     "~16 GB"),
        ],
        "nim": None,
        "vllm_extra_flags": ["--gpu-memory-utilization", "0.85", "--allowed-local-media-path", "/tmp"],
        "vllm_max_model_len": 32768,
    },
    "QW3-32B": {
        "variants": [
            ("Qwen3-VL-32B Instruct",   "Qwen3-VL-32B-Instruct",     "Qwen/Qwen3-VL-32B-Instruct",     "~64 GB"),
            ("Qwen3-VL-32B FP8",        "Qwen3-VL-32B-Instruct-FP8", "Qwen/Qwen3-VL-32B-Instruct-FP8", "~33 GB"),
            ("Qwen3-VL-32B Thinking",   "Qwen3-VL-32B-Thinking",     "Qwen/Qwen3-VL-32B-Thinking",     "~64 GB"),
        ],
        "nim": None,
        "vllm_extra_flags": ["--gpu-memory-utilization", "0.93", "--allowed-local-media-path", "/tmp"],
        "vllm_max_model_len": 16384,
    },
    # ── Nemotron-Nano-12B-v2-VL (gated — HF_TOKEN with nvidia org required) ──────
    # Requires vLLM nightly or compatible build; PyPI vLLM ≤0.11.0 unsupported.
    # Uses opencv video backend (not PyAV). Gradio sends video as file:// URL.
    # NVFP4-QAD variant requires special vLLM build — use BF16 or FP8 for demos.
    "NEM-12B": {
        "variants": [
            ("Nem-12B BF16", "NVIDIA-Nemotron-Nano-12B-v2-VL-BF16", "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16", "~26 GB"),
            ("Nem-12B FP8",  "NVIDIA-Nemotron-Nano-12B-v2-VL-FP8",  "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-FP8",  "~13 GB"),
        ],
        "nim": None,
        "vllm_extra_flags": [
            "--media-io-kwargs", '{"video": {"fps": 2, "num_frames": 128}}',
            "--video-pruning-rate", "0.75",
            "--allowed-local-media-path", "/tmp",
        ],
        "vllm_max_model_len": 32768,
        # VLLM_VIDEO_LOADER_BACKEND=opencv: required for Nemotron video decoding.
        # Nemotron's vLLM integration does not support PyAV; opencv is the only supported backend.
        # FLASHINFER_DISABLE_VERSION_CHECK=1: bypasses mismatch between flashinfer Python package
        # (0.5.3) and flashinfer-cubin (0.6.8.post1) installed by cosmos-reason2 uv sync.
        "vllm_env": {
            "VLLM_VIDEO_LOADER_BACKEND": "opencv",
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
        },
    },
}

# ── Config ──────────────────────────────────────────────────────────────────
HOME          = os.path.expanduser("~")
PATH_EXTRA    = f"{HOME}/.local/bin:{HOME}/.cargo/bin"
# HF_HOME=/tmp/hf-home: avoids root-owned ~/.cache/huggingface/ on Brev/shadeform instances.
# byo_video_setup.py may run as a non-root user where ~/.cache is owned by root (prior root
# invocation). Redirecting to /tmp/hf-home ensures write access for model cache, frpc binary,
# and HF modules. Set early so all downstream ENV copies inherit it.
HF_HOME_DIR   = os.environ.get("HF_HOME", "/tmp/hf-home")
os.makedirs(HF_HOME_DIR, exist_ok=True)
ENV           = {**os.environ, "PATH": f"{PATH_EXTRA}:{os.environ.get('PATH', '')}",
                 "PYTHONUNBUFFERED": "1",
                 "HF_HOME": HF_HOME_DIR,
                 "HF_MODULES_CACHE": f"{HF_HOME_DIR}/modules"}
HF_TOKEN      = os.environ.get("HF_TOKEN", "")
NGC_API_KEY   = os.environ.get("NGC_API_KEY", "")
MODEL_SIZE    = os.environ.get("MODEL_SIZE", "C3-2B").upper()
# .upper() normalises input but breaks mixed-case keys. Remap known exceptions.
_MODEL_SIZE_FIX = {"C3-SUPER": "C3-super"}
MODEL_SIZE = _MODEL_SIZE_FIX.get(MODEL_SIZE, MODEL_SIZE)
# Cosmos3-Reasoner uses cosmos-reason2 working dir until a dedicated repo is published.
# Set COSMOS_DIR env var to override if the repo path changes.
REASON2_DIR   = os.environ.get("COSMOS_DIR", f"{HOME}/cosmos-reason2")
MODELS_BASE   = f"{REASON2_DIR}/models"
GRADIO_PORT   = int(os.environ.get("GRADIO_PORT", "7860"))
GRADIO_APP    = "/tmp/gradio_cr2_byo.py"
URL_FILE      = "/tmp/gradio_url.txt"
LOG_FILE      = "/tmp/gradio_demo.log"
# MAXLEN-001: 32768 is the minimum required for video queries. Do not reduce below this.
VLLM_MAX_MODEL_LEN   = int(os.environ.get("VLLM_MAX_MODEL_LEN", "32768"))
INFERENCE_BACKEND    = os.environ.get("INFERENCE_BACKEND", "hf")
# VRAM flags — set during GPU detection; declare defaults here
ULTRA_LOW_VRAM = False
LOW_VRAM       = False
# Cost tracking
BREV_RATE_PER_HOUR = float(os.environ.get("BREV_RATE_PER_HOUR", "0"))
SETUP_START        = time.time()

def credits_spent():
    if BREV_RATE_PER_HOUR <= 0:
        return ""
    elapsed = time.time() - SETUP_START
    cost = BREV_RATE_PER_HOUR * elapsed / 3600
    return f" | Credits: ${cost:.3f}"

if MODEL_SIZE not in _MODEL_CONFIGS:
    print(f"  ✗  MODEL_SIZE={MODEL_SIZE} not supported. Use C3-2B, C3-8B, C3-32B, 2B, 8B, 32B, NEM-12B, QW3-2B, QW3-8B, or QW3-32B.")
    sys.exit(1)

_cfg = _MODEL_CONFIGS[MODEL_SIZE]

# Primary variant (first in list) drives MODEL_DIR/MODEL_NAME defaults
_primary_label, _primary_dirname, _primary_hf_id, _ = _cfg["variants"][0]
MODEL_DIR  = os.environ.get("MODEL_DIR",  f"{MODELS_BASE}/{_primary_dirname}")
MODEL_NAME = _primary_hf_id

# ── MODEL_ID override (arbitrary HF model, bypasses MODEL_SIZE lookup) ───────
MODEL_ID = os.environ.get("MODEL_ID", "")
if MODEL_ID:
    MODEL_NAME = MODEL_ID
    # Only fall back to "custom" if MODEL_SIZE wasn't explicitly set to a known config key.
    # Prevents MODEL_ID from silently discarding a valid MODEL_SIZE (e.g. NEM-12B, C3-8B).
    if MODEL_SIZE not in _MODEL_CONFIGS:
        # Unknown size: derive directory name from MODEL_ID. Keep hyphens — Linux supports them.
        _model_dir_name = MODEL_ID.split("/")[-1]
        MODEL_DIR  = os.path.join(MODELS_BASE, _model_dir_name)
        MODEL_SIZE = "custom"
        # Estimate size hint from model ID
        _size_hint = "~8GB" if "8B" in MODEL_ID else "~16GB" if ("32B" in MODEL_ID or "14B" in MODEL_ID) else "~4GB"
        # Override _cfg to a minimal single-variant config
        _cfg = {
            "repo": MODEL_ID.split("/")[-1],
            "variants": [(_model_dir_name, _model_dir_name, MODEL_ID, _size_hint)],
            "nim": False,
        }
        _variant_labels = MODEL_ID
    # When MODEL_SIZE is a known config key: MODEL_DIR already set correctly from config.
    # Only MODEL_NAME is updated to the explicit MODEL_ID override.

# ── Dashboard: 9-step progress checklist ─────────────────────────────────────
STEP_LABELS = [
    "GPU detect + VRAM tier",
    "HF auth + token validate",
    "NGC API key",
    "uv install",
    "cosmos-reason2 repo",
    "uv sync + CUDA libs",
    "PyAV + Gradio + requests",
    "Model weights download",
    "Gradio launch",
]
STEPS_DONE = []

def print_dashboard():
    print("\n── Setup Progress ──────────────────────────────", flush=True)
    elapsed = int(time.time() - SETUP_START)
    print(f"  Elapsed: {elapsed//60}m {elapsed%60}s{credits_spent()}", flush=True)
    for i, label in enumerate(STEP_LABELS, 1):
        if i in STEPS_DONE:
            marker = "✅"
        elif i == (max(STEPS_DONE) + 1 if STEPS_DONE else 1):
            marker = "⟳ "
        else:
            marker = "—"
        print(f"  [{marker}] Step {i}: {label}", flush=True)
    print("────────────────────────────────────────────────\n", flush=True)

print_dashboard()

# ── Pre-step: kill old Gradio so VRAM measurement is accurate ────────────────
subprocess.run(["bash", "-c", f"fuser -k {GRADIO_PORT}/tcp 2>/dev/null || true"])
time.sleep(2)

# ── Step 1: GPU check ─────────────────────────────────────────────────────────
header("Step 1 — GPU detect", eta="<5s")
NVIDIA_SMI_RETRIES = 3
_smi_rc = 1
_smi_out = ""
for _attempt in range(NVIDIA_SMI_RETRIES):
    _smi_rc, _smi_out = run_cmd(
        ["nvidia-smi", "--query-gpu=name,memory.free,memory.total",
         "--format=csv,noheader,nounits"]
    )
    if _smi_rc == 0 and _smi_out.strip():
        break
    if _attempt < NVIDIA_SMI_RETRIES - 1:
        warn(f"nvidia-smi attempt {_attempt + 1} failed — retrying in 5s")
        time.sleep(5)
else:
    if _smi_rc != 0 or not _smi_out.strip():
        warn("nvidia-smi failed after 3 attempts — defaulting to LOW_VRAM tier")
        gpu_name   = "unknown"
        vram_free  = 0
        vram_total = 0
        LOW_VRAM   = True
        tier_name  = "low-VRAM (fallback)"
        gradio_fps = 4
        max_pixels = 131072
        prefill_tps = 15

if _smi_rc == 0 and _smi_out.strip():
    gpu_line   = _smi_out.strip().splitlines()[0]
    parts_     = [p.strip() for p in gpu_line.split(",")]
    gpu_name   = parts_[0]
    vram_free  = int(parts_[1].split()[0])
    vram_total = int(parts_[2].split()[0])

    _gpu_upper  = gpu_name.upper()
    _h100_class = any(tag in _gpu_upper for tag in ("H100", "A100", "H200", "GB200"))
    if _h100_class and vram_free >= 60000:
        tier_name   = "H100/A100"
        gradio_fps  = 8
        max_pixels  = 1048576
        prefill_tps = 90
    elif vram_free >= 40000:
        tier_name   = "high-VRAM"
        gradio_fps  = 8
        max_pixels  = 524288
        prefill_tps = 23
    elif vram_free >= 8000:
        tier_name   = "low-VRAM (<40GB)"
        gradio_fps  = 4
        max_pixels  = 131072
        prefill_tps = 15
        LOW_VRAM    = True
    else:
        LOW_VRAM       = True
        ULTRA_LOW_VRAM = True
        gradio_fps     = 1
        max_pixels     = 65536
        prefill_tps    = 10
        tier_name      = "ULTRA-LOW-VRAM"

if ULTRA_LOW_VRAM:
    warn("RTX consumer GPU detected (<8GB VRAM free). Demo will run but may OOM on videos >5s at 720p. Pre-resize to 360p before upload.")

if not MODEL_ID:
    _variant_labels = " → ".join(lbl for lbl, _, _, _ in _cfg["variants"])
    if _cfg.get("nim"):
        _variant_labels += f" → NIM-{MODEL_SIZE}"
else:
    _variant_labels = MODEL_ID

ok(f"{gpu_name}  {vram_free:,} MiB free / {vram_total:,} MiB total")
ok(f"MODEL_SIZE: {MODEL_SIZE}  |  variants: {_variant_labels}")
ok(f"VRAM tier: {tier_name}  |  fps={gradio_fps}, max_pixels={max_pixels:,}, prefill_tps={prefill_tps}")
STEPS_DONE.append(1)
print_dashboard()

# ── Step 2: HF token ──────────────────────────────────────────────────────────
header("Step 2 — HuggingFace auth", eta="<5s")
hf_cache = os.path.expanduser("~/.cache/huggingface/token")
if INFERENCE_BACKEND == "nim_local":
    info("nim_local backend — HF auth not required (NIM container ships the model). Skipping Step 2/2b.")
elif HF_TOKEN:
    ok(f"HF_TOKEN set ({len(HF_TOKEN)} chars)")
elif os.path.exists(hf_cache):
    ok(f"HF token found at ~/.cache/huggingface/token")
    with open(hf_cache) as f:
        HF_TOKEN = f.read().strip()
    ENV["HF_TOKEN"] = HF_TOKEN
else:
    print("  ✗  HF_TOKEN not set and no cached token found.")
    print("     Run: export HF_TOKEN=hf_... and re-run this script.")
    sys.exit(1)

# ── Step 2b: Validate HF token ────────────────────────────────────────────────
if INFERENCE_BACKEND != "nim_local":
    header("Step 2b — Validate HF token", eta="<2s")
    _hf_req = urllib.request.Request(
        "https://huggingface.co/api/whoami",
        headers={"Authorization": f"Bearer {HF_TOKEN}"}
    )
    try:
        with urllib.request.urlopen(_hf_req, timeout=10) as _resp:
            if _resp.status == 200:
                ok("HF token valid")
            else:
                print(f"  ✗  HF token returned HTTP {_resp.status} — run 'huggingface-cli login' on this instance")
                sys.exit(1)
    except Exception as _hf_err:
        warn(f"HF token check failed ({_hf_err}) — continuing, will fail at download if token is bad")

STEPS_DONE.append(2)
print_dashboard()

# ── Step 3: NGC API key check (for NIM mode) ──────────────────────────────────
header("Step 3 — NGC API key (NIM mode)", eta="<5s")
# nim_local backend = local Docker container (this sprint); _cfg["nim"] = NVCF cloud catalog (NIM hosted)
if INFERENCE_BACKEND == "nim_local":
    if not NGC_API_KEY:
        print("  ✗  INFERENCE_BACKEND=nim_local requires NGC_API_KEY (export NGC_API_KEY=nvapi-...)"); sys.exit(1)
    if not NGC_API_KEY.startswith("nvapi-"):
        warn("NGC_API_KEY does not start with 'nvapi-' — docker pull nvcr.io may fail")
    ok(f"NGC_API_KEY set ({len(NGC_API_KEY)} chars) — nim_local Docker mode enabled")
elif _cfg["nim"]:
    if NGC_API_KEY:
        if NGC_API_KEY.startswith("nvapi-"):
            ok(f"NGC_API_KEY set ({len(NGC_API_KEY)} chars, nvapi- prefix) — NIM-{MODEL_SIZE} enabled")
        else:
            warn(f"NGC_API_KEY set but does not start with 'nvapi-' — NIM calls may fail")
    else:
        info(f"NGC_API_KEY not set — NIM-{MODEL_SIZE} will be skipped in Gradio UI")
        info("To enable NIM: export NGC_API_KEY=nvapi-...")
else:
    info(f"MODEL_SIZE={MODEL_SIZE} has no NIM endpoint in catalog — NIM step skipped")
STEPS_DONE.append(3)

# ── Step 4: uv ────────────────────────────────────────────────────────────────
header("Step 4 — uv package manager", eta="<5s if cached, ~10s first time")
rc, _ = run_cmd(["uv", "--version"], env=ENV)
if rc == 0:
    _, ver = run_cmd(["uv", "--version"], env=ENV)
    ok(f"uv already installed ({ver.strip()})")
else:
    run("Installing uv  (~10s)")
    t0 = time.time()
    rc = stream_cmd(["bash", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh"], env=ENV)
    if rc != 0:
        print("  ✗  uv install failed"); sys.exit(1)
    ok(f"uv installed in {time.time()-t0:.0f}s")
STEPS_DONE.append(4)

# ── Step 5: cosmos-reason2 repo ───────────────────────────────────────────────
header("Step 5 — cosmos-reason2 repo", eta="<5s if cached, ~15s first time")
if os.path.exists(f"{REASON2_DIR}/.git"):
    ok(f"cosmos-reason2 already cloned at {REASON2_DIR}")
else:
    run("Cloning cosmos-reason2  (~15s)")
    t0 = time.time()
    rc = stream_cmd(
        ["git", "clone", "https://github.com/nvidia-cosmos/cosmos-reason2.git", REASON2_DIR],
        env=ENV
    )
    if rc != 0:
        print("  ✗  git clone failed"); sys.exit(1)
    ok(f"Cloned in {time.time()-t0:.0f}s")
STEPS_DONE.append(5)

# ── Step 6: Python dependencies ───────────────────────────────────────────────
header("Step 6 — Python dependencies (uv sync)", eta="<5s if cached, ~2-3 min first time")
venv_marker = f"{REASON2_DIR}/.venv/lib"
if os.path.exists(venv_marker):
    ok("virtualenv already present — skipping uv sync")
else:
    _extras = os.environ.get("COSMOS_EXTRAS", "cu128")
    run(f"Running uv sync --extra {_extras}  (~2-3 min)")
    t0 = time.time()
    rc, out = run_cmd(["uv", "sync", "--extra", _extras], cwd=REASON2_DIR, env=ENV, timeout=600)
    if rc != 0:
        run(f"{_extras} failed, trying uv sync without extras")
        rc, out = run_cmd(["uv", "sync"], cwd=REASON2_DIR, env=ENV, timeout=600)
    if rc != 0:
        print("  ✗  uv sync failed:", out[-500:]); sys.exit(1)
    ok(f"Dependencies installed in {time.time()-t0:.0f}s")

# ── Step 6b: vLLM version pin for CUDA 12.8 ──────────────────────────────────
# vLLM version matrix for cu128 environments (Hyperstack H100, driver 570.x = CUDA 12.8):
#   vLLM 0.11.0 → requires torch 2.8.0  (ABI mismatch: cu128 uv sync installs torch 2.9.0)
#   vLLM 0.14.0 → requires torch 2.9.1+cu128 ✅ compatible with CUDA 12.8 driver
#   vLLM 0.20.1 → requires torch 2.11.0+cu130 (needs CUDA 13.0 driver = driver 575.x+)
# Rule: driver_major < 575 → pin vLLM 0.14.0. Driver ≥ 575 → allow latest vLLM.
try:
    import subprocess as _sp_cuda
    _smi = _sp_cuda.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        timeout=10, text=True
    ).strip().split(".")[0:2]
    _driver_major = int(_smi[0])
    if _driver_major < 575:  # < CUDA 13.0 threshold → cu128 pin required
        header("Step 6b — vLLM pin for CUDA 12.8", eta="~30-60s")
        rc_vllm, out_vllm = run_cmd(
            ["uv", "pip", "install", "vllm==0.14.0"],
            cwd=REASON2_DIR, env=ENV, timeout=180
        )
        if rc_vllm == 0:
            ok("vLLM pinned to 0.14.0 (torch 2.9.1+cu128) — compatible with CUDA 12.8 driver")
        else:
            warn(f"vLLM 0.14.0 pin failed — check pip output: {out_vllm[-300:]}")
            warn("vLLM ABI mismatch may cause ImportError at startup")
except Exception:
    pass  # CUDA check is best-effort; if nvidia-smi fails, proceed as-is

# ── Step 6c: ninja build system (required by flashinfer / torch compile) ─────
try:
    import subprocess as _sp_ninja
    _sp_ninja.check_output(["ninja", "--version"], timeout=5, stderr=_sp_ninja.DEVNULL)
except Exception:
    header("Step 6c — ninja build system", eta="<5s")
    try:
        import subprocess as _sp_ninja2
        _sp_ninja2.run(["apt-get", "install", "-y", "ninja-build"],
                       timeout=60, check=True, capture_output=True)
        ok("ninja-build installed")
    except Exception as _e_ninja:
        warn(f"ninja install failed ({_e_ninja}) — vLLM may fail with flashinfer JIT")

STEPS_DONE.append(6)
print_dashboard()

# ── Step 7: PyAV ──────────────────────────────────────────────────────────────
header("Step 7 — PyAV video backend", eta="<5s if cached, ~10s first time")
rc, av_check = run_cmd(
    ["uv", "run", "python", "-c", "import av; print(av.__version__)"],
    cwd=REASON2_DIR, env=ENV
)
if rc == 0 and "16.1.0" in av_check:
    ok(f"PyAV already installed ({av_check.strip()})")
else:
    run("Installing av==16.1.0  (~10s)")
    t0 = time.time()
    rc, out = run_cmd(["uv", "pip", "install", "av==16.1.0"], cwd=REASON2_DIR, env=ENV)
    if rc != 0:
        print("  ✗  av install failed:", out); sys.exit(1)
    ok(f"PyAV installed in {time.time()-t0:.0f}s")

# ── Step 8: Gradio + requests ─────────────────────────────────────────────────
header("Step 8 — Gradio + requests", eta="<5s if cached, ~30s first time")
rc, gr_check = run_cmd(
    ["uv", "run", "python", "-c", "import gradio; print(gradio.__version__)"],
    cwd=REASON2_DIR, env=ENV
)
if rc == 0:
    ok(f"Gradio already installed ({gr_check.strip()})")
else:
    run("Installing gradio  (~30s)")
    t0 = time.time()
    rc, out = run_cmd(["uv", "pip", "install", "gradio"], cwd=REASON2_DIR, env=ENV, timeout=120)
    if rc != 0:
        print("  ✗  gradio install failed:", out); sys.exit(1)
    ok(f"Gradio installed in {time.time()-t0:.0f}s")

rc, rq_check = run_cmd(
    ["uv", "run", "python", "-c", "import requests; print(requests.__version__)"],
    cwd=REASON2_DIR, env=ENV
)
if rc == 0:
    ok(f"requests already installed ({rq_check.strip()})")
else:
    run("Installing requests  (~5s)")
    rc, out = run_cmd(["uv", "pip", "install", "requests"], cwd=REASON2_DIR, env=ENV)
    if rc != 0:
        warn(f"requests install failed — NIM API mode unavailable: {out}")
    else:
        ok("requests installed")

STEPS_DONE.append(7)

# ── Step 7b: Gradio frpc binary (required for public share link) ─────────────
# frpc_linux_amd64_v0.3 is required for Gradio's SHARE=true tunnel.
# On fresh instances, ~/.cache/huggingface/gradio/ may be missing or root-owned.
# We pre-download to $HF_HOME/gradio/frpc/ so Gradio finds it without auth.
_frpc_dir  = os.path.join(HF_HOME_DIR, "gradio", "frpc")
_frpc_path = os.path.join(_frpc_dir, "frpc_linux_amd64_v0.3")
if not os.path.exists(_frpc_path):
    header("Step 7b — Gradio frpc binary (share link tunnel)", eta="~5s")
    os.makedirs(_frpc_dir, exist_ok=True)
    _frpc_url = "https://cdn-media.huggingface.co/frpc-gradio-0.3/frpc_linux_amd64"
    try:
        urllib.request.urlretrieve(_frpc_url, _frpc_path)
        os.chmod(_frpc_path, 0o755)
        ok(f"frpc binary downloaded to {_frpc_path}")
    except Exception as _frpc_err:
        warn(f"frpc download failed ({_frpc_err}) — public Gradio share link may not work")
else:
    ok(f"frpc binary already present ({_frpc_path})")

# ── Step 9: Model weights ──────────────────────────────────────────────────────

def download_model(model_name, model_dir, size_hint, dl_env):
    """Download one HF model. Returns True on success."""
    model_marker  = os.path.join(model_dir, "model.safetensors")
    config_marker = os.path.join(model_dir, "config.json")
    if os.path.exists(model_marker) or os.path.exists(config_marker):
        size_mb = 0
        if os.path.exists(model_marker):
            size_mb = os.path.getsize(model_marker) // (1024 * 1024)
        info_str = f"{size_mb:,} MB" if size_mb else "already present"
        ok(f"Weights already downloaded ({model_name}, {info_str})")
        return True

    run(f"Downloading {model_name}  ({size_hint}, may take several minutes)")
    info("Progress below — download continues even if it looks stalled:")
    t0 = time.time()
    MAX_RETRIES = 5
    rc = 1
    for attempt in range(1, MAX_RETRIES + 1):
        rc = stream_cmd(
            ["uv", "run", "hf", "download", model_name,
             "--local-dir", model_dir],
            cwd=REASON2_DIR, env=dl_env, prefix="HF │ "
        )
        if rc == 0:
            break
        if attempt < MAX_RETRIES:
            print(f"  ✗  Attempt {attempt} failed. Retry {attempt}/{MAX_RETRIES} after 30s", flush=True)
            for _t in range(30):
                time.sleep(1)
                if _t % 10 == 9:
                    _elapsed = int(time.time() - SETUP_START)
                    print(f"  ⟳  Retry in {30-_t-1}s | Elapsed: {_elapsed//60}m {_elapsed%60}s{credits_spent()}", flush=True)
        else:
            print(f"  ✗  All {MAX_RETRIES} attempts failed for {model_name}.")
            return False
    elapsed = time.time() - t0
    ok(f"Downloaded {model_name} in {elapsed/60:.1f} min")
    return True


dl_env = {**ENV, "HF_TOKEN": HF_TOKEN}

if INFERENCE_BACKEND == "nim_local":
    info("nim_local backend — skipping HF weights download (NIM container ships the model)")
else:
    for i, (var_label, var_dirname, var_hf_id, var_size) in enumerate(_cfg["variants"]):
        step_label = f"Step 9{'abcde'[i]} — {var_label} weights ({var_dirname})"
        header(step_label, eta=f"<5s if cached, longer first time ({var_size})")
        var_dir = os.path.join(MODELS_BASE, var_dirname)
        ok_ = download_model(var_hf_id, var_dir, var_size, dl_env)
        if not ok_ and i == 0:
            sys.exit(1)  # primary variant is required
        elif not ok_:
            warn(f"{var_label} download failed — will fall back to HF on first Gradio use")

STEPS_DONE.append(9)
print_dashboard()

# ── Step 9-NIM: Launch NIM container (nim_local backend only) ────────────────
if INFERENCE_BACKEND == "nim_local":
    header("Step 9-NIM — Launch NIM Docker container", eta="~30s if image cached, 10-30 min first pull")
    # Resolve NIM image short id from MODEL_ID. Examples:
    #   nvidia/Cosmos-Reason2-8B  -> cosmos-reason2-8b
    #   nvidia/Cosmos-Reason2-2B  -> cosmos-reason2-2b
    #   nvidia/Cosmos-Reason2-32B -> cosmos-reason2-32b
    _nim_short = (MODEL_ID.split("/")[-1] if "/" in MODEL_ID else MODEL_ID).lower()
    _nim_short = os.environ.get("NIM_MODEL_SHORT", _nim_short)
    _nim_image = os.environ.get("NIM_IMAGE", f"nvcr.io/nim/nvidia/{_nim_short}:latest")
    _nim_port = int(os.environ.get("NIM_PORT", "8000"))
    _nim_launch = "/tmp/nim_launch.sh"
    if not os.path.exists(_nim_launch):
        print(f"  ✗  {_nim_launch} not found — deploy nim_launch.sh first"); sys.exit(1)
    info(f"Image: {_nim_image} | Port: {_nim_port} | Container: cosmos-nim")
    _nim_env = {
        **ENV,
        "NGC_API_KEY":     NGC_API_KEY,
        "MODEL":           _nim_short,
        "IMAGE":           _nim_image,
        "PORT":            str(_nim_port),
        "CONTAINER_NAME":  "cosmos-nim",
        "LOCAL_NIM_CACHE": os.environ.get("LOCAL_NIM_CACHE", os.path.expanduser("~/.cache/nim")),
    }
    rc = stream_cmd(["bash", _nim_launch], env=_nim_env, prefix="NIM │ ")
    if rc != 0:
        print(f"  ✗  NIM launch failed (rc={rc}). Logs: /tmp/nim_launch.log + docker logs cosmos-nim"); sys.exit(1)
    # Point Gradio at the NIM container by overriding VLLM_BASE_URL.
    os.environ["VLLM_BASE_URL"] = f"http://localhost:{_nim_port}/v1"
    ok(f"NIM container live at http://localhost:{_nim_port}/v1")

# ── Step 9b: vLLM server auto-start (BUG-VLLM-AUTOSTART) ─────────────────────
# In vLLM mode Gradio connects to localhost:8000. If vLLM isn't running, the first
# inference request fails with "Connection refused". Start it here, before Gradio.
VLLM_LOG_FILE = "/tmp/vllm_server.log"
if INFERENCE_BACKEND == "vllm":
    _vllm_ready = False
    try:
        urllib.request.urlopen("http://localhost:8000/v1/models", timeout=3)
        _vllm_ready = True
        ok("vLLM already running on :8000 — reusing")
    except Exception:
        pass

    if not _vllm_ready:
        header("Step 9b — Start vLLM server", eta="~60-120s for model load")
        _vllm_flags  = _cfg.get("vllm_extra_flags", ["--gpu-memory-utilization", "0.85"])
        _vllm_maxlen = _cfg.get("vllm_max_model_len", VLLM_MAX_MODEL_LEN)
        _vllm_cmd = [
            f"{REASON2_DIR}/.venv/bin/vllm", "serve", MODEL_DIR,
            "--served-model-name", MODEL_NAME,
            "--port", "8000",
            "--dtype", "auto",
            "--trust-remote-code",
            "--max-model-len", str(_vllm_maxlen),
        ] + _vllm_flags

        _vllm_proc_env = {**ENV, **_cfg.get("vllm_env", {})}
        run(f"Launching vLLM server for {MODEL_SIZE} (log: {VLLM_LOG_FILE})")
        with open(VLLM_LOG_FILE, "w") as _vf:
            subprocess.Popen(_vllm_cmd, cwd=REASON2_DIR, env=_vllm_proc_env,
                             stdout=_vf, stderr=subprocess.STDOUT)

        # 32B models need ~5 min for torch.compile + CUDA graph warmup.
        _LARGE_MODEL_SIZES = {"32B", "C3-32B", "C3-super", "QW3-32B"}
        VLLM_TIMEOUT = 420 if MODEL_SIZE in _LARGE_MODEL_SIZES else 180
        t_vllm = time.time()
        while time.time() - t_vllm < VLLM_TIMEOUT:
            try:
                urllib.request.urlopen("http://localhost:8000/v1/models", timeout=3)
                _vllm_ready = True
                break
            except Exception:
                time.sleep(5)

        if not _vllm_ready:
            print(f"  ✗  vLLM server did not start within {VLLM_TIMEOUT}s. Check {VLLM_LOG_FILE}")
            sys.exit(1)
        ok(f"vLLM ready in {int(time.time() - t_vllm)}s")

# ── Step 10: Launch Gradio ────────────────────────────────────────────────────
header("Step 10 — Launch Gradio web demo", eta="~5-10s for model load")

if not os.path.exists(GRADIO_APP):
    print(f"  ✗  {GRADIO_APP} not found — deploy gradio_cr2_byo.py first"); sys.exit(1)

if os.path.exists(URL_FILE):
    os.remove(URL_FILE)

launch_env = {
    **ENV,
    "MODEL_SIZE":         MODEL_SIZE,
    "MODEL_DIR":          MODEL_DIR,
    "MODEL_NAME":         MODEL_NAME,
    "GRADIO_PORT":        str(GRADIO_PORT),
    "GRADIO_SHARE":       "true",
    "PYTHONUNBUFFERED":   "1",
    "HF_TOKEN":           HF_TOKEN,
    "NGC_API_KEY":        NGC_API_KEY,
    "LOW_VRAM":           "true" if LOW_VRAM else "false",
    "GRADIO_FPS":         str(gradio_fps),
    "GRADIO_MAX_PIXELS":  str(max_pixels),
    "GRADIO_PREFILL_TPS": str(prefill_tps),
    "INFERENCE_BACKEND":  INFERENCE_BACKEND,
    "VLLM_BASE_URL":      os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
    "VLLM_API_KEY":       os.environ.get("VLLM_API_KEY", "EMPTY"),
    "COSMOS_EXTRAS":      os.environ.get("COSMOS_EXTRAS", "cu128"),
    "FLASHINFER_DISABLE_VERSION_CHECK": "1",
    # MAXLEN-001: always pass explicitly — never rely on vLLM default (8192 breaks video queries)
    "VLLM_MAX_MODEL_LEN": str(_cfg.get("vllm_max_model_len", VLLM_MAX_MODEL_LEN)),
    # PRELOAD-001: skip HF preload when using vLLM backend
    "SKIP_HF_PRELOAD":    "1" if INFERENCE_BACKEND == "vllm" else "0",
}

run(f"Starting Cosmos Reason2 {MODEL_SIZE} demo on port {GRADIO_PORT}")

proc = subprocess.Popen(
    ["uv", "run", "python", "-u", "/tmp/gradio_cr2_byo.py"],
    cwd=REASON2_DIR,
    env=launch_env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

# BUG-SHARE-TIMEOUT: When gradio.live is firewalled (locked-down Brev orgs,
# air-gapped hosts), the share link never appears and the loop used to terminate
# the child Gradio process after 300s — destroying a working local-only demo.
# Fix: detect explicit "Could not create share link" or "Running on local URL"
# signals and fall back to a host-reachable local URL instead of killing.
url = None
local_url = None
share_failed = False
url_pattern        = re.compile(r'(https?://[^\s"\']+gradio\.live[^\s"\']*)')
local_url_pattern  = re.compile(r'Running on local URL:\s+(http://[^\s]+)')
URL_CAPTURE_TIMEOUT = 300
t_launch = time.time()
with open(LOG_FILE, "w") as log:
    for line in proc.stdout:
        log.write(line)
        log.flush()
        stripped = line.rstrip()
        if stripped:
            print(f"     {DIM}{stripped}{RESET}", flush=True)
        m = url_pattern.search(stripped)
        if m:
            url = m.group(1).rstrip(".")
            break
        ml = local_url_pattern.search(stripped)
        if ml and not local_url:
            local_url = ml.group(1).rstrip("/")
        if "Could not create share link" in stripped:
            share_failed = True
            if local_url:
                break
        if time.time() - t_launch > URL_CAPTURE_TIMEOUT:
            print(f"  ⚠  No gradio.live URL after {URL_CAPTURE_TIMEOUT}s — falling back to local URL.")
            break

def _detect_host_ip():
    env_ip = os.environ.get("BYO_VIDEO_LOCAL_HOST")
    if env_ip:
        return env_ip
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(2)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        if ip and not ip.startswith("127."):
            return ip
    except Exception:
        pass
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=2)
        for tok in out.stdout.split():
            if "." in tok and not tok.startswith("127."):
                return tok
    except Exception:
        pass
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if not ip.startswith("127."):
            return ip
    except Exception:
        pass
    return None

if not url:
    if not local_url:
        print("  ✗  Gradio printed no URL (neither share nor local). Check /tmp/gradio_demo.log")
        proc.terminate()
        sys.exit(1)
    host_ip = _detect_host_ip()
    if host_ip:
        url = local_url.replace("0.0.0.0", host_ip).replace("127.0.0.1", host_ip)
    else:
        url = local_url
    reason = "share link unavailable (host firewalled)" if share_failed else "share link timed out"
    print(f"  ⚠  {reason}. Using local URL: {url}")
    print(f"     {DIM}If your machine can't reach {url} directly, run on your client:{RESET}")
    print(f"     {DIM}  ssh -L {GRADIO_PORT}:localhost:{GRADIO_PORT} <user@host>{RESET}")
    print(f"     {DIM}then open http://localhost:{GRADIO_PORT}/ in your browser.{RESET}")

# BUG-LIVENESS: probe Gradio /info before declaring live.
# Writing URL_FILE before confirming the process survived causes stale live declarations.
# We poll the local port (not the public URL) because frpc tunnel may lag by a few seconds.
_live_flag = "/tmp/gradio_live.flag"
_probe_ok = False
for _probe_attempt in range(10):
    try:
        urllib.request.urlopen(f"http://localhost:{GRADIO_PORT}/", timeout=3)
        _probe_ok = True
        break
    except Exception:
        time.sleep(2)

if not _probe_ok:
    print("  ✗  Gradio process launched but / probe failed after 20s — process may have crashed.")
    print(f"     Check {LOG_FILE} for errors.")
    sys.exit(1)

with open(URL_FILE, "w") as f:
    f.write(url + "\n")

with open(_live_flag, "w") as f:
    f.write(url + "\n")

ok("Demo server up, public tunnel established")
STEPS_DONE.append(9)
print_dashboard()

# ── Final: print clickable hyperlink ────────────────────────────────────────
print(flush=True)
print(f"{BOLD}{'─'*62}{RESET}", flush=True)
print(f"{BOLD}  Cosmos Reason2 {MODEL_SIZE} Demo — Ready{RESET}", flush=True)
print(f"{'─'*62}", flush=True)
print(f"  {BOLD}URL:{RESET}  {hyperlink(url)}", flush=True)
print(f"  {DIM}Upload any MP4 → select checkpoint → Run Inference{RESET}", flush=True)
print(f"  {DIM}Run All Variants: {_variant_labels}{RESET}", flush=True)
print(f"  {DIM}Results: /tmp/byo_video_reason2_results.json{RESET}", flush=True)
print(f"  {DIM}Benchmark: /tmp/byo_video_benchmark.json{RESET}", flush=True)
print(f"  {DIM}Link valid for 72h. Kill instance when done.{RESET}", flush=True)
if _cfg["nim"] and not NGC_API_KEY:
    print(f"  {YELLOW}⚠  NIM-{MODEL_SIZE} mode requires NGC_API_KEY=nvapi-...{RESET}", flush=True)
print(f"{'─'*62}", flush=True)
print(flush=True)

# BUG-STDOUT-PIPE: Keep stdout pipe alive so Gradio's print(flush=True) calls
# don't get BrokenPipeError. Closing the read end here caused every inference
# request to fail at Step 1 — the first yield fired, then the next print()
# raised BrokenPipeError and the generator died. Drain thread keeps it open.
import threading as _thr

def _drain_gradio_stdout(fh, path):
    try:
        with open(path, "a") as f:
            for line in fh:
                f.write(line)
                f.flush()
    except Exception:
        pass

_thr.Thread(target=_drain_gradio_stdout, args=(proc.stdout, LOG_FILE), daemon=True).start()

# Stay alive while Gradio runs — without this, the subprocess gets SIGHUP when
# the screen session's controlling process exits.
try:
    proc.wait()
except KeyboardInterrupt:
    proc.terminate()
    proc.wait()
