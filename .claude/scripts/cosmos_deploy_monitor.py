#!/usr/bin/env python3
"""
cosmos_deploy_monitor.py — Local deployment monitoring agent for /byo-video pipeline.

Manages the pipeline on an existing Brev instance: BUILDING wait → script deploy →
setup launch → Gradio live detection. Outputs a formatted checklist each poll cycle
so every Monitor notification shows full pipeline state.

If the instance goes UNHEALTHY, waits 3 min, then rotates to the next provider
in the priority list. When all providers are exhausted, exits 2 so the main agent
can AskUserQuestion.

Exit codes:
  0  DONE     — Gradio live. URL in /tmp/cosmos_deploy_state.json.
  1  ERROR    — Unrecoverable. Message in state file.
  2  INPUT    — Needs user decision. Options in state file.

Usage:
  python3 cosmos_deploy_monitor.py \\
    --instance c3r-nano \\
    --model-id nvidia/Cosmos3-Nano-Reasoner \\
    --model-size C3-8B \\
    --backend vllm \\
    --rate 3.70 \\
    --provider gpu-h100-sxm.1gpu-16vcpu-200gb \\
    --provider-label "H100 SXM"
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import time

STATE_FILE  = "/tmp/cosmos_deploy_state.json"
LIVE_FLAG   = "/tmp/gradio_live.flag"
SETUP_LOG   = "/tmp/byo_video_setup.log"
SCRIPTS_DIR = os.path.expanduser("~/.claude/scripts")

PROVIDER_PRIORITY = [
    {"type": "gpu-h100-sxm.1gpu-16vcpu-200gb", "label": "H100 SXM",       "rate": 3.70, "gpu": "H100"},
    {"type": "hyperstack_A100",                 "label": "A100-80GB",       "rate": 2.20, "gpu": "A100"},
    {"type": "hyperstack_H100",                 "label": "H100 Hyperstack", "rate": 3.70, "gpu": "H100"},
    {"type": "scaleway_A40",                    "label": "A40",             "rate": 1.10, "gpu": "A40"},
]

EXPECTED_SECS = {
    "vllm": {
        "2B": 900, "8B": 1200, "32B": 2400,
        "C3-2B": 900, "C3-8B": 1200, "C3-32B": 2400,
        "NEM-12B": 1200,
        "QW3-2B": 720, "QW3-8B": 900, "QW3-32B": 1800,
    },
    "hf": {"2B": 600, "8B": 720},
}

# Display labels for the 10 steps (script uses 1-7 + 9 + 9b + 10)
STEP_LABELS = {
    1: "GPU detect + VRAM tier",
    2: "HF auth + token validate",
    3: "NGC API key",
    4: "uv package manager",
    5: "cosmos-reason2 repo",
    6: "uv sync + CUDA libs",
    7: "PyAV + Gradio + requests",
    9: "Model weights download",
    10: "Gradio launch",
}


# Models that don't require an HF token (fully public)
HF_PUBLIC_MODELS = {"QW3-2B", "QW3-8B", "QW3-32B"}


def is_public_model(model_size):
    return model_size in HF_PUBLIC_MODELS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_local(args_list, timeout=30):
    result = subprocess.run(args_list, capture_output=True, text=True, timeout=timeout)
    return result.stdout.strip(), result.returncode


def brev_ls(instance):
    """Return (status, build, shell) or (None, None, None) if not found."""
    out, _ = _run_local(["brev", "ls"], timeout=20)
    for line in out.splitlines():
        if instance in line:
            parts = line.split()
            if len(parts) >= 4:
                return parts[1], parts[2], parts[3]
    return None, None, None


def brev_exec(instance, remote_cmd, timeout=60):
    """Run a command on a Brev instance. Returns (stdout, returncode)."""
    result = subprocess.run(
        ["brev", "exec", instance, remote_cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.stdout.strip(), result.returncode


def brev_create(instance, provider):
    gpu_flag = provider["gpu"]
    out, rc = _run_local(
        ["brev", "create", instance, "--gpu-name", gpu_flag, "--type", provider["type"]],
        timeout=60,
    )
    return out, rc


def brev_delete(instance):
    _run_local(["brev", "delete", instance], timeout=30)


def deploy_hf_token(instance, state, done_steps, current_step):
    """Read local HF token, deploy to instance, validate via whoami API.
    Exits 2 if token is missing or invalid; exits 1 on hard deploy failure."""
    local_path = os.path.expanduser("~/.cache/huggingface/token")

    if not os.path.exists(local_path):
        state["exit_code"] = 2
        state["message"] = (
            "HF token not found at ~/.cache/huggingface/token.\n"
            "Copy your token from huggingface.co/settings/tokens, then run:\n"
            "  pbpaste > ~/.cache/huggingface/token\n"
            "Re-run the monitor to continue."
        )
        state["options"] = [
            "Get token at huggingface.co/settings/tokens → copy → pbpaste > ~/.cache/huggingface/token  then retry",
            "Cancel deployment",
        ]
        write_state(state)
        print_checklist(state, done_steps, current_step, "")
        sys.exit(2)

    with open(local_path, "r") as f:
        token = f.read().strip()

    if not token:
        state["exit_code"] = 2
        state["message"] = "~/.cache/huggingface/token is empty — paste a valid token first."
        state["options"] = ["pbpaste > ~/.cache/huggingface/token  then retry", "Cancel deployment"]
        write_state(state)
        print_checklist(state, done_steps, current_step, "")
        sys.exit(2)

    state["last_action"] = "Deploying HF token to instance..."
    print_checklist(state, done_steps, current_step, "")
    write_state(state)

    # Ensure remote cache dir exists
    brev_exec(instance, "mkdir -p ~/.cache/huggingface", timeout=15)

    # Deploy token via base64 (avoids shell quoting issues with token characters)
    b64_token = base64.b64encode(token.encode()).decode()
    write_cmd = (
        "python3 -c \""
        "import base64,os; "
        "p=os.path.expanduser('~/.cache/huggingface/token'); "
        f"open(p,'w').write(base64.b64decode('{b64_token}').decode()); "
        "os.chmod(p,0o600)"
        "\""
    )
    _, rc = brev_exec(instance, write_cmd, timeout=30)
    if rc != 0:
        state["last_action"] = "⚠️ Token deploy failed — retrying..."
        print_checklist(state, done_steps, current_step, "")
        _, rc = brev_exec(instance, write_cmd, timeout=30)
        if rc != 0:
            state["phase"]     = "ERROR"
            state["exit_code"] = 1
            state["message"]   = "Failed to deploy HF token to instance after 2 attempts"
            write_state(state)
            sys.exit(1)

    # Validate token via HF whoami API
    state["last_action"] = "Validating HF token with HuggingFace..."
    print_checklist(state, done_steps, current_step, "")
    write_state(state)

    validate_cmd = (
        'TOKEN=$(cat ~/.cache/huggingface/token) && '
        'curl -sf -H "Authorization: Bearer $TOKEN" https://huggingface.co/api/whoami-v2'
    )
    whoami_out, rc = brev_exec(instance, validate_cmd, timeout=30)

    if rc == 0 and whoami_out.strip().startswith("{"):
        try:
            info   = json.loads(whoami_out)
            user   = info.get("name", "unknown")
            orgs   = [o["name"] for o in info.get("orgs", [])]
            org_s  = ", ".join(orgs[:3]) or "none"
            state["last_action"] = f"✓ HF token valid (user: {user}, orgs: [{org_s}])"
        except (json.JSONDecodeError, KeyError):
            state["last_action"] = "✓ HF token validated"
    else:
        state["exit_code"] = 2
        state["message"] = (
            "HF token validation failed — token may be expired or lack nvidia org access.\n"
            "Get a fresh token at huggingface.co/settings/tokens, then:\n"
            "  pbpaste > ~/.cache/huggingface/token"
        )
        state["options"] = [
            "Refresh token: huggingface.co/settings/tokens → copy → pbpaste > ~/.cache/huggingface/token  then retry",
            "Cancel deployment",
        ]
        write_state(state)
        print_checklist(state, done_steps, current_step, "")
        sys.exit(2)

    print_checklist(state, done_steps, current_step, "")
    write_state(state)


def write_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def elapsed_fmt(start):
    s = int(time.time() - start)
    return f"{s // 60}m {s % 60:02d}s"


def eta_fmt(start, model_size, backend):
    elapsed = int(time.time() - start)
    total   = EXPECTED_SECS.get(backend, {}).get(model_size, 1200)
    remain  = max(0, total - elapsed)
    if remain == 0:
        return "any moment"
    return f"~{remain // 60}m"


def parse_setup_log(log_text, prev_done: set, prev_current, step_start_ts: dict):
    """Parse setup log. Updates step_start_ts in-place for timing.
    Returns (done_steps, current_step, has_error, last_line)."""
    done    = set()
    current = None
    has_err = False
    lines   = [l for l in log_text.splitlines() if l.strip()]
    now     = time.time()

    for line in lines:
        if "[✓] Step" in line:
            try:
                raw = line.split("Step")[1].split(":")[0].strip()
                num = int(raw.rstrip("abcdefghij"))
                done.add(num)
                # Record completion time if not already done
                if num not in prev_done:
                    if str(num) not in step_start_ts:
                        step_start_ts[str(num)] = now
                    step_start_ts[f"{num}_done"] = now
            except (ValueError, IndexError):
                pass
        elif "[→] Step" in line:
            try:
                raw = line.split("Step")[1].split(":")[0].strip()
                current = int(raw.rstrip("abcdefghij"))
                # Record start time for this step on first sight
                if str(current) not in step_start_ts:
                    step_start_ts[str(current)] = now
            except (ValueError, IndexError):
                pass
        if ("  ✗  " in line or "exit status 1" in line
                or "Traceback (most recent call last)" in line
                or any(exc in line for exc in (
                    "NameError:", "TypeError:", "AttributeError:", "ImportError:",
                    "RuntimeError:", "ModuleNotFoundError:", "FileNotFoundError:",
                    "KeyError:", "ValueError:", "OSError:",
                ))):
            has_err = True

    # Step became current this cycle but wasn't tracked
    if current and str(current) not in step_start_ts:
        step_start_ts[str(current)] = now

    last = lines[-1][:90] if lines else ""
    return done, current, has_err, last


def print_checklist(state, done_steps, current_step, last_log_line):
    """Print the full checklist. Lines within 200ms batch into one Monitor notification."""
    phase    = state["phase"]
    elapsed  = elapsed_fmt(state["start_ts"])
    eta      = eta_fmt(state["start_ts"], state["model_size"], state["backend"])
    instance = state["instance"]
    label    = state.get("provider_label", "?")
    rate     = state.get("rate", 0)
    model    = state.get("model_id", "?")
    backend  = state.get("backend", "?")
    msize    = state.get("model_size", "?")
    action   = state.get("last_action", "")

    PHASE_ORDER = ["BUILDING", "DEPLOY_SCRIPTS", "SETUP", "GRADIO_LIVE", "DONE"]

    def mark(target_phase):
        ci = PHASE_ORDER.index(phase)       if phase       in PHASE_ORDER else 0
        ti = PHASE_ORDER.index(target_phase) if target_phase in PHASE_ORDER else 99
        if ci > ti:  return "✓"
        if ci == ti: return "→"
        return " "

    # Phase-level elapsed (BUILDING + DEPLOY_SCRIPTS)
    build_elapsed  = ""
    deploy_elapsed = ""
    if state.get("phase_start_ts"):
        ps = int(time.time() - state["phase_start_ts"])
        ph_str = f"{ps // 60}m {ps % 60:02d}s"
        if phase == "BUILDING":
            build_elapsed  = f"  ({ph_str} — typical ~3-5m)"
        elif phase == "DEPLOY_SCRIPTS":
            deploy_elapsed = f"  ({ph_str} — ~30-120s)"

    buf = [
        f"Cosmos Deploy Monitor — {instance}  (elapsed: {elapsed} | ETA: {eta})",
        f"  {label} · ${rate:.2f}/hr · {model} ({msize}, {backend})",
        "──────────────────────────────────────────────────────────────",
        f"  [{mark('BUILDING')}] Wait for SHELL READY{build_elapsed}",
        f"  [{mark('DEPLOY_SCRIPTS')}] Deploy scripts{deploy_elapsed}",
    ]

    if phase in ("SETUP", "GRADIO_LIVE", "DONE"):
        buf.append(f"  [✓] Setup launched")
        all_done = phase in ("GRADIO_LIVE", "DONE")
        step_starts = state.get("step_start_ts", {})
        for num, label_s in STEP_LABELS.items():
            if num in done_steps or all_done:
                # Show how long the step took if recorded
                s_start = step_starts.get(str(num))
                s_done  = step_starts.get(f"{num}_done")
                if s_start and s_done:
                    took = int(s_done - s_start)
                    buf.append(f"       ✓ Step {num}: {label_s}  ({took}s)")
                else:
                    buf.append(f"       ✓ Step {num}: {label_s}")
            elif num == current_step:
                s_start = step_starts.get(str(num))
                running = f"  ({int(time.time() - s_start)}s)" if s_start else ""
                buf.append(f"       → Step {num}: {label_s}{running}")
            else:
                buf.append(f"       [ ] Step {num}: {label_s}")
    else:
        buf.append(f"  [{mark('SETUP')}] Setup (byo_video_setup.py)")

    buf.append(f"  [{mark('GRADIO_LIVE')}] Gradio live")
    buf.append("──────────────────────────────────────────────────────────────")

    if action:
        buf.append(f"  ↳ {action}")
    if last_log_line and phase == "SETUP":
        buf.append(f"  log: {last_log_line}")

    if state.get("gradio_url"):
        buf.append(f"  🎉 {state['gradio_url']}")

    print("\n".join(buf), flush=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance",       required=True)
    ap.add_argument("--model-id",       required=True)
    ap.add_argument("--model-size",     default="2B")
    ap.add_argument("--backend",        default="vllm", choices=["vllm", "hf"])
    ap.add_argument("--rate",           type=float, default=3.70)
    ap.add_argument("--provider",       default="gpu-h100-sxm.1gpu-16vcpu-200gb")
    ap.add_argument("--provider-label", default="H100 SXM")
    args = ap.parse_args()

    # Find this provider's index in the priority list for rotation
    prov_idx = next(
        (i for i, p in enumerate(PROVIDER_PRIORITY) if p["type"] == args.provider), 0
    )

    state = {
        "phase":          "BUILDING",
        "phase_start_ts": time.time(),
        "start_ts":       time.time(),
        "instance":       args.instance,
        "model_id":       args.model_id,
        "model_size":     args.model_size,
        "backend":        args.backend,
        "rate":           args.rate,
        "provider":       args.provider,
        "provider_label": args.provider_label,
        "gradio_url":     None,
        "last_action":    "Waiting for SHELL READY...",
        "exit_code":      None,
        "message":        None,
    }
    write_state(state)

    scripts_deployed = False
    token_handled    = is_public_model(args.model_size)  # skip for public models
    setup_launched   = False
    unhealthy_start  = None
    done_steps: set  = set()
    current_step     = None
    step_start_ts: dict = {}
    state["step_start_ts"] = step_start_ts

    # ── Poll loop ─────────────────────────────────────────────────────────────
    while True:
        try:
            brev_status, brev_build, brev_shell = brev_ls(args.instance)
        except subprocess.TimeoutExpired:
            time.sleep(15)
            continue

        # ── UNHEALTHY / CREATE_FAILED detection ──────────────────────────────
        build_failed = brev_build in ("UNHEALTHY", "FAILED", "CREATE_FAILED")

        # CREATE_FAILED = never provisioned → rotate immediately, no recovery window
        if brev_build == "CREATE_FAILED":
            print(f"✗ {args.provider_label} CREATE_FAILED — rotating provider immediately", flush=True)
            brev_delete(args.instance)
            time.sleep(10)

            next_providers = PROVIDER_PRIORITY[prov_idx + 1:]
            if not next_providers:
                state["phase"]     = "ERROR"
                state["exit_code"] = 2
                state["message"]   = (
                    f"All {len(PROVIDER_PRIORITY)} providers failed (CREATE_FAILED) starting with "
                    f"{args.provider_label}. Please choose an action."
                )
                state["options"] = ["Try again with same provider", "Cancel deployment"]
                write_state(state)
                print_checklist(state, done_steps, current_step, "")
                sys.exit(2)

            next_p              = next_providers[0]
            prov_idx           += 1
            args.provider       = next_p["type"]
            args.provider_label = next_p["label"]
            args.rate           = next_p["rate"]
            state["provider"]       = next_p["type"]
            state["provider_label"] = next_p["label"]
            state["rate"]           = next_p["rate"]
            state["last_action"]    = f"✗ CREATE_FAILED — trying {next_p['label']}"
            print(f"  → trying {next_p['label']} ({next_p['type']})", flush=True)
            _, rc = brev_create(args.instance, next_p)
            if rc != 0:
                state["last_action"] = f"⚠️ brev create {next_p['label']} also failed — will retry next cycle"
            unhealthy_start  = None
            scripts_deployed = False
            token_handled    = is_public_model(args.model_size)
            setup_launched   = False
            done_steps       = set()
            current_step     = None
            state["phase"]   = "BUILDING"
            write_state(state)
            time.sleep(30)
            continue

        if build_failed:
            if unhealthy_start is None:
                unhealthy_start = time.time()
                state["last_action"] = f"⚠️ UNHEALTHY — 3-min recovery window starting"

            wait_remaining = 180 - int(time.time() - unhealthy_start)
            if wait_remaining > 0:
                state["phase"]       = "BUILDING"
                state["last_action"] = f"⚠️ UNHEALTHY — checking again in 30s ({wait_remaining}s left in window)"
                print_checklist(state, done_steps, current_step, "")
                write_state(state)
                time.sleep(30)
                continue

            # Recovery window expired — rotate provider
            print(f"✗ {args.provider_label} UNHEALTHY (3-min window expired) — rotating provider", flush=True)
            brev_delete(args.instance)
            time.sleep(15)

            next_providers = PROVIDER_PRIORITY[prov_idx + 1:]
            if not next_providers:
                state["phase"]    = "ERROR"
                state["exit_code"] = 2
                state["message"]  = (
                    f"All {len(PROVIDER_PRIORITY)} providers exhausted after UNHEALTHY on "
                    f"{args.provider_label}. Please choose an action."
                )
                state["options"] = ["Try again with same provider", "Cancel deployment"]
                write_state(state)
                print_checklist(state, done_steps, current_step, "")
                sys.exit(2)

            next_p           = next_providers[0]
            prov_idx        += 1
            args.provider    = next_p["type"]
            args.provider_label = next_p["label"]
            args.rate        = next_p["rate"]
            state["provider"]       = next_p["type"]
            state["provider_label"] = next_p["label"]
            state["rate"]           = next_p["rate"]
            state["last_action"]    = f"✗ {args.provider_label} — rotating to {next_p['label']}"

            print(f"  → trying {next_p['label']} ({next_p['type']})", flush=True)
            _, rc = brev_create(args.instance, next_p)
            if rc != 0:
                # Create failed — try next provider on next iteration
                prov_idx -= 1  # will be re-incremented next UNHEALTHY cycle
            unhealthy_start  = None
            scripts_deployed = False
            token_handled    = is_public_model(args.model_size)
            setup_launched   = False
            done_steps       = set()
            current_step     = None
            state["phase"]   = "BUILDING"
            write_state(state)
            time.sleep(30)
            continue

        else:
            unhealthy_start = None  # reset on clean status

        # ── Instance not found: provision it ─────────────────────────────────
        if brev_status is None:
            state["phase"]          = "BUILDING"
            state["phase_start_ts"] = time.time()
            state["last_action"]    = f"Instance not found — provisioning {args.provider_label}..."
            scripts_deployed = False
            token_handled    = is_public_model(args.model_size)
            setup_launched   = False
            print_checklist(state, done_steps, current_step, "")
            current_p = next(
                (p for p in PROVIDER_PRIORITY if p["type"] == args.provider),
                PROVIDER_PRIORITY[prov_idx],
            )
            _, rc = brev_create(args.instance, current_p)
            if rc != 0:
                state["last_action"] = f"⚠️ brev create failed — will retry next cycle"
            write_state(state)
            time.sleep(30)
            continue

        # ── BUILDING: wait for SHELL READY ────────────────────────────────────
        if brev_shell != "READY":
            if state["phase"] != "BUILDING":
                state["phase_start_ts"] = time.time()
            state["phase"]       = "BUILDING"
            state["last_action"] = f"Building... (BUILD={brev_build})"
            print_checklist(state, done_steps, current_step, "")
            write_state(state)
            time.sleep(30)
            continue

        # ── SHELL READY: deploy scripts ───────────────────────────────────────
        if not scripts_deployed:
            if state["phase"] != "DEPLOY_SCRIPTS":
                state["phase_start_ts"] = time.time()
            state["phase"]       = "DEPLOY_SCRIPTS"
            state["last_action"] = "SHELL READY — deploying scripts..."
            print_checklist(state, done_steps, current_step, "")

            script_names = ("byo_video_setup", "gradio_cr2_byo")
            for idx, name in enumerate(script_names, 1):
                path = os.path.join(SCRIPTS_DIR, f"{name}.py")
                if not os.path.exists(path):
                    state["phase"]     = "ERROR"
                    state["exit_code"] = 1
                    state["message"]   = f"Script not found locally: {path}"
                    write_state(state)
                    print(f"ERROR: {state['message']}", flush=True)
                    sys.exit(1)

                # Progress line fires an immediate Monitor notification
                state["last_action"] = f"Uploading {name}.py ({idx}/{len(script_names)})..."
                print_checklist(state, done_steps, current_step, "")
                write_state(state)

                t0 = time.time()
                try:
                    cp_result = subprocess.run(
                        ["brev", "copy", path, f"{args.instance}:/tmp/{name}.py"],
                        capture_output=True, text=True, timeout=120,
                    )
                    rc, cp_stderr = cp_result.returncode, cp_result.stderr
                except subprocess.TimeoutExpired:
                    rc, cp_stderr = 1, "brev copy timed out after 120s"
                if rc != 0:
                    time.sleep(5)
                    try:
                        cp_result = subprocess.run(
                            ["brev", "copy", path, f"{args.instance}:/tmp/{name}.py"],
                            capture_output=True, text=True, timeout=120,
                        )
                        rc, cp_stderr = cp_result.returncode, cp_result.stderr
                    except subprocess.TimeoutExpired:
                        rc, cp_stderr = 1, "brev copy timed out after 120s"
                    if rc != 0:
                        state["phase"]     = "ERROR"
                        state["exit_code"] = 1
                        state["message"]   = (
                            f"Failed to deploy {name}.py after 2 attempts "
                            f"(brev copy error: {cp_stderr[:200]})"
                        )
                        write_state(state)
                        print(f"ERROR: {state['message']}", flush=True)
                        sys.exit(1)

                took = int(time.time() - t0)
                state["last_action"] = f"✓ {name}.py uploaded ({took}s) — {idx}/{len(script_names)}"
                print_checklist(state, done_steps, current_step, "")
                write_state(state)

            scripts_deployed     = True
            state["last_action"] = "Scripts deployed ✓"
            write_state(state)

        # ── HF token: deploy and validate for gated models ───────────────────
        if scripts_deployed and not token_handled:
            deploy_hf_token(args.instance, state, done_steps, current_step)
            token_handled        = True
            state["last_action"] = "Credentials ready — launching setup..."
            write_state(state)

        # ── Launch setup in background ────────────────────────────────────────
        if not setup_launched and token_handled:
            state["phase"]       = "SETUP"
            state["last_action"] = "Launching setup in background..."
            print_checklist(state, done_steps, current_step, "")

            inner = (
                f"export INFERENCE_BACKEND={args.backend} "
                f"MODEL_ID={args.model_id} "
                f"BREV_RATE_PER_HOUR={args.rate} "
                f"PATH=~/.local/bin:~/.cargo/bin:$PATH && "
                f"python3 /tmp/byo_video_setup.py"
            )
            launch_cmd = f"nohup bash -c \"{inner}\" > {SETUP_LOG} 2>&1 &"
            try:
                brev_exec(args.instance, launch_cmd, timeout=30)
            except subprocess.TimeoutExpired:
                pass  # fire-and-forget; nohup process continues on the instance
            setup_launched       = True
            state["last_action"] = "Setup launched — watching log"
            write_state(state)
            time.sleep(20)
            continue

        # ── SETUP running: tail log, check live flag ──────────────────────────
        if state["phase"] == "SETUP":
            try:
                log_raw, _ = brev_exec(
                    args.instance, f"tail -60 {SETUP_LOG} 2>/dev/null", timeout=30
                )
            except subprocess.TimeoutExpired:
                log_raw = ""
            done_steps, current_step, has_err, last_line = parse_setup_log(
                log_raw, done_steps, current_step, step_start_ts
            )
            state["step_start_ts"] = step_start_ts

            # Check live flag
            try:
                flag_out, flag_rc = brev_exec(
                    args.instance, f"cat {LIVE_FLAG} 2>/dev/null", timeout=15
                )
            except subprocess.TimeoutExpired:
                flag_out, flag_rc = "", 1
            url = flag_out.strip()
            if flag_rc == 0 and url.startswith("http"):
                state["gradio_url"]  = url
                state["phase"]       = "GRADIO_LIVE"
                state["last_action"] = f"Gradio live ✓"
                done_steps           = set(STEP_LABELS.keys())
                print_checklist(state, done_steps, None, "")
                state["exit_code"]   = 0
                write_state(state)
                print(f"DONE: {url}", flush=True)
                sys.exit(0)

            # Check if setup process is still running (after 60s startup grace period)
            setup_elapsed = int(time.time() - state["start_ts"])
            if setup_elapsed > 60:
                try:
                    proc_out, _ = brev_exec(
                        args.instance, "pgrep -f byo_video_setup.py", timeout=15
                    )
                except subprocess.TimeoutExpired:
                    proc_out = "timeout"  # assume alive on timeout
                if not proc_out.strip():
                    state["phase"]     = "ERROR"
                    state["exit_code"] = 1
                    state["message"]   = (
                        f"Setup process exited without Gradio going live. "
                        f"Last log: {last_line or '(no log output)'}"
                    )
                    write_state(state)
                    print_checklist(state, done_steps, current_step, last_line)
                    print(f"ERROR: {state['message']}", flush=True)
                    sys.exit(1)

            # Check for setup failure
            if has_err:
                state["last_action"] = f"⚠️ Error in setup log: {last_line}"

                # Check if vLLM or critical step failed
                critical_failure = any(
                    kw in log_raw
                    for kw in (
                        "sys.exit(1)", "CUDA out of memory", "No space left", "SIGKILL",
                        "Traceback (most recent call last)",
                        "NameError:", "TypeError:", "AttributeError:", "ImportError:",
                        "RuntimeError:", "ModuleNotFoundError:", "FileNotFoundError:",
                        "KeyError:", "ValueError:", "OSError:",
                    )
                )
                if critical_failure:
                    state["phase"]    = "ERROR"
                    state["exit_code"] = 1
                    state["message"]  = f"Setup failed: {last_line}"
                    write_state(state)
                    print_checklist(state, done_steps, current_step, last_line)
                    sys.exit(1)
            else:
                state["last_action"] = last_line or "setup running..."

            print_checklist(state, done_steps, current_step, last_line)
            write_state(state)
            time.sleep(30)
            continue

        time.sleep(30)


if __name__ == "__main__":
    main()
