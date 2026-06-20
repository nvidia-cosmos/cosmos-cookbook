#!/usr/bin/env python3
"""
byo_video_runtime_monitor.py — Runtime observer for /byo-video skill.

Polls a remote Gradio host (via SSH) every POLL_INTERVAL seconds.
Captures process health, GPU stats, log tail (since last offset),
and detects errors via a rule catalog. Saves session metrics for
later review.

Usage:
  poll    — long-running daemon (run via nohup)
  status  — print current state JSON
  alerts  — print alert stream (--since-line N to skip seen)
  metrics — summarize inference history
  stop    — kill the daemon via PID file

Output (all local /tmp):
  byo_video_runtime_state.json     — current snapshot
  byo_video_runtime_alerts.jsonl   — append-only alerts
  byo_video_session_metrics.jsonl  — append-only per-inference records
  byo_video_runtime_monitor.log    — daemon's own log
  byo_video_runtime_monitor.pid    — PID for stop
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

POLL_INTERVAL = 30
LOG_TAIL_LINES = 400
SSH_KEY = os.path.expanduser("~/.ssh/id_ed25519")
SSH_TIMEOUT = 15
HANG_THRESHOLD_S = int(os.environ.get("BYO_VIDEO_HANG_THRESHOLD_S", "180"))

STATE_FILE   = Path("/tmp/byo_video_runtime_state.json")
ALERTS_FILE  = Path("/tmp/byo_video_runtime_alerts.jsonl")
METRICS_FILE = Path("/tmp/byo_video_session_metrics.jsonl")
LOG_FILE     = Path("/tmp/byo_video_runtime_monitor.log")
PID_FILE     = Path("/tmp/byo_video_runtime_monitor.pid")
OFFSET_FILE  = Path("/tmp/byo_video_runtime_log_offset.json")

ERROR_RULES = [
    {
        "name": "cuda_oom",
        "patterns": [r"CUDA out of memory", r"OutOfMemoryError", r"torch\.cuda\.OutOfMemoryError"],
        "severity": "critical",
        "summary": "GPU ran out of memory during inference.",
        "fix": "In Advanced Settings, lower GRADIO_MAX_PIXELS (try 65536 or 32768), shorten the video, or drop fps to 1.",
    },
    {
        "name": "pyav_decode",
        "patterns": [r"av\.error", r"InvalidDataError", r"Could not open video", r"Error reading video"],
        "severity": "high",
        "summary": "Video could not be decoded by PyAV.",
        "fix": "Re-encode as MP4/H.264: ffmpeg -i input.mov -c:v libx264 -pix_fmt yuv420p output.mp4. Or try cosmos-reason2/data/sample.mp4.",
    },
    {
        "name": "preprocess_empty",
        "patterns": [r"IndexError.*preprocess", r"got 0 frames", r"empty video", r"sampled 0 frames", r"num_frames=0"],
        "severity": "high",
        "summary": "Video preprocessing returned 0 frames.",
        "fix": "Lower fps in Advanced Settings, check the video is >1s and not corrupt, or try a different file.",
    },
    {
        "name": "shape_mismatch",
        "patterns": [r"size mismatch for", r"shape '.*' is invalid", r"RuntimeError.*shape", r"Expected size"],
        "severity": "high",
        "summary": "Tensor shape mismatch in model forward pass.",
        "fix": "Re-select the checkpoint in Advanced Settings → Checkpoint dropdown. May be a partial model load.",
    },
    {
        "name": "vllm_disconnect",
        "patterns": [r"Connection refused.*8000", r"Connection refused.*vllm", r"vLLM .*not responding", r"ECONNREFUSED.*8000"],
        "severity": "critical",
        "summary": "Inference server not responding on port 8000 (vLLM or local NIM container).",
        "fix": "If vLLM mode: ssh <host> 'pkill -f \"vllm serve\"' then restart Gradio. If NIM mode: ssh <host> 'docker ps -a | grep cosmos-nim' — if exited, ssh <host> 'docker logs --tail 100 cosmos-nim' for the failure reason, then re-run bash /tmp/nim_launch.sh <NGC_API_KEY>.",
    },
    {
        "name": "nim_local_container_down",
        "patterns": [r"\[NIM\] Container not responding at"],
        "severity": "critical",
        "summary": "Local NIM Docker container is not reachable from Gradio.",
        "fix": "ssh <host> 'docker ps -a --filter name=cosmos-nim'; if container is Exited, run ssh <host> 'docker logs --tail 200 cosmos-nim' to see why; relaunch via ssh <host> 'bash /tmp/nim_launch.sh <NGC_API_KEY>'. Confirm port 8000 is mapped: ssh <host> 'curl -sf localhost:8000/v1/models'.",
    },
    {
        "name": "nim_local_image_unauthorized",
        "patterns": [r"unauthorized.*nvcr\.io", r"denied.*nvcr\.io", r"403.*nvcr\.io.*cosmos-reason2"],
        "severity": "critical",
        "summary": "NGC denied pull of cosmos-reason2 NIM image — key invalid or model not allowlisted.",
        "fix": "Confirm NGC_API_KEY is current (nvapi- prefix), then 'docker login nvcr.io --username $oauthtoken --password-stdin'. If still denied, the model variant may require allowlisting via build.nvidia.com.",
    },
    {
        # NIM-8B-FP8-THINK-EOS bug: greedy decode (temperature=0) on the FP8-quantized
        # cosmos-reason2-8b NIM emits `<think>` then EOS, finishing in ≤3 tokens with no
        # actual answer. Gradio shows the bare `<think>` opener and nothing else.
        "name": "nim_local_think_eos_truncation",
        "patterns": [r"\[vllm done\]\s+[\d.]+s\s+·\s+[123]\s+tok"],
        "severity": "high",
        "summary": "NIM 8B emitted only a few tokens — likely the <think>+EOS bug at greedy decode.",
        "fix": "Raise temperature ≥ 0.3 in the Gradio Advanced Settings panel. The Gradio app already clamps internally for nim_local, but if the slider was force-pushed to 0 the bug returns. Confirm with: docker logs cosmos-nim 2>&1 | tail -20 and inspect the actual completion.",
    },
    {
        "name": "process_killed",
        "patterns": [r"out of memory: Killed process", r"oom-killer"],
        "severity": "critical",
        "summary": "A process was killed by the OOM killer (system RAM exhausted).",
        "fix": "Free system RAM, reduce concurrent workloads, or use a host with more RAM. Setup needs a full restart.",
    },
    {
        "name": "broken_pipe",
        "patterns": [r"BrokenPipeError"],
        "severity": "high",
        "summary": "Gradio's stdout pipe broke — generator died mid-inference.",
        "fix": "Stale gradio_cr2_byo.py on the remote. Redeploy from ~/.claude/scripts/gradio_cr2_byo.py.",
    },
    {
        "name": "torch_compile_fail",
        "patterns": [r"torch\._dynamo.*BackendCompilerFailed", r"CompilationError"],
        "severity": "high",
        "summary": "torch.compile failed during model load or first forward pass.",
        "fix": "Set TORCH_COMPILE_DISABLE=1 and restart, or wait — first compile takes 2-5 min on 32B models.",
    },
    {
        "name": "nim_api_error",
        "patterns": [r"\[NIM ERROR\]"],
        "severity": "high",
        "summary": "NIM API call failed (NVCF endpoint).",
        "fix": "Check NGC_API_KEY (must start with nvapi-), verify network can reach integrate.api.nvidia.com, confirm the model is in the NVCF catalog. NIM 2B = nvidia/cosmos-reason2-2b.",
    },
    {
        "name": "vllm_runtime_error",
        "patterns": [r"\[VLLM ERROR\]", r"\[vllm error\]"],
        "severity": "high",
        "summary": "vLLM inference call failed.",
        "fix": "ssh <host> 'curl -sf localhost:8000/v1/models' to check vLLM. If down, restart it. Check /tmp/vllm_server.log on the remote for crash reason.",
    },
    {
        "name": "generic_traceback",
        "patterns": [r"^Traceback \(most recent call last\):"],
        "severity": "medium",
        "summary": "An unhandled exception was raised.",
        "fix": "Read the full traceback in /tmp/gradio_demo.log on the remote.",
    },
]

# Hang detection: temporal, not pattern-based. Inference is "started" when any of
# these markers appears, "ended" when a completion or error marker appears.
INFER_START_MARKERS = [
    re.compile(r"\[infer\]\s+model="),       # HF backend kicks off
    re.compile(r"\[vllm\]\s+endpoint="),     # vLLM backend kicks off
    re.compile(r"\[nim\]\s+NGC_API_KEY"),    # NIM backend kicks off
]
INFER_END_MARKERS = [
    re.compile(r"\[done\]\s+[\d.]+s"),        # HF success
    re.compile(r"\[vllm done\]\s+[\d.]+s"),   # vLLM success
    re.compile(r"\[nim done\]\s+[\d.]+s"),    # NIM success
    re.compile(r"\[NIM ERROR\]"),             # NIM failure (counts as terminated)
    re.compile(r"\[VLLM ERROR\]"),            # vLLM failure
]

# gradio_cr2_byo.py emits one of three completion lines per inference:
#   HF backend:   "[done] 41.9s · 44 tok · ttft=15.08s"
#   vLLM backend: "[vllm done] 30.5s · 287 tok out"
#   NIM backend:  "[nim done] 12.3s · 156 tok out"
# Plus a prefill marker: "[infer] Prefilling 297 tokens ..."
HF_DONE_RE     = re.compile(r"\[done\]\s+([\d.]+)s\s+·\s+(\d+)\s+tok\s+·\s+ttft=([\d.]+)s")
VLLM_DONE_RE   = re.compile(r"\[(vllm|nim) done\]\s+([\d.]+)s\s+·\s+(\d+)\s+tok")
PREFILL_RE     = re.compile(r"\[infer\]\s+Prefilling\s+([\d,]+)\s+tokens")


def ssh_exec(remote, cmd):
    full = ["ssh", "-i", SSH_KEY, "-o", f"ConnectTimeout={SSH_TIMEOUT}",
            "-o", "StrictHostKeyChecking=no", remote, cmd]
    try:
        r = subprocess.run(full, capture_output=True, text=True, timeout=SSH_TIMEOUT + 10)
        return r.returncode, (r.stdout + r.stderr)
    except subprocess.TimeoutExpired:
        return -1, "<ssh timeout>"
    except Exception as e:
        return -1, f"<ssh error: {e}>"


_OFFSET_DEFAULTS = {
    "log_offset": 0,
    "result_mtime": 0,
    "last_infer_start_ts": 0,
    "last_completion_ts": 0,
    "hang_alert_fired_for_start_ts": 0,
}


def _load_offset_record(remote):
    """Returns dict with all keys in _OFFSET_DEFAULTS. Migrates old bare-int schema."""
    rec = dict(_OFFSET_DEFAULTS)
    if not OFFSET_FILE.exists():
        return rec
    try:
        d = json.loads(OFFSET_FILE.read_text())
        stored = d.get(remote, 0)
        if isinstance(stored, int):
            rec["log_offset"] = stored
        elif isinstance(stored, dict):
            for k in _OFFSET_DEFAULTS:
                if k in stored:
                    rec[k] = stored[k]
    except Exception:
        pass
    return rec


def _save_offset_record(remote, **updates):
    """Merge `updates` into the remote's offset record."""
    rec = _load_offset_record(remote)
    rec.update(updates)
    d = {}
    if OFFSET_FILE.exists():
        try:
            d = json.loads(OFFSET_FILE.read_text())
        except Exception:
            pass
    d[remote] = rec
    OFFSET_FILE.write_text(json.dumps(d))


def load_offset(remote):
    return _load_offset_record(remote)["log_offset"]


def load_result_mtime(remote):
    return _load_offset_record(remote)["result_mtime"]


def save_offset(remote, offset):
    _save_offset_record(remote, log_offset=offset)


def save_result_mtime(remote, mtime):
    _save_offset_record(remote, result_mtime=mtime)


def append_alert(alert):
    with ALERTS_FILE.open("a") as f:
        f.write(json.dumps(alert) + "\n")


def append_metric(metric):
    with METRICS_FILE.open("a") as f:
        f.write(json.dumps(metric) + "\n")


def write_state(state):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def log(msg):
    with LOG_FILE.open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def detect_errors(new_log_text):
    alerts = []
    seen_rules = set()
    lines = new_log_text.split("\n")
    for rule in ERROR_RULES:
        if rule["name"] in seen_rules:
            continue
        for pat in rule["patterns"]:
            r = re.compile(pat)
            for i, line in enumerate(lines):
                if r.search(line):
                    seen_rules.add(rule["name"])
                    ctx_start = max(0, i - 3)
                    ctx_end = min(len(lines), i + 4)
                    context = "\n".join(lines[ctx_start:ctx_end])
                    alerts.append({
                        "ts": int(time.time()),
                        "rule": rule["name"],
                        "severity": rule["severity"],
                        "summary": rule["summary"],
                        "fix": rule["fix"],
                        "trigger_line": line.strip()[:300],
                        "context": context[:1200],
                    })
                    break
            if rule["name"] in seen_rules:
                break
    return alerts


def detect_inferences(new_log_text, session_id, model_id):
    """Parse completed inferences from log text. Each [done]/[vllm done]/[nim done]
    line yields one metric record. Prefill token count is carried from the most
    recent [infer] Prefilling N tokens line. gen_s is computed as total - ttft
    when ttft is available (HF backend only); otherwise null."""
    metrics = []
    pending_prefill = None
    for line in new_log_text.split("\n"):
        m_pre = PREFILL_RE.search(line)
        if m_pre:
            try:
                pending_prefill = int(m_pre.group(1).replace(",", ""))
            except Exception:
                pending_prefill = None
            continue
        m_hf = HF_DONE_RE.search(line)
        if m_hf:
            total_s = float(m_hf.group(1))
            tokens = int(m_hf.group(2))
            ttft_s = float(m_hf.group(3))
            metrics.append({
                "ts": int(time.time()),
                "session_id": session_id,
                "model_id": model_id,
                "backend": "hf",
                "prefill_tokens": pending_prefill,
                "total_s": total_s,
                "ttft_s": ttft_s,
                "gen_s": round(max(0.0, total_s - ttft_s), 2),
                "tokens": tokens,
            })
            pending_prefill = None
            continue
        m_v = VLLM_DONE_RE.search(line)
        if m_v:
            metrics.append({
                "ts": int(time.time()),
                "session_id": session_id,
                "model_id": model_id,
                "backend": m_v.group(1),
                "prefill_tokens": pending_prefill,
                "total_s": float(m_v.group(2)),
                "ttft_s": None,
                "gen_s": None,
                "tokens": int(m_v.group(3)),
            })
            pending_prefill = None
    return metrics


def to_int(s):
    try:
        return int(s.strip().split()[0])
    except Exception:
        return None


def to_float(s):
    try:
        return float(s.strip().split()[0])
    except Exception:
        return None


def count_lines(p):
    if not p.exists():
        return 0
    try:
        return sum(1 for _ in p.open())
    except Exception:
        return 0


RESULT_JSON_PATH = "/tmp/byo_video_reason2_results.json"
LOG_BYTE_CAP = 65536  # last 64KB of log per poll, max


def poll_once(remote, rate, session_id, model_id):
    # tail -c 65536 returns raw bytes from end-of-file (not lines). Combined
    # with log_size and last_offset we slice the actual NEW bytes locally —
    # avoids the duplicate-metric bug where detect_inferences re-fired on every
    # [done] line in the tail each time log_size grew.
    bundle = (
        "echo '---PROCS---'; pgrep -af gradio_cr2_byo | head -3; "
        "echo '---PORT---'; ss -tlnp 2>/dev/null | grep 7860 | head -1; "
        "echo '---GPU---'; nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.free,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>&1 | head -1; "
        "echo '---LOGSIZE---'; stat -c %s /tmp/gradio_demo.log 2>/dev/null || echo 0; "
        f"echo '---LOGTAIL---'; tail -c {LOG_BYTE_CAP} /tmp/gradio_demo.log 2>/dev/null; "
        f"echo '---RESULT_MTIME---'; stat -c %Y {RESULT_JSON_PATH} 2>/dev/null || echo 0; "
        f"echo '---RESULT---'; cat {RESULT_JSON_PATH} 2>/dev/null"
    )
    rc, out = ssh_exec(remote, bundle)
    state = {
        "ts": int(time.time()),
        "remote": remote,
        "rate_per_hour": rate,
        "session_id": session_id,
        "model_id": model_id,
        "ssh_ok": rc == 0,
        "gradio_alive": False,
        "port_listening": False,
        "gpu": {},
        "log_size": 0,
        "alerts_total": count_lines(ALERTS_FILE),
        "metrics_total": count_lines(METRICS_FILE),
    }
    if rc != 0:
        state["error"] = f"SSH failed (rc={rc}): {out[:500]}"
        log(f"SSH failed: {out[:200]}")
        write_state(state)
        return state

    sections = {}
    cur = None
    for line in out.split("\n"):
        if line.startswith("---") and line.endswith("---"):
            cur = line.strip("- ").strip()
            sections[cur] = []
        elif cur is not None:
            sections[cur].append(line)
    procs = "\n".join(sections.get("PROCS", []))
    port = "\n".join(sections.get("PORT", []))
    gpu_line = "\n".join(sections.get("GPU", [])).strip()
    try:
        log_size = int("\n".join(sections.get("LOGSIZE", [])).strip() or "0")
    except Exception:
        log_size = 0
    log_tail = "\n".join(sections.get("LOGTAIL", []))

    state["gradio_alive"] = "gradio_cr2_byo" in procs and bool(procs.strip())
    state["port_listening"] = "LISTEN" in port and "7860" in port
    state["log_size"] = log_size

    parts = [p.strip() for p in gpu_line.split(",")]
    if len(parts) >= 5:
        state["gpu"] = {
            "name": parts[0],
            "util_pct": to_int(parts[1]),
            "mem_used_mib": to_int(parts[2]),
            "mem_free_mib": to_int(parts[3]),
            "mem_total_mib": to_int(parts[4]),
            "temp_c": to_int(parts[5]) if len(parts) > 5 else None,
            "power_w": to_float(parts[6]) if len(parts) > 6 else None,
        }

    # Result JSON capture: gradio_cr2_byo writes the full result (prompt, response,
    # timing, token counts, status) to /tmp/byo_video_reason2_results.json on each
    # inference completion, OVERWRITING the previous result. We track its mtime so
    # we only parse it when it advances. Pair it with the most recent log [done]
    # line in this poll so prompt/response attach to the right metric record.
    try:
        result_mtime = int("\n".join(sections.get("RESULT_MTIME", [])).strip() or "0")
    except Exception:
        result_mtime = 0
    result_text = "\n".join(sections.get("RESULT", [])).strip()
    parsed_result = None
    last_result_mtime = load_result_mtime(remote)
    if result_mtime > last_result_mtime and result_text:
        try:
            parsed_result = json.loads(result_text)
            save_result_mtime(remote, result_mtime)
        except Exception as e:
            log(f"result json parse failed: {e}")

    last_offset = load_offset(remote)
    # Slice only NEW bytes. log_tail is the last LOG_BYTE_CAP bytes from the
    # remote; new_content is the suffix of that representing what was added
    # since last poll. Truncation (log_size < last_offset) → reset to full tail.
    new_content = ""
    log_changed = False
    if log_size > last_offset:
        delta = log_size - last_offset
        new_content = log_tail[-delta:] if len(log_tail) >= delta else log_tail
        save_offset(remote, log_size)
        log_changed = True
    elif log_size < last_offset:
        new_content = log_tail
        save_offset(remote, log_size)
        log_changed = True

    if log_changed:
        for a in detect_errors(new_content):
            append_alert(a)
            log(f"ALERT [{a['severity']}] {a['rule']}: {a['summary']}")
            state["last_alert"] = a
        new_metrics = detect_inferences(new_content, session_id, model_id)
        if parsed_result and new_metrics:
            new_metrics[-1]["prompt"] = (parsed_result.get("prompt") or "")[:2000]
            new_metrics[-1]["response"] = (parsed_result.get("response") or "")[:8000]
            new_metrics[-1]["status"] = parsed_result.get("status", "unknown")
            # Fallback for prefill_tokens: if the [infer] Prefilling N tokens
            # marker arrived in a different poll than the [done] marker (so
            # detect_inferences's local pending_prefill was None), the result
            # JSON's tokens_in field is authoritative.
            if new_metrics[-1].get("prefill_tokens") is None:
                tokens_in = parsed_result.get("tokens_in")
                if tokens_in is not None:
                    new_metrics[-1]["prefill_tokens"] = tokens_in
            parsed_result = None  # consumed
        for m in new_metrics:
            append_metric(m)
            ttft_str = f"{m['ttft_s']}s" if m.get('ttft_s') is not None else "n/a"
            has_text = " +text" if m.get("response") else ""
            log(f"INFER backend={m['backend']} total={m['total_s']}s ttft={ttft_str} tokens={m['tokens']}{has_text}")

    # Result JSON updated but no [done] line in log window — synthesize a metric
    # from the result JSON alone (covers the case where log offset already passed
    # the [done] line in a prior poll, or log was truncated).
    if parsed_result:
        synth = {
            "ts": int(time.time()),
            "session_id": session_id,
            "model_id": parsed_result.get("model") or model_id,
            "backend": "from_result_json",
            "prefill_tokens": parsed_result.get("tokens_in"),
            "total_s": parsed_result.get("infer_time_s"),
            "ttft_s": parsed_result.get("ttft_s"),
            "gen_s": None,
            "tokens": parsed_result.get("tokens_out"),
            "prompt": (parsed_result.get("prompt") or "")[:2000],
            "response": (parsed_result.get("response") or "")[:8000],
            "status": parsed_result.get("status", "unknown"),
        }
        append_metric(synth)
        log(f"INFER (from result json) tokens_out={synth['tokens']} ttft={synth['ttft_s']}s")

    if not state["gradio_alive"] and last_offset > 0:
        a = {
            "ts": int(time.time()),
            "rule": "process_disappeared",
            "severity": "critical",
            "summary": "Gradio process is no longer running.",
            "fix": "Re-run /byo-video to restart, or manually relaunch Gradio. Check /tmp/gradio_demo.log on remote for the crash reason.",
            "trigger_line": "(no gradio_cr2_byo PID found)",
            "context": "",
        }
        append_alert(a)
        state["last_alert"] = a
        log("ALERT: gradio_cr2_byo process disappeared")

    # ── Hang detection (temporal) ─────────────────────────────────────────────
    # If we observe a backend start marker but no end marker has arrived within
    # HANG_THRESHOLD_S, fire one inference_hung alert. Resets when next
    # completion lands.
    rec = _load_offset_record(remote)
    last_start_ts = rec["last_infer_start_ts"]
    last_end_ts   = rec["last_completion_ts"]
    last_alerted_for = rec["hang_alert_fired_for_start_ts"]
    now = int(time.time())

    if log_changed and new_content:
        saw_start = any(p.search(new_content) for p in INFER_START_MARKERS)
        saw_end   = any(p.search(new_content) for p in INFER_END_MARKERS)
        if saw_start and now > last_start_ts:
            last_start_ts = now
            _save_offset_record(remote, last_infer_start_ts=last_start_ts)
        if saw_end:
            last_end_ts = now
            _save_offset_record(remote, last_completion_ts=last_end_ts)

    state["last_infer_start_ts"] = last_start_ts
    state["last_completion_ts"]  = last_end_ts

    if last_start_ts > last_end_ts:
        elapsed_since_start = now - last_start_ts
        state["pending_inference_s"] = elapsed_since_start
        if elapsed_since_start > HANG_THRESHOLD_S and last_alerted_for != last_start_ts:
            a = {
                "ts": now,
                "rule": "inference_hung",
                "severity": "high",
                "summary": f"Inference started {elapsed_since_start}s ago but no completion observed.",
                "fix": "Likely network or API hang (NIM mode without reachable integrate.api.nvidia.com, vLLM not responding, or stuck preprocess). Check the latest backend marker in /tmp/gradio_demo.log on remote, kill the Gradio process, and relaunch.",
                "trigger_line": f"(no [done]/[vllm done]/[nim done] within {HANG_THRESHOLD_S}s of last [infer]/[vllm]/[nim] start marker)",
                "context": log_tail[-1000:],
            }
            append_alert(a)
            state["last_alert"] = a
            _save_offset_record(remote, hang_alert_fired_for_start_ts=last_start_ts)
            log(f"ALERT: inference_hung — {elapsed_since_start}s since last start, no completion")

    state["alerts_total"] = count_lines(ALERTS_FILE)
    state["metrics_total"] = count_lines(METRICS_FILE)
    write_state(state)
    return state


def cmd_poll(args):
    PID_FILE.write_text(str(os.getpid()))

    def _stop(signum, frame):
        log(f"signal {signum} received, exiting")
        try:
            PID_FILE.unlink()
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    log(f"runtime monitor started (pid={os.getpid()}, remote={args.remote}, rate=${args.rate}/hr, session={args.session_id})")
    try:
        while True:
            try:
                poll_once(args.remote, args.rate, args.session_id or args.remote, args.model_id or "unknown")
            except Exception as e:
                log(f"poll error: {e}")
            time.sleep(POLL_INTERVAL)
    finally:
        try:
            PID_FILE.unlink()
        except Exception:
            pass


def cmd_status(args):
    if not STATE_FILE.exists():
        print("No state file. Is the runtime monitor running?")
        sys.exit(1)
    print(STATE_FILE.read_text())


def cmd_alerts(args):
    if not ALERTS_FILE.exists():
        print("No alerts.")
        return
    lines = ALERTS_FILE.read_text().splitlines()
    start = max(0, args.since_line)
    if start >= len(lines):
        return
    for i, line in enumerate(lines[start:], start=start):
        try:
            a = json.loads(line)
            print(f"[{i}] {a.get('severity','?').upper():9s}  {a.get('rule','?'):25s}  {a.get('summary','?')}")
            if args.verbose:
                print(f"      fix: {a.get('fix','')}")
                if a.get("trigger_line"):
                    print(f"      trigger: {a['trigger_line']}")
        except Exception:
            pass


def cmd_metrics(args):
    if not METRICS_FILE.exists():
        print("No metrics.")
        return
    lines = METRICS_FILE.read_text().splitlines()
    print(f"Total inferences: {len(lines)}")
    if not lines:
        return
    totals, ttfts, gens, toks = [], [], [], []
    by_backend = {}
    for line in lines:
        try:
            m = json.loads(line)
            totals.append(m.get("total_s", 0))
            if m.get("ttft_s") is not None:
                ttfts.append(m["ttft_s"])
            if m.get("gen_s") is not None:
                gens.append(m["gen_s"])
            toks.append(m.get("tokens", 0))
            by_backend[m.get("backend", "?")] = by_backend.get(m.get("backend", "?"), 0) + 1
        except Exception:
            pass
    if totals:
        print(f"By backend: {by_backend}")
        print(f"Total min/avg/max: {min(totals):.2f}s / {sum(totals)/len(totals):.2f}s / {max(totals):.2f}s")
        if ttfts:
            print(f"TTFT  min/avg/max: {min(ttfts):.2f}s / {sum(ttfts)/len(ttfts):.2f}s / {max(ttfts):.2f}s  (n={len(ttfts)})")
        if gens:
            print(f"Gen   min/avg/max: {min(gens):.2f}s / {sum(gens)/len(gens):.2f}s / {max(gens):.2f}s  (n={len(gens)})")
        print(f"Tokens min/avg/max: {min(toks)} / {sum(toks)//len(toks)} / {max(toks)}")


def cmd_stop(args):
    if not PID_FILE.exists():
        print("Not running (no PID file).")
        return
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to {pid}")
    except Exception as e:
        print(f"Stop failed: {e}")


def main():
    p = argparse.ArgumentParser(description="Runtime monitor for /byo-video skill")
    sp = p.add_subparsers(dest="cmd", required=True)

    pp = sp.add_parser("poll", help="Run polling daemon (use nohup)")
    pp.add_argument("--remote", required=True, help="user@host SSH target")
    pp.add_argument("--rate", type=float, default=0.0, help="$/hr cost (0 for self-hosted SSH targets)")
    pp.add_argument("--session-id", default=None)
    pp.add_argument("--model-id", default=None)
    pp.set_defaults(func=cmd_poll)

    sp.add_parser("status", help="Print current state JSON").set_defaults(func=cmd_status)
    sp.add_parser("stop", help="Stop the daemon via PID file").set_defaults(func=cmd_stop)

    pa = sp.add_parser("alerts", help="Print alert stream")
    pa.add_argument("--since-line", type=int, default=0)
    pa.add_argument("--verbose", "-v", action="store_true")
    pa.set_defaults(func=cmd_alerts)

    sp.add_parser("metrics", help="Summarize inference history").set_defaults(func=cmd_metrics)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
