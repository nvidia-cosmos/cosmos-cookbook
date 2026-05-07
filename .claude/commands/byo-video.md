# /byo-video — Cosmos BYO-Video Demo

**Single-model deployment skill.** Deploys any supported model (Cosmos Reason2, Nemotron-Nano-12B-v2-VL, Qwen3-VL, etc.) to a Gradio web UI at a `gradio.live` public URL — user uploads video in browser.

For multi-model side-by-side comparison, use `/vlm-race` (separate skill, separate instance required).

Default output: **Gradio web UI at a `gradio.live` public URL** — user uploads video in browser.

**Canonical scripts (stable, versioned — do not read from /tmp/):**
- `~/.claude/scripts/gradio_cr2_byo.py`  — Gradio app (all supported models)
- `~/.claude/scripts/byo_video_setup.py` — Bootstrap + launch script

---

**Primary launch command (all environments):**
```bash
python3 /tmp/byo_video_setup.py
```
This script shows live step-by-step progress with ETAs for every install stage, then prints a clickable hyperlink to the Gradio UI. The URL is also written to `/tmp/gradio_url.txt` for agent capture. Deploy it to the instance before running (see deploy section below).

---

## SKILL PROTOCOL — Hybrid: main session + observer

The main session owns **PHASE 0–1** (pre-checks + picker). These phases are interactive;
`AskUserQuestion` resolves all user decisions before anything runs on a GPU.

Immediately after the picker answers are resolved, the main session spawns a **background
observer** for **PHASE 2–6** (provision → shell ready → deploy scripts → setup → Gradio live).
The observer handles all long-running work autonomously, keeping the main session free for
conversation. On completion the observer writes `/tmp/byo_video_observer_result.json`; the
main session reads it via the task completion notification and either displays the final URL
panel or fires `AskUserQuestion` for recovery.

**All user decisions happen in PHASE 1 before the observer spawns.** This includes the
"restart stopped instance vs provision fresh" choice — PHASE 0 data is used to build the Q3
options dynamically (see PICKER section).

**Never use external chat or notification systems in this skill.** All user communication
happens in the Claude Code UI — panels, `AskUserQuestion`, and completion text.

### PHASE 0 — Pre-checks

0. **Clear stale session artifacts first** (one Bash call before anything else):
   ```bash
   rm -f /tmp/byo_video_observer_result.json /tmp/byo_video_progress.json
   ```
   These files persist across Claude Code sessions. If not cleared, the progress cron will
   read a prior session's result and report a false success or failure. This step is mandatory
   and must run before `brev ls`.

1. Run `brev ls` via Bash. Capture the full output.
2. Note stopped H100/H200 instances — these become options in the PHASE 1 picker (Q3).
3. Note any RUNNING instances (may be reusable).

Display initial panel:

```
╔══════════════════════════════════════════════════════════════╗
║  Cosmos BYO-Video  ·  Pre-checks                             ║
╠══════════════════════════════════════════════════════════════╣
║  Existing instances:                                         ║
║    [list each: name | status | GPU | type]                   ║
║    (or "None found")                                         ║
╚══════════════════════════════════════════════════════════════╝
```

### PHASE 1 — Picker

Load `AskUserQuestion` via ToolSearch: `query: "select:AskUserQuestion"`, then fire all 3
questions in a single call (see PICKER section below for the full call).

**Q3 always offers four options.** The first slot is dynamic; the remaining three are fixed:

- **Slot 1 (dynamic):** If PHASE 0 found exactly one stopped instance with a compatible GPU for
  the selected model, use: `{ label: "Restart <name> (<GPU>)", description: "Resume stopped — faster to SHELL READY" }`
  mapping to `DEPLOY_TARGET=brev:<name>`. If multiple stopped instances match, pick the one
  whose GPU tier best fits MODEL_SIZE (prefer H200 for 32B/C3-32B, H100 for all others).
  If no stopped instances match, use `{ label: "New Brev instance", description: "Agent provisions appropriate GPU tier" }`.
- **Slot 2 (fixed):** `{ label: "New Brev instance", description: "Agent provisions appropriate GPU tier for your model" }` — always present unless Slot 1 is already "New Brev".
- **Slot 3 (fixed):** `{ label: "SSH target", description: "Provide user@host or IP — any GPU machine you can SSH into" }` → follow-up AskUserQuestion for host.
- **Slot 4 (fixed):** `{ label: "Local machine", description: "Run on this Mac — agent checks nvidia-smi locally first" }` → `DEPLOY_TARGET=local`.

This ensures SSH and Local are always reachable regardless of how many Brev instances exist.

Once all answers are received, resolve `MODEL_ID`, `MODEL_SIZE`, `INFERENCE_BACKEND`,
`DEPLOY_TARGET` per the answer→env var mapping table.

If Q3 answer is "Existing Brev" or "SSH target", fire one follow-up `AskUserQuestion` to collect
the instance name or host.

Record `PROVISION_START_TS` immediately after all answers are in. Then spawn the observer
(see HANDOFF section) and stay available for conversation.

---

## PICKER — Main session PHASE 1

Run after PHASE 0. Load `AskUserQuestion` via ToolSearch first: `query: "select:AskUserQuestion"`

Then fire with all 3 questions in a single call:

```
AskUserQuestion({
  questions: [
    {
      question: "Backend?",
      header: "Backend",
      multiSelect: false,
      options: [
        { label: "vLLM (Recommended)", description: "~15–20 min setup, quantization support, fast inference" },
        { label: "HF Transformers", description: "~8–12 min setup, no quantization, simpler" },
        { label: "NIM (local Docker)", description: "Pull nvcr.io NIM container, serve OpenAI-compatible API on port 8000 — needs NGC_API_KEY + Docker on the target" }
      ]
    },
    {
      question: "Model?",
      header: "Model",
      multiSelect: false,
      options: [
        { label: "Cosmos Reason2 2B", description: "nvidia/Cosmos-Reason2-2B (public, ≥40GB VRAM)" },
        { label: "Cosmos Reason2 8B", description: "nvidia/Cosmos-Reason2-8B (public, ≥80GB VRAM)" },
        { label: "Cosmos Reason2 32B", description: "nvidia/Cosmos-Reason2-32B (public, ≥141GB VRAM — H200 required)" },
        { label: "Something else", description: "Cosmos 3 (private), Cosmos Transfer, Nemotron, non-NVIDIA models" }
      ]
    },
    {
      question: "Environment?",
      header: "Environment",
      multiSelect: false,
      options: [
        // Slot 1: best stopped Brev instance for selected model, OR "New Brev instance" if none
        { label: "Restart <name> (<GPU>)", description: "Resume stopped instance — faster to SHELL READY" },
        // OR if no stopped instance matches:
        // { label: "New Brev instance", description: "Agent provisions appropriate GPU tier for your model" },

        // Slot 2: always present (unless Slot 1 is already "New Brev instance")
        { label: "New Brev instance", description: "Agent provisions appropriate GPU tier for your model" },

        // Slots 3 & 4: always present — never omit these
        { label: "SSH target", description: "Provide user@host or IP — any GPU machine you can SSH into" },
        { label: "Local machine", description: "Run on this Mac — agent checks nvidia-smi locally" }
      ]
    }
  ]
})
```

**Answer → env var mapping:**

| Question | Answer | Sets |
|---|---|---|
| Q1 Backend | vLLM | `INFERENCE_BACKEND=vllm` |
| Q1 Backend | HF Transformers | `INFERENCE_BACKEND=hf` |
| Q1 Backend | NIM (local Docker) | `INFERENCE_BACKEND=nim_local` · `NIM_IMAGE=nvcr.io/nim/nvidia/<model-short-id>:latest` (resolved from MODEL_ID) · requires `NGC_API_KEY` |
| Q2 Model | Cosmos Reason2 2B | `MODEL_ID=nvidia/Cosmos-Reason2-2B` · `MODEL_SIZE=2B` |
| Q2 Model | Cosmos Reason2 8B | `MODEL_ID=nvidia/Cosmos-Reason2-8B` · `MODEL_SIZE=8B` |
| Q2 Model | Cosmos Reason2 32B | `MODEL_ID=nvidia/Cosmos-Reason2-32B` · `MODEL_SIZE=32B` |
| Q2 Model | Something else | Fire SOMETHING ELSE sub-picker (see below) |
| Q3 Env | New Brev instance | `DEPLOY_TARGET=brev:new` |
| Q3 Env | Existing Brev | Follow-up AskUserQuestion: "Instance name?" → `DEPLOY_TARGET=brev:<name>` |
| Q3 Env | SSH target | Follow-up AskUserQuestion: "user@host or IP?" → `DEPLOY_TARGET=ssh:<user@host>` |
| Q3 Env | Local machine | `DEPLOY_TARGET=local` |

**SOMETHING ELSE sub-picker** — fire immediately when user selects "Something else" for Q2:

```
AskUserQuestion({
  questions: [
    {
      question: "Which model?",
      header: "Model",
      multiSelect: false,
      options: [
        { label: "Cosmos3-Nano-Reasoner", description: "nvidia/Cosmos3-Nano-Reasoner (8B, private, ≥40GB VRAM, HF_TOKEN required)" },
        { label: "Cosmos3-Reasoner-32B", description: "nvidia/Cosmos3-Reasoner-32B (32B, gated, nvidia org + HF_TOKEN required)" },
        { label: "Cosmos Transfer 2.5", description: "nvidia/Cosmos-Transfer2.5 (generation model, ≥80GB VRAM)" },
        { label: "Nemotron-Nano-12B-v2-VL", description: "nvidia/Nemotron-Nano-12B-v2-VL-BF16 (12B, gated, vLLM only, ≥40GB VRAM)" },
        { label: "Non-NVIDIA models", description: "Best-effort only, not officially supported" }
      ]
    }
  ]
})
```

| Selection | Sets |
|---|---|
| Cosmos3-Nano-Reasoner | `MODEL_ID=nvidia/Cosmos3-Nano-Reasoner` · `MODEL_SIZE=C3-8B` |
| Cosmos3-Reasoner-32B | `MODEL_ID=nvidia/Cosmos3-Reasoner-32B` · `MODEL_SIZE=C3-32B` |
| Cosmos Transfer 2.5 | `MODEL_ID=nvidia/Cosmos-Transfer2.5` · `MODEL_SIZE=32B` |
| Nemotron-Nano-12B-v2-VL | `MODEL_ID=nvidia/Nemotron-Nano-12B-v2-VL-BF16` · `MODEL_SIZE=NEM-12B` |
| Non-NVIDIA models | Show disclaimer inline, then fire Qwen sub-picker |

**Non-NVIDIA disclaimer (display inline before sub-picker):**
> ⚠️ Non-NVIDIA models: NVIDIA does not officially support or guarantee setup for third-party models. This is best-effort only.

**Qwen sub-picker:**

```
AskUserQuestion({
  questions: [
    {
      question: "Which non-NVIDIA model?",
      header: "Model",
      multiSelect: false,
      options: [
        { label: "Qwen3-VL-2B-Instruct", description: "Qwen/Qwen3-VL-2B-Instruct (public, ~8GB VRAM)" },
        { label: "Qwen3-VL-8B-Instruct", description: "Qwen/Qwen3-VL-8B-Instruct (public, ~20GB VRAM)" },
        { label: "Qwen3-VL-32B-Instruct", description: "Qwen/Qwen3-VL-32B-Instruct (public, ~64GB VRAM)" }
      ]
    }
  ]
})
```

| Selection | Sets |
|---|---|
| Qwen3-VL-2B-Instruct | `MODEL_ID=Qwen/Qwen3-VL-2B-Instruct` · `MODEL_SIZE=QW3-2B` |
| Qwen3-VL-8B-Instruct | `MODEL_ID=Qwen/Qwen3-VL-8B-Instruct` · `MODEL_SIZE=QW3-8B` |
| Qwen3-VL-32B-Instruct | `MODEL_ID=Qwen/Qwen3-VL-32B-Instruct` · `MODEL_SIZE=QW3-32B` |

Once `MODEL_ID`, `MODEL_SIZE`, `INFERENCE_BACKEND`, `DEPLOY_TARGET` are all resolved: record
`PROVISION_START_TS` and spawn the observer (see HANDOFF section below).

---

## EXECUTION PROTOCOL — Phase 1 (main session) + Phases 2–6 (observer)

The main session executes only PHASE 0–1. After the picker resolves all answers it spawns the
observer and stays free for conversation.

**PROVISION_START_TS** is recorded in PHASE 1 immediately after all picker answers are in and
passed to the observer as part of the spawn prompt.

The observer handles PHASE 2–6. It outputs clean checklist panels — commands run silently
underneath. See **OBSERVER PROTOCOL** section for the observer's checklist format and
per-phase instructions.

---

### HANDOFF — Spawn observer after PHASE 1

After all picker answers are resolved, display a brief handoff message and spawn the observer:

```
Setting up your demo — observer is running in the background (~15–20 min).
I'll show you the Gradio URL when it's live. Feel free to keep chatting.
```

Spawn:

```
Agent({
  subagent_type: "general-purpose",
  run_in_background: True,
  name: "byo-video-observer",
  prompt: """
OBSERVER TASK — Cosmos BYO-Video PHASE 2–6

You are a background observer. Read the full OBSERVER PROTOCOL section of the /byo-video
skill. Execute PHASE 2 through PHASE 6 exactly as described there.

Inherited state (fill in actual values):
  DEPLOY_TARGET:       <brev:new | brev:<name> | ssh:<user@host> | local>
  MODEL_ID:            <model_id>
  MODEL_SIZE:          <model_size>
  INFERENCE_BACKEND:   <backend>
  RATE:                <rate_per_hour>
  PROVISION_START_TS:  <epoch_seconds>

On success write to /tmp/byo_video_observer_result.json on the LOCAL machine:
  {"status":"live","url":"<gradio_url>","elapsed_s":<N>,"cost":<N>,"instance":"<name>","rate":<rate>,"gpu":"<gpu_label>","model_id":"<model_id>","model_size":"<model_size>","backend":"<backend>"}

On unrecoverable failure write:
  {"status":"failed","phase":<N>,"error":"<one-line error>","instance":"<name>"}

Then exit.
"""
})
```

**After spawning the observer, immediately start the progress loop:**

Load `CronCreate` via ToolSearch: `query: "select:CronCreate"`, then create:

```
CronCreate({
  cron: "*/1 * * * *",
  prompt: "Check /byo-video observer progress: read /tmp/byo_video_progress.json and /tmp/byo_video_observer_result.json on the local machine; display a one-line status update with the current checklist phase and elapsed time. If result JSON exists with status='live', display the LIVE final panel and cancel this cron job (load CronDelete via ToolSearch first). If result JSON exists with status='failed', fire AskUserQuestion with recovery options (see HALT-AND-ASK) and cancel this cron job.",
  recurring: true
})
```

Store the returned job ID as `PROGRESS_LOOP_ID`. Cancel it with `CronDelete(PROGRESS_LOOP_ID)` when the observer completes (success or failure).

After spawning: stay available for conversation.
When the task completion notification fires, OR when the progress loop detects a result JSON, read `/tmp/byo_video_observer_result.json`:
- `status == "live"` → display the LIVE final panel (see PHASE 6 completion template). Cancel the progress loop.
- `status == "failed"` → fire `AskUserQuestion` with recovery options (see HALT-AND-ASK). Cancel the progress loop.

---

### OBSERVER PROTOCOL — Standing Rule

> **OBSERVER RULE: No silent exits.**
> On ANY unrecoverable error — SSH failure, brev create failure, vLLM error, setup script error, all providers exhausted — the observer MUST:
> 1. Write `/tmp/byo_video_observer_result.json` on the local machine with `{"status":"failed","phase":<N>,"error":"<one-line>","instance":"<name>"}`.
> 2. Then `exit`.
>
> Never exit without writing this file. The main session's progress loop reads it every minute — an absent file means "still running," so a silent exit leaves the user waiting forever.

---

### OBSERVER PROTOCOL — PHASE 2: PROVISION

**This phase and all subsequent phases run inside the observer. Do not prompt the user.**

**HF_TOKEN:** Auto-read by the setup script from `~/.cache/huggingface/token` on the remote
instance. Do NOT ask. Do NOT pass as a CLI arg.

**For `DEPLOY_TARGET=brev:new`:**
1. Look up MODEL_SIZE in MODEL_GPU_REQUIREMENTS to select provider type.
2. Run `brev create <name> --type <provider_type>` — one Bash call.
   Name convention: `cr2-<modelsize>-<timestamp-short>` (e.g., `cr2-2b-0505`)
3. `brev create` blocks until shell ready. Proceed to PHASE 4.

**CRITICAL — `cloudCredId` error halt:** If `brev create` output contains `cloudCredId or workspaceGroupId must be specified on request`, do NOT rotate to a fallback provider — this error is an org-level credential gap that applies to all provider types. Write failure JSON immediately and exit:
`{"status":"failed","phase":2,"error":"Brev org missing cloud credential — brev create blocked for all providers. Fix in Brev dashboard org settings.","instance":"<name>"}`

**For `DEPLOY_TARGET=brev:<name>` (restart stopped instance):**
1. Run `brev start <name>` — one Bash call.
2. Poll `brev ls` every 30s until STATUS = RUNNING and SHELL = READY. Then proceed to PHASE 4.

**For `DEPLOY_TARGET=ssh:<user@host>`:**
1. Verify: `ssh -i ~/.ssh/id_ed25519 <user@host> "echo ok"`
2. Proceed directly to PHASE 4 on success.

**Observer checklist** — emit at the start of PHASE 2 and reprint (with updates) on each
phase transition and every 30s poll:

```
╔══════════════════════════════════════════════════════════════╗
║  Cosmos BYO-Video · setting up...  (elapsed: Xm Ys)          ║
║  <GPU label> · $<rate>/hr · <MODEL_ID> (<MODEL_SIZE>, <be>)  ║
╠══════════════════════════════════════════════════════════════╣
║  [→] Provisioning instance                                   ║
║  [ ] Shell ready                                             ║
║  [ ] Scripts deployed                                        ║
║  [ ] Installing dependencies          ETA ~8 min             ║
║  [ ] Downloading model weights        ETA ~5 min             ║
║  [ ] Starting Gradio                  ETA ~1 min             ║
╠══════════════════════════════════════════════════════════════╣
║  Cost so far: $0.00  |  Est. total setup: ~$<low>–$<high>    ║
║  Brev dashboard: https://brev.dev                            ║
╚══════════════════════════════════════════════════════════════╝
```

Mark each row `[✓]` when complete, `[→]` when active. Update elapsed and cost each reprint.
Raw bash output is never surfaced in the checklist.

Write progress to local machine `/tmp/byo_video_progress.json`:
```json
{"phase": 2, "status": "provisioning", "elapsed_s": 0, "instance": "<name>", "checklist": {"provision": "active", "shell": "pending", "scripts": "pending", "deps": "pending", "weights": "pending", "gradio": "pending"}}
```

Cost estimates by GPU tier:
- H100 SXM ($3.54/hr): 10 min setup ~$0.59 · 20 min ~$1.18
- H200 SXM ($4.20/hr): 20 min setup ~$1.40 · 30 min ~$2.10

---

### OBSERVER PROTOCOL — PHASE 3: WAIT FOR SHELL READY

Applies when `brev create` did not already block until shell ready (e.g., restart path).
Poll `brev ls` every 30s. Update the checklist `[→] Shell ready` row each poll.

**On UNHEALTHY:**
1. Update checklist: `[!] UNHEALTHY — recovering (3 min window)`.
2. Poll every 30s for 3 minutes.
3. If recovered: run `brev exec <name> "nvidia-smi"` — if GPU responds, continue.
4. If still UNHEALTHY after 3 minutes:
   - Try `brev reset <name>`. If succeeds → re-enter poll loop.
   - If reset fails: `brev delete <name>`, rotate to next provider (PROVIDER FALLBACK table).
     Emit one line: `✗ <type> UNHEALTHY — deleted, trying <next-type>`. Re-enter PHASE 2.
   - If all providers exhausted: write failure result and exit.
     `{"status":"failed","phase":3,"error":"All providers exhausted — UNHEALTHY","instance":"<last-name>"}`

On each poll, write progress to local machine `/tmp/byo_video_progress.json`:
```json
{"phase": 3, "status": "waiting_shell", "elapsed_s": <N>, "instance": "<name>", "checklist": {"provision": "done", "shell": "active", "scripts": "pending", "deps": "pending", "weights": "pending", "gradio": "pending"}}
```

On SHELL READY: update checklist `[✓] Shell ready`, proceed to PHASE 4.

---

### OBSERVER PROTOCOL — PHASE 4: DEPLOY SCRIPTS

Read and base64-encode each script. One Bash call per file. One Bash call per deploy.

**Step 0 — Auto-deploy HF token (before scripts, always):**
Check if a cached HF token exists locally. If so, deploy it to the instance so the setup script can authenticate without user interaction.
```bash
# Check local token
cat ~/.cache/huggingface/token 2>/dev/null || echo "NO_TOKEN"
```
If a token is found (starts with `hf_`):
```bash
brev exec <name> "mkdir -p ~/.cache/huggingface"
brev exec <name> "echo <token> > ~/.cache/huggingface/token"
```
This prevents the Step 2 HF auth failure that requires a full setup restart.

```bash
# Step 1: read + encode byo_video_setup.py (Bash call 1)
python3 -c "import base64, sys; sys.stdout.write(base64.b64encode(open('<HOME>/.claude/scripts/byo_video_setup.py','rb').read()).decode())"
# → capture B64_SETUP

# Step 2: deploy byo_video_setup.py to instance (Bash call 2)
brev exec <name> "python3 -c \"import base64; open('/tmp/byo_video_setup.py','wb').write(base64.b64decode('<B64_SETUP>'))\""
```
(Replace `<HOME>` with the absolute path to your home directory — `python3` does not expand `~` inside a string literal.)

**Step 3** — Check file size before choosing deploy method for gradio_cr2_byo.py:
```bash
wc -c ~/.claude/scripts/gradio_cr2_byo.py
```

**Step 4a** — If size ≤ 100352 bytes (98KB): base64 encode and deploy:
```bash
python3 -c "import base64, sys; sys.stdout.write(base64.b64encode(open('<HOME>/.claude/scripts/gradio_cr2_byo.py','rb').read()).decode())"
```
→ capture B64_GRADIO, then:
```bash
brev exec <name> "python3 -c \"import base64; open('/tmp/gradio_cr2_byo.py','wb').write(base64.b64decode('<B64_GRADIO>'))\""
```

**Step 4b** — If size > 100352 bytes (>98KB): use brev copy (avoids brev exec argument buffer overflow):
```bash
brev copy ~/.claude/scripts/gradio_cr2_byo.py <name>:/tmp/gradio_cr2_byo.py
```

Update checklist: `[✓] Scripts deployed`

Write progress to local machine `/tmp/byo_video_progress.json`:
```json
{"phase": 4, "status": "scripts_deployed", "elapsed_s": <N>, "instance": "<name>", "checklist": {"provision": "done", "shell": "done", "scripts": "done", "deps": "pending", "weights": "pending", "gradio": "pending"}}
```

---

### OBSERVER PROTOCOL — PHASE 4-NIM: NIM-LOCAL ADDENDUM

**Only when `INFERENCE_BACKEND=nim_local`. Otherwise skip to Phase 5.**

The NIM (local Docker) backend pulls a NIM container from `nvcr.io/nim/nvidia/<model-short>:latest` and runs it on the target's port 8000. The Gradio app talks to the container via the standard OpenAI-compatible client (same code path as vLLM, just a different `VLLM_BASE_URL`). Works on any reachable target — Brev, SSH host, or local Docker — provided NGC_API_KEY and Docker are present. The user supplies the target via the Phase 1 picker; the skill never assumes a default host.

**Source of truth for available VLM NIMs — fetch this URL every time the user selects NIM:**

> [https://docs.nvidia.com/nim/vision-language-models/latest/introduction.html](https://docs.nvidia.com/nim/vision-language-models/latest/introduction.html)

The `~/.claude/scripts/nim_catalog.py` helper does this automatically (and is also deployed to `/tmp/nim_catalog.py` on the target). The agent must:

1. **In the Phase 1 picker (main session)** — when the user selects "NIM (local Docker)" as the backend, run `python3 ~/.claude/scripts/nim_catalog.py upstream` to refresh the canonical model list from the URL above. If a model name appears upstream that is NOT in `KNOWN_VLM_NIMS` (the slug map inside `nim_catalog.py`), surface a one-line note to the user (e.g., *"Heads-up: docs added `Foo VLM`; the byo-video skill doesn't know its nvcr.io short-id yet — file a one-line PR adding it to KNOWN_VLM_NIMS, or use Custom Checkpoint ID."*).
2. **In observer Phase 4 (script deploy)** — deploy `nim_catalog.py` alongside the other scripts (see deploy block below).
3. **In runtime debug (the morning runtime-agent loop)** — re-run `nim_catalog.py upstream` each session start to detect upstream catalog changes (deprecations, new models, version bumps).

**Runtime NIM swap — out-of-band (not via a Gradio button):**

A previous version wired a "Switch to selected NIM" button inside Gradio that streamed `nim_launch.sh` output to the browser. It was removed because a 5-15 min docker pull + 1-3 min vLLM warmup blows past Gradio's streaming heartbeat, so the UI froze even when the backend was making progress. The dropdown still lists every VLM NIM in the upstream catalog (so the user knows their options), but the swap itself is performed out-of-band by one of:

1. **Runtime agent path (preferred)** — when the user says *"switch the NIM to X"*, Claude runs the SSH commands below, watches `docker logs -f cosmos-nim`, confirms `/v1/models` is back up, and tells the user to refresh the Gradio page. Gradio re-queries `/v1/models` on every page load and picks up the new served model id automatically.

2. **Manual SSH path (no Claude needed)** — the Gradio info panel in `nim_local` mode displays this snippet directly:
   ```bash
   ssh <user@host>
   docker rm -f cosmos-nim
   MODEL=<short-id> CONTAINER_NAME=cosmos-nim PORT=8000 \
     bash /tmp/nim_launch.sh <NGC_API_KEY>
   # Then reload the Gradio page.
   ```
   Valid short-ids come from `python3 /tmp/nim_catalog.py list --no-probe` (or the static panel inside Gradio).

**Agent runbook for "switch the NIM" requests:**
1. Run `python3 ~/.claude/scripts/nim_catalog.py upstream` to refresh the catalog from `docs.nvidia.com/nim/vision-language-models/latest/introduction.html`. Surface any upstream model name not in `KNOWN_VLM_NIMS` as a one-line note.
2. Resolve the user's request (e.g. *"Cosmos Reason2 2B"*) to a short-id (e.g. `cosmos-reason2-2b`) via the slug map in `nim_catalog.py`.
3. SSH to the target. Run `docker rm -f cosmos-nim` then `MODEL=<short-id> bash /tmp/nim_launch.sh <NGC_API_KEY>`. Stream the log so the user can see progress.
4. Verify with `curl -sf http://localhost:8000/v1/models | jq '.data[0].id'` — confirm the served model id matches.
5. Tell the user to refresh the Gradio page.

**Required env on the target:**
- `NGC_API_KEY` (must start with `nvapi-`) — used by both `docker login nvcr.io` and the running container
- `HF_TOKEN` — **not needed**; NIM ships the model

**Deploy `nim_launch.sh` AND `nim_catalog.py` alongside the other scripts (Phase 4 step 1):**
```bash
# nim_launch.sh
brev exec <name> "python3 -c \"import base64; open('/tmp/nim_launch.sh','wb').write(base64.b64decode('<B64_NIM_SH>'))\""
brev exec <name> "chmod +x /tmp/nim_launch.sh"
# nim_catalog.py — used by Gradio at startup AND by the runtime agent
brev exec <name> "python3 -c \"import base64; open('/tmp/nim_catalog.py','wb').write(base64.b64decode('<B64_NIM_CATALOG>'))\""
```
(For SSH targets: replace `brev exec <name>` with `ssh -i ~/.ssh/id_ed25519 <user@host>`.)

**`byo_video_setup.py` handles the NIM launch automatically** when `INFERENCE_BACKEND=nim_local`:
- Step 2/2b — HF auth is **skipped**
- Step 3 — NGC_API_KEY is **mandatory** (script exits if missing)
- Step 9 (HF weights download) — **skipped**
- Step 9-NIM (new) — runs `bash /tmp/nim_launch.sh` which:
  1. Reuses an existing healthy container with the same name (idempotent)
  2. Otherwise: `docker login nvcr.io`, `docker pull` the image, `docker run -d` per official build.nvidia.com snippet (`--gpus all --ipc host --shm-size=32GB -e NGC_API_KEY -v $LOCAL_NIM_CACHE:/opt/nim/.cache -u $(id -u) -p 8000:8000`)
  3. Waits up to 1800s for `GET /v1/models` to respond (model download from NGC + load can take 10-15 min on first run for 8B)
- Step 10 — Gradio launches with `VLLM_BASE_URL=http://localhost:8000/v1`. The Gradio app auto-detects the served model name via `/v1/models` (so `_SERVER_MODEL_ID` matches the NIM-served id, e.g. `nvidia/cosmos-reason2-8b`).

**NIM image short-id resolution (in `byo_video_setup.py`):**
```
MODEL_ID=nvidia/Cosmos-Reason2-8B   →  short=cosmos-reason2-8b   →  nvcr.io/nim/nvidia/cosmos-reason2-8b:latest
MODEL_ID=nvidia/Cosmos-Reason2-2B   →  short=cosmos-reason2-2b   →  nvcr.io/nim/nvidia/cosmos-reason2-2b:latest
MODEL_ID=nvidia/Cosmos-Reason2-32B  →  short=cosmos-reason2-32b  →  nvcr.io/nim/nvidia/cosmos-reason2-32b:latest
```
Override at any time with `NIM_MODEL_SHORT` or full `NIM_IMAGE` env var.

**NIM-specific runtime constraint:** the container caps at 5 images per prompt. The Gradio app clamps `max_frames` to 5 when `INFERENCE_BACKEND=nim_local` (vs 8 for vLLM). For single-image inference this is a no-op.

**NIM-8B-FP8-THINK-EOS bug (greedy decode):** the FP8-quantized cosmos-reason2-8b NIM emits a bare `<think>` opener then an EOS-like token at `temperature=0`, finishing in 2-3 tokens with no reasoning trace and no final answer. Visible symptom in Gradio: response shows only `<think>` (or appears empty) and the run completes in <1s with `tok=2` or `tok=3` in `gradio_demo.log`. Workaround: keep temperature ≥ 0.3. The Gradio app defaults the slider to 0.6 in `nim_local` mode and clamps server-side calls to ≥0.3 as a safety net. Runtime monitor rule `nim_local_think_eos_truncation` flags any `[vllm done] X.Xs · 1|2|3 tok` line.

**Failure modes the runtime monitor catches** (`byo_video_runtime_monitor.py` rule names):
- `nim_local_container_down` — Gradio reports `[NIM] Container not responding at` (port 8000 unreachable)
- `nim_local_image_unauthorized` — NGC denied the pull (key invalid or model not allowlisted)
- `vllm_disconnect` — generic port-8000 failure (covers both vLLM and NIM-local container)

---

### OBSERVER PROTOCOL — PHASE 5: SETUP LAUNCH + LOG TAIL

**This phase runs inside the observer subagent, not the main session.**

Record `SETUP_DISPATCHED_AT` = now.

**Pre-launch VRAM check (mandatory — runs before setup):**
```bash
brev exec <name> "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1"
```
(For SSH: replace `brev exec <name>` with `ssh -i ~/.ssh/id_ed25519 <user@host>`.)

Look up `MODEL_GPU_REQUIREMENTS[MODEL_SIZE].min_vram` in MB (multiply GB values by 1000):
`2B`→40000 · `8B`→80000 · `32B`→141000 · `C3-8B`→40000 · `C3-32B`→141000 · `NEM-12B`→40000 · `QW3-2B`→8000 · `QW3-8B`→20000 · `QW3-32B`→64000 · `C3-super`→141000

If `free_vram_mb < min_vram_mb`:
  Write to `/tmp/byo_video_observer_result.json` on the LOCAL machine and exit immediately:
  ```json
  {"status":"failed","phase":5,"error":"insufficient_vram","vram_free_mb":<free>,"vram_needed_mb":<needed>,"model_size":"<MODEL_SIZE>","instance":"<name>"}
  ```
  Do NOT launch setup. The main session handles model switching via AskUserQuestion (see HALT-AND-ASK).

If `free_vram_mb >= min_vram_mb`: proceed with the user's requested model — do not substitute.

Launch setup (one Bash call — nohup so brev exec returns immediately). Pass `MODEL_NAME` and `MODEL_SIZE` explicitly so the setup script does not auto-select a different model:
```bash
brev exec <name> "nohup bash -c 'export INFERENCE_BACKEND=<backend> MODEL_ID=<model_id> MODEL_NAME=<model_id> MODEL_SIZE=<model_size> BREV_RATE_PER_HOUR=<rate> PATH=~/.local/bin:~/.cargo/bin:$PATH && python3 /tmp/byo_video_setup.py > /tmp/byo_video_setup.log 2>&1' &"
```

**Log tail loop (every 30s):** Tail the log, print a compact status line (not a full panel —
the observer is headless). Look for step markers and errors. On each poll, write progress to local machine `/tmp/byo_video_progress.json`:
```json
{"phase": 5, "status": "<active_step>", "elapsed_s": <N>, "instance": "<name>", "checklist": {"provision": "done", "shell": "done", "scripts": "done", "deps": "active|done", "weights": "active|done|pending", "gradio": "pending"}}
```
(Update `deps` to `"done"` after Step 6 completes, `weights` to `"done"` after Step 9 completes.)

```bash
brev exec <name> "tail -60 /tmp/byo_video_setup.log 2>/dev/null || echo '(log not yet written)'"
```

Step markers to detect (scan log for these strings):

```
Step 1: GPU detect       Step 2: HF auth         Step 3: NGC API key
Step 4: uv               Step 5: cosmos-reason2  Step 6: uv sync
Step 7: PyAV/Gradio      Step 9: weights          Step 10: Gradio launch
```

**Error detection:** Scan for `Traceback`, `Error:`, `exit status 1`, `FAILED`.

On error:
1. Retry once: re-run the setup launch command, re-enter log tail loop.
2. If error recurs: write failure result and exit:
   ```json
   {"status":"failed","phase":5,"error":"<one-line error>","instance":"<name>"}
   ```
   Write to `/tmp/byo_video_observer_result.json` on the local machine, then exit.
   The main session handles user recovery via `AskUserQuestion`.

---

### OBSERVER PROTOCOL — PHASE 6: GRADIO LIVE + URL CAPTURE

Update checklist: `[→] Starting Gradio`. Poll every 30s for the live flag. On each poll, write progress to local machine `/tmp/byo_video_progress.json`:
```json
{"phase": 6, "status": "waiting_gradio", "elapsed_s": <N>, "instance": "<name>", "checklist": {"provision": "done", "shell": "done", "scripts": "done", "deps": "done", "weights": "done", "gradio": "active"}}
```

```bash
brev exec <name> "cat /tmp/gradio_live.flag 2>/dev/null"
```

When flag is present, read the URL:
```bash
brev exec <name> "cat /tmp/gradio_url.txt 2>/dev/null"
```

**Compute total cost:** `(time.time() - PROVISION_START_TS) / 3600 * rate_per_hour`

Write success result to `/tmp/byo_video_observer_result.json` on the **local machine**:
```json
{"status":"live","url":"<gradio_url>","elapsed_s":<N>,"cost":<computed>,"instance":"<name>","rate":<rate>,"gpu":"<gpu_label>","model_id":"<model_id>","model_size":"<model_size>","backend":"<backend>"}
```
Then exit. The main session reads this file on task completion and displays the final panel.

**LIVE final panel** (displayed by main session on success):

```
╔══════════════════════════════════════════════════════════════╗
║  Cosmos BYO-Video  ·  LIVE  ✓                                ║
║  <GPU label> · $<rate>/hr · elapsed: <N>m <S>s               ║
╠══════════════════════════════════════════════════════════════╣
║  [✓] Pre-checks                                              ║
║  [✓] Model & environment selected                            ║
║  [✓] Instance: <name> (<GPU type>)                           ║
║  [✓] SHELL READY                                             ║
║  [✓] Scripts deployed                                        ║
║  [✓] Setup complete                                          ║
║  [✓] Gradio live                                             ║
╠══════════════════════════════════════════════════════════════╣
║  URL:  <gradio_url>                                          ║
║  Total setup cost: ~$<computed>  |  Link valid 72h           ║
║  Kill the Brev instance when you're done to stop billing.    ║
╚══════════════════════════════════════════════════════════════╝
```

Do NOT auto-terminate the instance.

---

### POST-LIVE — Runtime Monitor (optional, opt-in)

Once Gradio is live, the user is on their own — but the Gradio UI swallows most server-side
errors as a generic "Error" toast. Errors at preprocess, prefill, or generate stages are
fully visible in `/tmp/gradio_demo.log` on the remote, but invisible in the browser.

The **runtime monitor** is a lightweight polling daemon (`~/.claude/scripts/byo_video_runtime_monitor.py`)
that watches the remote Gradio process, GPU stats, and log tail; pattern-matches errors
against a rule catalog (cuda_oom, pyav_decode, preprocess_empty, shape_mismatch,
vllm_disconnect, process_killed, broken_pipe, torch_compile_fail, generic_traceback,
process_disappeared); and saves per-inference metrics for later review.

After the LIVE panel, fire `AskUserQuestion` to offer the monitor:

```
AskUserQuestion: "Gradio is live. Spawn the runtime monitor to watch for errors and capture session metrics?"
Options:
  "Yes (Recommended)"  — "Polls every 30s; surfaces errors with suggested fixes; saves all inference metrics for later questions"
  "Skip"               — "No background watcher. Spawn later with: python3 ~/.claude/scripts/byo_video_runtime_monitor.py poll --remote <host>"
```

If yes, launch the daemon:

```bash
nohup python3 ~/.claude/scripts/byo_video_runtime_monitor.py poll \
  --remote <user@host> --rate <rate_per_hour> \
  --session-id <session_name> --model-id <model_id> \
  > /tmp/byo_video_runtime_monitor.out 2>&1 &
```

Then schedule a 1-min cron progress loop that reads the alert stream:

```
CronCreate({
  cron: "*/1 * * * *",
  prompt: "Read /tmp/byo_video_runtime_state.json and /tmp/byo_video_runtime_alerts.jsonl. Compare alert line count to /tmp/byo_video_runtime_alerts_seen.txt (default 0). For each new alert: print one-line summary 'severity rule: summary — fix'. Update the seen counter. Then print one-line state: 'gradio=alive|dead | GPU util=N% mem=Xused/Yfree | alerts=N metrics=N'. If gradio_alive=false, fire AskUserQuestion (load via ToolSearch) with options 'Restart Gradio' / 'Investigate log' / 'Abort'. Cancel this cron when user dismisses the monitor.",
  recurring: true
})
```

**Output files (all local /tmp):**
- `byo_video_runtime_state.json` — current snapshot (overwritten each poll)
- `byo_video_runtime_alerts.jsonl` — append-only alert stream
- `byo_video_session_metrics.jsonl` — append-only inference history (prompt, response, ttft_s, gen_s, tokens, model_id)

**Querying session data later:**
```bash
python3 ~/.claude/scripts/byo_video_runtime_monitor.py status
python3 ~/.claude/scripts/byo_video_runtime_monitor.py alerts -v
python3 ~/.claude/scripts/byo_video_runtime_monitor.py metrics
```

**Stopping the monitor:**
```bash
python3 ~/.claude/scripts/byo_video_runtime_monitor.py stop
```
Also cancel the cron loop (`CronDelete <id>`) when stopping.

The monitor is read-only on the remote. Polling cost: one SSH bundle every 30s, ~1KB.

---

### SSH DEPLOYMENTS (non-Brev)

PHASE 0–4 run in the main session with SSH commands replacing `brev exec`. After scripts are
deployed, the same HANDOFF applies — spawn the observer with `DEPLOY_TARGET=ssh:<user@host>`.
The observer runs PHASE 5–6 via SSH instead of `brev exec`.

Deploy scripts:
```bash
ssh -i ~/.ssh/id_ed25519 <user@host> "python3 -c \"import base64; open('/tmp/byo_video_setup.py','wb').write(base64.b64decode('<B64>'))\""
ssh -i ~/.ssh/id_ed25519 <user@host> "python3 -c \"import base64; open('/tmp/gradio_cr2_byo.py','wb').write(base64.b64decode('<B64>'))\""
```

Launch setup:
```bash
ssh -i ~/.ssh/id_ed25519 <user@host> "nohup bash -c 'export INFERENCE_BACKEND=<backend> MODEL_ID=<model_id> MODEL_NAME=<model_id> MODEL_SIZE=<model_size> && python3 /tmp/byo_video_setup.py > /tmp/byo_video_setup.log 2>&1' &"
```

Tail logs:
```bash
ssh -i ~/.ssh/id_ed25519 <user@host> "tail -60 /tmp/byo_video_setup.log 2>/dev/null"
```

URL capture:
```bash
ssh -i ~/.ssh/id_ed25519 <user@host> "cat /tmp/gradio_live.flag 2>/dev/null"
ssh -i ~/.ssh/id_ed25519 <user@host> "cat /tmp/gradio_url.txt 2>/dev/null"
```

---

---

## HALT-AND-ASK PROTOCOL

During PHASE 0–1 (main session): all user decisions are captured in the picker. If an edge case
requires a follow-up question, fire `AskUserQuestion` before spawning the observer.

During PHASE 2–6 (observer): the observer never calls `AskUserQuestion`. On unrecoverable
failure it writes `/tmp/byo_video_observer_result.json` and exits. The main session reads the
result on task completion notification and fires `AskUserQuestion` for recovery.

**Load AskUserQuestion** before firing: `ToolSearch({ query: "select:AskUserQuestion" })`.

### Exception triggers and question templates

**Observer failure (any phase) — triggered by main session progress loop detecting failure JSON:**

Read `/tmp/byo_video_progress.json` for last-known state before firing AskUserQuestion.

```
AskUserQuestion: "Observer failed at Phase <N>: <error>.
Last checklist state: [✓] provision [✓] shell [→] scripts [ ] deps [ ] weights [ ] gradio
What next?"
Options:
  "Retry — spawn a new observer on the same instance"
  "Provision a new instance (current will be deleted)"
  "Abort — delete instance and exit"
```

(Replace the checklist state line with actual values read from `/tmp/byo_video_progress.json` — use `[✓]` for `"done"`, `[→]` for `"active"`, `[ ]` for `"pending"`.)

On "Retry": spawn a new observer with the same state, `DEPLOY_TARGET=brev:<existing-name>`.
On "New instance": delete the failed instance, spawn a new observer with `DEPLOY_TARGET=brev:new`.
On "Abort": `brev delete <name>`, exit skill.

**Insufficient VRAM — triggered by main session detecting `"insufficient_vram"` in failure JSON:**

Read `vram_free_mb` and `vram_needed_mb` from the failure JSON. Build options from MODEL_GPU_REQUIREMENTS — include only models whose `min_vram_mb` ≤ `vram_free_mb`.

```
AskUserQuestion: "The requested model (<MODEL_SIZE>, needs ~<vram_needed_mb/1000>GB VRAM) won't fit on <instance>
(<vram_free_mb/1000>GB free). Switch to a model that fits, or abort?"
Options (show only what fits — examples for 40GB free):
  "Cosmos Reason2 2B (needs ~40GB)"    → MODEL_ID=nvidia/Cosmos-Reason2-2B, MODEL_SIZE=2B
  "Cosmos3-Nano-Reasoner (needs ~40GB)" → MODEL_ID=nvidia/Cosmos3-Nano-Reasoner, MODEL_SIZE=C3-8B
  "Abort — exit without deleting instance"
```

On model switch: update MODEL_ID, MODEL_SIZE, MODEL_NAME to the new selection, respawn observer.
On Abort: exit skill. Do not delete the instance (user may want it for other work).

**Rule:** Never silently substitute a different model. If the requested model does not fit and there
is no confirmed user-approved alternative, always ask. A silent downgrade is a broken demo.

**Rule:** For any exception not listed here, apply the closest matching template. The goal is
one clear question with 2–3 concrete options. Never ask open-ended questions mid-deployment.

---

### MODEL_GPU_REQUIREMENTS — Read before provisioning any instance

| MODEL_SIZE | Min VRAM | Brev type | Rate | Avoid |
|---|---|---|---|---|
| `2B` | 40 GB | `gpu-h100-sxm.1gpu-16vcpu-200gb` | $3.54/hr | — |
| `8B` | 80 GB | `gpu-h100-sxm.1gpu-16vcpu-200gb` | $3.54/hr | A40 (48GB too tight) |
| `32B` | 141 GB | `gpu-h200-sxm.1gpu-16vcpu-200gb` | $4.20/hr | H100 single-GPU |
| `C3-8B` | 40 GB | `gpu-h100-sxm.1gpu-16vcpu-200gb` | $3.54/hr | — |
| `C3-32B` | 141 GB | `gpu-h200-sxm.1gpu-16vcpu-200gb` | $4.20/hr | H100 single-GPU |
| `NEM-12B` | 40 GB | `gpu-h100-sxm.1gpu-16vcpu-200gb` | $3.54/hr | — |
| `QW3-2B` | 8 GB | Any GPU ≥8GB | varies | — |
| `QW3-8B` | 20 GB | `gpu-h100-sxm.1gpu-16vcpu-200gb` | $3.54/hr | — |
| `QW3-32B` | 64 GB | `gpu-h100-sxm.1gpu-16vcpu-200gb` | $3.54/hr | — |
| `C3-super` | TBD (32B) | `gpu-h200-sxm.1gpu-16vcpu-200gb` | $4.20/hr | — |

> **C3-super** is a native MODEL_SIZE key in both `byo_video_setup.py` and `gradio_cr2_byo.py`. Requires H200 SXM (141GB VRAM). VLLM_TIMEOUT is 420s for this size (32B torch.compile takes ~5 min).

**32B and C3-32B — H200 only.** H100 80GB is insufficient. If the user has an existing H100 and selects 32B: warn them and offer to provision an H200.

**H200 fallback order for 32B/C3-32B:**

| Priority | Type | Rate |
|---|---|---|
| 0 | Existing stopped H200 (from brev ls) | existing rate |
| 1 | `gpu-h200-sxm.1gpu-16vcpu-200gb` | $4.20/hr |
| 2 | `digitalocean_H200_sxm5` | $4.13/hr |

### PROVIDER FALLBACK — Auto-rotation on provisioning failure

Rotate silently (no AskUserQuestion) on UNHEALTHY after 3 min, `brev create` error, or reset failure. Emit one line per rotation: `✗ <type> failed — trying <next-type>`.

**Standard (all models except 32B/C3-32B):**

| Priority | Type | GPU | Rate |
|---|---|---|---|
| 1 | `gpu-h100-sxm.1gpu-16vcpu-200gb` | H100 SXM 80GB | $3.54/hr |
| 2 | `hyperstack_A100_80G` | A100 80GB | $1.62/hr |
| 3 | `hyperstack_H100` | H100 80GB | $2.28/hr |
| 4 | `scaleway_A40` | A40 48GB | $1.10/hr (2B LOW_VRAM only) |

If rotating to a tier >$1.50/hr more expensive than the current one: display a one-line cost warning, wait 30s (allows interruption), then proceed.

Only AskUserQuestion when all providers exhausted — present the failure summary and options.

---

---

## Supported models

| Model | Size | Min VRAM | MODEL_SIZE | Use case |
|---|---|---|---|---|
| Cosmos Reason2 BF16 | 2B VLM | 40 GB | `2B` | Video understanding: robotics, AV, Metropolis |
| Cosmos Reason2 FP8 | 2B VLM | 24 GB | `2B` | Same, quantized |
| Cosmos Reason2 BF16 | 8B VLM | 80 GB | `8B` | Higher quality video understanding |
| Cosmos Reason2 BF16 | 32B VLM | 141 GB | `32B` | Public; H200 SXM minimum (H100 80GB insufficient) |
| Cosmos3-Nano-Reasoner | 8B VLM | 40 GB | `C3-8B` | Public; was Cosmos3-Reasoner-8B-Private |
| Cosmos3-Reasoner 2B/32B | 2B/32B | 40/80+ GB | `C3-2B`, `C3-32B` | Gated HF_TOKEN; nvidia org required |
| Nemotron-Nano-12B-v2-VL BF16 | 12B VLM | 40 GB | `NEM-12B` | vLLM-only; gated; opencv backend |
| Nemotron-Nano-12B-v2-VL FP8 | 12B VLM | 24 GB | `NEM-12B` | FP8 quantized |
| Qwen3-VL-2B Instruct/FP8/Thinking | 2B VLM | 8 GB | `QW3-2B` | Public (no HF_TOKEN); vLLM-only |
| Qwen3-VL-8B Instruct/FP8/Thinking | 8B VLM | 20 GB | `QW3-8B` | Public; higher quality |
| Qwen3-VL-32B Instruct/FP8/Thinking | 32B VLM | 64+ GB | `QW3-32B` | Public; best quality |
| Cosmos Transfer2.5 | Gen | 80 GB+ | — | Video-to-video generation, sim2real |
| Cosmos Predict2 | Gen | 80 GB+ | — | World model generation |

Transfer2.5 and Predict2 are datacenter-only (H100/A100 80GB+).

### Nemotron-Nano-12B-v2-VL notes

- **Gated model** — HF_TOKEN required with nvidia org access
- **vLLM 0.14.0+** — vLLM 0.11.0 is ABI-incompatible with torch 2.9.0+cu128. `byo_video_setup.py` auto-pins to 0.14.0 on CUDA 12.8 (driver < 575). Do not pin lower.
- **opencv backend** — `VLLM_VIDEO_LOADER_BACKEND=opencv` is injected automatically. PyAV is not supported.
- **file:// video protocol** — Gradio copies video to `/tmp/gradio_upload.mp4` and sends `file:///tmp/gradio_upload.mp4` to vLLM. `--allowed-local-media-path /tmp` is required (set automatically by setup script NEM-12B config).
- **FLASHINFER bypass** — `FLASHINFER_DISABLE_VERSION_CHECK=1` is injected automatically (flashinfer package/cubin version mismatch in cosmos-reason2 venv).
- **NVFP4-QAD variant** — requires a special vLLM build; not available in standard setup. Use BF16 or FP8.
- **Launch**: `MODEL_SIZE=NEM-12B INFERENCE_BACKEND=vllm python3 /tmp/byo_video_setup.py`

### Qwen3-VL notes

- **Public model** — no HF_TOKEN required
- **vLLM only** — uses same `file://` video URL protocol as Nemotron; HF inference backend not supported
- **`--allowed-local-media-path /tmp`** required — set automatically in QW3-* configs
- **Variant picker** — Gradio checkpoint dropdown shows Instruct, FP8, and Thinking for each size
- **Thinking variant** — extended reasoning mode; max_tokens should be ≥2048 for best results
- **VRAM**: 2B~8GB, 8B~20GB, 32B~64GB (FP8 halves these)
- **Launch**: `MODEL_SIZE=QW3-8B INFERENCE_BACKEND=vllm python3 /tmp/byo_video_setup.py`

---

## VLM NIM Catalog (for `INFERENCE_BACKEND=nim_local`)

Canonical catalog: `~/.claude/scripts/nim_catalog.py` → `KNOWN_VLM_NIMS`. The agent walks BOTH `https://docs.nvidia.com/nim/vision-language-models/latest/introduction.html` AND every versioned release-notes page in `KNOWN_RELEASE_VERSIONS` to keep older containers (e.g. Cosmos Reason1 7B) discoverable when a newer release drops them from the introduction table. When a name appears upstream that is NOT in `KNOWN_VLM_NIMS`, surface it to the user as a one-line PR-this hint.

Two catalogs to keep separate:

| Catalog | URL | What it lists | Auth |
|---|---|---|---|
| **Self-host Docker images** (used by `nim_local`) | `nvcr.io/nim/<vendor>/<short-id>:latest` (entitlements at `docs.nvidia.com/nim/...`) | Containers you can `docker pull` and run on your own GPU | NGC_API_KEY (`nvapi-…`) |
| **Hosted serverless API** (separate path) | `integrate.api.nvidia.com/v1/chat/completions` (catalog at `build.nvidia.com`) | Subset of NIMs NVIDIA serves for you | NGC_API_KEY |

CR2-2B is in the Docker catalog only. CR2-8B is in both. The "[NIM] Skipped — not in public NVCF catalog" message in Gradio refers to the hosted-API catalog and does NOT mean the local container is unavailable.

### Video-capable NIMs (relevant to `/byo-video`)

| Family | Short-id | Min VRAM | Notes |
|---|---|---|---|
| Cosmos Reason2 | `cosmos-reason2-2b` | 20 GB | FP8; reasoning; temp ≥ 0.3 to avoid `<think>+EOS` bug at greedy decode. Container only — not on hosted API. |
| Cosmos Reason2 | `cosmos-reason2-8b` | 40 GB | FP8; reasoning; Efficient Video Sampling (EVS); same temp constraint as 2B. Container + hosted API. |
| Cosmos Reason2 | `cosmos-reason2-32b` | 80 GB | May require allowlisting via build.nvidia.com. |
| Cosmos Reason1 | `cosmos-reason1-7b` | 24 GB | Older release; preserved via release-notes walk. |
| Nemotron | `nemotron-3-nano-omni-30b-a3b-reasoning` | 40 GB | Specialized container; release 1.7.0+. |
| Nemotron | `nemotron-nano-12b-v2-vl` | 40 GB | Image+video; vLLM-style backend. |
| Mistral | `ministral-14b-instruct-2512` | 40 GB | Tool calling; 100k context on L40S. |
| Qwen | `qwen3.5-35b-a3b` | 40 GB | MoE; high-concurrency video constraints. |
| Qwen | `qwen3.5-122b-a10b` | 140 GB | MoE; KV cache saturation risk on long video. Multi-GPU. |
| Qwen | `qwen3.5-397b-a17b` | 400 GB | Video not enabled by default — config flag required. Multi-node. |
| Qwen | `qwen3.6-35b-a3b` | 40 GB | MoE; SGLang backend. |
| Gemma | `gemma-4-31b-it` | 40 GB | Structured output not supported. |

### Image-only NIMs (filtered OUT of `/byo-video` by `list_video_nims()`)

`mistral-medium-3.5-128b` · `mistral-small-4-119b-2603` · `mistral-small-3.2-24b-instruct-2506` · `mistral-large-3-675b-instruct-2512` · `kimi-k2.5` · `kimi-k2.6` · `qwen3.6-27b` · `llama-3.1-nemotron-nano-vl-8b-v1` · `llama-3.2-11b-vision-instruct` · `llama-3.2-90b-vision-instruct` · `llama-4-maverick-17b-128e-instruct` · `llama-4-scout-17b-16e-instruct` · `nemotron-parse-v1.2` · `nemotron-3-content-safety`

Adding a new NIM:

1. Find its row in either the latest [introduction](https://docs.nvidia.com/nim/vision-language-models/latest/introduction.html) or a [release notes](https://docs.nvidia.com/nim/vision-language-models/1.7.0/release-notes.html) page.
2. Read the model card linked from that row — note image short-id, served model ID, VRAM minimum, and whether the card mentions multi-frame video input.
3. Append a new `NimImage(...)` line to `KNOWN_VLM_NIMS` in `~/.claude/scripts/nim_catalog.py` with `supports_video=` set correctly. Append the new release version to `KNOWN_RELEASE_VERSIONS` if it's newer than the latest entry.
4. Mirror the change to `<repo>/.claude/scripts/nim_catalog.py` and commit.

Querying from the agent or runbook:

```bash
python3 ~/.claude/scripts/nim_catalog.py upstream      # raw upstream model-name list
python3 ~/.claude/scripts/nim_catalog.py list --no-probe  # full known catalog as JSON
python3 -c "from nim_catalog import list_video_nims; \
            [print(n.short_id, n.min_vram_mb) for n in list_video_nims()]"
```

---

## VRAM auto-selection

The setup script (`byo_video_setup.py`) handles all of this automatically. Rules as of 2026-04-21:

**Model selection** — always CR2-2B for the live demo (8B fails the <60s inference target on non-H100 GPUs). Force 8B via `MODEL_NAME=nvidia/Cosmos-Reason2-8B` env var if quality > speed.
- ≥ 40000 MiB free → `nvidia/Cosmos-Reason2-2B`
- < 40000 MiB → CR2-2B with LOW_VRAM mode (fps=1, reduced resolution)

**FPS/pixel tier** — detected by GPU name (not free VRAM), because workstation GPUs like RTX PRO 6000 have large VRAM but slower prefill compute:
| Tier | Condition | fps | max_pixels | Expected inference |
|---|---|---|---|---|
| H100/A100 | GPU name contains H100, A100, H200, GB200 | 2 | 1,048,576 | ~44s (validated H100) |
| High-VRAM (non-H100) | ≥40GB free, non-H100 (RTX PRO, A40, A30, etc.) | 1 | 524,288 | ~54s (validated RTX PRO 6000 97GB) |
| Low-VRAM | <40GB free | 1 | 131,072 | ~80-120s |
| Ultra-Low-VRAM | <8GB free (RTX 5070, GTX 3090, etc.) | 1 | 65,536 | ~120-180s; may OOM on videos >5s at 720p — pre-resize to 360p |

**Pre-kill**: setup script kills any process on port 7860 BEFORE measuring VRAM, so stale processes don't skew the tier selection.

Transfer2.5 / Predict2: abort if < 80000. Do not proceed.

---

## Environment detection (LOCAL FIRST)

Check in this order:

1. **Local GPU**: `nvidia-smi` succeeds → local path, no cloud needed
2. **Brev**: `brev ls` succeeds → Brev path
3. **Nebius**: `NEBIUS_ENDPOINT` is set → Nebius path
4. **None**: ask "Which environment?"

---

## Primary flow — Brev web demo (most common)

### Step 1 — Create or reuse instance

```bash
# Create new instance — use model-aware GPU type (see MODEL_GPU_REQUIREMENTS)
# For 32B: brev create <name> --gpu-name H200 --type gpu-h200-sxm.1gpu-16vcpu-200gb
# For ≤8B: brev create <name> --gpu-name H100 --type gpu-h100-sxm.1gpu-16vcpu-200gb

# Or list existing
brev ls
```

Wait for STATUS: RUNNING. Then get the public IP:
```bash
brev exec <name> "curl -s ifconfig.me"
```

### Step 2 — Bootstrap

Run once per fresh instance. All commands via `brev exec <name> "<cmd>"`.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone cosmos-reason2 (provides model code + sample video)
git clone https://github.com/nvidia-cosmos/cosmos-reason2.git ~/cosmos-reason2

# Install dependencies
cd ~/cosmos-reason2 && export PATH=~/.local/bin:~/.cargo/bin:$PATH && uv sync --extra cu128

# Install PyAV (required on Hyperstack — FFmpeg not in PATH)
uv pip install "av==16.1.0"

# Download model weights (CR2-2B ~8GB, first-run only, ~5-10 min on cloud bandwidth)
export HF_TOKEN=hf_...
uv run huggingface-cli download nvidia/Cosmos-Reason2-2B \
  --local-dir ~/cosmos-reason2/models/Cosmos-Reason2-2B
```

### Step 3 — Deploy scripts and launch

Two scripts must be present on the instance:
- `/tmp/byo_video_setup.py` — shows live progress + ETAs, launches Gradio, prints clickable URL
- `/tmp/gradio_cr2_byo.py` — the Gradio app itself (called by setup script)

Both live at `~/.claude/scripts/` on your local machine (canonical, versioned). Deploy via base64:

```bash
# Deploy both scripts to the instance
for script in byo_video_setup gradio_cr2_byo; do
  B64=$(base64 -i ~/.claude/scripts/${script}.py | tr -d '\n')
  brev exec <name> "python3 -c \"import base64; open('/tmp/${script}.py','wb').write(base64.b64decode('${B64}'))\""
done
```

Then run setup (streams live progress to this terminal).
Note: pass env vars as separate exports in the remote quoted command — HF_TOKEN is auto-read by the script from `~/.cache/huggingface/token` on the instance (do not pass it manually):
```bash
brev exec <name> "export INFERENCE_BACKEND=vllm MODEL_ID=nvidia/Cosmos-Reason2-2B BREV_RATE_PER_HOUR=3.70 PATH=~/.local/bin:~/.cargo/bin:$PATH && python3 /tmp/byo_video_setup.py"
```

The script will:
1. Print a 9-step setup dashboard and begin executing
2. Detect GPU + VRAM tier (H100 / High-VRAM / Low-VRAM / Ultra-Low-VRAM)
3. Validate HF token (whoami check — fails fast if expired)
4. Install any missing deps (uv, repos, PyAV, Gradio)
5. Download model weights with retry logic
6. Launch Gradio and print a **clickable hyperlink** to the public `gradio.live` URL
7. Report elapsed time and credits spent throughout

### Step 4 — The URL appears at the end

Example output:
```
──────────────────────────────────────────────────────────────
  Cosmos Reason2 Demo — Ready
──────────────────────────────────────────────────────────────
  URL:  https://xxxxxxxxxxxx.gradio.live
  Upload any MP4 → type a prompt → click Run Inference
  Link valid for 72h. Kill instance when done.
──────────────────────────────────────────────────────────────
```

The URL is an OSC 8 hyperlink — click it directly in iTerm2 or Terminal.app (macOS). Valid for 72 hours.

### Step 5 — Kill instance when done

Delete the instance from the Brev dashboard or run `brev delete <name>`. Do NOT auto-terminate.

---

## Local web demo (no cloud billing)

Same Gradio app, runs on the user's own GPU machine.

Prerequisites:
```bash
export HF_TOKEN=hf_...
# Check VRAM — must be ≥40GB for CR2-2B
nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1
```

Bootstrap (once):
```bash
git clone https://github.com/nvidia-cosmos/cosmos-reason2.git ~/cosmos-reason2
cd ~/cosmos-reason2
uv sync --extra cu128
uv pip install "av==16.1.0" gradio
uv run huggingface-cli download nvidia/Cosmos-Reason2-2B \
  --local-dir ~/cosmos-reason2/models/Cosmos-Reason2-2B
```

Launch:
```bash
export MODEL_DIR=~/cosmos-reason2/models/Cosmos-Reason2-2B
export MODEL_NAME=nvidia/Cosmos-Reason2-2B
cd ~/cosmos-reason2
uv run python /tmp/gradio_cr2_byo.py
```

Open `http://localhost:7860` in any browser. No kill alert needed — local machine.

**Claude auth on local:** If running via Claude Code on a headless SSH machine, check auth first:
```bash
claude auth status
```
If `loggedIn: false`: `claude auth login --console` (API/console.anthropic.com users) or `claude auth login` (Claude.ai). On SSH — a URL prints; open it in your local browser.

---

## Headless inference (programmatic / no browser)

For CI or batch use where no browser is needed. Results go to JSON only.

```bash
export HF_TOKEN=hf_...
export BYO_VIDEO=/path/to/video.mp4
export MODEL_DIR=~/cosmos-reason2/models/Cosmos-Reason2-2B
export MODEL_NAME=nvidia/Cosmos-Reason2-2B
export OUT_FILE=/tmp/byo_video_reason2_results.json
export PROVIDER=brev  # or local
cd ~/cosmos-reason2
uv run python /tmp/smoke_cr2_byo.py
```

Results at `/tmp/byo_video_reason2_results.json`.

---

## Nebius (OpenAI-compatible API endpoint)

Nebius runs Reason2 as a vLLM serving endpoint — no SSH, no Gradio needed. Useful for API integration testing, not for interactive browser demos.

```python
from openai import OpenAI
client = OpenAI(base_url="https://<instance>.nebius.ai/v1", api_key="<nebius-key>")
response = client.chat.completions.create(
    model="nvidia/Cosmos-Reason2-2B",
    messages=[{"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,<b64>"}},
        {"type": "text", "text": "Describe what is happening in this video."}
    ]}]
)
```

Write result to `/tmp/byo_video_reason2_results.json`. Delete the endpoint when done.

---

## Gradio app script

Canonical source: **`~/.claude/scripts/gradio_cr2_byo.py`** (versioned 2026-04-21).

Features (as of 2026-04-21):
- **Checkpoint selector** — Advanced Settings accordion has a preset dropdown (CR2-2B base, CR2-2B FP8, CR2-8B NVFP4, NIM 2B) plus a custom model ID field. Any HF model ID or local path is accepted.
- **On-demand load/unload** — switching checkpoints does `del model → gc.collect() → cuda.empty_cache()` before loading the next variant. VRAM is confirmed free before loading.
- **Run All Variants button** — sequential benchmark: FP8 → NVFP4 → NIM, each unloaded before the next. Results saved to `/tmp/byo_video_benchmark.json`.
- **Right-side status panel** — replaces the grey loading box. Shows Step N/5 WIP tracker (resolve / load / preprocess / prefill / generate) with ✅/⟳/— per step, plus live token metrics: prefill count, generated count (running), TTFT, inference time.
- **NIM mode** — calls NVCF API (`https://integrate.api.nvidia.com/v1/chat/completions`) via NGC_API_KEY (nvapi- prefix). Extracts up to 8 JPEG frames from the video and sends them as image content. No local model load needed.
- LOW_VRAM mode, PyAV backend, qwen_vl_utils pipeline, auto-cap, results JSON — all retained from 2026-04-20.

Deploy to instances via base64 as shown in the deploy section.

**Do not embed the source here.** The canonical file is the source of truth.

### Checkpoint selector usage

| Method | How |
|---|---|
| Preset (base, FP8, NVFP4, NIM) | Advanced Settings → Checkpoint dropdown |
| Custom HF ID | Advanced Settings → Custom Checkpoint ID field (overrides dropdown) |
| Custom local path | Same custom field — accepts `/path/to/model` |
| Env var (headless) | `export CR2_CHECKPOINT=nvidia/Cosmos-Reason2-2B-FP8` before setup |

For multi-variant benchmarking without the UI, click **Run All Variants** — runs FP8 → NVFP4 → NIM sequentially.

### NIM mode requirements

| Item | Value |
|---|---|
| NGC_API_KEY | `nvapi-...` prefix (set in env before setup, passed to Gradio) |
| Model identifier | `nvidia/cosmos-reason2-2b` (NVCF catalog) |
| Video input | Up to 8 JPEG frames extracted by PyAV at selected fps |
| Auth header | `Authorization: Bearer $NGC_API_KEY` |
| Output | Streamed via SSE, same JSON results format as HF path |

Set `NGC_API_KEY` before running `byo_video_setup.py` — it passes it through to the Gradio process env.

### Setup script — multi-variant mode

Set `MULTI_VARIANT=true` to also pre-download FP8 and NVFP4 model weights during setup:

```bash
export HF_TOKEN=hf_...
export NGC_API_KEY=nvapi-...
export MULTI_VARIANT=true
python3 /tmp/byo_video_setup.py
```

Without `MULTI_VARIANT=true`, only the base model downloads. FP8/NVFP4 will download from HF on first use in Gradio (with HF_TOKEN).

### Env vars (single-model mode)

| Var | Default | Notes |
|---|---|---|
| `MODEL_NAME` | `nvidia/Cosmos-Reason2-2B` | HF model ID to load at startup |
| `MODEL_DIR` | `~/cosmos-reason2/models/Cosmos-Reason2-2B` | Local path |
| `HF_TOKEN` | — | Required for gated model download |
| `NGC_API_KEY` | — | Required for NIM mode (`nvapi-` prefix) |
| `MULTI_VARIANT` | false | `true` = also download FP8 + NVFP4 during setup |
| `GRADIO_PORT` | 7860 | Gradio server port |
| `GRADIO_SHARE` | true | Set `false` to disable public link |
| `GRADIO_FPS` | tier-based | Passed by setup script |
| `GRADIO_MAX_PIXELS` | tier-based | Passed by setup script |
| `GRADIO_PREFILL_TPS` | tier-based | Passed by setup script |
| `LOW_VRAM` | auto | Set `true` to force low-VRAM mode |
| `OUT_FILE` | `/tmp/byo_video_reason2_results.json` | Per-run results JSON path |

---

## Cosmos cookbook structure (as of 2026-04-17)

The cookbook repo restructured. **`deploy/` directory no longer exists.** Old shell script paths are invalid.

| What | New location |
|---|---|
| Brev Reason2 setup script | `docs/getting_started/brev/reason2/setup_script.sh` |
| Worker safety recipe (Python) | `docs/recipes/inference/reason2/worker_safety/worker_safety.py` |
| Transfer2.5 real augmentation | `docs/recipes/inference/transfer2_5/inference-real-augmentation/inference.md` |
| Predict2 ITS | `docs/recipes/inference/predict2/inference-its/inference.md` |

For BYO-video inference, **do not use cookbook scripts** — use `gradio_cr2_byo.py` (web demo) or `smoke_cr2_byo.py` (headless) directly against the `cosmos-reason2` repo environment.

---

## PyAV backend patch (always apply)

Required on Hyperstack and similar cloud images — FFmpeg is not in system PATH so torchcodec fails. Already embedded in both `gradio_cr2_byo.py` and `smoke_cr2_byo.py`.

If writing a new inference script, prepend this before loading the processor:

```python
from transformers import video_processing_utils
from transformers.video_utils import load_video as _load_video

def _patched_fetch_videos(self, video_url_or_urls, sample_indices_fn=None):
    if isinstance(video_url_or_urls, list):
        return list(zip(*[
            _patched_fetch_videos(self, x, sample_indices_fn=sample_indices_fn)
            for x in video_url_or_urls
        ]))
    return _load_video(video_url_or_urls, backend="pyav", sample_indices_fn=sample_indices_fn)

video_processing_utils.BaseVideoProcessor.fetch_videos = _patched_fetch_videos
```

---

## Timing benchmarks

| Environment | GPU | Model | Tier | Load | Inference | Pass <60s? |
|---|---|---|---|---|---|---|
| Brev Hyperstack | H100 PCIe | CR2-2B | fps=2, 1M px | 1.7s | 44.1s | ✅ |
| Brev Hyperstack | H100 PCIe | CR2-2B | fps=2, 1M px | 10.6s | 42.9s | ✅ |
| Brev H100 | H100 | CR2-2B-FP8 | fps=8, 1M px | TBD | TBD | ⏳ pending smoke gate |
| Brev H100 | H100 | CR2-8B-NVFP4 | fps=8, 1M px | TBD | TBD | ⏳ pending smoke gate |
| Brev H100 | H100 | NIM 2B (NVCF) | 8 frames | N/A | TBD | ⏳ pending smoke gate |

Notes:
- "Inference" = preprocess + prefill + decode (TTFT-dominated for video inference)
- CR2-8B fails the <60s target on all non-H100 GPUs tested; use CR2-2B for demos
- FP8/NVFP4/NIM timing benchmarks TBD — update after smoke gate

---

## Dynamic Reconfiguration — Switching Model Class at Runtime

**When invoked with `<instance> <model-class>` arguments**, the skill changes the active model without touching the static launch scripts.

Usage:
```
/byo-video <instance-name> 8B
/byo-video <instance-name> 2B
/byo-video <user@host> 8B
```

### Models on disk (vLLM instance)

| Size | Local path | HF ID |
|---|---|---|
| 2B FP8 | `~/cosmos-reason2/models/Cosmos-Reason2-2B-FP8` | `nvidia/Cosmos-Reason2-2B-FP8` |
| 2B BF16 | `~/cosmos-reason2/models/Cosmos-Reason2-2B` | `nvidia/Cosmos-Reason2-2B` |
| 8B NVFP4 | `~/cosmos-reason2/models/Cosmos-Reason2-8B-NVFP4` | `nvidia/Cosmos-Reason2-8B-NVFP4` |
| 8B FP8 | `~/cosmos-reason2/models/Cosmos-Reason2-8B-FP8` | `nvidia/Cosmos-Reason2-8B-FP8` |
| 8B BF16 | `~/cosmos-reason2/models/Cosmos-Reason2-8B` | `nvidia/Cosmos-Reason2-8B` |
| 32B BF16 | public (May 2026) — no HF_TOKEN, ~66 GB | `nvidia/Cosmos-Reason2-32B` |
| 32B AV | public (May 2026) — no HF_TOKEN, ~66 GB | `nvidia/Cosmos-Reason2-32B-AV` |

### Reconfiguration procedure (agent runs all steps)

**Step 1 — Kill Gradio:**
```bash
brev exec <instance> "pkill -f gradio_cr2_byo"
```
Then wait 3s, verify port is free:
```bash
brev exec <instance> "ss -tlnp | grep 7860"
```

**Step 2 — Kill vLLM server:**
```bash
brev exec <instance> "pkill -f 'vllm serve'"
```
Wait 5s for GPU memory to release:
```bash
brev exec <instance> "nvidia-smi --query-gpu=memory.used --format=csv,noheader"
```

**Step 3 — Deploy updated gradio script (if changed):**
```bash
B64=$(base64 -i ~/.claude/scripts/gradio_cr2_byo.py | tr -d '\n')
brev exec <instance> "python3 -c \"import base64; open('/tmp/gradio_cr2_byo.py','wb').write(base64.b64decode('${B64}'))\""
```

**Step 4 — Start vLLM with target model:**

For 2B FP8 (vLLM, real FP8 kernels):
```bash
brev exec <instance> "nohup bash -c 'source ~/.ngc_env && cd ~/cosmos-reason2 && .venv/bin/vllm serve models/Cosmos-Reason2-2B-FP8 --served-model-name nvidia/Cosmos-Reason2-2B-FP8 --port 8000 --dtype auto --trust-remote-code --max-model-len 8192 --gpu-memory-utilization 0.90 > /tmp/vllm_server.log 2>&1 &'"
```

For 8B NVFP4 (current default):
```bash
brev exec <instance> "nohup bash -c 'source ~/.ngc_env && cd ~/cosmos-reason2 && .venv/bin/vllm serve models/Cosmos-Reason2-8B-NVFP4 --served-model-name nvidia/Cosmos-Reason2-8B-NVFP4 --port 8000 --dtype auto --trust-remote-code --max-model-len 8192 --gpu-memory-utilization 0.85 > /tmp/vllm_server.log 2>&1 &'"
```

For 8B FP8:
```bash
brev exec <instance> "nohup bash -c 'source ~/.ngc_env && cd ~/cosmos-reason2 && .venv/bin/vllm serve models/Cosmos-Reason2-8B-FP8 --served-model-name nvidia/Cosmos-Reason2-8B-FP8 --port 8000 --dtype auto --trust-remote-code --max-model-len 8192 --gpu-memory-utilization 0.85 > /tmp/vllm_server.log 2>&1 &'"
```

For 32B BF16 (needs smaller max-model-len to fit 80GB — download first):
```bash
brev exec <instance> "nohup bash -c 'source ~/.ngc_env && cd ~/cosmos-reason2 && .venv/bin/vllm serve models/Cosmos-Reason2-32B --served-model-name nvidia/Cosmos-Reason2-32B --port 8000 --dtype bfloat16 --trust-remote-code --max-model-len 4096 --gpu-memory-utilization 0.95 > /tmp/vllm_server.log 2>&1 &'"
```

**Step 5 — Wait for vLLM ready (poll /v1/models, up to 120s):**
```bash
brev exec <instance> "for i in \$(seq 1 24); do curl -sf http://localhost:8000/v1/models && break || sleep 5; done"
```

**Step 6 — Start Gradio with correct MODEL_SIZE:**

Do NOT edit `launch_gradio_vllm.sh`. Run inline:
```bash
brev exec <instance> "nohup bash -c 'source ~/.ngc_env && cd ~/cosmos-reason2 && INFERENCE_BACKEND=vllm VLLM_BASE_URL=http://localhost:8000/v1 MODEL_SIZE=<SIZE> .venv/bin/python /tmp/gradio_cr2_byo.py > /tmp/gradio_demo.log 2>&1 &'"
```

Replace `<SIZE>` with `2B`, `8B`, or `32B`.

**Step 7 — Get URL:**
```bash
brev exec <instance> "sleep 20 && cat /tmp/gradio_url.txt"
```

### Downloading missing models

If the target model isn't on disk yet, download before starting vLLM:
```bash
brev exec <instance> "cd ~/cosmos-reason2 && .venv/bin/huggingface-cli download nvidia/Cosmos-Reason2-8B-FP8 --local-dir models/Cosmos-Reason2-8B-FP8"
```
HF_TOKEN required for gated models. `Cosmos-Reason2-8B-FP8` is public.

### 32B feasibility notes

- **For /byo-video: H200 is the minimum viable GPU.** H100 single-GPU (80GB) is insufficient for comfortable inference under the skill's time constraints. Use `gpu-h200-sxm.1gpu-16vcpu-200gb` (141GB VRAM) — empirically confirmed with c3r-32b.
- 32B BF16 weights are ~66GB, but vLLM requires additional VRAM for KV cache, activations, and overhead. On H100 80GB you can technically load with `--max-model-len 4096 --gpu-memory-utilization 0.95`, but this leaves almost no room for KV cache and will OOM on longer video sequences.
- CR2-32B is **public** as of May 2026 — no HF_TOKEN required. Download: `huggingface-cli download nvidia/Cosmos-Reason2-32B --local-dir models/Cosmos-Reason2-32B`.
- In vLLM mode with 8B loaded, `run_all_variants` will send 32B requests to the 8B server — the table `Notes` column will show `vLLM serves Cosmos-Reason2-8B-NVFP4` to flag the mismatch.
- For true 32B benchmarking: restart vLLM with 32B model using this skill, then run `Run All Variants` with `MODEL_SIZE=32B`.

---

## Common failure modes

| Symptom | Fix |
|---|---|
| Port 7860 unreachable | Check `ss -tlnp | grep 7860` on instance. If not listening, Gradio failed to start — check `/tmp/gradio_demo.log`. |
| Video upload fails in browser | Gradio temp dir issue. Try with sample.mp4 via Examples button first. |
| Black frames / torchcodec error | PyAV patch not applied. Confirm `gradio_cr2_byo.py` was deployed (not a custom script). |
| OOM during inference | VRAM too low. Kill other GPU processes first. CR2-2B needs ~10GB GPU RAM peak. |
| `uv sync --extra cu128` fails | Wrong CUDA driver. Check `nvidia-smi` shows CUDA 12.x driver. |
| `claude auth status` → loggedIn: false | `claude auth login --console` (API users). On SSH: copy URL, open in local browser. |
| Wrong VRAM tier selected (old Gradio using memory) | Setup script now kills port 7860 BEFORE measuring VRAM. Re-run setup to get clean tier. |
| Inference >60s on high-VRAM workstation GPU | GPU name doesn't match H100/A100/H200/GB200 → fps=1 tier applies. If inference still slow, check that `GRADIO_FPS=1` is in the Gradio process env. |
| HF 429 rate limit on download | Script retries 5× with 30s sleep. Common on shared cloud IPs. Usually succeeds by attempt 3-4. |
| `brev login` fails with EOF | `brev login` requires a browser handoff — it cannot run in a non-TTY context. Open a separate terminal tab, run `brev login` there, complete the browser prompt, then return. |
| vLLM Connection refused on first inference | `byo_video_setup.py` now auto-starts vLLM before Gradio (Step 9b). If running the Gradio script manually, start vLLM first: `nohup .venv/bin/vllm serve <model_dir> --port 8000 ... &` then poll `curl localhost:8000/v1/models`. |
| Nemotron: `no module named 'mamba_ssm'` or `selective_scan_cuda` | vLLM PyPI build doesn't include mamba-ssm. Use vLLM nightly Docker: `vllm/vllm-openai:nightly-8bff831f0aa239006f34b721e63e1340e3472067` or `nvcr.io/nvidia/vllm:25.12.post1-py3`. |
| Nemotron: `video_url not supported` or `unsupported content type` | vLLM version doesn't support `video_url` message type. Requires vLLM nightly; PyPI ≤0.11.0 unsupported. |
| Nemotron: 400 error from vLLM on inference | Check that `--allowed-local-media-path /tmp` is in the vLLM serve command (set automatically by `byo_video_setup.py`). |
