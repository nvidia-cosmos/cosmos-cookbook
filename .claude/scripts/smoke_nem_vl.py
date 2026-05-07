#!/usr/bin/env python3
"""
Smoke test for Nemotron/Qwen3-VL vLLM endpoint.

Usage:
    python3 smoke_nem_vl.py                          # default: localhost:8000
    VLLM_BASE_URL=http://host:8000/v1 python3 smoke_nem_vl.py
    TEST_VIDEO=/path/to/video.mp4 python3 smoke_nem_vl.py

Exit codes:
    0 = PASS
    1 = FAIL (model error, empty response, or HTTP error)
    2 = SKIP (endpoint not reachable)
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
import shutil
import tempfile

VLLM_URL  = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
TIMEOUT_S = int(os.environ.get("SMOKE_TIMEOUT_S", "120"))

VIDEO_PATH = os.environ.get("TEST_VIDEO", "/tmp/smoke_test_video.mp4")

# Small public-domain MP4 (Big Buck Bunny 5-second clip, ~120 KB)
_SAMPLE_URL = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"

def _download_video():
    if os.path.exists(VIDEO_PATH) and os.path.getsize(VIDEO_PATH) > 1000:
        return True
    print(f"  Downloading test video → {VIDEO_PATH} ...", end=" ", flush=True)
    try:
        with urllib.request.urlopen(_SAMPLE_URL, timeout=30) as resp:
            with open(VIDEO_PATH, "wb") as f:
                f.write(resp.read(5 * 1024 * 1024))  # cap at 5 MB
        print("done")
        return True
    except Exception as e:
        print(f"FAILED ({e})")
        return False


def _get_serving_model():
    try:
        req = urllib.request.Request(f"{VLLM_URL}/models")
        with urllib.request.urlopen(req, timeout=10) as r:
            data = json.loads(r.read())
            ids = [m["id"] for m in data.get("data", [])]
            return ids[0] if ids else None
    except Exception:
        return None


def _check_reachable():
    try:
        urllib.request.urlopen(f"{VLLM_URL}/models", timeout=5)
        return True
    except Exception:
        return False


def run_smoke():
    print("=" * 60)
    print("  Nemotron/Qwen3-VL vLLM Smoke Test")
    print(f"  Endpoint: {VLLM_URL}")
    print("=" * 60)

    # 1. Reachability
    print("\n[1/4] Checking endpoint reachability ...", end=" ", flush=True)
    if not _check_reachable():
        print("UNREACHABLE")
        print(f"\n  SKIP — {VLLM_URL} is not responding. Is vLLM running?")
        sys.exit(2)
    print("OK")

    # 2. Discover model ID
    print("[2/4] Discovering serving model ...", end=" ", flush=True)
    model_id = _get_serving_model()
    if not model_id:
        print("FAIL — /v1/models returned no models")
        sys.exit(1)
    print(f"OK ({model_id})")

    # 3. Video
    print("[3/4] Preparing test video ...", end=" ", flush=True)
    if not _download_video():
        print("FAIL — could not obtain test video")
        sys.exit(1)
    print(f"OK ({VIDEO_PATH}, {os.path.getsize(VIDEO_PATH)//1024} KB)")

    # 4. Inference
    file_url = f"file://{VIDEO_PATH}"
    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video_url", "video_url": {"url": file_url}},
                    {"type": "text", "text": "In one sentence, what is happening in this video?"},
                ],
            }
        ],
        "max_tokens": 128,
        "temperature": 0.0,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{VLLM_URL}/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    print(f"[4/4] Running inference (timeout {TIMEOUT_S}s) ...")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            elapsed = time.time() - t0
            raw = r.read()
            data = json.loads(raw)
    except urllib.error.HTTPError as e:
        elapsed = time.time() - t0
        err_body = e.read().decode(errors="replace")
        print(f"\n  FAIL — HTTP {e.code} after {elapsed:.1f}s")
        print(f"  Error body: {err_body[:500]}")
        sys.exit(1)
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  FAIL — {type(e).__name__}: {e} (after {elapsed:.1f}s)")
        sys.exit(1)

    # Validate response
    choices = data.get("choices", [])
    if not choices:
        print(f"\n  FAIL — No choices in response: {json.dumps(data)[:300]}")
        sys.exit(1)

    text = choices[0].get("message", {}).get("content", "").strip()
    if not text:
        print(f"\n  FAIL — Empty content in response: {json.dumps(data)[:300]}")
        sys.exit(1)

    # Check for vLLM error markers in the content
    err_markers = ["[VLLM ERROR]", "500 Internal", "400 Client Error", "RuntimeError"]
    for marker in err_markers:
        if marker in text:
            print(f"\n  FAIL — Error marker '{marker}' found in response text")
            print(f"  Content: {text[:300]}")
            sys.exit(1)

    usage = data.get("usage", {})
    print(f"\n  PASS in {elapsed:.1f}s")
    print(f"  Model:   {model_id}")
    print(f"  Tokens:  {usage.get('prompt_tokens', '?')} in / {usage.get('completion_tokens', '?')} out")
    print(f"  Content: {text[:200]}")

    # Write results
    result = {
        "status": "PASS",
        "model": model_id,
        "elapsed_s": round(elapsed, 2),
        "content": text,
        "usage": usage,
    }
    out_path = "/tmp/smoke_nem_vl_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Results → {out_path}")
    print("=" * 60)
    sys.exit(0)


if __name__ == "__main__":
    run_smoke()
