#!/usr/bin/env python3
"""
nim_catalog.py — VLM NIM catalog helper for /byo-video.

Single source of truth: the agents always re-check
  https://docs.nvidia.com/nim/vision-language-models/latest/introduction.html
PLUS every versioned release-notes page in KNOWN_RELEASE_VERSIONS, so the
catalog reflects what NVIDIA ships AND what older NIM containers are still
pullable from nvcr.io. Walking release notes is required because
introduction.html only lists models in the CURRENT release; older models
(e.g. Cosmos Reason 1 7B from <1.5) drop off its table even though their
images still work.

How it works:
  1. fetch_supported_models() — GETs introduction.html and every
     release-notes page; merges the model-name columns. Per-page parse is
     fault-tolerant; if any URL fails, the others still contribute.
     No nvcr.io paths or served model IDs are exposed at this level
     (those live in per-model cards), so KNOWN_VLM_NIMS below maps display
     names to image short-IDs and per-NIM metadata (VRAM, video support).
  2. KNOWN_VLM_NIMS — the slug map. Update when a new VLM family is
     added upstream (the agents are told to do this when fetch finds a
     name not in the map). Each entry carries supports_video — set False
     for image-only NIMs so /byo-video filters them out of its dropdown.
  3. list_available_nims(ngc_api_key, vram_mb=None) — intersects the
     upstream catalog with the slug map, then probes each candidate
     image via `docker manifest inspect` to confirm the user's
     NGC_API_KEY actually has access. Optionally filters by min VRAM.
  4. list_video_nims() — quick filter for /byo-video population.
  5. CLI: `python3 nim_catalog.py list [--vram MB]` → prints JSON.

Network failures fall back to the slug map alone (no upstream filter).
"""
from __future__ import annotations

import argparse
import html
import json
import os
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field, asdict
from typing import List, Optional

DOCS_URL = "https://docs.nvidia.com/nim/vision-language-models/latest/introduction.html"

# Versioned release-notes pages. introduction.html only lists models in the
# CURRENT release; older models (e.g. Cosmos Reason 1 7B from <1.5) drop off
# its table even though their containers are still pullable from nvcr.io.
# Fetching every known release-notes page and unioning keeps older Cosmos +
# Llama variants discoverable. Add new versions here as NVIDIA ships them.
RELEASE_NOTES_URL_FMT = "https://docs.nvidia.com/nim/vision-language-models/{version}/release-notes.html"
KNOWN_RELEASE_VERSIONS = [
    "1.0.0", "1.1.0", "1.2.0", "1.3.0", "1.4.0",
    "1.5.0", "1.6.0", "1.7.0",
]


@dataclass
class NimImage:
    short_id: str           # e.g. "cosmos-reason2-8b"
    image: str              # e.g. "nvcr.io/nim/nvidia/cosmos-reason2-8b:latest"
    family: str             # e.g. "Cosmos Reason2"
    label: str              # human-readable label for the dropdown
    served_model_id: str    # e.g. "nvidia/cosmos-reason2-8b" (best-effort)
    min_vram_mb: int = 0    # 0 if unknown
    supports_video: bool = True   # True = video frames; False = single image only
    notes: str = ""


# Slug + min-VRAM map — translates display names from introduction.html
# into actual nvcr.io short-IDs. Conservative VRAM minimums; refine per
# model card. Add new entries here when the upstream catalog grows.
#
# supports_video: which models accept multi-frame video input vs single
# images only. Sourced from each model card on
# https://docs.nvidia.com/nim/vision-language-models/1.7.0/release-notes.html
# (cross-referenced against build.nvidia.com per-model docs). When in
# doubt, set False; agents can always upgrade after reading the card.
#
KNOWN_VLM_NIMS = [
    # ── Cosmos Reason2 family (NVIDIA, video-native) ─────────────────────
    NimImage("cosmos-reason2-2b",  "nvcr.io/nim/nvidia/cosmos-reason2-2b:latest",
             "Cosmos Reason2", "Cosmos Reason2 2B (NIM)",
             "nvidia/cosmos-reason2-2b", min_vram_mb=20000, supports_video=True,
             notes="FP8 quantized; reasoning VLM; speculative decoding; "
                   "needs temperature ≥ 0.3 to avoid <think>+EOS bug at greedy decode. "
                   "Container only — NOT on build.nvidia.com hosted API."),
    NimImage("cosmos-reason2-8b",  "nvcr.io/nim/nvidia/cosmos-reason2-8b:latest",
             "Cosmos Reason2", "Cosmos Reason2 8B (NIM)",
             "nvidia/cosmos-reason2-8b", min_vram_mb=40000, supports_video=True,
             notes="FP8 quantized; reasoning VLM; Efficient Video Sampling (EVS); "
                   "needs temperature ≥ 0.3 to avoid <think>+EOS bug at greedy decode. "
                   "Container + build.nvidia.com hosted API."),
    NimImage("cosmos-reason2-32b", "nvcr.io/nim/nvidia/cosmos-reason2-32b:latest",
             "Cosmos Reason2", "Cosmos Reason2 32B (NIM)",
             "nvidia/cosmos-reason2-32b", min_vram_mb=80000, supports_video=True,
             notes="May require allowlisting via build.nvidia.com (not in default NGC catalog)."),
    NimImage("cosmos-reason1-7b",  "nvcr.io/nim/nvidia/cosmos-reason1-7b:latest",
             "Cosmos Reason1 7B", "Cosmos Reason1 7B (NIM)",
             "nvidia/cosmos-reason1-7b", min_vram_mb=24000, supports_video=True),

    # ── Nemotron family (NVIDIA) ─────────────────────────────────────────
    NimImage("nemotron-3-nano-omni-30b-a3b-reasoning",
             "nvcr.io/nim/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:latest",
             "NVIDIA Nemotron 3 Nano Omni",
             "Nemotron 3 Nano Omni 30B (NIM)",
             "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
             min_vram_mb=40000, supports_video=True,
             notes="Specialized container variant; release 1.7.0+."),
    NimImage("nemotron-nano-12b-v2-vl",
             "nvcr.io/nim/nvidia/nemotron-nano-12b-v2-vl:latest",
             "Nemotron Nano 12B v2 VL", "Nemotron Nano 12B v2 VL (NIM)",
             "nvidia/nemotron-nano-12b-v2-vl",
             min_vram_mb=40000, supports_video=True),
    NimImage("nemotron-parse-v1.2",
             "nvcr.io/nim/nvidia/nemotron-parse-v1.2:latest",
             "Nemotron-Parse-v1.2", "Nemotron Parse v1.2 (NIM)",
             "nvidia/nemotron-parse-v1.2", min_vram_mb=24000,
             supports_video=False,
             notes="Image parsing only; document-extraction VLM. Not for /byo-video."),
    NimImage("nemotron-3-content-safety",
             "nvcr.io/nim/nvidia/nemotron-3-content-safety:latest",
             "Nemotron-3-Content-Safety", "Nemotron 3 Content Safety (NIM)",
             "nvidia/nemotron-3-content-safety", min_vram_mb=24000,
             supports_video=False,
             notes="Safety classifier; not a generative VLM."),

    # ── Llama family (Meta) ──────────────────────────────────────────────
    NimImage("llama-3.1-nemotron-nano-vl-8b-v1",
             "nvcr.io/nim/nvidia/llama-3.1-nemotron-nano-vl-8b-v1:latest",
             "Llama 3.1 Nemotron Nano VL 8B v1",
             "Llama 3.1 Nemotron Nano VL 8B (NIM)",
             "nvidia/llama-3.1-nemotron-nano-vl-8b-v1",
             min_vram_mb=40000, supports_video=False,
             notes="Image input only; no multi-frame video."),
    NimImage("llama-3.2-11b-vision-instruct",
             "nvcr.io/nim/meta/llama-3.2-11b-vision-instruct:latest",
             "Llama 3.2 11B Vision Instruct",
             "Llama 3.2 11B Vision Instruct (NIM)",
             "meta/llama-3.2-11b-vision-instruct",
             min_vram_mb=40000, supports_video=False,
             notes="Image input only."),
    NimImage("llama-3.2-90b-vision-instruct",
             "nvcr.io/nim/meta/llama-3.2-90b-vision-instruct:latest",
             "Llama 3.2 90B Vision Instruct",
             "Llama 3.2 90B Vision Instruct (NIM)",
             "meta/llama-3.2-90b-vision-instruct",
             min_vram_mb=160000, supports_video=False,
             notes="Image input only. Multi-GPU only (~160GB+ VRAM)."),
    NimImage("llama-4-maverick-17b-128e-instruct",
             "nvcr.io/nim/meta/llama-4-maverick-17b-128e-instruct:latest",
             "Llama 4 Maverick 17B 128E Instruct",
             "Llama 4 Maverick 17B (NIM)",
             "meta/llama-4-maverick-17b-128e-instruct",
             min_vram_mb=80000, supports_video=False),
    NimImage("llama-4-scout-17b-16e-instruct",
             "nvcr.io/nim/meta/llama-4-scout-17b-16e-instruct:latest",
             "Llama 4 Scout 17B 16E Instruct",
             "Llama 4 Scout 17B (NIM)",
             "meta/llama-4-scout-17b-16e-instruct",
             min_vram_mb=80000, supports_video=False),

    # ── Mistral family ───────────────────────────────────────────────────
    NimImage("mistral-medium-3.5-128b",
             "nvcr.io/nim/mistralai/mistral-medium-3.5-128b:latest",
             "Mistral Medium 3.5", "Mistral Medium 3.5 128B (NIM)",
             "mistralai/mistral-medium-3.5-128b",
             min_vram_mb=140000, supports_video=False,
             notes="Image + text only; no video. Multi-GPU."),
    NimImage("mistral-small-4-119b-2603",
             "nvcr.io/nim/mistralai/mistral-small-4-119b-2603:latest",
             "Mistral-Small-4-119B-2603", "Mistral Small 4 119B (NIM)",
             "mistralai/mistral-small-4-119b-2603",
             min_vram_mb=140000, supports_video=False,
             notes="Text + image only; no video. Multi-GPU."),
    NimImage("mistral-small-3.2-24b-instruct-2506",
             "nvcr.io/nim/mistralai/mistral-small-3.2-24b-instruct-2506:latest",
             "Mistral Small 3.2 24B Instruct 2506",
             "Mistral Small 3.2 24B (NIM)",
             "mistralai/mistral-small-3.2-24b-instruct-2506",
             min_vram_mb=60000, supports_video=False),
    NimImage("ministral-14b-instruct-2512",
             "nvcr.io/nim/mistralai/ministral-14b-instruct-2512:latest",
             "Ministral 3 14B Instruct 2512",
             "Ministral 3 14B (NIM)",
             "mistralai/ministral-14b-instruct-2512",
             min_vram_mb=40000, supports_video=True,
             notes="Tool calling; 100k context on L40S."),
    NimImage("mistral-large-3-675b-instruct-2512",
             "nvcr.io/nim/mistralai/mistral-large-3-675b-instruct-2512:latest",
             "Mistral Large 3 675B Instruct 2512",
             "Mistral Large 3 675B (NIM)",
             "mistralai/mistral-large-3-675b-instruct-2512",
             min_vram_mb=400000, supports_video=False,
             notes="Frontier-scale; multi-node inference."),

    # ── Qwen family (Alibaba) ────────────────────────────────────────────
    NimImage("qwen3.5-35b-a3b",
             "nvcr.io/nim/qwen/qwen3.5-35b-a3b:latest",
             "Qwen3.5", "Qwen3.5 35B A3B (NIM)",
             "qwen/qwen3.5-35b-a3b",
             min_vram_mb=40000, supports_video=True,
             notes="MoE; high-concurrency video constraints noted in card."),
    NimImage("qwen3.5-122b-a10b",
             "nvcr.io/nim/qwen/qwen3.5-122b-a10b:latest",
             "Qwen3.5", "Qwen3.5 122B A10B (NIM)",
             "qwen/qwen3.5-122b-a10b",
             min_vram_mb=140000, supports_video=True,
             notes="MoE; KV cache saturation risk on long video. Multi-GPU."),
    NimImage("qwen3.5-397b-a17b",
             "nvcr.io/nim/qwen/qwen3.5-397b-a17b:latest",
             "Qwen3.5", "Qwen3.5 397B A17B (NIM)",
             "qwen/qwen3.5-397b-a17b",
             min_vram_mb=400000, supports_video=True,
             notes="Video input not enabled by default — config flag required. Multi-node."),
    NimImage("qwen3.6-27b",
             "nvcr.io/nim/qwen/qwen3.6-27b:latest",
             "Qwen3.6-27B", "Qwen3.6 27B (NIM)",
             "qwen/qwen3.6-27b",
             min_vram_mb=40000, supports_video=False,
             notes="Image + text only; no video."),
    NimImage("qwen3.6-35b-a3b",
             "nvcr.io/nim/qwen/qwen3.6-35b-a3b:latest",
             "Qwen3.6-35B-A3B", "Qwen3.6 35B A3B (NIM)",
             "qwen/qwen3.6-35b-a3b",
             min_vram_mb=40000, supports_video=True,
             notes="MoE; SGLang backend; video concurrency constraints."),

    # ── Moonshot Kimi family ─────────────────────────────────────────────
    NimImage("kimi-k2.5",
             "nvcr.io/nim/moonshotai/kimi-k2.5:latest",
             "Kimi-K2.5", "Kimi K2.5 (NIM)",
             "moonshotai/kimi-k2.5",
             min_vram_mb=140000, supports_video=False,
             notes="Text + image only; no video. Multi-GPU."),
    NimImage("kimi-k2.6",
             "nvcr.io/nim/moonshotai/kimi-k2.6:latest",
             "Kimi-K2.6", "Kimi K2.6 (NIM)",
             "moonshotai/kimi-k2.6",
             min_vram_mb=140000, supports_video=False,
             notes="Does not support video workloads. Multi-GPU."),

    # ── Google Gemma family ──────────────────────────────────────────────
    NimImage("gemma-4-31b-it",
             "nvcr.io/nim/google/gemma-4-31b-it:latest",
             "Gemma 4 31B Instruct", "Gemma 4 31B Instruct (NIM)",
             "google/gemma-4-31b-it",
             min_vram_mb=40000, supports_video=True,
             notes="Structured output not supported on this NIM."),
]


def list_video_nims() -> List["NimImage"]:
    """Subset of KNOWN_VLM_NIMS that accepts multi-frame video input.
    Use this for /byo-video dropdown population — image-only NIMs would
    error out on the standard frame-extraction pipeline."""
    return [n for n in KNOWN_VLM_NIMS if n.supports_video]


def _fetch_url_body(url: str, timeout: int = 10) -> str:
    """Fetch a URL with the standard User-Agent. Returns "" on any failure."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "byo-video-skill nim_catalog/1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def _extract_model_names_from_table(body: str) -> List[str]:
    """Pull model names from the first column of every <tr> in the page.
    Skips version-string cells and "Release notes" link cells."""
    names: List[str] = []
    _version_re = re.compile(r"^\d+(\.\d+){1,3}")
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.DOTALL | re.IGNORECASE):
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL | re.IGNORECASE)
        if not cells:
            continue
        raw = cells[0]
        # Prefer text inside <a>...</a> if present (link text == model name).
        m = re.search(r"<a[^>]*>(.*?)</a>", raw, re.DOTALL | re.IGNORECASE)
        text = m.group(1) if m else raw
        text = re.sub(r"<[^>]+>", "", text)
        text = html.unescape(text).strip()
        if not text or text.lower() in ("model", ""):
            continue
        if _version_re.match(text) or text.lower().startswith("release notes"):
            continue
        names.append(text)
    return names


def fetch_supported_models(timeout: int = 10, include_release_notes: bool = True) -> List[str]:
    """Return the union of model names listed across:
      - https://docs.nvidia.com/nim/vision-language-models/latest/introduction.html
      - every release-notes page in KNOWN_RELEASE_VERSIONS (when
        include_release_notes is True — default).

    Walking the release notes is what keeps older models like Cosmos Reason 1
    7B in the catalog even though they no longer appear on introduction.html.

    Returns [] on total network failure (callers must tolerate that)."""
    seen: set = set()
    out: List[str] = []

    body = _fetch_url_body(DOCS_URL, timeout=timeout)
    for n in _extract_model_names_from_table(body):
        if n not in seen:
            seen.add(n); out.append(n)

    if include_release_notes:
        for version in KNOWN_RELEASE_VERSIONS:
            url = RELEASE_NOTES_URL_FMT.format(version=version)
            rn_body = _fetch_url_body(url, timeout=timeout)
            if not rn_body:
                continue
            for n in _extract_model_names_from_table(rn_body):
                if n not in seen:
                    seen.add(n); out.append(n)

    return out


def _matches_family(name: str, family: str) -> bool:
    """Loose match: introduction.html sometimes prints 'Cosmos Reason2'
    while we want to match all sizes (2B/8B/32B). Compare lowercased
    prefix-form, ignoring punctuation differences."""
    norm = lambda s: re.sub(r"[^a-z0-9]+", "", s.lower())
    return norm(family) in norm(name) or norm(name) in norm(family)


def probe_image(image: str, timeout: int = 30) -> bool:
    """Return True if `docker manifest inspect <image>` succeeds. Requires a
    prior `docker login nvcr.io` to have populated ~/.docker/config.json."""
    try:
        r = subprocess.run(
            ["docker", "manifest", "inspect", image],
            capture_output=True, text=True, timeout=timeout,
        )
        return r.returncode == 0
    except Exception:
        return False


def list_available_nims(
    ngc_api_key: Optional[str] = None,
    vram_mb: Optional[int] = None,
    use_upstream: bool = True,
    do_probe: bool = True,
) -> List[NimImage]:
    """Return the subset of KNOWN_VLM_NIMS that:
      - has a family name present on DOCS_URL (when use_upstream and reachable),
      - fits in vram_mb if given,
      - passes a `docker manifest inspect` probe when do_probe.
    `ngc_api_key` is informational only — `docker login nvcr.io` must already
    have been done for probing to work."""
    upstream_names: List[str] = fetch_supported_models() if use_upstream else []
    out: List[NimImage] = []
    for nim in KNOWN_VLM_NIMS:
        if upstream_names and not any(_matches_family(n, nim.family) for n in upstream_names):
            continue
        if vram_mb is not None and nim.min_vram_mb and nim.min_vram_mb > vram_mb:
            continue
        if do_probe and not probe_image(nim.image):
            continue
        out.append(nim)
    return out


def main(argv: List[str]) -> int:
    p = argparse.ArgumentParser(description="VLM NIM catalog helper")
    sp = p.add_subparsers(dest="cmd", required=True)
    pl = sp.add_parser("list", help="List available NIMs as JSON")
    pl.add_argument("--vram", type=int, default=None,
                    help="Filter by max VRAM available on target (MB)")
    pl.add_argument("--no-upstream", action="store_true",
                    help="Skip fetching docs.nvidia.com (use slug map only)")
    pl.add_argument("--no-probe", action="store_true",
                    help="Skip `docker manifest inspect` probes")
    sp.add_parser("upstream", help="Print model names from docs.nvidia.com")
    args = p.parse_args(argv)

    if args.cmd == "upstream":
        names = fetch_supported_models()
        print(json.dumps(names, indent=2))
        return 0
    if args.cmd == "list":
        nims = list_available_nims(
            ngc_api_key=os.environ.get("NGC_API_KEY"),
            vram_mb=args.vram,
            use_upstream=not args.no_upstream,
            do_probe=not args.no_probe,
        )
        print(json.dumps([asdict(n) for n in nims], indent=2))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
