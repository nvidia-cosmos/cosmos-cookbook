# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Video Quality Evaluation Pipeline for Cosmos-generated videos.

Evaluates each generated video across up to three complementary layers:

  Layer 1 — Perceptual similarity  (LPIPS, SSIM, PSNR)  requires --ref_dir
  Layer 2 — Semantic text alignment (CLIP)               requires --prompts
  Layer 3 — Physical plausibility  (Cosmos Reason 1)     requires --reason_ckpt

Any subset of layers is valid. The script computes a weighted aggregate score
for each video and classifies it as PASS, BORDERLINE, or REJECT, attributing
which layer(s) drove a low score.

For distributional quality (FVD/FID), use the existing scripts in
  scripts/metrics/qualitative/fvd_fid/
on the full generated set before running this per-video pipeline.

Example — all layers:
  python evaluate.py \\
      --pred_dir ./generated \\
      --ref_dir  ./reference \\
      --prompts  prompts.txt \\
      --reason_ckpt ./checkpoints/Cosmos-Reason1-7B-Reward \\
      --output   report.json

Example — VLM + CLIP only (no reference videos needed):
  python evaluate.py \\
      --pred_dir ./generated \\
      --prompts  prompts.txt \\
      --reason_ckpt ./checkpoints/Cosmos-Reason1-7B-Reward \\
      --output   report.json

Example — perceptual only (no prompts, no model download needed):
  python evaluate.py \\
      --pred_dir ./generated \\
      --ref_dir  ./reference \\
      --output   report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import decord
import numpy as np
import torch
from PIL import Image as PILImage
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PASS_LABEL = "PASS"
BORDERLINE_LABEL = "BORDERLINE"
REJECT_LABEL = "REJECT"

# Weights for each layer's contribution to the aggregate score.
# Must sum to 1.0 across the layers that are actually active.
DEFAULT_WEIGHTS = {
    "perceptual": 0.30,
    "semantic": 0.35,
    "vlm": 0.35,
}

CLIP_FRAMES = 8  # frames sampled per video for CLIP alignment
LPIPS_FRAMES = 16  # frames sampled per video for LPIPS/SSIM


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class LayerResult:
    score: float  # 0-1, higher is better
    raw: dict = field(default_factory=dict)  # metric-specific raw values


@dataclass
class VideoResult:
    video: str
    aggregate_score: float
    verdict: str  # PASS | BORDERLINE | REJECT
    active_layers: list[str]
    failure_reasons: list[str]
    perceptual: Optional[LayerResult] = None
    semantic: Optional[LayerResult] = None
    vlm: Optional[LayerResult] = None


# ---------------------------------------------------------------------------
# Video loading helpers
# ---------------------------------------------------------------------------


def load_frames_rgb(video_path: str, n_frames: int) -> np.ndarray:
    """Sample n_frames evenly from a video. Returns (N, H, W, 3) uint8 array."""
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    total = len(vr)
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()  # (N, H, W, 3)
    return frames


def frames_to_tensor(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert (N, H, W, 3) uint8 numpy to (N, 3, H, W) float32 in [-1, 1]."""
    t = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 127.5 - 1.0
    return t.to(device)


def resize_frames(frames: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize (N, H, W, 3) frames to (N, size[0], size[1], 3)."""
    resized = []
    for f in frames:
        img = PILImage.fromarray(f).resize((size[1], size[0]), PILImage.BICUBIC)
        resized.append(np.array(img))
    return np.stack(resized)


# ---------------------------------------------------------------------------
# Layer 1 — Perceptual (LPIPS, SSIM, PSNR)
# ---------------------------------------------------------------------------


def _import_perceptual():
    """Lazy import so users without lpips/skimage can still use other layers."""
    try:
        import lpips
        from skimage.metrics import peak_signal_noise_ratio as psnr_fn
        from skimage.metrics import structural_similarity as ssim_fn

        return lpips, ssim_fn, psnr_fn
    except ImportError as e:
        print(f"[perceptual] Missing dependency: {e}. Install with: pip install lpips scikit-image")
        return None, None, None


def compute_perceptual_layer(
    pred_path: str,
    ref_path: str,
    lpips_model,
    ssim_fn,
    psnr_fn,
    device: torch.device,
) -> LayerResult:
    pred_frames = load_frames_rgb(pred_path, LPIPS_FRAMES)
    ref_frames = load_frames_rgb(ref_path, LPIPS_FRAMES)

    # Match spatial resolution to the smaller of the two
    h = min(pred_frames.shape[1], ref_frames.shape[1])
    w = min(pred_frames.shape[2], ref_frames.shape[2])
    pred_frames = resize_frames(pred_frames, (h, w))
    ref_frames = resize_frames(ref_frames, (h, w))

    pred_t = frames_to_tensor(pred_frames, device)  # (N, 3, H, W) in [-1,1]
    ref_t = frames_to_tensor(ref_frames, device)

    # LPIPS (lower is better → invert to [0,1] higher-is-better)
    with torch.no_grad():
        lpips_scores = []
        for p, r in zip(pred_t, ref_t):
            val = lpips_model(p.unsqueeze(0), r.unsqueeze(0)).item()
            lpips_scores.append(val)
    mean_lpips = float(np.mean(lpips_scores))
    lpips_score = max(0.0, 1.0 - mean_lpips)  # LPIPS in [0,~1], invert

    # SSIM / PSNR (both higher is better, SSIM already in [0,1])
    ssim_vals, psnr_vals = [], []
    pred_u8 = ((pred_frames).astype(np.float32))
    ref_u8 = ((ref_frames).astype(np.float32))
    for p, r in zip(pred_u8, ref_u8):
        ssim_vals.append(ssim_fn(p, r, channel_axis=2, data_range=255))
        psnr_vals.append(psnr_fn(r, p, data_range=255))
    mean_ssim = float(np.mean(ssim_vals))
    mean_psnr = float(np.mean(psnr_vals))
    # Normalise PSNR: 0 dB → 0, 40+ dB → 1
    psnr_score = float(np.clip(mean_psnr / 40.0, 0.0, 1.0))

    # Aggregate perceptual score: equal weight across three signals
    layer_score = (lpips_score + mean_ssim + psnr_score) / 3.0

    return LayerResult(
        score=layer_score,
        raw={
            "lpips": mean_lpips,
            "lpips_score": lpips_score,
            "ssim": mean_ssim,
            "psnr_db": mean_psnr,
            "psnr_score": psnr_score,
        },
    )


# ---------------------------------------------------------------------------
# Layer 2 — Semantic (CLIP text-video alignment)
# ---------------------------------------------------------------------------


def _import_clip():
    try:
        import clip

        return clip
    except ImportError:
        print("[semantic] Missing dependency: clip. Install with: pip install git+https://github.com/openai/CLIP.git")
        return None


def load_clip_model(device: torch.device):
    clip = _import_clip()
    if clip is None:
        return None, None, None
    model, preprocess = clip.load("ViT-B/32", device=device)
    return clip, model, preprocess


def compute_semantic_layer(
    pred_path: str,
    prompt: str,
    clip_lib,
    clip_model,
    clip_preprocess,
    device: torch.device,
) -> LayerResult:
    frames = load_frames_rgb(pred_path, CLIP_FRAMES)

    # Encode frames
    frame_tensors = []
    for f in frames:
        img = PILImage.fromarray(f)
        frame_tensors.append(clip_preprocess(img))
    frame_batch = torch.stack(frame_tensors).to(device)

    # Encode text
    text_tokens = clip_lib.tokenize([prompt], truncate=True).to(device)

    with torch.no_grad():
        image_features = clip_model.encode_image(frame_batch)
        text_features = clip_model.encode_text(text_tokens)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Cosine similarity per frame, then average
        sims = (image_features @ text_features.T).squeeze(-1)  # (N,)
        mean_sim = sims.mean().item()

    # CLIP cosine similarity is typically in [0.1, 0.35] for good matches.
    # Normalise to [0, 1]: below 0.1 → 0, above 0.35 → 1.
    clip_score = float(np.clip((mean_sim - 0.10) / 0.25, 0.0, 1.0))

    return LayerResult(
        score=clip_score,
        raw={"cosine_similarity": mean_sim, "clip_score": clip_score},
    )


# ---------------------------------------------------------------------------
# Layer 3 — VLM Judge (Cosmos Reason 1 — 7B Reward model)
# ---------------------------------------------------------------------------


def load_reason_model(checkpoint_path: str, device: torch.device):
    """Load Cosmos Reason 1-7B-Reward model and processor."""
    try:
        import mediapy as media
        import qwen_vl_utils  # noqa: F401
        from transformers import AutoModelForVision2Seq, AutoProcessor
    except ImportError as e:
        print(f"[vlm] Missing dependency: {e}. Install with: pip install mediapy qwen-vl-utils transformers")
        return None, None

    processor = AutoProcessor.from_pretrained(checkpoint_path)
    model = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        device_map="auto",
        use_cache=False,
    )
    model.eval()
    return model, processor


# System prompt matches the one used during Cosmos Reason 1 training.
_REASON_SYSTEM_PROMPT = """You are a helpful video analyzer. The goal is to identify artifacts and anomalies in the video. Watch carefully and focus on the following aspects:

* Gravity (e.g. a ball cannot fly in the air)
* Collision (e.g. two objects cannot penetrate each other)
* Object interaction (e.g. an object cannot move without any apparent reason)
* Fluid dynamics (e.g. a liquid cannot flow through a solid object)
* Object permanence (e.g. an object cannot suddenly appear, disappear or change its shape)
* Common sense (e.g. an object should be functional and useful)
* Cause-and-effect (e.g. a door cannot open without any apparent reason)
* Human motion (e.g. a person's body cannot morph and the joints cannot move in impossible ways)

Here are some examples of non-artifacts you should not include in your analysis:

* Being an animated video, such as a cartoon, does not automatically make it artifacts.
* The video has no sound. Do not make any conclusions based on sound.
* Ignore any lighting, shadows, blurring, and camera effects.
* Avoid judging based on overall impression, artistic style, or background elements.

Begin your response with a single word: "Yes" or "No"."""

# Ask the same physical-plausibility question from three different angles
# (multi-judge pattern): aggregate their scores rather than trusting one call.
_REASON_QUERIES = [
    "Does the video contain any anomalies or artifacts?",
    "Are there any violations of physical laws such as gravity or collision in this video?",
    "Does any object in the video behave in a physically implausible way?",
]


def _score_one_query(video_path: str, query: str, model, processor) -> float:
    """Return Cosmos Reason 1 'no-anomaly' score (0-1, higher=better) for one query."""
    import mediapy as media
    import qwen_vl_utils
    from qwen_vl_utils.vision_process import smart_resize

    temporal_patch_size = 2
    target_num_tokens = 9216
    patch_size = 14
    min_pixels = 16 * (patch_size * 2) ** 2 * temporal_patch_size
    max_pixels = target_num_tokens * (patch_size * 2) ** 2 * temporal_patch_size

    with open(video_path, "rb") as f:
        mp4_bytes = f.read()
    video = media.decompress_video(mp4_bytes)
    total_frames = video.shape[0]
    video_fps = video.metadata.fps

    interval = max(1, (total_frames - 1) // 160 + 1)
    idx = np.arange(0, total_frames, interval)
    video_frames = video[idx]
    nframes = len(idx)
    sample_fps = video_fps / interval

    frames_to_remove = nframes % temporal_patch_size
    if frames_to_remove:
        video_frames = video_frames[:-frames_to_remove]

    nframes, h, w, _ = video_frames.shape
    max_pixels_per_frame = max_pixels // nframes
    rh, rw = smart_resize(h, w, min_pixels=min_pixels, max_pixels=max_pixels_per_frame)
    pil_frames = [
        PILImage.fromarray(f).resize((rw, rh), PILImage.BICUBIC) for f in video_frames
    ]

    messages = [
        [
            {"role": "system", "content": _REASON_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": pil_frames, "fps": sample_fps},
                    {"type": "text", "text": query},
                ],
            },
        ]
    ]

    text = processor.apply_chat_template(messages[0], tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = qwen_vl_utils.process_vision_info(
        messages, return_video_kwargs=True
    )
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        **(video_kwargs or {}),
    )
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
        yes_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_id = processor.tokenizer.encode("No", add_special_tokens=False)[0]
        no_score = torch.softmax(torch.tensor([logits[yes_id], logits[no_id]]), dim=0)[1].item()

    return float(no_score)


def compute_vlm_layer(video_path: str, model, processor) -> LayerResult:
    """Run all three Cosmos Reason queries and aggregate their scores."""
    query_scores = []
    for query in _REASON_QUERIES:
        score = _score_one_query(video_path, query, model, processor)
        query_scores.append(score)

    # Majority-weighted aggregate: mean is sufficient for three soft scores
    aggregate = float(np.mean(query_scores))

    return LayerResult(
        score=aggregate,
        raw={
            "query_scores": {q: s for q, s in zip(_REASON_QUERIES, query_scores)},
            "aggregate": aggregate,
        },
    )


# ---------------------------------------------------------------------------
# Aggregate + classification
# ---------------------------------------------------------------------------


def aggregate_and_classify(
    results: dict[str, Optional[LayerResult]],
    weights: dict[str, float],
    threshold_pass: float,
    threshold_reject: float,
) -> tuple[float, str, list[str], list[str]]:
    """
    Compute weighted aggregate score, verdict, and failure reasons.

    Returns (aggregate_score, verdict, active_layers, failure_reasons).
    """
    active_layers = [k for k, v in results.items() if v is not None]

    if not active_layers:
        return 0.0, REJECT_LABEL, [], ["no evaluation layers were active"]

    # Re-weight to sum to 1 over active layers only
    raw_weights = {k: weights.get(k, 1.0) for k in active_layers}
    total_w = sum(raw_weights.values())
    norm_weights = {k: v / total_w for k, v in raw_weights.items()}

    aggregate = sum(results[k].score * norm_weights[k] for k in active_layers)

    if aggregate >= threshold_pass:
        verdict = PASS_LABEL
    elif aggregate < threshold_reject:
        verdict = REJECT_LABEL
    else:
        verdict = BORDERLINE_LABEL

    failure_reasons = []
    for layer in active_layers:
        score = results[layer].score
        if score < threshold_reject:
            failure_reasons.append(f"{layer} layer failed (score={score:.3f})")
        elif score < threshold_pass:
            failure_reasons.append(f"{layer} layer borderline (score={score:.3f})")

    return float(aggregate), verdict, active_layers, failure_reasons


# ---------------------------------------------------------------------------
# Prompt loading
# ---------------------------------------------------------------------------


def load_prompts(prompts_file: str, video_stems: list[str]) -> dict[str, str]:
    """
    Load prompts from a file. Supports two formats:

    1. Plain text — one prompt per line, matched to videos by sort order.
    2. JSON / JSONL — maps video filename stem to prompt string:
         {"video_001": "a robot picking up a cup", ...}
         or one JSON object per line.
    """
    path = Path(prompts_file)
    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_file}")

    text = path.read_text().strip()

    # Try JSON object
    if text.startswith("{"):
        mapping = json.loads(text)
        return {stem: mapping.get(stem, "") for stem in video_stems}

    # Try JSONL
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if lines and lines[0].startswith("{"):
        mapping = {}
        for line in lines:
            obj = json.loads(line)
            mapping.update(obj)
        return {stem: mapping.get(stem, "") for stem in video_stems}

    # Plain text — match by order
    prompt_list = lines
    return {
        stem: prompt_list[i] if i < len(prompt_list) else ""
        for i, stem in enumerate(video_stems)
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def run_evaluation(args) -> list[VideoResult]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pred_dir = Path(args.pred_dir)
    pred_videos = sorted(pred_dir.glob("*.mp4"))
    if not pred_videos:
        print(f"No .mp4 files found in {pred_dir}")
        sys.exit(1)

    video_stems = [v.stem for v in pred_videos]

    # --- Resolve reference videos ---
    ref_map: dict[str, Path] = {}
    if args.ref_dir:
        ref_dir = Path(args.ref_dir)
        for v in pred_videos:
            ref_path = ref_dir / v.name
            if ref_path.exists():
                ref_map[v.stem] = ref_path
        if not ref_map:
            print(f"[perceptual] Warning: --ref_dir supplied but no matching filenames found in {ref_dir}")

    # --- Load perceptual models ---
    lpips_model = ssim_fn = psnr_fn = None
    if ref_map:
        lpips_lib, ssim_fn, psnr_fn = _import_perceptual()
        if lpips_lib is not None:
            lpips_model = lpips_lib.LPIPS(net="alex").to(device)
            lpips_model.eval()
            print("[perceptual] LPIPS model loaded")

    # --- Load CLIP ---
    clip_lib = clip_model = clip_preprocess = None
    prompts: dict[str, str] = {}
    if args.prompts:
        clip_lib, clip_model, clip_preprocess = load_clip_model(device)
        if clip_lib is not None:
            prompts = load_prompts(args.prompts, video_stems)
            print(f"[semantic] CLIP loaded, {len(prompts)} prompts found")

    # --- Load Cosmos Reason ---
    reason_model = reason_processor = None
    if args.reason_ckpt:
        print(f"[vlm] Loading Cosmos Reason 1 from {args.reason_ckpt} …")
        reason_model, reason_processor = load_reason_model(args.reason_ckpt, device)
        if reason_model is not None:
            print("[vlm] Cosmos Reason 1 loaded")

    weights = {
        "perceptual": args.weight_perceptual,
        "semantic": args.weight_semantic,
        "vlm": args.weight_vlm,
    }

    results: list[VideoResult] = []

    for pred_path in tqdm(pred_videos, desc="Evaluating"):
        stem = pred_path.stem
        layer_results: dict[str, Optional[LayerResult]] = {
            "perceptual": None,
            "semantic": None,
            "vlm": None,
        }

        # Layer 1 — Perceptual
        if lpips_model is not None and stem in ref_map:
            try:
                layer_results["perceptual"] = compute_perceptual_layer(
                    str(pred_path), str(ref_map[stem]),
                    lpips_model, ssim_fn, psnr_fn, device,
                )
            except Exception as e:
                print(f"  [perceptual] Error on {stem}: {e}")

        # Layer 2 — Semantic
        if clip_model is not None and prompts.get(stem):
            try:
                layer_results["semantic"] = compute_semantic_layer(
                    str(pred_path), prompts[stem],
                    clip_lib, clip_model, clip_preprocess, device,
                )
            except Exception as e:
                print(f"  [semantic] Error on {stem}: {e}")

        # Layer 3 — VLM
        if reason_model is not None:
            try:
                layer_results["vlm"] = compute_vlm_layer(str(pred_path), reason_model, reason_processor)
            except Exception as e:
                print(f"  [vlm] Error on {stem}: {e}")

        aggregate, verdict, active_layers, failure_reasons = aggregate_and_classify(
            layer_results, weights, args.threshold_pass, args.threshold_reject
        )

        results.append(
            VideoResult(
                video=str(pred_path),
                aggregate_score=aggregate,
                verdict=verdict,
                active_layers=active_layers,
                failure_reasons=failure_reasons,
                perceptual=layer_results["perceptual"],
                semantic=layer_results["semantic"],
                vlm=layer_results["vlm"],
            )
        )

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_report(results: list[VideoResult], args) -> dict:
    total = len(results)
    pass_count = sum(1 for r in results if r.verdict == PASS_LABEL)
    borderline_count = sum(1 for r in results if r.verdict == BORDERLINE_LABEL)
    reject_count = sum(1 for r in results if r.verdict == REJECT_LABEL)

    def layer_mean(layer: str) -> Optional[float]:
        scores = [getattr(r, layer).score for r in results if getattr(r, layer) is not None]
        return float(np.mean(scores)) if scores else None

    report = {
        "summary": {
            "total_videos": total,
            "pass": pass_count,
            "borderline": borderline_count,
            "reject": reject_count,
            "pass_rate": pass_count / total if total else 0.0,
            "mean_aggregate_score": float(np.mean([r.aggregate_score for r in results])),
            "mean_perceptual_score": layer_mean("perceptual"),
            "mean_semantic_score": layer_mean("semantic"),
            "mean_vlm_score": layer_mean("vlm"),
            "config": {
                "threshold_pass": args.threshold_pass,
                "threshold_reject": args.threshold_reject,
                "weight_perceptual": args.weight_perceptual,
                "weight_semantic": args.weight_semantic,
                "weight_vlm": args.weight_vlm,
            },
        },
        "videos": [],
    }

    for r in results:
        entry = {
            "video": r.video,
            "aggregate_score": r.aggregate_score,
            "verdict": r.verdict,
            "active_layers": r.active_layers,
            "failure_reasons": r.failure_reasons,
        }
        for layer in ("perceptual", "semantic", "vlm"):
            lr: Optional[LayerResult] = getattr(r, layer)
            if lr is not None:
                entry[layer] = {"score": lr.score, **lr.raw}
        report["videos"].append(entry)

    return report


def print_summary(report: dict) -> None:
    s = report["summary"]
    print("\n" + "=" * 60)
    print("VIDEO QUALITY EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total videos evaluated : {s['total_videos']}")
    print(f"  PASS                   : {s['pass']} ({s['pass_rate']:.1%})")
    print(f"  BORDERLINE             : {s['borderline']}")
    print(f"  REJECT                 : {s['reject']}")
    print(f"  Mean aggregate score   : {s['mean_aggregate_score']:.3f}")
    for layer in ("perceptual", "semantic", "vlm"):
        mean = s.get(f"mean_{layer}_score")
        if mean is not None:
            print(f"  Mean {layer:12s} score : {mean:.3f}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-layer video quality evaluation for Cosmos-generated videos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pred_dir", required=True, help="Directory of generated .mp4 videos to evaluate")
    parser.add_argument("--ref_dir", default=None, help="Directory of reference .mp4 videos (enables Layer 1 — perceptual metrics)")
    parser.add_argument("--prompts", default=None, help="Text prompts file (enables Layer 2 — CLIP alignment). Plain text (one per line) or JSON/JSONL mapping stem→prompt")
    parser.add_argument("--reason_ckpt", default=None, help="Path to Cosmos-Reason1-7B-Reward checkpoint (enables Layer 3 — VLM judge)")
    parser.add_argument("--output", default="quality_report.json", help="Output JSON report path")
    parser.add_argument("--threshold_pass", type=float, default=0.65, help="Aggregate score ≥ this → PASS")
    parser.add_argument("--threshold_reject", type=float, default=0.35, help="Aggregate score < this → REJECT")
    parser.add_argument("--weight_perceptual", type=float, default=DEFAULT_WEIGHTS["perceptual"], help="Weight for Layer 1 (perceptual)")
    parser.add_argument("--weight_semantic", type=float, default=DEFAULT_WEIGHTS["semantic"], help="Weight for Layer 2 (semantic)")
    parser.add_argument("--weight_vlm", type=float, default=DEFAULT_WEIGHTS["vlm"], help="Weight for Layer 3 (VLM)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not Path(args.pred_dir).exists():
        print(f"Error: --pred_dir not found: {args.pred_dir}")
        sys.exit(1)

    if args.threshold_reject >= args.threshold_pass:
        print("Error: --threshold_reject must be less than --threshold_pass")
        sys.exit(1)

    results = run_evaluation(args)
    report = build_report(results, args)
    print_summary(report)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))
    print(f"Full report written to: {output_path}")


if __name__ == "__main__":
    main()