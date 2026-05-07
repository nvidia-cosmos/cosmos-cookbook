#!/usr/bin/env python3
"""Headless BYO-video smoke test for Cosmos Reason2.
Usage: BYO_VIDEO=/path/to/video.mp4 MODEL_NAME=nvidia/Cosmos-Reason2-2B python smoke_cr2_byo.py
"""
import os, json, time, warnings
warnings.filterwarnings("ignore")

import torch
import transformers

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
print("[smoke] PyAV backend patch applied", flush=True)

BYO_VIDEO = os.environ.get("BYO_VIDEO", "/home/shadeform/cosmos-reason2/assets/sample.mp4")
MODEL_NAME = os.environ.get("MODEL_NAME", "nvidia/Cosmos-Reason2-2B")
MODEL_DIR  = os.environ.get("MODEL_DIR", MODEL_NAME)
OUT_FILE   = os.environ.get("OUT_FILE", "/tmp/byo_video_reason2_results.json")
PROMPT     = os.environ.get("REASON2_PROMPT", "Describe this video in detail.")

import subprocess
gpu_info = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,memory.free,memory.total", "--format=csv,noheader"],
    text=True
).strip()
print(f"[smoke] GPU: {gpu_info}", flush=True)
print(f"[smoke] Model: {MODEL_NAME}", flush=True)
print(f"[smoke] Video: {BYO_VIDEO}", flush=True)

PIXELS_PER_TOKEN = 32 ** 2

t0 = time.time()
print("[smoke] Loading model...", flush=True)
model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_DIR,
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa",
)
processor = transformers.Qwen3VLProcessor.from_pretrained(MODEL_DIR)
processor.image_processor.size = {
    "shortest_edge": 256 * PIXELS_PER_TOKEN,
    "longest_edge":  4096 * PIXELS_PER_TOKEN,
}
processor.video_processor.size = {
    "shortest_edge": 256 * PIXELS_PER_TOKEN,
    "longest_edge":  4096 * PIXELS_PER_TOKEN,
}
load_time = time.time() - t0
print(f"[smoke] Model loaded in {load_time:.1f}s", flush=True)

conversation = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant that analyzes videos."}]},
    {"role": "user",   "content": [
        {"type": "video", "video": BYO_VIDEO},
        {"type": "text",  "text": PROMPT},
    ]},
]

t1 = time.time()
inputs = processor.apply_chat_template(
    conversation, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt", fps=4,
)
inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
print("[smoke] Inputs prepared, generating...", flush=True)

with torch.inference_mode():
    out_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)

n = inputs["input_ids"].shape[1]
response = processor.decode(out_ids[0][n:], skip_special_tokens=True)
infer_time = time.time() - t1

result = {
    "recipe":      "reason2/byo_video",
    "provider":    os.environ.get("PROVIDER", "local"),
    "model":       MODEL_NAME,
    "gpu":         gpu_info,
    "byo_video":   BYO_VIDEO,
    "prompt":      PROMPT,
    "response":    response,
    "load_time_s": round(load_time, 1),
    "infer_time_s": round(infer_time, 1),
    "status":      "success",
}

with open(OUT_FILE, "w") as f:
    json.dump(result, f, indent=2)

print(f"\n[smoke] Done in {infer_time:.1f}s inference / {load_time:.1f}s load")
print(f"[smoke] Response: {response[:200]}")
print(f"[smoke] Results written to {OUT_FILE}")
print(json.dumps(result, indent=2))
