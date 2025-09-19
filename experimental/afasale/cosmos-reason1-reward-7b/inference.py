#!/usr/bin/env python3
"""
Standalone inference script for video anomaly detection model.
Given a single video, produces a yes/no prediction for whether it contains anomalies/artifacts.
"""

import argparse

import mediapy as media
import numpy as np
import qwen_vl_utils
import torch
from PIL import Image as PILImage
from qwen_vl_utils.vision_process import smart_resize
from transformers.models.auto.modeling_auto import AutoModelForVision2Seq
from transformers.models.auto.processing_auto import AutoProcessor

# System prompt used during training
SYSTEM_PROMPT = """You are a helpful video analyzer. The goal is to identify artifacts and anomalies in the video. Watch carefully and focus on the following aspects:

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

USER_PROMPT = "Does the video contain any anomalies or artifacts?"


def decode_video_for_inference(
    video_path: str,
    temporal_patch_size: int = 2,
    target_num_tokens: int = 9216,
    frame_count_range: tuple[int, int] = (40, 160),
) -> tuple[list[PILImage.Image], float]:
    """Decode video file for inference, similar to training preprocessing."""
    patch_size = 14
    min_height_width = 56
    min_pixels = 16 * (patch_size * 2) ** 2 * temporal_patch_size
    max_pixels = target_num_tokens * (patch_size * 2) ** 2 * temporal_patch_size

    # Read video file
    with open(video_path, "rb") as f:
        mp4_bytes = f.read()

    video = media.decompress_video(mp4_bytes)
    total_frames = video.shape[0]
    video_fps = video.metadata.fps  # type: ignore

    # Calculate downsampling interval to get frames within target range
    min_frames_range, max_frames_range = frame_count_range
    interval = max(1, (total_frames - 1) // max_frames_range + 1)
    if interval != 1:
        print(f"Video downsampled from {total_frames} to {total_frames // interval} frames")

    # Downsample frames
    idx = np.arange(0, total_frames, interval)
    video_frames = video[idx, ...]
    nframes = len(idx)
    sample_fps = video_fps / interval

    # Make frame count divisible by temporal_patch_size
    frames_to_remove = nframes % temporal_patch_size
    if frames_to_remove != 0:
        video_frames = video_frames[:-frames_to_remove, ...]

    # Resize frames
    nframes, height, width, _ = video_frames.shape
    max_pixels_per_frame = max_pixels // nframes
    resized_height, resized_width = smart_resize(
        height,
        width,
        min_pixels=min_pixels,
        max_pixels=max_pixels_per_frame,
    )

    list_of_pil_images = [
        PILImage.fromarray(x).resize((resized_width, resized_height), resample=PILImage.Resampling.BICUBIC)
        for x in video_frames
    ]

    return list_of_pil_images, sample_fps


def load_model_and_processor(checkpoint_path: str):
    """Load the trained model and processor."""
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


def predict_video_anomaly(video_path: str, model, processor) -> tuple[str, float]:
    """Predict if video contains anomalies. Returns (prediction, score) where score is softmax prob of 'No'."""

    # Process video
    video_frames, video_fps = decode_video_for_inference(video_path)

    # Prepare message format (same as training)
    messages = [
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_frames, "fps": video_fps},
                    {"type": "text", "text": USER_PROMPT},
                ],
            },
        ]
    ]

    # Apply chat template
    text = processor.apply_chat_template(messages[0], tokenize=False, add_generation_prompt=True)

    # Process inputs
    image_inputs, video_inputs, video_kwargs = qwen_vl_utils.process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        **(video_kwargs if video_kwargs else {}),
    )

    # Move to device
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Generate prediction
    with torch.no_grad():
        # Get logits for next token prediction
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last token logits

        # Get tokens for "Yes" and "No"
        yes_token_id = processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_token_id = processor.tokenizer.encode("No", add_special_tokens=False)[0]

        # Compare logits for Yes vs No
        yes_logit = logits[yes_token_id].item()
        no_logit = logits[no_token_id].item()

        # Compute softmax score - probability of "No" (video has no anomalies)
        no_score = torch.softmax(torch.tensor([yes_logit, no_logit]), dim=0)[1].item()

        prediction = "Bad" if yes_logit > no_logit else "Good"
        return prediction, no_score


def main():
    parser = argparse.ArgumentParser(description="Video reward model inference")
    parser.add_argument("video_path", help="Path to input video file")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint")

    args = parser.parse_args()

    print(f"Loading model from {args.checkpoint}...")
    model, processor = load_model_and_processor(args.checkpoint)

    print(f"Processing video: {args.video_path}")
    prediction, score = predict_video_anomaly(args.video_path, model, processor)

    print(f"Physical accuracy: {prediction}")
    print(f"Score (high is good): {score:.4f}")
    return prediction, score


if __name__ == "__main__":
    main()
