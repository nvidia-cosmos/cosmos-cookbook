import torch
import torchvision.transforms.functional
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple
import numpy as np


def tensor_to_pil_images(video_tensor):
    """
    Convert a video tensor of shape (C, T, H, W) or (T, C, H, W) to a list of PIL images.

    Args:
        video_tensor (torch.Tensor): Video tensor with shape (C, T, H, W) or (T, C, H, W)

    Returns:
        list[PIL.Image.Image]: List of PIL images
    """
    # Check tensor shape and convert if needed
    if video_tensor.shape[0] == 3 and video_tensor.shape[1] > 3:  # (C, T, H, W)
        # Convert to (T, C, H, W)
        video_tensor = video_tensor.permute(1, 0, 2, 3)

    # Convert to numpy array with shape (T, H, W, C)
    video_np = video_tensor.permute(0, 2, 3, 1).cpu().numpy()

    # Ensure values are in the right range for PIL (0-255, uint8)
    if video_np.dtype == np.float32 or video_np.dtype == np.float64:
        if video_np.max() <= 1.0:
            video_np = (video_np * 255).astype(np.uint8)
        else:
            video_np = video_np.astype(np.uint8)

    # Convert each frame to a PIL image
    pil_images = [Image.fromarray(frame) for frame in video_np]

    return pil_images


def overlay_text(
    images: List[Image.Image],
    fps: float,
    border_height: int = 28,  # this is due to patch size of 28
    temporal_path_size: int = 2,  # Number of positions to cycle through
    font_size: int = 20,
    font_color: str = "white",
) -> Tuple[List[Image.Image], List[float]]:
    """
    Overlay text on a list of PIL images with black border.
    The timestamp position cycles through available positions.

    Args:
        images: List of PIL images to process
        fps: Frames per second
        border_height: Height of the black border in pixels (default: 28)
        temporal_path_size: Number of positions to cycle through (default: 2)
        font_size: Font size for the text (default: 20)
        font_color: Color of the text (default: "white")

    Returns:
        List of PIL images with text overlay
        List of timestamps
    """

    # Try to use DejaVu Sans Mono font for better readability
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)

    # Process each image
    processed_images = []

    for i, image in enumerate(images):
        # Get original dimensions
        width, height = image.size

        # Create new image with black border at the bottom
        new_height = height + border_height
        new_image = Image.new("RGB", (width, new_height), color="black")

        # Paste original image at the top
        new_image.paste(image, (0, 0))

        # Draw text on the black border
        draw = ImageDraw.Draw(new_image)

        # Calculate timestamp for current frame
        total_seconds = i / fps
        text = f"{total_seconds:.2f}s"

        # Get text dimensions
        try:
            # Get text bounding box
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            # Fallback for older PIL versions
            text_width, text_height = draw.textsize(text, font=font)

        # Define available positions (cycling through horizontal positions)
        position_idx = i % temporal_path_size
        section_width = width // temporal_path_size

        # Calculate x position based on cycling position
        section_center_x = position_idx * section_width + section_width // 2
        text_x = section_center_x - text_width // 2

        # Ensure text doesn't go outside bounds
        text_x = max(0, min(text_x, width - text_width))

        # Center vertically in the border
        text_y = height + (border_height - text_height) // 2

        # Draw the single timestamp
        draw.text((text_x, text_y), text, fill=font_color, font=font)

        processed_images.append(new_image)

    return processed_images, [i / fps for i in range(len(images))]


# You can also replace the MODEL_PATH by a safetensors folder path mentioned above
MODEL_PATH = "nvidia/Cosmos-Reason1.1-7B"
VIDEO_PATH = "d0034acb53ca4274dab2dd47c9818db8b7de760ea09dabca1e45563e.mp4"
FPS = 2.0

llm = LLM(
    model=MODEL_PATH,
    limit_mm_per_prompt={"image": 10, "video": 10},
)

sampling_params = SamplingParams(
    temperature=0.6,
    top_p=0.95,
    repetition_penalty=1.05,
    max_tokens=4096,
)

video_messages = [
    {"role": "system", "content": """"Please provide captions of all the events in the video with timestamps using the following format: 
     <start time> <end time> caption of event 1.\n<start time> <end time> caption of event 2.\n
    At each frame, the timestamp is embedded at the bottom of the video. You need to extract the timestamp and answer the user question."""},
    {"role": "user", "content": [
            {"type": "text", "text": (
                    "Describe the notable events in the provided video."
                )
            },
            {
                "type": "video", 
                "video": VIDEO_PATH,
                "fps": FPS,
                "total_pixels": 28 * 28 * 8092,  # 8092 vision tokens.
            }
        ]
    },
]

# Here we use video messages as a demonstration
messages = video_messages

processor = AutoProcessor.from_pretrained(MODEL_PATH)
prompt = processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
video_inputs_with_timestamp = []
for video in video_inputs:
    images = tensor_to_pil_images(video)
    images_with_timestamp, _ = overlay_text(images, FPS)
    tensors = [torchvision.transforms.functional.pil_to_tensor(img) for img in images_with_timestamp]
    tensors = torch.stack(tensors, dim=0)
    video_inputs_with_timestamp.append(tensors)
video_inputs = video_inputs_with_timestamp

mm_data = {}
if image_inputs is not None:
    mm_data["image"] = image_inputs
if video_inputs is not None:
    mm_data["video"] = video_inputs

llm_inputs = {
    "prompt": prompt,
    "multi_modal_data": mm_data,

    # FPS will be returned in video_kwargs
    "mm_processor_kwargs": video_kwargs,
}

outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
generated_text = outputs[0].outputs[0].text

print(generated_text)
