import base64
import logging
import os
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from qwen_vl_utils import fetch_video  # type: ignore

DEFAULT_FPS = 4
MAX_VIDEO_DURATION = 5  # Maximum video duration in seconds


class MediaProcessor:
    """Handles both image and video processing operations"""

    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        self.frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)

    def process_image(self, image_path: str) -> str:
        """Process a single image file"""
        try:
            # Validate image file
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Convert to RGB
            img = Image.open(image_path).convert("RGB")

            # encode image with base64
            output_buffer = BytesIO()
            img.save(output_buffer, format="png")
            byte_data = output_buffer.getvalue()
            base64_str = base64.b64encode(byte_data).decode("utf-8")
            base64_str = f"data:image/png;base64,{base64_str}"

            return base64_str
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            raise e

    def process_video(self, video_path: str) -> str:
        """
        Process a video file and return a list of base64 encoded frames
        Follow Qwen official repo steps: https://github.com/QwenLM/Qwen2.5-VL/blob/main/README.md?plain=1#L800,
        except using png format over jpeg to avoid compression artifacts.
        """
        try:
            # initialize temp_video_path为None
            temp_video_path = None

            # Get video information
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # Calculate required number of frames
            max_frames = int(MAX_VIDEO_DURATION * fps)
            if total_frames > max_frames:
                logging.info(
                    f"Video is longer than {MAX_VIDEO_DURATION} seconds."
                    f"Truncating to first {MAX_VIDEO_DURATION} seconds."
                )
                # Create temporary file
                temp_video_path = os.path.join(self.temp_dir, "temp_cropped.mp4")
                # Use ffmpeg to crop video
                os.system(f"ffmpeg -y -i {video_path} -t {MAX_VIDEO_DURATION} -c copy {temp_video_path}")
                video_path = temp_video_path

            video_data = fetch_video({"video": video_path, "fps": DEFAULT_FPS})  # TCHW, float32, [0, 255]
            video_data = video_data.permute(0, 2, 3, 1).numpy().astype(np.uint8)  # THWC, uint8, [0, 255]

            # encode image with base64
            base64_frames = []
            for frame in video_data:
                img = Image.fromarray(frame)
                output_buffer = BytesIO()
                img.save(output_buffer, format="jpeg")  # use png to avoid compression artifacts
                byte_data = output_buffer.getvalue()
                base64_str = base64.b64encode(byte_data).decode("utf-8")
                base64_frames.append(base64_str)
            base64_frames = f"data:video/jpeg;base64,{','.join(base64_frames)}"

            # Clean up temporary file
            if temp_video_path and os.path.exists(temp_video_path):
                os.remove(temp_video_path)

        except Exception as e:
            logging.error(f"Error processing video: {e}")
            raise e
        return base64_frames
