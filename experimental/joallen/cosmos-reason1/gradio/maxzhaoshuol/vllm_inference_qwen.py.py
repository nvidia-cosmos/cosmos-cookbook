#!/usr/bin/env python3
"""
Standalone inference module for warehouse video analysis.
Independent of Streamlit and other UI frameworks.
"""

import base64
import os
from io import BytesIO

import numpy as np
from openai import OpenAI
from PIL import Image

# Import vision processing utilities
from projects.cosmos.reason1.utils.reason1_vision_process import process_vision_info

# Model configuration
MODEL_TO_IP_DICT = {
    "cosmos-reason1.1": "https://b5k2m9x7-cosmos-reason1p1-h200.xenon.lepton.run/v1",
    "qwen2.5-vl-7b": "https://b5k2m9x7-qwen2p5-vl-7b-instruct.xenon.lepton.run/v1",
}

MODEL_TO_API_KEY_DICT = {
    "cosmos-reason1.1": "mhjNXXvPbCcyHWhWhhzqDRQOnmrwW73S",
    "qwen2.5-vl-7b": "nc0fvvb4gf2sbTp93MaPfovmi6BMHzmz",
}

def get_model_name_for_api(selected_model):
    """Get the appropriate model name for API calls based on selected model"""
    if "cosmos" in selected_model.lower():
        return "nvidia/Cosmos-Reason1-7B"
    elif "qwen2.5-vl-7b" in selected_model.lower():
        return "Qwen/Qwen2.5-VL-7B-Instruct"
    elif "internlm" in selected_model.lower():
        return "internlm/Intern-S1-FP8"
    else:
        # Default fallback
        return "nvidia/Cosmos-Reason1-7B"

def prepare_message_for_vllm(content_messages, max_num_vision_tokens=None, timestamp_video=None, fps_value=2):
    """
    The frame extraction logic for videos in `vLLM` differs from that of `qwen_vl_utils`.
    Here, we utilize `qwen_vl_utils` to extract video frames, with the `media_type` of the video explicitly set to `video/jpeg`.
    By doing so, vLLM will no longer attempt to extract frames from the input base64-encoded images.
    """
    vllm_messages, fps_list = [], []
    for message in content_messages:
        message_content_list = message["content"]
        if not isinstance(message_content_list, list):
            vllm_messages.append(message)
            continue

        new_content_list = []
        for part_message in message_content_list:
            if 'video' in part_message:
                part_message['fps'] = fps_value
                video_message = [{'content': [part_message]}]
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    video_message, 
                    return_video_kwargs=True,
                    max_num_vision_tokens=max_num_vision_tokens,
                    timestamp_video=timestamp_video or False
                )
                assert video_inputs is not None, "video_inputs should not be None"
                
                # Handle different types returned by process_vision_info
                video_data = video_inputs.pop()
                try:
                    # If it's a tensor
                    video_input = video_data.permute(0, 2, 3, 1).numpy().astype(np.uint8)  # type: ignore
                except AttributeError:
                    # If it's already a list of PIL Images or numpy arrays
                    if isinstance(video_data, list):
                        video_input = [np.array(img) for img in video_data]
                    else:
                        video_input = [np.array(video_data)]
                
                fps_list.extend(video_kwargs.get('fps', []))

                # encode image with base64
                base64_frames = []
                for frame in video_input:
                    img = Image.fromarray(frame)
                    output_buffer = BytesIO()
                    img.save(output_buffer, format="jpeg")
                    byte_data = output_buffer.getvalue()
                    base64_str = base64.b64encode(byte_data).decode("utf-8")
                    base64_frames.append(base64_str)

                part_message = {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/jpeg;base64,{','.join(base64_frames)}"}
                }
            elif 'image' in part_message:
                # Process image using vision processing
                image_message = [{'content': [part_message]}]
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    image_message, 
                    return_video_kwargs=True,
                    max_num_vision_tokens=max_num_vision_tokens,
                    timestamp_video=timestamp_video or False
                )
                
                if image_inputs is not None and len(image_inputs) > 0:
                    # Process the image input
                    image_data = image_inputs[0]  # Get first image
                    
                    # Convert to PIL Image if it's not already
                    if hasattr(image_data, 'convert'):
                        img = image_data
                    else:
                        img = Image.fromarray(np.array(image_data))
                    
                    # Encode as base64
                    output_buffer = BytesIO()
                    img.save(output_buffer, format="jpeg")
                    byte_data = output_buffer.getvalue()
                    base64_str = base64.b64encode(byte_data).decode("utf-8")
                    
                    part_message = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"}
                    }
                else:
                    # Fallback: process image directly from file path
                    image_path = part_message.get('image', '')
                    if image_path and os.path.exists(image_path):
                        img = Image.open(image_path).convert('RGB')
                        output_buffer = BytesIO()
                        img.save(output_buffer, format="jpeg")
                        byte_data = output_buffer.getvalue()
                        base64_str = base64.b64encode(byte_data).decode("utf-8")
                        
                        part_message = {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_str}"}
                        }
            
            new_content_list.append(part_message)
        message["content"] = new_content_list
        vllm_messages.append(message)
    return vllm_messages, {'fps': fps_list}

def inference(
    media_path, 
    user_prompt, 
    system_prompt, 
    model_name="cosmos-reason1.1",
    media_type="video", 
    max_num_vision_tokens=None, 
    timestamp_video=False, 
    fps_value=4, 
    temperature=0.01, 
    seed=None
):
    """
    Run inference on media (video or image) using the specified model.
    
    Args:
        media_path (str): Path to the media file
        user_prompt (str): User prompt for the model
        system_prompt (str): System prompt for the model
        model_name (str): Name of the model to use
        media_type (str): Type of media ("video" or "image")
        max_num_vision_tokens (int, optional): Maximum number of vision tokens
        timestamp_video (bool): Whether to add timestamps to video
        fps_value (int): FPS for video processing
        temperature (float): Temperature for sampling
        seed (int, optional): Random seed for reproducibility
    
    Returns:
        str: Model response
    """
    # Validate model
    if model_name not in MODEL_TO_IP_DICT:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_TO_IP_DICT.keys())}")
    
    # Prepare media content based on type
    if media_type == "video":
        media_content = [{"type": "video", "video": media_path}]
    else:  # image
        media_content = [{"type": "image", "image": media_path}]
    
    # Add text prompt
    media_content.append({"type": "text", "text": user_prompt})
    
    # Prepare messages in Qwen format
    content_messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user", 
            "content": media_content
        }
    ]
    
    # Convert to vLLM format using official Qwen function
    vllm_messages, video_kwargs = prepare_message_for_vllm(
        content_messages,
        max_num_vision_tokens=max_num_vision_tokens,
        timestamp_video=timestamp_video,
        fps_value=fps_value,
    )
    
    # Get model configuration
    selected_port = MODEL_TO_IP_DICT[model_name]
    selected_api_key = MODEL_TO_API_KEY_DICT[model_name]
    api_model_name = get_model_name_for_api(model_name)

    openai_api_base = f"{selected_port}"

    client = OpenAI(
        api_key=selected_api_key,
        base_url=openai_api_base,
    )

    # Prepare API call parameters
    api_params = {
        "model": api_model_name,
        "messages": vllm_messages,
        "temperature": temperature,
        "extra_body": {
            "mm_processor_kwargs": video_kwargs
        }
    }
    
    # Add seed if provided
    if seed is not None:
        api_params["seed"] = seed
    
    # Make API call
    chat_response = client.chat.completions.create(**api_params)
    
    response = chat_response.choices[0].message.content
    if response is None:
        raise RuntimeError("No response received from the model")
    
    return response

def list_available_models():
    """Return list of available models."""
    return list(MODEL_TO_IP_DICT.keys())

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python standalone_inference.py <video_path> <user_prompt> <system_prompt> [model_name]")
        print(f"Available models: {list_available_models()}")
        sys.exit(1)
    
    video_path = sys.argv[1]
    user_prompt = sys.argv[2]
    system_prompt = sys.argv[3]
    model_name = sys.argv[4] if len(sys.argv) > 4 else "cosmos-reason1.1"
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        sys.exit(1)
    
    print(f"Running inference on {video_path} with model {model_name}...")
    
    try:
        response = inference(
            media_path=video_path,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            model_name=model_name,
            temperature=0.01,
            seed=42
        )
        print(f"Response: {response}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)