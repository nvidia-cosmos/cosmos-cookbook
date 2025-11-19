from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import os
import pickle
import argparse

# User prompts dictionary
USER_PROMPTS = {
    "cube": """You should find the following 3 events in the input video
            Event 1: grasping the red cube.
            Event 2: releasing the red cube.
            Event 3: grasping the green cube.
            Extract the exact timestamps for each event.""",
    "cube_bridge": """You should find the following 2 events in the input video
                    Event 1: grasping the cube.
                    Event 2: releasing the cube.
                    Extract the exact timestamps for each event.""",
    "nut": """You should find the following 3 events in the input video
            Event 1: Picking up the red cylinder from the table.
            Event 2: Placing the red cylinder in the blue tray.
            Event 3: Picking up the yellow bowl from the table.
            Extract the exact timestamps for each event.""",
    "toaster": """You should find the following 4 events in the input video
                Event 1: grasping bread.
                Event 2: releasing bread.
                Event 3: pushing the toaster.
                Event 4: releasing the toaster.
                Extract the exact timestamps for each event.""",
    "chips": """You should find the following 2 events in the input video
            Event 1: grasping a bag of chips.
            Event 2: releasing a bag of chips.
            Extract the exact timestamps for each event.""",
    "fork": """You should find the following 4 events in the input video
            Event 1: grasping a fork.
            Event 2: releasing a fork.
            Event 3: grasping a bowl.
            Event 4: releasing a bowl.
            Extract the exact timestamps for each event.""",
    "cup": """You should find the following 3 events in the input video
            Event 1: grasping a cup.
            Event 2: grasping a rag.
            Event 3: releasing a rag.
            Extract the exact timestamps for each event.""",
}

# System prompt (common for all)
SYSTEM_PROMPT = """You are a specialized behavior analyst. Your task is to analyze the video and identify MULTIPLE discrete events with precise timestamps. At each frame, the timestamp is embedded at the bottom of the video. You need to extract the timestamp and answer the user question
                    CRITICAL REQUIREMENTS:
                    1. Extract timestamps from the bottom of each frame
                    2. Extract timestamps for USER-DEFINED events

                    Answer the question in the following format:
                    <think>
                    I will analyze the video systematically:
                    1. First, identify ALL visible timestamps throughout the video
                    2. Identify USER-DEFINED events
                    3. Extract timestamps for identified USER-DEFINED events. There will be different timestamps for each video.
                    4. Always answer in English

                    Event 1: <start time> - <end time> - Event | reasoning
                    Event 2: <start time> - <end time> - Event | reasoning 
                    Event 3: <start time> - <end time> - Event | reasoning

                    [Continue for all events identified]
                    </think>

                    <answer>
                    Event 1: <start time> - <end time> Specific Event | detailed explanation.
                    Event 2: <start time> - <end time> Specific Event | detailed explanation.
                    Event 3: <start time> - <end time> Specific Event | detailed explanation.
                    [Continue for all events identified]
                    </answer>"""


def get_video_files(video_dir, extensions=('.mp4', '.avi', '.mov', '.mkv')):
    """Get all video files from the directory"""
    video_files = []
    for file in os.listdir(video_dir):
        if file.lower().endswith(extensions):
            video_files.append(os.path.join(video_dir, file))
    return sorted(video_files)


def process_video(llm, processor, sampling_params, video_path, fps, user_prompt, total_pixels=6422528):
    """Process a single video and return the generated text"""
    video_messages = [
        {"role": "system", 
        "content": [
            {
                "type": "text",
                "text": SYSTEM_PROMPT
            }
        ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "fps": fps,
                    "total_pixels": total_pixels
                },
                {
                    "type": "text",
                    "text": user_prompt
                }
            ]
        }
    ]
    
    # Process the video
    prompt = processor.apply_chat_template(
        video_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(video_messages, return_video_kwargs=True)

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,
        "mm_processor_kwargs": video_kwargs,
    }

    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text
    
    return generated_text


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run vLLM inference on videos with different prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Demo-based mode:
  python process_video_vllm_uni.py --model_path /path/to/model --prompt cube --output_dir results
  python process_video_vllm_uni.py --model_path /path/to/model --prompt chips --num_trials 5 --output_dir results
  python process_video_vllm_uni.py --model_path /path/to/model --prompt toaster --start_demo 0 --end_demo 5 --output_dir results
  
  # Video directory mode:
  python process_video_vllm_uni.py --model_path /path/to/model --prompt cube --video_dir /path/to/videos --output_dir results
  python process_video_vllm_uni.py --model_path /path/to/model --prompt fork --video_dir /path/to/videos --output_dir results
        """
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )
    parser.add_argument(
        "--prompt",
        choices=list(USER_PROMPTS.keys()),
        required=True,
        help="User prompt type to use"
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Directory containing video files to process (alternative to demo-based mode)"
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=10,
        help="Number of trials per video (default: 10)",
    )
    parser.add_argument(
        "--fps_list",
        type=int,
        nargs='+',
        default=[4, 8, 12],
        help="List of FPS values to test (default: 4 8 12)",
    )
    parser.add_argument(
        "--start_demo",
        type=int,
        default=0,
        help="Starting demo number (default: 0) - only for demo-based mode"
    )
    parser.add_argument(
        "--end_demo",
        type=int,
        default=10,
        help="Ending demo number (exclusive, default: 10) - only for demo-based mode"
    )
    parser.add_argument(
        "--video_path_template",
        type=str,
        default="/mnt/pvc/datasets/videos_bridge_small/02_2023-05-05_10-36-06_traj_group0_traj{demo_num}_images0.mp4",
        help="Video path template with {demo_num} placeholder - only for demo-based mode"
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        required=False,
        help="Maximum model length",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (default: 0.6)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p sampling parameter (default: 0.95)",
    )
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.15,
        help="Repetition penalty (default: 1.15)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate (default: 4096)",
    )
    parser.add_argument(
        "--total_pixels",
        type=int,
        default=6422528,
        help="Total pixels for video processing (default: 6422528)",
    )
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default=None,
        help="CUDA_VISIBLE_DEVICES to set (e.g., '0,1'). If not specified, uses current environment setting.",
    )
    return parser.parse_args()


def main():
    """Main function to process all videos"""
    args = parse_args()
    
    # Set CUDA_VISIBLE_DEVICES if specified
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
        print(f"Set CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    user_prompt = USER_PROMPTS[args.prompt]
    
    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"  Model path: {args.model_path}")
    print(f"  Prompt: {args.prompt}")
    print(f"  FPS list: {args.fps_list}")
    print(f"  Trials per video: {args.num_trials}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Temperature: {args.temperature}")
    print(f"  Top-p: {args.top_p}")
    print(f"  Repetition penalty: {args.repetition_penalty}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Total pixels: {args.total_pixels}")
    if args.max_model_len:
        print(f"  Max model length: {args.max_model_len}")
    
    # Determine which mode to use
    if args.video_dir:
        print(f"  Mode: Video Directory")
        print(f"  Video directory: {args.video_dir}")
        
        # Get all video files from the directory
        video_files = get_video_files(args.video_dir)
        
        if not video_files:
            print(f"\nError: No video files found in {args.video_dir}")
            return
        
        print(f"  Found {len(video_files)} video files")
        videos_to_process = [(os.path.basename(vp).rsplit('.', 1)[0], vp) for vp in video_files]
    else:
        print(f"  Mode: Demo-based")
        print(f"  Demo range: {args.start_demo} to {args.end_demo-1}")
        print(f"  Video template: {args.video_path_template}")
        
        # Generate demo-based video paths
        videos_to_process = []
        for demo_num in range(args.start_demo, args.end_demo):
            video_path = args.video_path_template.format(demo_num=demo_num)
            if os.path.exists(video_path):
                videos_to_process.append((f"demo{demo_num}", video_path))
            else:
                print(f"  Warning: Video not found: {video_path}")
    
    print(f"{'='*60}\n")
    
    # Initialize the model once
    print(f"Loading model from: {args.model_path}")
    if args.max_model_len:
        llm = LLM(
        model=args.model_path,
        limit_mm_per_prompt={"image": 10, "video": 10},
        max_model_len=args.max_model_len if args.max_model_len else None
        )
    else:
        llm = LLM(
        model=args.model_path,
        limit_mm_per_prompt={"image": 10, "video": 10},
        )
        
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_tokens=args.max_tokens,
    )
    
    # Initialize processor once
    processor = AutoProcessor.from_pretrained(args.model_path)
    
    # Process videos
    for video_name, video_path in videos_to_process:
        result_dict = {}
        result_dict["video_path"] = video_path
        result_dict["prompt_type"] = args.prompt
        
        print(f"\nProcessing: {video_name}")
        print(f"  Video: {video_path}")
        
        for fps in args.fps_list:
            print(f"  FPS {fps}:")
            fps_key = f"fps{fps}"
            result_dict[fps_key] = []
            
            for trial_num in range(1, args.num_trials + 1):
                print(f"    Trial {trial_num}/{args.num_trials}")
                
                try:
                    # Process the video
                    output_text = process_video(
                        llm, processor, sampling_params, 
                        video_path, fps, user_prompt, args.total_pixels
                    )
                    
                    result_dict[fps_key].append(output_text)
                    
                    # Save result_dict after each trial
                    output_file = os.path.join(args.output_dir, f'results_{video_name}.pkl')
                    with open(output_file, 'wb') as f:
                        pickle.dump(result_dict, f)
                    
                    print(f"      Saved to: {output_file}")
                    
                except Exception as e:
                    print(f"      Error in trial {trial_num}: {str(e)}")
                    continue
    
    print("\nAll processing completed!")


if __name__ == "__main__":
    main()

