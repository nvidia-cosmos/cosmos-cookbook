import os
import time
import urllib.parse
import pickle
import argparse
from pathlib import Path
from openai import OpenAI

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


def get_video_files(video_dir, extensions=('.mp4', '.avi', '.mov', '.mkv', '.webm')):
    """Get all video files from the directory"""
    video_files = []
    for file in os.listdir(video_dir):
        if file.lower().endswith(extensions):
            video_files.append(os.path.join(video_dir, file))
    return sorted(video_files)


def process_video(client, video_path, model_name, max_tokens, user_prompt):
    """Process a single video using OpenAI API"""
    # Create file:// URI for the video
    video_uri = "file://" + urllib.parse.quote(str(video_path), safe="/:")
    
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "video_url", "video_url": {"url": video_uri}}
            ],
        }
    ]
    
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_tokens
    )
    latency = time.time() - t0
    
    return resp.choices[0].message.content, latency


def main():
    """Main function to process videos using OpenAI API"""
    parser = argparse.ArgumentParser(
        description="Process videos using OpenAI-compatible API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Demo-based mode:
  python process_video_openai_api.py --model Qwen/Qwen3-VL-235B-A22B-Instruct --prompt cube --output_dir results
  python process_video_openai_api.py --model nvidia/Cosmos-Reason2-30B-A3B-v1.0 --prompt chips --num_trials 5 --output_dir results
  python process_video_openai_api.py --model Qwen/Qwen3-VL-8B-Instruct --prompt pour --start_demo 0 --end_demo 5 --output_dir results
  
  # Video directory mode:
  python process_video_openai_api.py --model Qwen/Qwen3-VL-235B-A22B-Instruct --prompt cube --video_dir /path/to/videos --output_dir results
  python process_video_openai_api.py --model nvidia/Cosmos-Reason2-30B-A3B-v1.0 --prompt chips --video_dir /path/to/videos --output_dir results
        """
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results"
    )
    parser.add_argument(
        "--model", 
        type=str,
        required=True,
        help="Model name to use (e.g., Qwen/Qwen3-VL-235B-A22B-Instruct)"
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
        help="Number of trials per video (default: 10)"
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
        "--api_base",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for OpenAI-compatible API (default: http://localhost:8000/v1)"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default="EMPTY",
        help="API key for the service (default: EMPTY)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Maximum tokens for completion (default: 8192)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="API request timeout in seconds (default: 3600)"
    )
    
    args = parser.parse_args()
    
    # Initialize OpenAI client
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base,
        timeout=args.timeout
    )
    
    user_prompt = USER_PROMPTS[args.prompt]
    
    print(f"\n{'='*60}")
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  API Base: {args.api_base}")
    print(f"  Prompt: {args.prompt}")
    print(f"  Trials per video: {args.num_trials}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Output directory: {args.output_dir}")
    
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
    
    # Process videos
    for video_name, video_path in videos_to_process:
        result_dict = {}
        
        print(f"\nProcessing: {video_name}")
        print(f"  Video: {video_path}")
        
        for fps in [8]:
            print(f"  FPS {fps}:")
            result_dict["fps"+str(fps)] = []
            for trial_num in range(args.num_trials):
                print(f"  Trial {trial_num+1}/{args.num_trials}", end="")
                
                try:
                    # Process the video
                    output_text, latency = process_video(
                        client, video_path, args.model, args.max_tokens, user_prompt
                    )
                    
                    result_dict["fps"+str(fps)].append(output_text)
                    result_dict["video_path"] = video_path
                    result_dict["model"] = args.model
                    result_dict["prompt_type"] = args.prompt
                    
                    # Save result_dict after each trial
                    os.makedirs(args.output_dir, exist_ok=True)
                    output_file = os.path.join(args.output_dir, f'results_{video_name}.pkl')
                    with open(output_file, 'wb') as f:
                        pickle.dump(result_dict, f)
                    
                    print(f" - Latency: {latency:.2f}s - Saved to: {output_file}")
                    
                except Exception as e:
                    print(f" - Error: {str(e)}")
                    continue
    
    print("\nAll processing completed!")


if __name__ == "__main__":
    main()

