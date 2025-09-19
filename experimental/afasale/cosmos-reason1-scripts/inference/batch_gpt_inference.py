"""Batch GPT inference script to process JSON files with captions and generate reasoning traces.

Example:
```shell
./scripts/batch_gpt_inference.py --input-dir ./outputs/cam_front/test_03/ --output-dir ./reasoning_outputs/
```
"""

from transformers import pipeline
import torch
import json
import argparse
import os
from pathlib import Path

def get_json_files(directory):
    """Get all JSON files from a directory recursively."""
    json_files = []
    for file_path in Path(directory).rglob('*.json'):
        if file_path.is_file():
            json_files.append(str(file_path))
    return sorted(json_files)


def process_caption_with_gpt(caption, pipe):
    """Process a single caption with GPT to generate reasoning trace."""
    system_prompt = "You are a precise, grounded reasoning assistant trained on visual-language tasks, especially car dashcam video analysis. Your goal is to convert a dense, grounded caption describing a dashcam video into a clear, step-by-step thinking trace (Chain of Thought). You must reason only from what is visible in the video."
    
    prompt = f"""Given the following dense caption from a car dashcam video, break it into a 4-step reasoning trace: \
    Caption: \
    <{caption}> \
    Step 1: <Identify key visible objects and infrastructure.> \
    Step 2: <Describe the spatial relationships and observed actions.> \
    Step 3: <Infer grounded purpose or function of the scene (no speculation).> \
    Step 4: <Summarize into a dense caption.> \
    
    Provide only the 4 reasoning steps."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    
    outputs = pipe(
        messages,
        max_new_tokens=4096,
    )
    
    return outputs[0]["generated_text"][-1]["content"]


def main():
    parser = argparse.ArgumentParser(description="Process JSON files with captions through GPT for reasoning traces")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing JSON files to process")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save GPT reasoning trace outputs")
    parser.add_argument("--model", type=str, default="openai/gpt-oss-20b", help="GPT model to use")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Get all JSON files
    json_files = get_json_files(args.input_dir)
    if not json_files:
        print(f"[red]Error: No JSON files found in {args.input_dir}[/red]")
        return
    
    print(f"Found {len(json_files)} JSON files to process")

    # Initialize GPT pipeline
    print(f"Loading model: {args.model}")
    pipe = pipeline(
        "text-generation",
        model=args.model,
        torch_dtype="auto",
        device_map="auto",
    )
    print("Model loaded successfully!")

    # Process each JSON file
    for i, json_path in enumerate(json_files, 1):
        print(f"\n[{i}/{len(json_files)}] Processing: {json_path}")
        
        try:
            # Load JSON file
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract caption from response field
            if 'response' not in data:
                print(f"[yellow]Warning: No 'response' field found in {json_path}, skipping[/yellow]")
                continue
                
            caption = data['response']
            print(f"Caption preview: {caption[:100]}...")
            
            # Process with GPT
            reasoning_trace = process_caption_with_gpt(caption, pipe)
            
            # Create output file path
            json_filename = Path(json_path).stem
            output_path = output_dir / f"{json_filename}_reasoning.json"
            
            # Create output data
            output_data = {
                "original_file": json_path,
                "video": data.get("video", ""),
                "original_prompt": data.get("prompt", ""),
                "original_response": caption,
                "gpt_reasoning_trace": reasoning_trace,
                "gpt_model": args.model
            }
            
            # Save reasoning trace
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"[green]✓ Saved reasoning trace to: {output_path}[/green]")
            print(f"Reasoning preview: {reasoning_trace[:200]}...")
            
        except Exception as e:
            print(f"[red]✗ Error processing {json_path}: {str(e)}[/red]")
            continue

    print(f"\n[green]Completed processing {len(json_files)} JSON files![/green]")


if __name__ == "__main__":
    main()
