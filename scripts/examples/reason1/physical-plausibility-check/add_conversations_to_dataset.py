#!/usr/bin/env python3
"""Add conversation format to existing datasets.

This script converts datasets with caption/video_url/pc format
to the conversation format required for training.
"""

import argparse
import json
from pathlib import Path

import datasets
import yaml
from cosmos_reason1_utils.text import PromptConfig, create_conversation
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Input dataset directory"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output dataset directory"
    )
    parser.add_argument(
        "--prompt_path", type=str, required=True,
        help="Path to prompt YAML file"
    )
    args = parser.parse_args()

    # Load prompt template
    print(f"📝 Loading prompt from: {args.prompt_path}")
    with open(args.prompt_path, 'r') as f:
        prompt_config = PromptConfig.model_validate(yaml.safe_load(f))
    
    system_prompt = prompt_config.system_prompt
    user_prompt = prompt_config.user_prompt

    # Load existing dataset
    print(f"📂 Loading dataset from: {args.input_dir}")
    dataset = datasets.load_from_disk(args.input_dir)
    print(f"✅ Loaded {len(dataset)} samples")
    print(f"Current features: {list(dataset.features.keys())}")

    # Convert to conversation format
    print("\n🔄 Converting to conversation format...")
    conversations = []
    
    for sample in tqdm(dataset, desc="Processing samples"):
        video_path = sample['video_url']
        pc_score = sample['pc']
        
        # Create conversation
        conversation = create_conversation(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            videos=[video_path],
            response=f"<answer>\n{pc_score}\n</answer>",
        )
        
        conversations.append(json.dumps(conversation))
    
    # Add conversations column to dataset
    dataset = dataset.add_column("conversations", conversations)
    
    print(f"\n✅ Added 'conversations' column")
    print(f"New features: {list(dataset.features.keys())}")
    
    # Save updated dataset
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n💾 Saving to: {output_dir}")
    dataset.save_to_disk(str(output_dir))
    
    print(f"\n✅ Dataset saved successfully!")
    print(f"\nSample conversation:")
    print(json.loads(dataset[0]['conversations']))


if __name__ == "__main__":
    main()


