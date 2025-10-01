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
Data preprocessing script for Spatial AI Warehouse dataset.

This script converts warehouse annotation data to LLaVA format and includes random subsampling functionality.
It processes the original train.json format and converts it to be compatible with custom_sft.py.
The script now includes SOM prompt functionality to add additional context during data loading.

Usage:
    python data_preprocess.py --input_file /path/to/train.json --output_file /path/to/output/train_llava.json --samples_per_category 20000
    python data_preprocess.py --input_file /path/to/train.json --output_file /path/to/output/train_llava.json --som_prompt "Custom SOM prompt text"
"""

import json
import os
import random
import re
import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator
import numpy as np
import matplotlib.pyplot as plt

# Import the prompt templates from dataset.py
QUESTION_TYPE_TO_ANSWER_FORMAT_PROMPT_NAIVE = {
    "left_right": '{query}\nAnswer with "left" or "right". Do not include any explanation or extra text.',
    "mcq": ("{query}\nPlease answer with only the integer number of the correct region"
            " the number should be one that is both shown in the image and mentioned in this question."
            " Do not include any explanation or extra text."),
    "count": '{query}\nRespond with only a integer number. Do not include any explanation or extra text.',
    "distance": '{query}\nRespond with only a floating point number. Do not include any explanation or extra text.',
}

# SOM prompt for additional context during data loading
DEFAULT_SOM_PROMPT = "The first image is the original, and the second is an overlay. Bright numeric IDs are labeled at the center of certain visual objects in the second image."

# Using argparse for command line argument parsing

# Try to import ijson for streaming, fall back to regular json if not available
try:
    import ijson
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False
    print("Warning: ijson not available. Using regular JSON loading (may cause memory issues with large files)")

def load_json_streaming(file_path: str) -> Iterator[Dict[str, Any]]:
    """Load JSON data in streaming fashion to avoid memory issues."""
    print(f"Loading data from {file_path} in streaming mode...")

    if HAS_IJSON:
        with open(file_path, 'rb') as file:
            # Parse JSON array items one by one
            parser = ijson.items(file, 'item')
            count = 0
            for item in parser:
                count += 1
                if count % 100000 == 0:
                    print(f"  Processed {count} entries...")
                yield item
            print(f"Total entries processed: {count}")
    else:
        # Fallback to regular JSON loading
        print("Warning: Using regular JSON loading - this may cause memory issues with large files")
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} entries")
        for item in data:
            yield item


def replace_masks_with_regions(text: str) -> str:
    """
    Replace <mask> tokens with Region [idx] format.

    Args:
        text: Input text containing <mask> tokens

    Returns:
        Text with <mask> replaced by Region [idx]
    """
    # Find all <mask> tokens and replace them with Region [idx]
    mask_count = 0
    def replace_mask(_match):
        nonlocal mask_count
        result = f"Region [{mask_count}]"
        mask_count += 1
        return result

    return re.sub(r'<mask>', replace_mask, text)


def update_image_text(text):
    """Remove image tags from text."""
    return (
        text
        .replace("<image>\n", "")
        .replace("<image>", "")
    )


def adjust_user_query_for_som(origin_content: str, som_prompt: str = DEFAULT_SOM_PROMPT) -> str:
    """
    Adjust user query by adding SOM prompt context.

    Args:
        origin_content: Original query content
        som_prompt: SOM prompt to prepend to the query

    Returns:
        Adjusted query with SOM prompt
    """
    # Remove existing image tags
    origin_content = origin_content.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
    origin_content = origin_content.replace("<video>\n", "").replace("\n<video>", "").replace("<video>", "")

    # Prepend SOM prompt
    adjusted_content = som_prompt + "\n" + origin_content
    return adjusted_content




def convert_to_llava_format(entry: Dict[str, Any], replace_masks: bool = True, is_train: bool = False, som_prompt: str = DEFAULT_SOM_PROMPT) -> Dict[str, Any]:
    """
    Convert a warehouse annotation entry to LLaVA format.

    Args:
        entry: Original warehouse annotation entry
        replace_masks: Whether to replace <mask> with Region [idx]
        is_train: Whether this is training data (affects image paths)
        som_prompt: SOM prompt to prepend to the query

    Returns:
        Entry converted to LLaVA format
    """
    # Extract basic information
    entry_id = entry.get("id", "")
    image_name = entry.get("image", "")
    conversations = entry.get("conversations", [])
    category = entry.get("category", "unknown")
    normalized_answer = entry.get("normalized_answer", "")

    # Create proper image paths
    if is_train:
        original_image_path = f"train/images/{image_name}"
        som_image_path = f"train/som_images/{Path(image_name).stem}.{entry_id}.png"
    else:
        original_image_path = f"val/images/{image_name}"
        som_image_path = f"val/som_images/{Path(image_name).stem}.{entry_id}.png"

    # Process conversations with proper formatting
    processed_conversations = []
    for conv in conversations:
        if conv["from"] == "human":
            # Apply QUESTION_TYPE_TO_ANSWER_FORMAT_PROMPT_NAIVE formatting
            query = conv["value"]
            if replace_masks:
                query = replace_masks_with_regions(query)

            # Remove <image> tags and add proper formatting
            query = update_image_text(query)

            # Apply SOM prompt adjustment
            query = adjust_user_query_for_som(query, som_prompt)

            # Apply the appropriate prompt template based on category
            if category in QUESTION_TYPE_TO_ANSWER_FORMAT_PROMPT_NAIVE:
                prompt_template = QUESTION_TYPE_TO_ANSWER_FORMAT_PROMPT_NAIVE[category]
                formatted_query = prompt_template.format(query=query)
            else:
                formatted_query = query

            processed_conv = {
                "from": "human",
                "value": "<image>\n" + formatted_query
            }
        elif conv["from"] == "gpt":
            # Use normalized_answer as the value for gpt, ensuring it's a string
            processed_conv = {
                "from": "gpt",
                "value": str(normalized_answer)
            }
        else:
            raise ValueError(f"Invalid from: {conv['from']}")

        processed_conversations.append(processed_conv)

    # Create LLaVA format entry
    llava_entry = {
        "id": entry_id,
        "images": [original_image_path, som_image_path],  # Both original and SOM images
        "conversations": processed_conversations,
        "category": category,
        "normalized_answer": str(normalized_answer),
    }

    return llava_entry


def analyze_data_distribution_streaming(file_path: str, target_categories: Optional[List[str]] = None) -> Dict[str, Any]:
    """Analyze the distribution of categories and normalized answers using streaming."""
    print("Analyzing data distribution in streaming mode...")

    category_counts = defaultdict(int)
    category_answer_ranges = defaultdict(list)
    total_entries = 0

    for entry in load_json_streaming(file_path):
        total_entries += 1

        # Filter by target categories if specified
        if target_categories is not None:
            category = entry.get('category', 'unknown')
            if category not in target_categories:
                continue

        category = entry.get('category', 'unknown')
        normalized_answer = entry.get('normalized_answer')

        category_counts[category] += 1

        if normalized_answer is not None:
            try:
                # Convert to float if possible
                answer_val = float(normalized_answer)
                category_answer_ranges[category].append(answer_val)
            except (ValueError, TypeError):
                pass

    print(f"Analyzed {total_entries} total entries")

    # Calculate statistics for each category
    category_stats = {}
    for category, answers in category_answer_ranges.items():
        if answers:
            category_stats[category] = {
                'count': category_counts[category],
                'answer_count': len(answers),
                'min_answer': min(answers),
                'max_answer': max(answers),
                'mean_answer': np.mean(answers),
                'std_answer': np.std(answers)
            }
        else:
            category_stats[category] = {
                'count': category_counts[category],
                'answer_count': 0,
                'min_answer': None,
                'max_answer': None,
                'mean_answer': None,
                'std_answer': None
            }

    return category_stats


def collect_entries_by_category_streaming(
    file_path: str,
    target_categories: Optional[List[str]] = None,
    max_entries_per_category: Optional[int] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """Collect entries by category using streaming to avoid memory issues."""
    print("Collecting entries by category in streaming mode...")

    category_entries = defaultdict(list)
    category_counts = defaultdict(int)
    total_processed = 0

    for entry in load_json_streaming(file_path):
        total_processed += 1

        category = entry.get('category', 'unknown')

        # Filter by target categories if specified
        if target_categories is not None and category not in target_categories:
            continue

        # Stop collecting if we have enough entries for this category
        if max_entries_per_category is not None and category_counts[category] >= max_entries_per_category:
            continue

        category_entries[category].append(entry)
        category_counts[category] += 1

        if total_processed % 100000 == 0:
            print(f"  Processed {total_processed} entries, collected: {dict(category_counts)}")

    print(f"Collection complete. Total processed: {total_processed}")
    print(f"Final category counts: {dict(category_counts)}")

    return dict(category_entries)


def sample_randomly(
    entries: List[Dict[str, Any]],
    target_size: int
) -> List[Dict[str, Any]]:
    """Sample entries randomly from the given list."""
    if len(entries) <= target_size:
        return entries.copy()

    return random.sample(entries, target_size)


def plot_distributions(data: List[Dict], output_file: str):
    """
    Plot distributions of categories and normalized answers.

    Args:
        data: List of data entries
        output_file: Full path to the output file (used to determine output directory and base filename)
    """
    output_dir = os.path.dirname(output_file)
    base_filename = os.path.splitext(os.path.basename(output_file))[0]
    # Prepare data for plotting
    category_counts = defaultdict(int)
    category_answers = defaultdict(list)

    for entry in data:
        if 'category' in entry and 'normalized_answer' in entry:
            category = entry['category']
            category_counts[category] += 1

            # Collect normalized answers (can be string or number)
            answer = entry['normalized_answer']
            category_answers[category].append(answer)

    # Plot 1: Category distribution
    categories = sorted(category_counts.keys())

    _, ax1 = plt.subplots(figsize=(12, 6))
    counts = [category_counts[c] for c in categories]
    bars = ax1.bar(categories, counts, alpha=0.7, color='skyblue')
    ax1.set_xticklabels(categories, rotation=45, ha='right', fontsize=10)
    ax1.set_title(f'Category Distribution - {base_filename}', fontsize=14)
    ax1.set_xlabel('Category', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Add count labels on bars
    for bar, count in zip(bars, counts, strict=False):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts)*0.01,
                str(count), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'category_distribution_{base_filename}.png'), dpi=300, bbox_inches='tight')
    print(f"Category distribution plot saved to: {os.path.join(output_dir, f'category_distribution_{base_filename}.png')}")
    plt.close()

    # Plot 2: Normalized answer distributions per category
    num_categories = len(category_answers)
    if num_categories > 0:
        # Use dynamic grid based on number of categories
        n_cols = min(3, num_categories)
        n_rows = (num_categories + n_cols - 1) // n_cols

        _, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()

        for idx, (category, answers) in enumerate(sorted(category_answers.items())):
            if idx < len(axes):
                ax = axes[idx]
                # Check if answers are numerical
                try:
                    # Try to convert answers to float to check if numerical
                    numerical_answers = [float(ans) for ans in answers]
                    is_numerical = True
                except (ValueError, TypeError):
                    is_numerical = False

                if is_numerical:
                    # Plot histogram for numerical data
                    ax.hist(numerical_answers, bins=10, alpha=0.7, color='skyblue')
                    ax.set_title(f'{category} (Numerical)', fontsize=12)
                    ax.set_xlabel('Normalized Answer', fontsize=10)
                    ax.set_ylabel('Count', fontsize=10)

                    # Add statistics
                    mean_val = np.mean(numerical_answers)
                    std_val = np.std(numerical_answers)
                    ax.text(0.95, 0.95,
                           f'Mean: {mean_val:.2f}\nStd: {std_val:.2f}\nTotal: {len(answers)}',
                           transform=ax.transAxes,
                           verticalalignment='top',
                           horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           fontsize=10)
                else:
                    # Plot bar chart for categorical data
                    answer_counts = defaultdict(int)
                    for ans in answers:
                        answer_counts[ans] += 1

                    # Sort by count for better visualization
                    sorted_answers = sorted(answer_counts.items(), key=lambda x: x[1], reverse=True)
                    labels = [ans for ans, _ in sorted_answers]
                    counts = [count for _, count in sorted_answers]

                    bars = ax.bar(range(len(labels)), counts, alpha=0.7, color='skyblue')
                    ax.set_xticks(range(len(labels)))
                    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
                    ax.set_title(f'{category} (Categorical)', fontsize=12)
                    ax.set_xlabel('Answer', fontsize=10)
                    ax.set_ylabel('Count', fontsize=10)

                    # Add total count
                    ax.text(0.95, 0.95,
                           f'Total: {len(answers)}',
                           transform=ax.transAxes,
                           verticalalignment='top',
                           horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           fontsize=10)

                ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(num_categories, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f'Normalized Answer Distribution by Category - {base_filename}', fontsize=14, y=0.95)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'answer_distribution_{base_filename}.png'), dpi=300, bbox_inches='tight')
        print(f"Answer distribution plot saved to: {os.path.join(output_dir, f'answer_distribution_{base_filename}.png')}")
        plt.close()

    # Print category distribution summary
    print("\nCategory Distribution Summary:")
    for category in categories:
        total = category_counts[category]
        print(f"\n{category}: {total}")

        # Print answer statistics if available
        if category in category_answers:
            answers = category_answers[category]
            print("  Answer Statistics:")
            try:
                # Try to convert answers to float to check if numerical
                numerical_answers = [float(ans) for ans in answers]
                print(f"    Mean: {np.mean(numerical_answers):.2f}")
                print(f"    Std: {np.std(numerical_answers):.2f}")
                print(f"    Min: {np.min(numerical_answers):.2f}")
                print(f"    Max: {np.max(numerical_answers):.2f}")
                print(f"    Total numerical answers: {len(answers)}")
            except (ValueError, TypeError):
                # For categorical answers, show value counts
                answer_counts = defaultdict(int)
                for ans in answers:
                    answer_counts[ans] += 1
                print("    Categorical answers:")
                for ans, count in sorted(answer_counts.items(), key=lambda x: (-x[1], x[0])):
                    percentage = (count / len(answers) * 100)
                    print(f"      {ans}: {count} ({percentage:.2f}%)")
                print(f"    Total categorical answers: {len(answers)}")


def save_annotations_to_jsonl(annotations: List[Dict[str, Any]], output_file: str) -> None:
    """Save annotations to JSONL format."""
    with open(output_file, "w") as f:
        for annotation in annotations:
            f.write(json.dumps(annotation) + "\n")


def preprocess_warehouse_data(
    input_file: str,
    output_file: str,
    samples_per_category: int = 20000,
    random_seed: int = 42,
    target_categories: Optional[List[str]] = None,
    generate_plots: bool = True,
    replace_masks: bool = True,
    no_sampling: bool = False,
    output_format: str = "json",
    som_prompt: str = DEFAULT_SOM_PROMPT
) -> None:
    """Main function to preprocess warehouse data with optional sampling."""

    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Create output directory
    output_dir = os.path.dirname(output_file)
    if output_dir:  # Only create directory if there's a directory path
        os.makedirs(output_dir, exist_ok=True)

    # Determine if this is a validation file (no sampling)
    is_val_file = "val" in input_file.lower() or no_sampling
    is_train = not is_val_file

    if is_val_file:
        print("Processing validation file - no sampling will be applied")
        samples_per_category = None

    # First pass: Analyze data distribution using streaming
    category_stats = analyze_data_distribution_streaming(input_file, target_categories)

    print("\nCategory statistics:")
    for category, stats in category_stats.items():
        print(f"\n{category}:")
        print(f"  Total entries: {stats['count']}")
        print(f"  Entries with answers: {stats['answer_count']}")
        if stats['min_answer'] is not None:
            print(f"  Answer range: {stats['min_answer']:.3f} - {stats['max_answer']:.3f}")
            print(f"  Answer mean: {stats['mean_answer']:.3f} ± {stats['std_answer']:.3f}")

    # Filter categories if specified
    if target_categories is not None:
        available_categories = set(category_stats.keys())
        target_categories_set = set(target_categories)
        valid_categories = available_categories.intersection(target_categories_set)

        if not valid_categories:
            raise ValueError(f"None of the target categories {target_categories} found in data")

        print(f"\nFiltering to categories: {sorted(valid_categories)}")
        target_categories = list(valid_categories)

    # Second pass: Collect entries by category using streaming
    if is_val_file:
        # For validation files, collect all entries without sampling
        max_entries_per_category = None
    else:
        # For training files, collect extra for sampling
        max_entries_per_category = samples_per_category * 2

    category_entries = collect_entries_by_category_streaming(
        input_file,
        target_categories,
        max_entries_per_category
    )

    # Process entries and convert to LLaVA format
    all_processed_entries = []

    if is_val_file:
        print("\nProcessing all entries without sampling...")
    else:
        print(f"\nProcessing and sampling {samples_per_category} entries per category using random strategy...")

    for category, entries in category_entries.items():
        print(f"\nProcessing category: {category}")
        print(f"  Available entries: {len(entries)}")

        if is_val_file:
            # Process all entries for validation
            processed_entries = []
            for entry in entries:
                llava_entry = convert_to_llava_format(entry, replace_masks, is_train, som_prompt)
                processed_entries.append(llava_entry)
            print(f"  Processed entries: {len(processed_entries)}")
        else:
            # Sample for training
            if len(entries) < samples_per_category:
                print(f"  Warning: Only {len(entries)} entries available, processing all")
                sampled_entries = entries
            else:
                # Sample randomly
                sampled_entries = sample_randomly(entries, samples_per_category)

            print(f"  Sampled entries: {len(sampled_entries)}")

            # Convert to LLaVA format
            processed_entries = []
            for entry in sampled_entries:
                llava_entry = convert_to_llava_format(entry, replace_masks, is_train, som_prompt)
                processed_entries.append(llava_entry)

            print(f"  Processed entries: {len(processed_entries)}")

        all_processed_entries.extend(processed_entries)

    # Shuffle the final dataset
    random.shuffle(all_processed_entries)

    # Save the processed data to the exact output_file path
    print(f"\nSaving {len(all_processed_entries)} entries to {output_file}")

    if output_format.lower() == "jsonl":
        save_annotations_to_jsonl(all_processed_entries, output_file)
    else:
        with open(output_file, 'w') as f:
            json.dump(all_processed_entries, f, indent=2)

    # Print final statistics
    print("\nFinal processing statistics:")
    final_category_counts = defaultdict(int)
    for entry in all_processed_entries:
        final_category_counts[entry.get('category', 'unknown')] += 1

    for category, count in sorted(final_category_counts.items()):
        print(f"  {category}: {count}")

    print(f"\nTotal processed entries: {len(all_processed_entries)}")
    if is_val_file:
        print("Processing mode: validation (no sampling)")
    else:
        print("Sampling strategy used: random")
    print(f"Saved to: {output_file}")

    # Generate distribution plots if requested
    if generate_plots:
        print("\nGenerating distribution plots...")
        plot_distributions(all_processed_entries, output_file)
        print("Distribution plots generated successfully!")

    # Create a sample entry for verification
    if all_processed_entries:
        sample_entry = all_processed_entries[0]
        output_dir = os.path.dirname(output_file)
        sample_path = os.path.join(output_dir, "sample_processed_entry.json")
        with open(sample_path, 'w') as f:
            json.dump(sample_entry, f, indent=2)
        print(f"\nSample processed entry saved to: {sample_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Data preprocessing for Spatial AI Warehouse dataset")
    parser.add_argument("--input_file", type=str, required=True, help="Path to input JSON file (train.json or val.json)")
    parser.add_argument("--output_file", type=str, required=True, help="Full path to output JSON file")
    parser.add_argument("--samples_per_category", type=int, default=20000, help="Number of samples per category (ignored for val files)")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--target_categories", nargs="*", help="Target categories to include")
    parser.add_argument("--generate_plots", action="store_true", default=True, help="Generate distribution plots")
    parser.add_argument("--replace_masks", action="store_true", default=True, help="Replace <mask> with Region [idx]")
    parser.add_argument("--no_sampling", action="store_true", help="Process all entries without sampling (useful for val files)")
    parser.add_argument("--output_format", type=str, default="json", choices=["json", "jsonl"], help="Output format (json or jsonl)")
    parser.add_argument("--som_prompt", type=str, default=DEFAULT_SOM_PROMPT, help="SOM prompt to prepend to queries")

    args = parser.parse_args()

    preprocess_warehouse_data(
        input_file=args.input_file,
        output_file=args.output_file,
        samples_per_category=args.samples_per_category,
        random_seed=args.random_seed,
        target_categories=args.target_categories,
        generate_plots=args.generate_plots,
        replace_masks=args.replace_masks,
        no_sampling=args.no_sampling,
        output_format=args.output_format,
        som_prompt=args.som_prompt
    )


if __name__ == "__main__":
    main()
