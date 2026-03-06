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

"""Evaluate a model on a dataset."""
import concurrent.futures
import glob
import json
import logging as log
import os
import re
import sysconfig
import time
from argparse import ArgumentParser
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import attrs
import yaml
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm

# Enable tokenizers parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Configure basic logging
_eval_log_level = getattr(log, os.environ.get("LOGLEVEL", "INFO").upper(), log.INFO)
log.basicConfig(
    level=_eval_log_level, format="%(asctime)s - %(levelname)s - %(message)s"
)

from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# Define type alias for clarity when using the real processor
Processor = AutoProcessor

from utils.model_download import download_checkpoint
from utils.output import OutputStructure, parse_letter_response, save_results_parallel


def check_python_headers():
    """
    Checks if Python headers are available, which are required for compiling
    kernels on the fly with openai-triton (used by VLLM).
    """
    include_path = sysconfig.get_path("include")
    # Check for Python.h which is the standard header file
    python_h = os.path.join(include_path, "Python.h")

    if not os.path.exists(python_h):
        error_msg = (
            f"Python headers not found at {include_path}. "
            "Please install python-dev/python3-dev to allow VLLM/Triton "
            "to compile kernels on the fly."
            "Example: sudo apt-get install python3.12-dev"
        )
        log.error(error_msg)
        raise RuntimeError(error_msg)

    log.info(f"Python headers found at {include_path}")


@attrs.define(slots=False)
class LlavaInputStructure:
    datasource: str
    media_id: str
    question: str
    question_idx: int
    media_paths: str
    media_mode: str
    correct_answer: str
    prompt: Optional[Union[str, List[Dict[str, Any]]]] = None

    @classmethod
    def from_dict(
        cls, datasource: str, qa_pair: Dict[str, Any]
    ) -> "LlavaInputStructure":
        return cls(
            datasource=datasource,
            media_id=qa_pair["media_id"],
            question=qa_pair["conversations"][1]["content"],
            correct_answer=qa_pair["conversations"][2]["content"],
            question_idx=qa_pair["id"],
            media_paths=qa_pair["media_paths"],
            media_mode=qa_pair["media_mode"],
            prompt=qa_pair["conversations"][:-1],
        )


def prepare_model_inputs_parallel(
    input_tasks: List[LlavaInputStructure],
    processor: Any,
    num_processes: int,
    vision_config: dict,
) -> List[Any]:
    """
    Prepares model inputs for a list of input tasks in parallel using threads.

    This function is suitable for I/O-bound tasks like reading files or
    processing data structures, as it uses ThreadPoolExecutor. It calls
    `prepare_single_model_input` for each task. Handles collecting results
    and filters out any tasks that failed during preparation.

    Args:
        input_tasks: A list of InputStructure objects to prepare inputs for.
        processor: The model's processor/tokenizer object.
        num_processes: The maximum number of threads to use for parallel execution.
        fps: The frames per second to associate with video inputs.

    Returns:
        A list of prepared model input objects. Tasks that failed during
        preparation are excluded from the list. The order of inputs
        corresponds to the order of successful tasks in the input list.
    """
    if not input_tasks:
        log.info("No input tasks to prepare model inputs for.")
        return []

    # Determine the number of workers, up to the number of tasks
    num_workers = min(num_processes, len(input_tasks))

    log.info(
        f"Preparing model inputs in parallel for {len(input_tasks)} tasks "
        f"using {num_workers} threads."
    )

    # Create a partial function with fixed arguments for the worker

    worker_fn = partial(
        prepare_single_model_input,
        processor=processor,
        vision_config=vision_config,
    )

    processed_inputs_with_index: List[Tuple[int, Any]] = []

    # Use ThreadPoolExecutor for potentially I/O-bound model input preparation
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit tasks and map futures to their original index
        future_to_idx = {
            executor.submit(worker_fn, task): i for i, task in enumerate(input_tasks)
        }

        # Use tqdm to track progress as futures complete
        for future in tqdm(
            concurrent.futures.as_completed(future_to_idx),
            total=len(future_to_idx),
            desc="Preparing model inputs",
        ):
            idx = future_to_idx[future]
            try:
                model_input = future.result()
                if model_input is not None:  # Only append successful results
                    processed_inputs_with_index.append((idx, model_input))
            except (ValueError, OSError, RuntimeError) as e:
                # This catch might be redundant if prepare_single_model_input handles errors,
                # but included for robustness.
                log.exception(
                    f"Unexpected error preparing model input for task {idx}: {e}. Skipping task."
                )

    # Sort successful results by original index and extract the inputs
    processed_inputs_with_index.sort(key=lambda x: x[0])
    inputs = [inp for idx, inp in processed_inputs_with_index]

    if len(inputs) < len(input_tasks):
        log.warning(
            f"Successfully prepared inputs for {len(inputs)} out of {len(input_tasks)} tasks."
        )

    return inputs


def prepare_single_model_input(
    input_task: LlavaInputStructure,
    processor: Any,
    vision_config: dict,
) -> Optional[Any]:
    """
    Worker function to prepare the input data for a single model inference task.

    Integrates the video path into the prompt structure and applies the model's
    chat template. Uses `process_vision_info` and formats the final model input
    based on whether a Hugging Face checkpoint is being used.

    Args:
        input_task: The InputStructure object for the task.
        processor: The model's processor or tokenizer object, used for template application
                   and input formatting.
        fps: The frames per second to associate with the video input.

    Returns:
        The prepared model input object or None
        if an error occurs during preparation.
    """
    # Add video information to the user message content
    # Assuming the user message is the second element and its content is text
    if (
        len(input_task.prompt) > 1
        and input_task.prompt[1]["role"] == "user"
        and isinstance(input_task.prompt[1]["content"], str)
    ):
        content = []
        media_mode = input_task.media_mode
        for media_path in input_task.media_paths:
            video_content = {
                "type": media_mode,
                media_mode: media_path,
            }
            # Add all key:value pairs from vision_config to the video_content dict
            for k, v in vision_config.items():
                video_content[k] = v
            content.append(video_content)
        content.append({"type": "text", "text": input_task.prompt[1]["content"]})
        input_task.prompt[1]["content"] = content

    # Apply the model's chat template to get the final text prompt
    # Use add_generation_prompt=True to include the prompt part that signals
    # the model to start generating the response.
    processed_text_prompt = processor.apply_chat_template(
        input_task.prompt, tokenize=False, add_generation_prompt=True
    )

    # Process vision information (image/video paths) from the prompt structure
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        input_task.prompt,
        image_patch_size=16,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    # Format for a potential custom model structure
    # Assuming video_inputs contains a list and we need the first item
    if not video_inputs and not image_inputs:
        log.error(
            f"No video or image inputs found for task: media_id={input_task.media_id}, question_idx={input_task.question_idx}. Cannot prepare model input."
        )
        return None

    if video_inputs:
        model_input = {
            "prompt": processed_text_prompt,
            "multi_modal_data": {"video": video_inputs},
            "mm_processor_kwargs": video_kwargs,
        }
    else:
        model_input = {
            "prompt": processed_text_prompt,
            "multi_modal_data": {"image": image_inputs},
        }

    log.debug(
        f"Prepared model input for task: media_id={input_task.media_id}, "
        f"question_idx={input_task.question_idx}"
    )
    return model_input


# === Model Definition Functions ===
def define_model(
    tokenizer_model_name: str,
    model_name: str,
    dtype: str,
    tp_size: int | None,
    max_length: int = 12800,
) -> tuple[LLM, Processor]:
    """
    Defines and loads the language model and its processor.

    Args:
        tokenizer_name: The name of the tokenizer (e.g., "qwen2.5-vl-7b").
        model_name: Name of the model.
        dtype: Data type for model weights ("bfloat16" or "float16").
        tp_size: Tensor parallel size for VLLM, or None to use defaults.
        max_length: Maximum sequence length for the model.

    Returns:
        A tuple containing the loaded model and its processor.
    """

    if os.path.isabs(model_name) and os.path.exists(model_name):
        checkpoint_output_dir = model_name

    else:
        hf_cache_dir = os.environ.get(
            "HF_HOME",
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
        )
        checkpoint_output_dir = os.path.join(hf_cache_dir, model_name)
        # Ensure the checkpoint directory exists
        os.makedirs(checkpoint_output_dir, exist_ok=True)

    # Download checkpoint if not already present
    download_checkpoint(model_name, checkpoint_output_dir)

    log.info("Using VLLM backend.")
    # Allow longer max model length in VLLM
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    # Initialize VLLM LLM object
    llm = LLM(
        model=checkpoint_output_dir,
        tokenizer=checkpoint_output_dir,
        trust_remote_code=True,
        dtype=dtype,
        # Specify multimedia limit per prompt (e.g., one video, zero images)
        limit_mm_per_prompt={"video": 1, "image": 1},
        tensor_parallel_size=tp_size,
        max_model_len=max_length,
        max_num_seqs=8,
        gpu_memory_utilization=0.85,
    )

    # Load processor from the same checkpoint directory
    processor: Processor = AutoProcessor.from_pretrained(
        checkpoint_output_dir, max_length=max_length
    )

    return llm, processor


# === Task Generation Functions ===


def make_tasks_from_single_media(
    output_json_fname: str,
    qa_pair: dict[str, Any],
    datasource_name: str,
) -> tuple[list[LlavaInputStructure], list[OutputStructure]]:
    """
    Creates InputStructure and OutputStructure objects for all questions related to a single media.

    Args:
        output_json_fname: The base filename for saving results for this media.
        qa_pairs: A list of question-answer dictionaries for the media.
        datasource_name: The name of the dataset the video belongs to.

    Returns:
        A tuple containing:
        - A list of InputStructure objects for questions needing evaluation.
        - A list of OutputStructure objects to store evaluation results.
    """
    input_questions: list[LlavaInputStructure] = []
    output_results: list[OutputStructure] = []

    # Create InputStructure for the current question
    input_questions.append(LlavaInputStructure.from_dict(datasource_name, qa_pair))
    # Create corresponding OutputStructure to store results
    output_results.append(
        OutputStructure(
            datasource=datasource_name,
            video_id=qa_pair["media_id"],
            # Get the correct answer (handles variations in dict key)
            correct_answer=qa_pair["conversations"][-1]["content"],
            output_json_fname=output_json_fname,
            prompt="",  # This will be filled later
        )
    )

    return input_questions, output_results


def make_all_tasks(
    datasets: dict,
    results_output_folder: str,
    answer_type: str,
    total_shard: int,
    shard_id: int,
) -> tuple[list[LlavaInputStructure], list[OutputStructure]]:
    """
    Gathers all evaluation tasks from the specified datasources. Supports loading from
    a list of datasource names (for S3 paths) or a Hugging Face dataset.

    Args:
        datasource_list: List of datasource names.
        results_output_folder: Base directory to save evaluation results.
        limit: Maximum number of tasks to gather across all datasources (-1 for no limit).

    Returns:
        A tuple containing:
        - A list of all InputStructure objects to be evaluated.
        - A list of all corresponding OutputStructure objects.
    """

    input_tasks: list[LlavaInputStructure] = []  # Stores all input tasks
    output_results: list[OutputStructure] = []  # Stores all output result objects

    # Process each datasource
    qa_pairs = []
    for datasource_name, datasource_config in datasets.items():
        log.info(f"Gathering tasks from dataset: {datasource_name}")

        media_dir = datasource_config.get("media_dir", None)
        annotation_path = datasource_config.get("annotation_path")
        system_prompt = datasource_config.get(
            "system_prompt",
            "You are a helpful assistant.",
        )

        if answer_type == "reasoning" and "<think>" not in system_prompt:
            system_prompt += "Answer the question using the following format:\n\n<think>\nYour reasoning.\n</think>\n\nWrite your final answer immediately after the </think> tag."

        # Check if video_path exists
        if media_dir and not os.path.exists(media_dir):
            log.error(f"Media path does not exist: {media_dir}")
            continue

        with open(annotation_path, "r") as f:
            annotations = json.load(f)

            for item in annotations:
                # question = item['conversations'][0]['value'].replace("<video>\n", "").replace("<video> \n", " ")
                question = re.sub(
                    r"(\n)?</?(image|video)>(\n)?",
                    "",
                    item["conversations"][0]["value"],
                ).strip()
                conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": item["conversations"][1]["value"]},
                ]

                images = item.get("image", None) or item.get("images", None)
                videos = item.get("video", None)

                if images:
                    if isinstance(images, str):
                        images = [images]
                    relative_media_paths = images
                    media_mode = "image"
                elif videos:
                    if isinstance(videos, str):
                        videos = [videos]
                    relative_media_paths = videos
                    media_mode = "video"
                else:
                    log.error(f"No media paths found for item: {item}")
                    continue

                if media_dir:
                    media_paths = [
                        os.path.join(media_dir, path) for path in relative_media_paths
                    ]
                else:
                    media_paths = relative_media_paths

                qa_pairs.append(
                    {
                        "media_id": relative_media_paths[0],
                        "id": item["id"],
                        "media_paths": media_paths,
                        "media_mode": media_mode,
                        "conversations": conversation,
                    }
                )

    shard_qa_pairs = qa_pairs[shard_id::total_shard]
    log.info(
        f"Sharding {len(qa_pairs)} tasks into {total_shard} shards, shard {shard_id} has {len(shard_qa_pairs)} tasks."
    )
    for qa_pair in shard_qa_pairs:
        output_json_fname = os.path.join(
            results_output_folder, datasource_name, f"{qa_pair['media_id']}.json"
        )
        os.makedirs(os.path.dirname(output_json_fname), exist_ok=True)
        input_task, output_result = make_tasks_from_single_media(
            output_json_fname, qa_pair, datasource_name
        )
        input_tasks.extend(input_task)
        output_results.extend(output_result)

    return input_tasks, output_results


# === Model Execution Functions ===


def run_model(
    model: LLM,
    inputs: list[str],  # VLLM generate takes list of prompts
    input_tasks: list[LlavaInputStructure],
    output_results: list[OutputStructure],
    stop_token_id: int,
    answer_type: str,
    max_retries: int = 3,
    max_tokens: int = 1024,
    temperature: float = 0.6,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    seed: int = 0,
) -> None:
    """
    Runs the VLLM model on the provided inputs and processes the outputs.
    Includes retry logic for tasks resulting in empty answers.

    Args:
        model: The loaded VLLM model.
        inputs: A list of prompt strings for the model.
        input_tasks: List of original InputStructure objects.
        output_results: List of OutputStructure objects to update with results.
        stop_token_id: The token ID to stop generation at.
        answer_type: Expected format of the answer ("letter" or "reasoning").
        max_retries: Maximum number of times to retry tasks with empty answers.
        max_tokens: Maximum number of tokens to generate per task.
        temperature: Sampling temperature.
        repetition_penalty: Penalty for repeating tokens.
        presence_penalty: Penalty for using tokens already present.
        frequency_penalty: Penalty based on token frequency.
        seed: Random seed for sampling.
    """
    # Configure sampling parameters based on the expected answer type
    if answer_type == "letter":
        # Use greedy decoding (temperature=0, top_k=1) for letter answers
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=10,  # Generate only a few tokens for a letter answer
            stop_token_ids=[stop_token_id],
            top_k=1,
            seed=seed,
        )
    else:  # answer_type == "reasoning" or "freeform"
        # Use specified sampling parameters for reasoning answers
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,  # Use top_p sampling
            stop_token_ids=[stop_token_id],
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            seed=seed,
        )

    log.info(f"Generating outputs for {len(inputs)} tasks using VLLM...")
    # Generate outputs for all inputs in a single batch
    list_of_requestoutput = model.generate(inputs, sampling_params)
    log.info(
        f"Finished VLLM generation. Received {len(list_of_requestoutput)} outputs."
    )

    empty_answer_indices: list[
        int
    ] = []  # Keep track of indices for tasks with empty answers

    # Process the initial generation outputs
    for i, (requestoutput, input_task, output_result) in enumerate(
        zip(list_of_requestoutput, input_tasks, output_results, strict=False)
    ):
        output_text = requestoutput.outputs[0].text  # Get the generated text
        # Parse the generated text based on the expected answer type
        if answer_type == "letter":
            answer, reasoning = parse_letter_response(output_text)
        elif answer_type == "reasoning":
            if "</think>" not in output_text:
                answer = output_text
                reasoning = ""
            else:
                reasoning, answer = output_text.split("</think>")
                answer = answer.strip()
                reasoning = reasoning.strip()
        else:
            answer = output_text
            reasoning = ""

        # Store the generated results in the OutputStructure object
        output_result.prompt = input_task.prompt  # Store the original prompt
        output_result.reasoning = reasoning
        output_result.answer = answer
        output_result.full_response = output_text

        # If the generated answer is empty, add its index to the retry list
        if not answer:
            empty_answer_indices.append(i)

    # --- Retry logic for empty answers ---
    current_empty_indices = empty_answer_indices  # Initialize retry list
    for retry_count in range(max_retries):
        if not current_empty_indices:
            log.info("No more empty answers. Retries finished.")
            break  # Exit retry loop if no empty answers remain

        log.info(
            f"Found {len(current_empty_indices)} empty answers. Retrying batch ({retry_count + 1}/{max_retries})..."
        )

        # Prepare inputs specifically for the tasks that had empty answers
        retry_inputs: list[str] = [inputs[i] for i in current_empty_indices]

        # Adjust sampling parameters for retries (increase max tokens and temperature)
        # This encourages the model to generate different responses
        sampling_params.max_tokens = max_tokens + 256 * (retry_count + 1)
        sampling_params.temperature = min(temperature + (0.05 * (retry_count + 1)), 0.9)
        log.info(
            f"  Retry {retry_count + 1} sampling params: max_tokens={sampling_params.max_tokens}, temperature={sampling_params.temperature}"
        )

        # Generate outputs for the retry batch
        retry_outputs = model.generate(retry_inputs, sampling_params)

        still_empty_indices: list[
            int
        ] = []  # List to hold indices that are still empty after this retry
        # Process the retry outputs
        for batch_idx, original_idx in enumerate(current_empty_indices):
            retry_text = (
                retry_outputs[batch_idx].outputs[0].text
            )  # Get the generated text from retry
            # Parse the generated text
            if answer_type == "letter":
                answer, reasoning = parse_letter_response(retry_text)
            else:
                if "</think>" not in output_text:
                    answer = output_text
                    reasoning = ""
                else:
                    reasoning, answer = output_text.split("</think>")
                    answer = answer.strip()
                    reasoning = reasoning.strip()

            # If a valid answer was obtained in this retry, update the result
            if answer:
                output_results[original_idx].reasoning = reasoning
                output_results[original_idx].answer = answer
                output_results[original_idx].full_response = retry_text
            else:
                # If still no answer, add to the list for the next retry
                still_empty_indices.append(original_idx)

        # Update the list of indices to retry for the next iteration
        current_empty_indices = still_empty_indices

    # Report any tasks that still have empty answers after all retries
    if current_empty_indices:
        log.warning(
            f"{len(current_empty_indices)} tasks still have empty answers after {max_retries} retries."
        )
        # Log details of tasks that failed to produce an answer
        for original_idx in current_empty_indices:
            log.warning(
                f"  - Task from video '{input_tasks[original_idx].media_id}' in datasource '{input_tasks[original_idx].datasource}' could not generate a valid answer."
            )


def run_evaluation_metrics(result_path):
    """
    Calculates the accuracy of the model on the model responses in the results directory.
    """
    correct_count = 0
    total_count = 0
    result_files = glob.glob(os.path.join(result_path, "**", "*.json"), recursive=True)
    for result_file in result_files:
        if "results.json" in result_file:
            continue
        with open(result_file, "r") as f:
            result = json.load(f)
            for item in result:
                if item["correct_answer"].lower() == item["answer"].lower():
                    correct_count += 1
                total_count += 1

    if total_count == 0:
        log.warning("No results found. Please check the results directory.")
        accuracy = 0
    else:
        accuracy = correct_count / total_count
    log.info(
        f"Total number of questions: {total_count}, Correct: {correct_count}, Accuracy: {accuracy}"
    )

    results = {
        "total_correct": correct_count,
        "total_questions": total_count,
        "accuracy": accuracy,
    }
    with open(os.path.join(result_path, "results.json"), "w") as f:
        json.dump(results, f)


# === Main Function and Script Entry Point ===
def main():
    """
    Main function to set up evaluation, run the model, and save results.
    """
    # Check for Python headers before proceeding
    check_python_headers()

    # --- Argument Parsing ---
    parser = ArgumentParser(description="Run video language model evaluation.")
    parser.add_argument(
        "--config_file",
        type=str,
        required=True,
        help="Path to YAML configuration file with model and evaluation parameters.",
    )

    # These arguments will remain as direct command-line options
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Base directory to save evaluation results.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="Name of the model to use for evaluation.",
    )

    parser.add_argument(
        "--total_shard",
        type=int,
        default=1,
        help="Total number of shards.",
    )

    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="Shard ID.",
    )

    args = parser.parse_args()

    # Load configuration from YAML file
    try:
        with open(args.config_file) as f:
            config = yaml.safe_load(f)
    except (FileNotFoundError, yaml.YAMLError, PermissionError) as e:
        log.exception(f"Error loading config file: {e}")
        return

    # Extract configuration sections
    model_config = config.get("model", {})
    eval_config = config.get("evaluation", {})
    gen_config = config.get("generation", {})
    vision_config = config.get("vision", {})

    # Retrieve datasets from config
    datasets = config.get("datasets")
    if not datasets:
        log.error("No datasets provided in config file")
        return

    # Convert datasets list to a temporary datasource file or use in-memory list
    # depending on what make_all_tasks expects
    if isinstance(datasets, dict):
        log.info(f"Using datasets from config: {', '.join(datasets.keys())}")
    else:
        log.error("'datasets' must be a dict in the config file")
        return

    # --- Model Configuration ---
    model_name = args.model_name or model_config.get("model_name", None)
    tokenizer_model_name = model_config.get("tokenizer_model_name", "qwen2.5-vl-7b")
    dtype = model_config.get("dtype", "bfloat16")
    tp_size = model_config.get("tp_size", 1)
    max_length = model_config.get("max_length", 32768)

    # --- Evaluation Parameters ---
    answer_type = eval_config.get("answer_type", "reasoning")
    num_processes = eval_config.get("num_processes", 80)
    seed = eval_config.get("seed", 1)

    # --- Generation Parameters ---
    max_retries = gen_config.get("max_retries", 10)
    max_tokens = gen_config.get("max_tokens", 1024)
    temperature = gen_config.get("temperature", 0)
    repetition_penalty = gen_config.get("repetition_penalty", 1.0)
    presence_penalty = gen_config.get("presence_penalty", 0.0)
    frequency_penalty = gen_config.get("frequency_penalty", 0.0)

    # Append dtype and seed to results directory for better organization
    results_output_base = f"{args.results_dir}"
    log.info(f"Results base directory updated to: {results_output_base}")

    # Log all effective arguments for reproducibility
    log.info("--- Script Configuration ---")
    log.info(f"  Config file: {args.config_file}")
    log.info(f"  Model: {model_name}")
    log.info(f"  Results directory: {args.results_dir}")
    log.info("--- Model Configuration ---")
    log.info(f"  Tokenizer model name: {tokenizer_model_name}")
    log.info(f"  Data type: {dtype}")
    log.info(f"  Tensor parallel size: {tp_size}")
    log.info(f"  Max length: {max_length}")
    log.info("--- Evaluation Configuration ---")
    log.info(f"  Answer type: {answer_type}")
    log.info(f"  Number of processes: {num_processes}")
    log.info(f"  Seed: {seed}")
    log.info("--- Vision Configuration ---")
    log.info(f"  Vision config: {vision_config}")
    log.info("--- Generation Configuration ---")
    log.info(f"  Max retries: {max_retries}")
    log.info(f"  Max tokens: {max_tokens}")
    log.info(f"  Temperature: {temperature}")
    log.info(f"  Repetition penalty: {repetition_penalty}")
    log.info(f"  Presence penalty: {presence_penalty}")
    log.info(f"  Frequency penalty: {frequency_penalty}")
    log.info("------------------------")
    # Define the final output path structure
    # {results_dir_base}/{model_name}/{answer_type}/{datasource}/{video_id}.json
    save_folder = model_config.get("save_folder", None)
    if save_folder:
        results_output_dir = os.path.join(results_output_base, save_folder)
    else:
        results_output_dir = os.path.join(
            results_output_base, os.path.basename(model_name.rstrip("/")), answer_type
        )
    log.info(f"Evaluation results will be saved to: {results_output_dir}")

    # === Step 1: Gather all tasks across all datasources and videos ===
    log.info("Starting task gathering...")
    start_time = time.time()
    # make_all_tasks now takes datasets directly
    input_tasks, output_results = make_all_tasks(
        datasets,  # Use the datasets list directly instead of a file
        results_output_dir,  # Pass the full results directory
        answer_type,
        args.total_shard,
        args.shard_id,
    )
    log.info(f"Initial number of tasks gathered: {len(input_tasks)}")

    # === Step 2: Load model and processor ===
    log.info("Loading model and processor...")
    start_time = time.time()

    # Define and load the actual model and processor
    model, processor = define_model(
        tokenizer_model_name,
        model_name,
        dtype,
        tp_size,
        max_length,
    )
    log.info(f"Time taken to load model: {time.time() - start_time:.2f} seconds")

    # === Step 3: Prepare model inputs ===
    log.info("Preparing model inputs in parallel...")
    start_time = time.time()
    # Prepare inputs based on the chosen backend (HF or VLLM)
    # This step tokenizes prompts and handles image/video encoding if needed
    inputs = prepare_model_inputs_parallel(
        input_tasks,
        processor,
        num_processes,
        vision_config,
    )
    log.info(f"Prepared inputs for {len(inputs)} tasks.")
    log.info(
        f"Time taken to prepare model inputs: {time.time() - start_time:.2f} seconds"
    )

    # === Step 4: Generate outputs using the model ===
    log.info("Generating outputs using the model...")
    start_time = time.time()

    # Run evaluation using the VLLM backend
    # Need the EOS token ID from the processor's tokenizer for VLLM stopping
    run_model(
        model,
        inputs,  # VLLM expects list of strings (prompts) directly
        input_tasks,
        output_results,
        processor.tokenizer.eos_token_id,  # Pass EOS token ID for VLLM stopping
        answer_type,
        max_retries,
        max_tokens,
        temperature,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        seed,
    )
    log.info(
        f"Time taken for model generation and output processing: {time.time() - start_time:.2f} seconds"
    )

    # === Step 5: Save results in parallel ===
    log.info("Saving results in parallel...")
    start_time = time.time()
    # Save the updated OutputStructure objects to JSON files
    save_results_parallel(output_results, num_processes=num_processes)
    log.info(f"Time taken to save results: {time.time() - start_time:.2f} seconds")
    log.info("Evaluation completed.")

    # === Step 6: Run evaluation metrics ===
    log.info("Running evaluation metrics...")
    start_time = time.time()
    run_evaluation_metrics(results_output_dir)
    log.info(
        f"Time taken to run evaluation metrics: {time.time() - start_time:.2f} seconds"
    )


# Script entry point
if __name__ == "__main__":
    main()
