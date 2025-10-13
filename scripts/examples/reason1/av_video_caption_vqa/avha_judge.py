#!/usr/bin/env -S uv run --script

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

# https://docs.astral.sh/uv/guides/scripts/#using-a-shebang-to-create-an-executable-file
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "openai",
# ]
# ///

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from misc_utils import (
    extract_tagged_text,
    get_base_filename,
    get_filename_prefix,
    get_list_of_files,
    iterate_with_timing_info,
    read_json_file,
    read_json_output_file,
    run_sharded_computation,
    string_to_int,
    write_json_file,
)
from model_openai import OpenAIModel

SCRIPT_DIR = Path(__file__).parent


SYSTEM_PROMPT = """
You are a helpful assistant who has been tasked with evaluating the reponses produced by
another LLM.  A series of questions were given to both humans and LLMs, and we want to
evaluate how closely the LLM response matches the human response.

The match should be scored on a scale between 1 and 5, according to the following rubric:

1: The responses don't match at all; the LLM gave a completely different answer to the human.

2: There is some overlap between the two responses, but overall the LLM response is substantially
different from the human response, and thus should be considered to be "wrong" overall.

3: The LLM got some things right, and some things wrong.

4: The responses are clearly similar, but there are still a few details where the LLM response
does not match the human response.

5: The responses match very closely.  Note that the phrasing may be different, but the match
should still be given a grade of 5 if they have the same meaning.  For example, when asked about
weather, the human may respond "clear blue skies", while the LLM may respond "the sky is sunny
and cloudless", but these mean the same thing.

Give your answer in the following format.  First, write a short description of why you think the
responses match or don't match.  If the responses don't match, be sure to explain why you think
they are different.  Then provide your score.
"""

USER_PROMPT = """
Given the question ``%s'', here are the two responses to compare:

*** Human response ***:
%s

*** LLM response ***:
%s

Provide your answer in the following format:

[Reason for giving the score.]

<answer>[Single digit between 1 and 5]</answer>
"""


def get_model():
    """Return a model object."""
    model = OpenAIModel()
    model.set_system_prompt(SYSTEM_PROMPT)
    return model


def run_inference(model, question, answer, llm_output):
    """Prompt the model with a question, and return the response."""
    if model is None:
        return "Duhhh... <answer>1</answer>"  # dry run

    prompt = USER_PROMPT % (question, answer, llm_output)
    try:
        response = model.generate(prompt)
    except Exception as e:
        print(f"Model called failed with exception {e}")
        print(f"{prompt=}")
        raise
    return response


def extract_score(answer: str) -> Optional[int]:
    """Extract a score from the LLM response."""
    score = extract_tagged_text(answer, "answer", fallback="[1-5]")
    if score is None:
        return None
    score = string_to_int(score)
    if score is None:
        return None
    if score < 1 or score > 5:
        return None
    return score


def compute_summary(all_scores: Any) -> Any:
    """Compute the summary (mean scores) for all scores."""
    summary = {}
    for k, rs in all_scores.items():
        num_results = len(rs)
        num_scores = 0
        total_score = 0
        for r in rs:
            score = r["score"]
            if score is not None:
                total_score += score
                num_scores += 1
        mean_score = total_score / (num_scores + 0.00001)
        success_rate = num_scores / (num_results + 0.00001)

        summary[k] = {
            "num_results": num_results,
            "num_scores": num_scores,
            "mean_score": mean_score,
            "success_rate": success_rate,
        }
    return summary


def process_outputs(
    model,
    output_file_list: list[str],
    output_dir: Path,
    answer_dir: Path,
    score_dir: Path,
    questions,
    force_reprocess: bool,
    shard_id: int,
):
    """Process the output for a single video, and record the results."""

    all_scores = {k: [] for k in questions}
    num_json_failures = 0
    prefix = f"s{shard_id}: "

    def update_all_scores(answer_scores):
        nonlocal all_scores
        for k in questions:
            all_scores[k].append(answer_scores[k])

    def process_output_fn(output_filename: str) -> bool:
        nonlocal num_json_failures
        nonlocal prefix

        base_name = get_filename_prefix(get_base_filename(output_filename))

        # Process a single answer.
        output_path = output_dir / output_filename
        answer_path = answer_dir / (base_name + ".label.json")
        score_path = score_dir / (base_name + ".score.json")

        # Skip if already processed (unless force flag is set)
        if score_path.exists():
            if not force_reprocess:
                print(f"{prefix}Skipping {output_path} (already processed)")
                print(f"{prefix}Loading previously computed scores from {score_path}.")
                answer_scores = read_json_file(score_path)
                update_all_scores(answer_scores)
                return False

        print(f"{prefix}Reading output file {output_path}.")
        llm_output = read_json_output_file(output_path)
        if llm_output is None:
            num_json_failures += 1
            llm_output = {}

        print(f"{prefix}Reading answer file {answer_path}.")
        gt_answer = read_json_file(answer_path)

        answer_scores = {}
        for k, question in questions.items():
            if k in gt_answer and k in llm_output:
                print(f"{prefix}Question {k}: ...")
                result = run_inference(
                    model,
                    question=question,
                    answer=gt_answer[k],
                    llm_output=llm_output[k],
                )
                # print(f"{prefix} Result = {result}")
                score = extract_score(result)
                print(f"{prefix}Question {k}: {score}")
                answer_scores[k] = {
                    "llm_answer": llm_output[k],
                    "human_answer": gt_answer[k],
                    "result": result,
                    "score": score,
                }
            elif k not in llm_output:
                answer_scores[k] = {
                    "result": "No answer from LLM.",
                    "score": None,
                }
            else:
                answer_scores[k] = {
                    "result": "No ground truth answer from human.",
                    "score": None,
                }

        # Write scores to json file.
        print(f"{prefix}Writing scores to {score_path}.")
        write_json_file(answer_scores, score_path)

        # Keep a table of all results and scores.
        update_all_scores(answer_scores)
        # current_summary = compute_summary(all_scores)
        # print(f"{prefix}current summary: {json.dumps(current_summary, indent=2)}")
        return True

    totalp = iterate_with_timing_info(
        output_file_list, process_output_fn, prefix_str=prefix
    )
    print(f"{prefix}Number of JSON failures: {num_json_failures}")
    print(f"{prefix}Processed {totalp} outputs.")
    print(f"{prefix}Finished.  Collected scores:")
    for k, ss in all_scores.items():
        print(f"{prefix}: {k}: {len(ss)}")

    return all_scores


def process_fn(eval_list: list, other_args: list, shard_id: int):
    """Pickleable function which wraps process_outputs."""
    (output_dir, answer_dir, score_dir, questions, dryrun, force_reprocess) = other_args

    try:
        model = get_model() if not dryrun else None
        scores = process_outputs(
            model,
            eval_list,
            output_dir=output_dir,
            answer_dir=answer_dir,
            score_dir=score_dir,
            questions=questions,
            force_reprocess=force_reprocess,
            shard_id=shard_id,
        )
    except Exception as e:
        print(f"Shard {shard_id} failed with exception {e}")
        raise
    return scores


def main():
    """Main function."""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate Cosmos-Reason1 model using configuration file"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="./output/baseline",
        help="Output directory for previously-computed LLM outputs.",
    )

    parser.add_argument(
        "--answer-dir",
        "-a",
        default="./eval/metas",
        help="Directory for ground-truth answers.",
    )

    parser.add_argument(
        "--score-dir",
        "-s",
        default="./scores/baseline",
        help="Directory to write final scores.",
    )

    parser.add_argument(
        "--question",
        "-q",
        default="./prompts/score_question.json",
        help="Question used to compute final scores.",
    )

    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Number of shards to use for parallel processing.",
    )

    parser.add_argument(
        "--dryrun", action="store_true", help="Do a dry run (not running the model)."
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of videos even if output files exist.",
    )

    args = parser.parse_args()
    print(f"Script arguments: {args}")

    # Resolve file path to output directory -- where previously computed LLM outputs are stored.
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = SCRIPT_DIR / output_dir
    print(f"Output directory set to: {output_dir}")

    # Resolve file path to the directory for ground-truth answers.
    answer_dir = Path(args.answer_dir)
    if not answer_dir.is_absolute():
        answer_dir = SCRIPT_DIR / answer_dir
    print(f"Answer directory set to: {answer_dir}")

    # Resolve file path to the directory to write final scores.
    score_dir = Path(args.score_dir)
    if not score_dir.is_absolute():
        score_dir = SCRIPT_DIR / score_dir
    score_dir.mkdir(parents=True, exist_ok=True)  # Create dir if necessary.
    print(f"Score directory set to: {score_dir}")

    # Load a file containing the questions to ask for scoring.
    question_file = Path(args.question)
    if not question_file.is_absolute():
        question_file = SCRIPT_DIR / question_file
    print(f"Loading scoring questions from {question_file}")
    questions = read_json_file(question_file)

    # Handle dry run.
    if args.dryrun:
        print("Dry run -- no model.")

    process_fn_other_args = (
        output_dir,
        answer_dir,
        score_dir,
        questions,
        args.dryrun,
        args.force,
    )

    def join_results_fn(result_list: list):
        # process_outputs returns a dictionary of lists.
        # concatenate all lists into one dictionary.
        all_scores = {k: [] for k in questions}
        for result_scores in result_list:
            if result_scores is None:
                continue
            for k in questions:
                if k in result_scores:
                    all_scores[k].extend(result_scores[k])
        return all_scores

    output_list = get_list_of_files(output_dir)
    print(f"Found {len(output_list)} items to analyze.")

    all_scores = run_sharded_computation(
        process_fn,
        join_results_fn,
        output_list,
        other_args=process_fn_other_args,
        num_shards=args.num_shards,
    )

    all_scores_file = score_dir / "scores.json"
    write_json_file(all_scores, all_scores_file)

    summary = compute_summary(all_scores)
    summary_file = score_dir / "summary.json"
    write_json_file(summary, summary_file)

    print(f"Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
