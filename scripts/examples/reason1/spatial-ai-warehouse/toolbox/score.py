"""
Scoring script for evaluation results.

This script works with results from both:
- evaluate.py (original complex version)
- evaluate_simple.py (simplified version)

Both scripts produce the same JSON output format, so this scoring script
is compatible with either evaluation approach.
"""

import argparse
import functools
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List


def parse_from_terminal(
    terminal_output: str, start_with_to_key: Dict[str, str]
) -> Dict[str, Any]:
    parsed_metrics = {}
    for line in terminal_output.splitlines():
        for start_with, key in start_with_to_key.items():
            if line.startswith(start_with):
                # Try both separators, fallback to original line if neither found
                if "=" in line:
                    m = line.split("=")[-1].strip()
                elif ":" in line:
                    m = line.split(":")[-1].strip()
                else:
                    m = line.strip()
                parsed_metrics[key] = m
                print(line)
                break
    return parsed_metrics


def official_compute_score_fn(
    post_processed_results: List[Dict[str, Any]],
    official_answer_path: str,
    official_evaluation_script: str,
) -> Dict[str, Any]:
    temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8")
    try:
        json.dump(post_processed_results, temp_file, ensure_ascii=False)
        temp_file.flush()
        temp_file_path = temp_file.name
    finally:
        temp_file.close()

    try:
        # Validation error message
        INVALID_SCRIPT_MSG = (
            f"Invalid evaluation script path: {official_evaluation_script}"
        )

        # Validate that the evaluation script exists and is a Python file
        if not os.path.exists(
            official_evaluation_script
        ) or not official_evaluation_script.endswith(".py"):
            raise ValueError(INVALID_SCRIPT_MSG)

        cmd = [
            "python",
            official_evaluation_script,
            "--gt_path",
            official_answer_path,
            "--pred_path",
            temp_file_path,
        ]

        print(" ".join(cmd))
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        output = result.stdout
        print(output)
        parsed_metrics = {}
        evaluation_match = re.search(
            r"===== EVALUATION RESULTS =====\n(.*?)\n===== OVERALL SUMMARY =====\n",
            output,
            re.DOTALL,
        )
        if not evaluation_match:
            print("Warning: Could not find EVALUATION RESULTS section in output")
            evaluation_results_part = ""
        else:
            evaluation_results_part = evaluation_match.group(1)

        start_with_to_key = {
            "Count": "count",
            "Distance": "distance",
            "left_right": "left_right",
            "mcq": "mcq",
        }
        parsed_metrics.update(
            parse_from_terminal(evaluation_results_part, start_with_to_key)
        )

        if "===== OVERALL SUMMARY =====" in output:
            overall_summary_part = output.split("===== OVERALL SUMMARY =====")[-1]
        else:
            print("Warning: Could not find OVERALL SUMMARY section in output")
            overall_summary_part = ""
        start_with_to_key = {
            "Final Weighted Score": "final_weighted_score",
            "Quantitative": "quantitative",
            "Qualitative": "qualitative",
            "Overall": "overall",
        }
        parsed_metrics.update(
            parse_from_terminal(overall_summary_part, start_with_to_key)
        )
        return parsed_metrics
    finally:
        os.unlink(temp_file_path)


def evaluate(
    raw_results_dir: str,
    eval_answer_key: str,
    compute_score_fn: Callable[[str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Evaluate results from evaluation scripts (both evaluate.py and evaluate_simple.py).

    Expected JSON structure per file:
    [
        {
            "datasource": "warehouse",
            "video_id": "annotation_id",
            "prompt": null,
            "correct_answer": "ground_truth",
            "reasoning": "",
            "answer": "model_prediction",
            "full_response": "complete_model_response",
            "is_correct": true
        }
    ]
    """
    json_files = glob.glob(os.path.join(raw_results_dir, "*.json"))
    post_processed_results = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
            for sample in data:
                post_processed_results.append(
                    {
                        "id": sample["video_id"],
                        "normalized_answer": sample[eval_answer_key],
                    }
                )
    return compute_score_fn(
        post_processed_results=post_processed_results,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--official_answer_path", type=str, required=True)
    parser.add_argument("--official_evaluation_script", type=str, required=True)
    args = parser.parse_args()

    results_dir = args.results_dir

    metrics_path = os.path.join(results_dir, "metrics.json")
    if os.path.exists(metrics_path):
        print("metrics.json already exists, skip evaluation")
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    else:
        metrics = list()

    compute_score_fn = functools.partial(
        official_compute_score_fn,
        official_answer_path=args.official_answer_path,
        official_evaluation_script=args.official_evaluation_script,
    )

    # Expected directory structure (same for both evaluate.py and evaluate_simple.py):
    # results_dir/
    #   └── model_name/
    #       └── answer_type/  # "naive" or "reasoning"
    #           ├── annotation_id1.json
    #           ├── annotation_id2.json
    #           └── ...
    results_dir_path_obj = Path(results_dir)

    if not results_dir_path_obj.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    model_names = [p.name for p in results_dir_path_obj.iterdir() if p.is_dir()]

    if not model_names:
        print(f"Warning: No model directories found in {results_dir}")
    model_name_answer_types = [
        (entry_metrics["model_name"], entry_metrics["answer_type"])
        for entry_metrics in metrics
    ]
    for model_name in model_names:
        answer_types = [
            p.name
            for p in results_dir_path_obj.joinpath(model_name).iterdir()
            if p.is_dir()
        ]
        # only allow "naive" and "reasoning"
        assert set(answer_types).issubset({"naive", "reasoning"})
        for answer_type in answer_types:
            if (model_name, answer_type) in model_name_answer_types:
                print(f"skip {model_name} {answer_type} because it already exists")
                continue
            eval_answer_key = (
                "full_response"
                if answer_type == "naive"
                else "answer"
                if answer_type == "reasoning"
                else None
            )
            try:
                entry_metrics = evaluate(
                    raw_results_dir=os.path.join(results_dir, model_name, answer_type),
                    eval_answer_key=eval_answer_key,
                    compute_score_fn=compute_score_fn,
                )
            except (
                FileNotFoundError,
                json.JSONDecodeError,
                subprocess.CalledProcessError,
            ) as e:
                print(f"Error evaluating {model_name} {answer_type}: {e}")
                continue
            metrics.append(
                {
                    "model_name": model_name,
                    "answer_type": answer_type,
                    "metrics": entry_metrics,
                }
            )

    print("=" * 100)
    print(metrics)
    print("=" * 100)
    print(f"save metrics to {os.path.join(results_dir, 'metrics.json')}")
    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)
