#!/usr/bin/env python3
"""
Evaluate Fine-tuned Cosmos Reason 1 Model on Transfer1 Evaluation Dataset

Usage:
    python3 scripts/evaluate_model.py \
        --model_path outputs/transfer1_sft/20251023145904/checkpoints/step_80/policy \
        --eval_dataset data/transfer1_split_with_conv/eval \
        --prompt_path prompts/video_reward.yaml \
        --output_dir eval_results
"""

import argparse
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter, defaultdict

import yaml
from datasets import load_from_disk
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


def parse_response(response):
    """Parse response to extract integer score from <answer></answer> tags."""
    try:
        # Try XML parsing first
        wrapped = f"<root>{response.strip()}</root>"
        root = ET.fromstring(wrapped)
        answer_element = root.find("answer")

        if answer_element is not None and answer_element.text:
            answer_text = answer_element.text.strip()
            try:
                answer_int = int(answer_text)
                # Ensure score is in valid range
                if 1 <= answer_int <= 5:
                    return answer_int
            except ValueError:
                pass

        # Try regex as fallback
        match = re.search(r"<answer>\s*(\d+)\s*</answer>", response)
        if match:
            try:
                answer_int = int(match.group(1))
                if 1 <= answer_int <= 5:
                    return answer_int
            except ValueError:
                pass

    except Exception:
        pass

    return None


def load_prompt_config(prompt_path):
    """Load prompt configuration from YAML file."""
    with open(prompt_path, 'r') as f:
        config = yaml.safe_load(f)
    return config.get('system_prompt', ''), config.get('user_prompt', '')


def run_inference_batch(llm, processor, video_paths, system_prompt, user_prompt, batch_size=4):
    """Run inference on a batch of videos."""
    sampling_params = SamplingParams(
        temperature=0.1,  # Low temperature for more deterministic outputs
        top_k=10,
        top_p=0.9,
        repetition_penalty=1.05,
        max_tokens=512,  # Shorter for evaluation
    )

    results = []
    
    for i in range(0, len(video_paths), batch_size):
        batch_videos = video_paths[i:i+batch_size]
        batch_inputs = []
        
        for video_path in batch_videos:
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "fps": 16,
                            "total_pixels": 8192 * 28 * 28,
                        },
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ]
            
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True
            )
            
            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs
            
            batch_inputs.append({
                "prompt": prompt,
                "multi_modal_data": mm_data,
                "mm_processor_kwargs": video_kwargs,
            })
        
        # Generate responses for batch
        outputs = llm.generate(batch_inputs, sampling_params=sampling_params)
        
        for output in outputs:
            response_text = output.outputs[0].text
            predicted_score = parse_response(response_text)
            results.append({
                'response': response_text,
                'predicted_score': predicted_score
            })
    
    return results


def calculate_metrics(predictions, ground_truths):
    """Calculate evaluation metrics."""
    # Filter out failed predictions
    valid_pairs = [(pred, gt) for pred, gt in zip(predictions, ground_truths) if pred is not None]
    
    if not valid_pairs:
        return None
    
    predictions_valid = [p for p, _ in valid_pairs]
    ground_truths_valid = [g for _, g in valid_pairs]
    
    # Exact accuracy
    exact_matches = sum(1 for pred, gt in valid_pairs if pred == gt)
    exact_accuracy = exact_matches / len(valid_pairs)
    
    # Accuracy within 1 point
    within_1 = sum(1 for pred, gt in valid_pairs if abs(pred - gt) <= 1)
    within_1_accuracy = within_1 / len(valid_pairs)
    
    # Mean Absolute Error
    mae = sum(abs(pred - gt) for pred, gt in valid_pairs) / len(valid_pairs)
    
    # Confusion matrix
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    for pred, gt in valid_pairs:
        confusion_matrix[gt][pred] += 1
    
    # Binary classification metrics (1 vs 5)
    # Ground truth: 1 = bad, 5 = good
    binary_predictions = [1 if p <= 2 else (5 if p >= 4 else 3) for p in predictions_valid]
    binary_ground_truth = [g for g in ground_truths_valid]
    
    # True positives, false positives, etc. for score 1 (bad videos)
    tp_bad = sum(1 for pred, gt in zip(binary_predictions, binary_ground_truth) if pred == 1 and gt == 1)
    fp_bad = sum(1 for pred, gt in zip(binary_predictions, binary_ground_truth) if pred == 1 and gt != 1)
    tn_bad = sum(1 for pred, gt in zip(binary_predictions, binary_ground_truth) if pred != 1 and gt != 1)
    fn_bad = sum(1 for pred, gt in zip(binary_predictions, binary_ground_truth) if pred != 1 and gt == 1)
    
    # True positives, false positives, etc. for score 5 (good videos)
    tp_good = sum(1 for pred, gt in zip(binary_predictions, binary_ground_truth) if pred == 5 and gt == 5)
    fp_good = sum(1 for pred, gt in zip(binary_predictions, binary_ground_truth) if pred == 5 and gt != 5)
    tn_good = sum(1 for pred, gt in zip(binary_predictions, binary_ground_truth) if pred != 5 and gt != 5)
    fn_good = sum(1 for pred, gt in zip(binary_predictions, binary_ground_truth) if pred != 5 and gt == 5)
    
    # Precision, Recall, F1 for bad videos
    precision_bad = tp_bad / (tp_bad + fp_bad) if (tp_bad + fp_bad) > 0 else 0
    recall_bad = tp_bad / (tp_bad + fn_bad) if (tp_bad + fn_bad) > 0 else 0
    f1_bad = 2 * precision_bad * recall_bad / (precision_bad + recall_bad) if (precision_bad + recall_bad) > 0 else 0
    
    # Precision, Recall, F1 for good videos
    precision_good = tp_good / (tp_good + fp_good) if (tp_good + fp_good) > 0 else 0
    recall_good = tp_good / (tp_good + fn_good) if (tp_good + fn_good) > 0 else 0
    f1_good = 2 * precision_good * recall_good / (precision_good + recall_good) if (precision_good + recall_good) > 0 else 0
    
    return {
        'total_samples': len(predictions),
        'valid_predictions': len(valid_pairs),
        'failed_predictions': len(predictions) - len(valid_pairs),
        'exact_accuracy': exact_accuracy,
        'within_1_accuracy': within_1_accuracy,
        'mean_absolute_error': mae,
        'confusion_matrix': dict(confusion_matrix),
        'binary_metrics': {
            'bad_videos': {
                'precision': precision_bad,
                'recall': recall_bad,
                'f1_score': f1_bad,
                'true_positives': tp_bad,
                'false_positives': fp_bad,
                'true_negatives': tn_bad,
                'false_negatives': fn_bad,
            },
            'good_videos': {
                'precision': precision_good,
                'recall': recall_good,
                'f1_score': f1_good,
                'true_positives': tp_good,
                'false_positives': fp_good,
                'true_negatives': tn_good,
                'false_negatives': fn_good,
            }
        },
        'score_distribution': {
            'predictions': dict(Counter(predictions_valid)),
            'ground_truth': dict(Counter(ground_truths_valid)),
        }
    }


def generate_html_report(results, metrics, output_path):
    """Generate HTML evaluation report."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Fine-tuned Model Evaluation Report</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #667eea; text-align: center; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 30px 0; }
        .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; }
        .metric-label { font-size: 14px; opacity: 0.9; }
        .metric-value { font-size: 32px; font-weight: bold; margin-top: 10px; }
        .confusion-matrix { margin: 30px 0; }
        .confusion-matrix table { border-collapse: collapse; width: 100%; max-width: 600px; margin: 20px auto; }
        .confusion-matrix th, .confusion-matrix td { border: 1px solid #ddd; padding: 12px; text-align: center; }
        .confusion-matrix th { background: #667eea; color: white; }
        .result-item { border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .correct { background: #e8f5e9; }
        .incorrect { background: #ffebee; }
        .failed { background: #fff3e0; }
        .score { font-size: 18px; font-weight: bold; display: inline-block; padding: 5px 15px; border-radius: 5px; margin: 5px; }
        .score.pred { background: #bbdefb; }
        .score.truth { background: #c8e6c9; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Fine-tuned Model Evaluation Report</h1>
        <p style="text-align: center; color: #666;">Cosmos Reason 1 - Transfer1 Dataset Evaluation</p>
"""

    # Metrics section
    html += """
        <h2>Overall Metrics</h2>
        <div class="metrics">
"""
    
    if metrics:
        html += f"""
            <div class="metric-card">
                <div class="metric-label">Exact Accuracy</div>
                <div class="metric-value">{metrics['exact_accuracy']:.1%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Within ±1 Accuracy</div>
                <div class="metric-value">{metrics['within_1_accuracy']:.1%}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Mean Absolute Error</div>
                <div class="metric-value">{metrics['mean_absolute_error']:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Valid Predictions</div>
                <div class="metric-value">{metrics['valid_predictions']}/{metrics['total_samples']}</div>
            </div>
        </div>
        
        <h2>Binary Classification Metrics</h2>
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-label">Bad Videos F1-Score</div>
                <div class="metric-value">{metrics['binary_metrics']['bad_videos']['f1_score']:.1%}</div>
                <div style="font-size: 12px; margin-top: 10px;">
                    Precision: {metrics['binary_metrics']['bad_videos']['precision']:.1%} | 
                    Recall: {metrics['binary_metrics']['bad_videos']['recall']:.1%}
                </div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Good Videos F1-Score</div>
                <div class="metric-value">{metrics['binary_metrics']['good_videos']['f1_score']:.1%}</div>
                <div style="font-size: 12px; margin-top: 10px;">
                    Precision: {metrics['binary_metrics']['good_videos']['precision']:.1%} | 
                    Recall: {metrics['binary_metrics']['good_videos']['recall']:.1%}
                </div>
            </div>
        </div>
"""

    # Confusion matrix
    if metrics and metrics.get('confusion_matrix'):
        html += """
        <div class="confusion-matrix">
            <h2>Confusion Matrix</h2>
            <table>
                <tr>
                    <th>Ground Truth \\ Predicted</th>
"""
        # Get all scores
        all_scores = sorted(set(list(metrics['confusion_matrix'].keys()) + 
                              [pred for preds in metrics['confusion_matrix'].values() for pred in preds.keys()]))
        
        for score in all_scores:
            html += f"<th>Score {score}</th>"
        html += "</tr>"
        
        for gt_score in all_scores:
            html += f"<tr><th>Score {gt_score}</th>"
            for pred_score in all_scores:
                count = metrics['confusion_matrix'].get(gt_score, {}).get(pred_score, 0)
                html += f"<td>{count}</td>"
            html += "</tr>"
        
        html += """
            </table>
        </div>
"""

    # Detailed results
    html += """
        <h2>Detailed Results</h2>
"""
    
    for i, result in enumerate(results[:50], 1):  # Show first 50 results
        video_name = Path(result['video_path']).name
        pred_score = result['predicted_score']
        gt_score = result['ground_truth']
        
        if pred_score is None:
            css_class = "failed"
            status = "Failed to Parse"
        elif pred_score == gt_score:
            css_class = "correct"
            status = "Correct"
        else:
            css_class = "incorrect"
            status = "Incorrect"
        
        html += f"""
        <div class="result-item {css_class}">
            <div><strong>{status} - Sample {i}</strong></div>
            <div style="color: #666; font-size: 12px; margin: 5px 0;">{video_name}</div>
            <div>
                <span class="score truth">Ground Truth: {gt_score}</span>
                <span class="score pred">Predicted: {pred_score if pred_score else 'N/A'}</span>
            </div>
            <details style="margin-top: 10px;">
                <summary style="cursor: pointer; color: #667eea;">Show Response</summary>
                <pre style="background: #f5f5f5; padding: 10px; border-radius: 5px; font-size: 12px; overflow-x: auto;">{result['response']}</pre>
            </details>
        </div>
"""
    
    if len(results) > 50:
        html += f'<p style="text-align: center; color: #666; margin: 20px 0;">... and {len(results) - 50} more results</p>'
    
    html += """
    </div>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model on evaluation dataset")
    parser.add_argument("--model_path", type=str, required=True, help="Path to fine-tuned model checkpoint")
    parser.add_argument("--eval_dataset", type=str, required=True, help="Path to evaluation dataset")
    parser.add_argument("--prompt_path", type=str, default="prompts/video_reward.yaml", help="Path to prompt config")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Output directory for results")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=" * 80)
    print("Fine-tuned Model Evaluation")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.eval_dataset}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    print()
    
    # Load prompt configuration
    print("Loading prompt configuration...")
    system_prompt, user_prompt = load_prompt_config(args.prompt_path)
    
    # Load evaluation dataset
    print("Loading evaluation dataset...")
    eval_dataset = load_from_disk(args.eval_dataset)
    print(f"   Loaded {len(eval_dataset)} samples")
    print()
    
    # Load fine-tuned model
    print("Loading fine-tuned model...")
    print("   (This may take a few minutes...)")
    llm = LLM(
        model=args.model_path,
        limit_mm_per_prompt={"image": 0, "video": 1},
        enforce_eager=True,
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    print("   Model loaded successfully")
    print()
    
    # Run inference
    print("Running inference...")
    video_paths = [sample['video_url'] for sample in eval_dataset]
    ground_truths = [sample['pc'] for sample in eval_dataset]
    
    start_time = time.time()
    inference_results = run_inference_batch(
        llm, processor, video_paths, system_prompt, user_prompt, 
        batch_size=args.batch_size
    )
    elapsed_time = time.time() - start_time
    
    print(f"   Inference completed in {elapsed_time:.1f} seconds")
    print(f"   Average: {elapsed_time/len(eval_dataset):.2f} seconds/sample")
    print()
    
    # Combine results
    results = []
    predictions = []
    for i, (sample, inference_result) in enumerate(zip(eval_dataset, inference_results)):
        result = {
            'video_path': sample['video_url'],
            'ground_truth': sample['pc'],
            'predicted_score': inference_result['predicted_score'],
            'response': inference_result['response'],
            'caption': sample['caption']
        }
        results.append(result)
        predictions.append(inference_result['predicted_score'])
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(predictions, ground_truths)
    
    if metrics:
        print()
        print("=" * 80)
        print("EVALUATION RESULTS")
        print("=" * 80)
        print(f"Total Samples:        {metrics['total_samples']}")
        print(f"Valid Predictions:    {metrics['valid_predictions']}")
        print(f"Failed Predictions:   {metrics['failed_predictions']}")
        print(f"Exact Accuracy:       {metrics['exact_accuracy']:.2%}")
        print(f"Within ±1 Accuracy:   {metrics['within_1_accuracy']:.2%}")
        print(f"Mean Absolute Error:  {metrics['mean_absolute_error']:.3f}")
        print()
        print("Binary Classification (Bad vs Good):")
        print(f"  Bad Videos F1:      {metrics['binary_metrics']['bad_videos']['f1_score']:.2%}")
        print(f"  Good Videos F1:     {metrics['binary_metrics']['good_videos']['f1_score']:.2%}")
        print("=" * 80)
    
    # Save results
    print()
    print("Saving results...")
    
    # Save JSON
    json_path = output_dir / "evaluation_results.json"
    with open(json_path, 'w') as f:
        json.dump({
            'metrics': metrics,
            'results': results,
            'config': {
                'model_path': args.model_path,
                'eval_dataset': args.eval_dataset,
                'num_samples': len(eval_dataset),
            }
        }, f, indent=2)
    print(f"   JSON: {json_path}")
    
    # Generate HTML report
    html_path = output_dir / "evaluation_report.html"
    generate_html_report(results, metrics, html_path)
    print(f"   HTML: {html_path}")
    
    print()
    print("=" * 80)
    print("Evaluation completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    print(f"Open the HTML report: {html_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()


