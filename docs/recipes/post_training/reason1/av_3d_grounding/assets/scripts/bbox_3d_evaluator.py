#!/usr/bin/env python3
# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
Evaluate 3D bounding box predictions against ground truth using IoU matching.

This module provides functionality to compare predicted 3D bounding boxes with ground truth
annotations. It uses the Hungarian algorithm for optimal matching and computes Intersection
over Union (IoU) metrics for evaluation.

The evaluation process:
1. Loads prediction and ground truth JSON files from specified directories
2. Matches files by filename between the two directories
3. For each matching file pair, performs optimal matching using Hungarian algorithm
4. Computes 3D IoU for matched boxes (only matching boxes with same labels)
5. Computes Average Precision (AP) metrics for both 2D and 3D boxes
6. Aggregates results across all files

Usage:
    python docs/recipes/post_training/reason1/av_3d_grounding/assets/scripts/bbox_3d_evaluator.py <pred_dir> <gt_dir> [--iou-threshold 0.5] [--verbose]

    Example:
        python path/to/bbox_3d_evaluator.py output_eval/text_qwen30b output_eval/text_gt --iou-threshold 0.5 --verbose

Input JSON Format:
    Each JSON file should contain:
    {
        "camera_params": {
            "fx": 359.0,
            "fy": 540.0,
            "cx": 641.9,
            "cy": 360.9
        },
        "annotations": [
            {
                "bbox_3d": [x, y, z, width, height, depth, roll, pitch, yaw],
                "label": "car"
            },
            ...
        ]
    }

    Note:
    - Only the first 6 values (x, y, z, width, height, depth) are used for 3D IoU calculation.
      Rotation (roll, pitch, yaw) is ignored for axis-aligned 3D IoU computation.
    - camera_params is optional but required for AP2D metrics. If present, it enables 2D projection
      and AP2D calculation.
    - Angles (roll, pitch, yaw) are assumed to be in radians.

Metrics:
    - AP3D: Average Precision for 3D boxes (averaged across IoU thresholds 0.5:0.05:0.95)
    - AP3D@0.5: Average Precision at IoU threshold 0.5
    - AP3D@0.75: Average Precision at IoU threshold 0.75
    - AP2D: Average Precision for 2D projected boxes (if camera_params available)
    - AP2D@0.5: Average Precision for 2D boxes at IoU threshold 0.5
    - AP2D@0.75: Average Precision for 2D boxes at IoU threshold 0.75
"""

import argparse
import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def load_boxes(
    json_path: Path,
) -> tuple[list[list[float]], list[str], dict[str, float] | None]:
    """
    Load bbox_3d and label arrays from a JSON annotation file.

    Args:
        json_path: Path to JSON file containing annotations

    Returns:
        Tuple of (boxes, labels, camera_params) where:
        - boxes: List of bbox_3d lists, each containing [x, y, z, width, height, depth, ...]
        - labels: List of label strings corresponding to each box
        - camera_params: Dictionary with keys 'fx', 'fy', 'cx', 'cy' or None if not present

    Raises:
        FileNotFoundError: If the JSON file doesn't exist
        json.JSONDecodeError: If the file is not valid JSON
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    boxes = [ann["bbox_3d"] for ann in data.get("annotations", [])]
    labels = [ann["label"] for ann in data.get("annotations", [])]
    camera_params = data.get("camera_params", None)
    return boxes, labels, camera_params


def normalize_label(label: str) -> str:
    """
    Normalize label by extracting base name and removing suffixes.

    Removes numeric suffixes and underscores (e.g., "Car_5" → "car", "van_100" → "van").
    Converts to lowercase for case-insensitive matching.

    Args:
        label: Original label string (e.g., "Car_5", "van_100", "truck_1")

    Returns:
        Normalized label (lowercase, base name only)

    Examples:
        "Car_5" → "car"
        "van_100" → "van"
        "truck_1" → "truck"
        "pedestrian" → "pedestrian"
    """
    label_lower = label.lower().strip()

    # Split by underscore and take the first part (base name)
    # This handles cases like "Car_5", "van_100", "truck_1", etc.
    base_name = label_lower.split("_")[0]

    return base_name


def project_3d_to_2d_bbox(
    bbox_3d: list[float],
    camera_params: dict[str, float],
) -> tuple[float, float, float, float] | None:
    """
    Project a 3D bounding box to 2D image coordinates and return 2D bounding box.

    Args:
        bbox_3d: 3D bounding box as [x, y, z, width, height, depth, roll, pitch, yaw]
                 where angles are in radians. If fewer than 9 values, rotation is assumed to be 0.
        camera_params: Dictionary with keys 'fx', 'fy', 'cx', 'cy'

    Returns:
        Tuple of (min_x, min_y, max_x, max_y) in pixel coordinates, or None if box is behind camera
    """
    if len(bbox_3d) < 6:
        return None

    x, y, z, x_size, y_size, z_size = bbox_3d[:6]
    # Default rotation to 0 if not provided (angles assumed to be in radians)
    if len(bbox_3d) >= 9:
        roll, pitch, yaw = bbox_3d[6], bbox_3d[7], bbox_3d[8]
    else:
        roll, pitch, yaw = 0.0, 0.0, 0.0
    hx, hy, hz = x_size / 2, y_size / 2, z_size / 2

    # 8 corners of the 3D box in local coordinates
    local_corners = [
        [hx, hy, hz],
        [hx, hy, -hz],
        [hx, -hy, hz],
        [hx, -hy, -hz],
        [-hx, hy, hz],
        [-hx, hy, -hz],
        [-hx, -hy, hz],
        [-hx, -hy, -hz],
    ]

    def rotate_xyz(
        _point: list[float], _pitch: float, _yaw: float, _roll: float
    ) -> list[float]:
        """Rotate point by pitch, yaw, roll angles (in radians)."""
        x0, y0, z0 = _point
        x1 = x0
        y1 = y0 * math.cos(_pitch) - z0 * math.sin(_pitch)
        z1 = y0 * math.sin(_pitch) + z0 * math.cos(_pitch)
        x2 = x1 * math.cos(_yaw) + z1 * math.sin(_yaw)
        y2 = y1
        z2 = -x1 * math.sin(_yaw) + z1 * math.cos(_yaw)
        x3 = x2 * math.cos(_roll) - y2 * math.sin(_roll)
        y3 = x2 * math.sin(_roll) + y2 * math.cos(_roll)
        z3 = z2
        return [x3, y3, z3]

    fx, fy, cx, cy = (
        camera_params["fx"],
        camera_params["fy"],
        camera_params["cx"],
        camera_params["cy"],
    )
    img_corners = []

    for corner in local_corners:
        rotated = rotate_xyz(corner, pitch, yaw, roll)
        X, Y, Z = rotated[0] + x, rotated[1] + y, rotated[2] + z
        if Z > 0:  # Only project points in front of camera
            x_2d = fx * (X / Z) + cx
            y_2d = fy * (Y / Z) + cy
            img_corners.append([x_2d, y_2d])

    if len(img_corners) == 0:
        return None  # All corners behind camera

    img_corners = np.array(img_corners)
    min_x = float(np.min(img_corners[:, 0]))
    min_y = float(np.min(img_corners[:, 1]))
    max_x = float(np.max(img_corners[:, 0]))
    max_y = float(np.max(img_corners[:, 1]))

    return (min_x, min_y, max_x, max_y)


def compute_2d_iou(
    box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]
) -> float:
    """
    Compute 2D Intersection over Union (IoU) for axis-aligned bounding boxes.

    Args:
        box1: First bounding box as (min_x, min_y, max_x, max_y)
        box2: Second bounding box as (min_x, min_y, max_x, max_y)

    Returns:
        IoU value between 0.0 and 1.0
    """
    min_x1, min_y1, max_x1, max_y1 = box1
    min_x2, min_y2, max_x2, max_y2 = box2

    # Calculate intersection
    inter_min_x = max(min_x1, min_x2)
    inter_min_y = max(min_y1, min_y2)
    inter_max_x = min(max_x1, max_x2)
    inter_max_y = min(max_y1, max_y2)

    if inter_max_x < inter_min_x or inter_max_y < inter_min_y:
        return 0.0

    inter_area = (inter_max_x - inter_min_x) * (inter_max_y - inter_min_y)

    # Calculate union
    area1 = (max_x1 - min_x1) * (max_y1 - min_y1)
    area2 = (max_x2 - min_x2) * (max_y2 - min_y2)
    union_area = area1 + area2 - inter_area

    return float(inter_area / union_area) if union_area > 0 else 0.0


def compute_3d_iou(box1: list[float], box2: list[float]) -> float:
    """
    Compute axis-aligned 3D Intersection over Union (IoU) ignoring rotation.

    This function computes the IoU of two 3D bounding boxes by treating them as
    axis-aligned boxes. Rotation information (roll, pitch, yaw) is ignored.

    Args:
        box1: First bounding box as [x, y, z, width, height, depth, ...]
        box2: Second bounding box as [x, y, z, width, height, depth, ...]

    Returns:
        IoU value between 0.0 and 1.0, where:
        - 0.0 means no overlap
        - 1.0 means perfect overlap
        - Returns 0.0 if union volume is zero

    Note:
        Only the first 6 values (center position and dimensions) are used.
        The boxes are assumed to be axis-aligned (rotation is ignored).
    """
    b1 = np.array(box1[:6])
    b2 = np.array(box2[:6])

    def to_min_max(b: np.ndarray) -> np.ndarray:
        """Convert center-size representation to min-max representation."""
        x, y, z, w, h, d = b
        return np.array(
            [x - w / 2, x + w / 2, y - h / 2, y + h / 2, z - d / 2, z + d / 2]
        )

    mm1 = to_min_max(b1)
    mm2 = to_min_max(b2)

    inter_min = np.maximum(mm1[[0, 2, 4]], mm2[[0, 2, 4]])
    inter_max = np.minimum(mm1[[1, 3, 5]], mm2[[1, 3, 5]])
    inter = np.maximum(inter_max - inter_min, 0)
    inter_vol = np.prod(inter)

    vol1 = np.prod(mm1[[1, 3, 5]] - mm1[[0, 2, 4]])
    vol2 = np.prod(mm2[[1, 3, 5]] - mm2[[0, 2, 4]])

    union = vol1 + vol2 - inter_vol
    return float(inter_vol / union) if union > 0 else 0.0


def compute_average_precision(
    pred_boxes: list[list[float]],
    pred_labels: list[str],
    gt_boxes: list[list[float]],
    gt_labels: list[str],
    iou_thresholds: list[float],
    compute_iou_fn: Callable[[list[float], list[float]], float],
    camera_params: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Compute Average Precision (AP) metric following COCO-style evaluation.

    Args:
        pred_boxes: List of predicted bounding boxes
        pred_labels: List of labels for predicted boxes
        gt_boxes: List of ground truth bounding boxes
        gt_labels: List of labels for ground truth boxes
        iou_thresholds: List of IoU thresholds to evaluate (e.g., [0.5, 0.55, ..., 0.95])
        compute_iou_fn: Function to compute IoU between two boxes
        camera_params: Camera parameters dict (required for 2D AP, None for 3D AP)

    Returns:
        Dictionary containing:
        - "ap": Average Precision averaged across all IoU thresholds
        - "ap_50": Average Precision at IoU threshold 0.5
        - "ap_75": Average Precision at IoU threshold 0.75
    """
    if len(gt_boxes) == 0:
        return {"ap": 0.0, "ap_50": 0.0, "ap_75": 0.0}

    if len(pred_boxes) == 0:
        return {"ap": 0.0, "ap_50": 0.0, "ap_75": 0.0}

    # Normalize labels
    gt_labels_normalized = [normalize_label(label) for label in gt_labels]
    pred_labels_normalized = [normalize_label(label) for label in pred_labels]

    # Group predictions and ground truth by label
    label_to_preds: dict[str, list[tuple[int, list[float]]]] = {}
    label_to_gts: dict[str, list[tuple[int, list[float]]]] = {}

    for i, (box, label_norm) in enumerate(
        zip(pred_boxes, pred_labels_normalized, strict=False)
    ):
        if label_norm not in label_to_preds:
            label_to_preds[label_norm] = []
        label_to_preds[label_norm].append((i, box))

    for i, (box, label_norm) in enumerate(
        zip(gt_boxes, gt_labels_normalized, strict=False)
    ):
        if label_norm not in label_to_gts:
            label_to_gts[label_norm] = []
        label_to_gts[label_norm].append((i, box))

    # Compute AP for each label and average
    aps_per_threshold: dict[float, list[float]] = {
        thresh: [] for thresh in iou_thresholds
    }

    for label_norm in set(gt_labels_normalized):
        if label_norm not in label_to_gts:
            continue

        label_gt_boxes = [box for _, box in label_to_gts[label_norm]]
        label_pred_boxes = [box for _, box in label_to_preds.get(label_norm, [])]

        if len(label_gt_boxes) == 0:
            continue

        if len(label_pred_boxes) == 0:
            # No predictions for this label, AP = 0
            for thresh in iou_thresholds:
                aps_per_threshold[thresh].append(0.0)
            continue

        # Compute IoU matrix for this label
        iou_matrix = np.zeros((len(label_gt_boxes), len(label_pred_boxes)))
        for i, gt_box in enumerate(label_gt_boxes):
            for j, pred_box in enumerate(label_pred_boxes):
                if camera_params is not None:
                    # For 2D AP, project boxes first
                    gt_2d = project_3d_to_2d_bbox(gt_box, camera_params)
                    pred_2d = project_3d_to_2d_bbox(pred_box, camera_params)
                    if gt_2d is None or pred_2d is None:
                        iou_matrix[i, j] = 0.0
                    else:
                        iou_matrix[i, j] = compute_2d_iou(gt_2d, pred_2d)
                else:
                    # For 3D AP, use 3D IoU
                    iou_matrix[i, j] = compute_iou_fn(gt_box, pred_box)

        # For each IoU threshold, compute AP
        for thresh in iou_thresholds:
            # Use Hungarian matching to find optimal assignment
            cost = 1.0 - iou_matrix
            cost[cost < 0] = 1e6  # Set negative costs (IoU > 1) to impossible
            cost[iou_matrix < thresh] = 1e6  # Set low IoU matches to impossible

            if len(label_gt_boxes) > 0 and len(label_pred_boxes) > 0:
                gt_idx, pred_idx = linear_sum_assignment(cost)

                # Count true positives (matched with IoU >= threshold)
                tp = 0
                for g, p in zip(gt_idx, pred_idx, strict=False):
                    if cost[g, p] < 1e6:  # Valid match
                        tp += 1

                # Compute precision and recall
                precision = (
                    tp / len(label_pred_boxes) if len(label_pred_boxes) > 0 else 0.0
                )
                # recall = tp / len(label_gt_boxes) if len(label_gt_boxes) > 0 else 0.0

                # AP is the same as precision when using optimal matching
                # (since we're matching optimally, precision = recall = AP for single threshold)
                ap = precision if len(label_gt_boxes) > 0 else 0.0
                aps_per_threshold[thresh].append(ap)
            else:
                aps_per_threshold[thresh].append(0.0)

    # Average AP across all labels for each threshold
    ap_per_threshold = {
        thresh: float(np.mean(aps)) if len(aps) > 0 else 0.0
        for thresh, aps in aps_per_threshold.items()
    }

    # Compute overall AP (average across all thresholds)
    overall_ap = (
        float(np.mean(list(ap_per_threshold.values()))) if ap_per_threshold else 0.0
    )

    return {
        "ap": overall_ap,
        "ap_50": ap_per_threshold.get(0.5, 0.0),
        "ap_75": ap_per_threshold.get(0.75, 0.0),
    }


def evaluate(
    pred_boxes: list[list[float]],
    pred_labels: list[str],
    gt_boxes: list[list[float]],
    gt_labels: list[str],
    iou_threshold: float = 0.5,
    camera_params: dict[str, float] | None = None,
) -> dict[str, float | list[float] | int | str]:
    """
    Evaluate predictions against ground truth using optimal matching (Hungarian algorithm).

    This function performs order-independent matching between predicted and ground truth
    boxes. It uses the Hungarian algorithm to find the optimal assignment that maximizes
    total IoU while ensuring:
    - Only boxes with matching labels can be matched
    - Each ground truth box can match at most one prediction
    - Each prediction can match at most one ground truth

    Args:
        pred_boxes: List of predicted bounding boxes, each as [x, y, z, w, h, d, ...]
        pred_labels: List of labels for predicted boxes
        gt_boxes: List of ground truth bounding boxes, each as [x, y, z, w, h, d, ...]
        gt_labels: List of labels for ground truth boxes
        iou_threshold: IoU threshold for considering a match as correct (default: 0.5)

    Returns:
        Dictionary containing:
        - "matched_ious": List of IoU values for all matched pairs
        - "matched_labels": List of (gt_label, pred_label) tuples for matched pairs
        - "mean_iou": Mean IoU across all matched pairs
        - "accuracy_percent": Percentage of ground truth boxes with IoU >= threshold
        - "label_accuracy_percent": Percentage of ground truth boxes with correct label match
        - "total_gt": Total number of ground truth boxes
        - "total_pred": Total number of predicted boxes
        - "matched_pairs": Number of successfully matched pairs
        - "ap_3d": Average Precision for 3D boxes (averaged across IoU thresholds 0.5:0.05:0.95)
        - "ap_3d_50": Average Precision for 3D boxes at IoU threshold 0.5
        - "ap_3d_75": Average Precision for 3D boxes at IoU threshold 0.75
        - "ap_2d": Average Precision for 2D boxes (if camera_params provided)
        - "ap_2d_50": Average Precision for 2D boxes at IoU threshold 0.5 (if camera_params provided)
        - "ap_2d_75": Average Precision for 2D boxes at IoU threshold 0.75 (if camera_params provided)

    Note:
        - If there are no ground truth boxes, accuracy is 100% if there are no predictions,
          otherwise 0%
        - Boxes with different normalized labels cannot be matched (cost = 1e6)
        - Label matching extracts base names by removing suffixes (e.g., "Car_5" → "car", "van_100" → "van")
        - Labels are matched case-insensitively after normalization
        - The matching is optimal in terms of maximizing total IoU
        - Label accuracy is always 100% for matched pairs since only same normalized-label boxes can match
        - AP metrics are computed using optimal matching per label class
    """
    n_pred = len(pred_boxes)
    n_gt = len(gt_boxes)

    if n_gt == 0:
        result = {
            "matched_ious": [],
            "matched_labels": [],
            "mean_iou": 0.0,
            "accuracy_percent": 100.0 if n_pred == 0 else 0.0,
            "label_accuracy_percent": 100.0 if n_pred == 0 else 0.0,
            "total_gt": 0,
            "total_pred": n_pred,
            "matched_pairs": 0,
            "ap_3d": 0.0,
            "ap_3d_50": 0.0,
            "ap_3d_75": 0.0,
        }
        if camera_params is not None:
            result["ap_2d"] = 0.0
            result["ap_2d_50"] = 0.0
            result["ap_2d_75"] = 0.0
        return result

    # Normalize labels for flexible matching (vehicle names grouped together)
    gt_labels_normalized = [normalize_label(label) for label in gt_labels]
    pred_labels_normalized = [normalize_label(label) for label in pred_labels]

    # Cost matrix (1 - IoU), but only for matching same normalized labels
    cost = np.ones((n_gt, n_pred))  # initialize large costs

    for i in range(n_gt):
        for j in range(n_pred):
            if (
                gt_labels_normalized[i] == pred_labels_normalized[j]
            ):  # normalized labels must match
                iou = compute_3d_iou(gt_boxes[i], pred_boxes[j])
                cost[i][j] = 1 - iou  # cost = 1 - IoU
            else:
                cost[i][j] = 1e6  # impossible match

    # Run Hungarian matching
    gt_idx, pred_idx = linear_sum_assignment(cost)

    matched_ious = []
    matched_labels = []
    correct = 0
    label_correct = 0

    for g, p in zip(gt_idx, pred_idx, strict=False):
        if cost[g][p] >= 1e6:  # label mismatch case
            continue

        iou = 1 - cost[g][p]
        matched_ious.append(iou)
        # Store original labels (not normalized) for reporting
        matched_labels.append((gt_labels[g], pred_labels[p]))

        # Label is already matched (since we skip label mismatches)
        # Check if normalized labels match (they should, but verify)
        if gt_labels_normalized[g] == pred_labels_normalized[p]:
            label_correct += 1

        if iou >= iou_threshold:
            correct += 1

    accuracy = (correct / len(gt_boxes)) * 100 if len(gt_boxes) > 0 else 0.0
    label_accuracy = (label_correct / len(gt_boxes)) * 100 if len(gt_boxes) > 0 else 0.0

    # Compute AP3D metrics
    iou_thresholds_3d = [0.5 + 0.05 * i for i in range(10)]  # 0.5:0.05:0.95
    ap_3d_results = compute_average_precision(
        pred_boxes,
        pred_labels,
        gt_boxes,
        gt_labels,
        iou_thresholds_3d,
        compute_3d_iou,
        camera_params=None,
    )

    result = {
        "matched_ious": matched_ious,
        "matched_labels": matched_labels,
        "mean_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
        "accuracy_percent": accuracy,
        "label_accuracy_percent": label_accuracy,
        "total_gt": len(gt_boxes),
        "total_pred": len(pred_boxes),
        "matched_pairs": len(matched_ious),
        "ap_3d": ap_3d_results["ap"],
        "ap_3d_50": ap_3d_results["ap_50"],
        "ap_3d_75": ap_3d_results["ap_75"],
    }

    # Compute AP2D metrics if camera parameters are available
    if camera_params is not None:
        ap_2d_results = compute_average_precision(
            pred_boxes,
            pred_labels,
            gt_boxes,
            gt_labels,
            iou_thresholds_3d,
            compute_3d_iou,
            camera_params=camera_params,
        )
        result["ap_2d"] = ap_2d_results["ap"]
        result["ap_2d_50"] = ap_2d_results["ap_50"]
        result["ap_2d_75"] = ap_2d_results["ap_75"]

    return result


def main() -> None:
    """
    Main function to compare prediction and ground truth JSON files file-by-file.

    This function:
    1. Parses command-line arguments
    2. Finds all JSON files in both prediction and ground truth directories
    3. Matches files by filename
    4. Evaluates each matching file pair
    5. Aggregates results and prints summary statistics

    Command-line arguments:
        pred_dir: Directory containing prediction JSON files
        gt_dir: Directory containing ground truth JSON files
        --iou-threshold: IoU threshold for correct matches (default: 0.5)
        --verbose: Print per-file results in addition to overall summary

    Output:
        Prints overall statistics including:
        - Number of files processed
        - Mean IoU across all matches
        - Overall accuracy percentage
        - Total ground truth and prediction counts

        If --verbose is set, also prints per-file results.

    Raises:
        ValueError: If either directory doesn't exist
    """
    parser = argparse.ArgumentParser(
        description="Evaluate 3D bounding box predictions against ground truth"
    )
    parser.add_argument(
        "pred_dir",
        type=str,
        help="Directory containing prediction JSON files",
    )
    parser.add_argument(
        "gt_dir",
        type=str,
        help="Directory containing ground truth JSON files",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for considering a match correct (default: 0.5)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-file results",
    )

    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)

    if not pred_dir.exists():
        raise ValueError(f"Prediction directory does not exist: {pred_dir}")
    if not gt_dir.exists():
        raise ValueError(f"Ground truth directory does not exist: {gt_dir}")

    # Find all JSON files in both directories
    pred_files = {f.name: f for f in pred_dir.glob("*.json")}
    gt_files = {f.name: f for f in gt_dir.glob("*.json")}

    # Find common files
    common_files = set(pred_files.keys()) & set(gt_files.keys())
    pred_only = set(pred_files.keys()) - set(gt_files.keys())
    gt_only = set(gt_files.keys()) - set(pred_files.keys())

    if not common_files:
        print("No matching files found between the two directories!")
        print(f"Prediction-only files: {len(pred_only)}")
        print(f"Ground truth-only files: {len(gt_only)}")
        return

    print(f"Found {len(common_files)} matching files")
    if pred_only:
        print(f"Warning: {len(pred_only)} files only in prediction directory")
    if gt_only:
        print(f"Warning: {len(gt_only)} files only in ground truth directory")
    print()

    # Evaluate each file pair
    per_file_results = []
    for filename in sorted(common_files):
        pred_path = pred_files[filename]
        gt_path = gt_files[filename]

        try:
            pred_boxes, pred_labels, pred_camera_params = load_boxes(pred_path)
            gt_boxes, gt_labels, gt_camera_params = load_boxes(gt_path)

            # Use camera params from GT if available, otherwise from predictions
            camera_params = (
                gt_camera_params if gt_camera_params is not None else pred_camera_params
            )

            result = evaluate(
                pred_boxes,
                pred_labels,
                gt_boxes,
                gt_labels,
                args.iou_threshold,
                camera_params=camera_params,
            )
            result["filename"] = filename
            per_file_results.append(result)

            if args.verbose:
                print(f"{filename}:")
                print(f"  Mean IoU: {result['mean_iou']:.4f}")
                print(f"  IoU Accuracy: {result['accuracy_percent']:.2f}%")
                print(f"  Label Accuracy: {result['label_accuracy_percent']:.2f}%")
                print(
                    f"  AP3D: {result['ap_3d']:.4f}, AP3D@0.5: {result['ap_3d_50']:.4f}, AP3D@0.75: {result['ap_3d_75']:.4f}"
                )
                if "ap_2d" in result:
                    print(
                        f"  AP2D: {result['ap_2d']:.4f}, AP2D@0.5: {result['ap_2d_50']:.4f}, AP2D@0.75: {result['ap_2d_75']:.4f}"
                    )
                print(
                    f"  GT boxes: {result['total_gt']}, Pred boxes: {result['total_pred']}, Matched: {result['matched_pairs']}"
                )
                print()

        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            print(f"Error processing {filename}: {e}")
            continue

    # Aggregate results
    if per_file_results:
        all_matched_ious = []
        total_gt = 0
        total_pred = 0
        total_correct = 0
        total_label_correct = 0
        total_matched = 0
        ap_3d_values = []
        ap_3d_50_values = []
        ap_3d_75_values = []
        ap_2d_values = []
        ap_2d_50_values = []
        ap_2d_75_values = []
        has_2d_metrics = False

        for result in per_file_results:
            all_matched_ious.extend(result["matched_ious"])
            total_gt += result["total_gt"]
            total_pred += result["total_pred"]
            total_matched += result["matched_pairs"]
            # Calculate correct from accuracy
            correct = int((result["accuracy_percent"] / 100.0) * result["total_gt"])
            total_correct += correct
            label_correct = int(
                (result["label_accuracy_percent"] / 100.0) * result["total_gt"]
            )
            total_label_correct += label_correct

            # Collect AP metrics
            ap_3d_values.append(result["ap_3d"])
            ap_3d_50_values.append(result["ap_3d_50"])
            ap_3d_75_values.append(result["ap_3d_75"])

            if "ap_2d" in result:
                has_2d_metrics = True
                ap_2d_values.append(result["ap_2d"])
                ap_2d_50_values.append(result["ap_2d_50"])
                ap_2d_75_values.append(result["ap_2d_75"])

        overall_result = {
            "num_files": len(per_file_results),
            "mean_iou": float(np.mean(all_matched_ious)) if all_matched_ious else 0.0,
            "iou_accuracy_percent": (total_correct / total_gt * 100)
            if total_gt > 0
            else 0.0,
            "label_accuracy_percent": (total_label_correct / total_gt * 100)
            if total_gt > 0
            else 0.0,
            "total_gt": total_gt,
            "total_pred": total_pred,
            "total_matched": total_matched,
            "total_correct": total_correct,
            "total_label_correct": total_label_correct,
            "ap_3d": float(np.mean(ap_3d_values)) if ap_3d_values else 0.0,
            "ap_3d_50": float(np.mean(ap_3d_50_values)) if ap_3d_50_values else 0.0,
            "ap_3d_75": float(np.mean(ap_3d_75_values)) if ap_3d_75_values else 0.0,
        }

        if has_2d_metrics:
            overall_result["ap_2d"] = (
                float(np.mean(ap_2d_values)) if ap_2d_values else 0.0
            )
            overall_result["ap_2d_50"] = (
                float(np.mean(ap_2d_50_values)) if ap_2d_50_values else 0.0
            )
            overall_result["ap_2d_75"] = (
                float(np.mean(ap_2d_75_values)) if ap_2d_75_values else 0.0
            )

        print("=" * 60)
        print("Overall Results:")
        print("=" * 60)
        print(json.dumps(overall_result, indent=2))

        if args.verbose:
            print("\nPer-file results:")
            print(json.dumps(per_file_results, indent=2))


if __name__ == "__main__":
    main()
