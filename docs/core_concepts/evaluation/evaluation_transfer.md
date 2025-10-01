# Transfer Model Evaluation (ControlNet / Cosmos Transfer)

Evaluate multi‑modality ControlNet models (e.g., Cosmos Transfer) for fidelity to control signals and overall video quality.

## Applicability of Predict Metrics

All metrics documented in [evaluation_predict.md](evaluation_predict.md) apply equally to Transfer (ControlNet) models. Use them alongside the ControlNet‑specific metrics below for a comprehensive evaluation.

## Core Metrics (Control Fidelity & Technical Quality)

### Blur SSIM (Structural Similarity Index Measure)

This metric measures perceptual similarity after applying identical blur to predicted and ground‑truth videos; it is robust to minor misalignment and can be reported per region.

#### How this metric works

1. Apply the same blur strength to both predicted and ground‑truth frames.
2. Compute SSIM considering luminance, contrast, and structure on blurred frames.
3. Average per‑pixel SSIM over frames; optionally compute FG/BG using masks.

### Canny‑F1 Score

This metric measures edge preservation accuracy; it treats edge detection as a binary classification task and reports F1 with precision/recall.

#### How this metric works

1. Extract Canny edge maps for predicted and ground‑truth frames
2. Define positives as “edge” and negatives as “non‑edge”
3. Compute: TP (edge in both), FP (edge only in pred), FN (edge only in GT)
4. F1 = 2 × (Precision × Recall) / (Precision + Recall); also report precision/recall
5. (Optional) FG/BG evaluation via region masks

### Depth RMSE (Root Mean Square Error)

This metric measures scale‑invariant depth error after median scaling; it supports log‑space computation and masking invalid values.

#### How this metric works

1. Use Scale‑Invariant RMSE (SI‑RMSE) for robustness to outliers
2. Median scaling: ratio = median(GT) / median(pred)
3. Compute RMSE after scaling: RMSE = sqrt(mean((GT − scaled_pred)²))
4. (Optional) compute in log‑space; mask zeros/invalid depth

### Seg mIOU (Mean Intersection Over Union)

This metric measures segmentation fidelity between predicted and ground‑truth masks with flexible matching strategies.

#### How this metric works

1. For each object/segment: IOU = Intersection / Union
2. Matching strategies: max‑IOU per GT segment or Hungarian (1‑to‑1 optimal assignment)
3. Report the mean IOU across matches and recall (GT segments detected above threshold)

### Dover Score (Video Quality Assessment)

This metric measures technical video quality, focusing on clarity, compression artifacts, and motion smoothness (not aesthetics).

#### How this metric works

1. Use DOVER (Disentangled Objective Video Quality Evaluator) on full videos.q
2. Assesses clarity/sharpness, compression artifacts, motion smoothness, and overall technical quality.
3. Returns a single quality score; compute for both predicted and ground‑truth videos for comparison.
