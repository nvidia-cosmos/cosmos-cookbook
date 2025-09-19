# Transfer Model Evaluation (ControlNet / Cosmos Transfer)

Evaluate multi‑modality ControlNet models (e.g., Cosmos Transfer) for fidelity to control signals and overall video quality.

## Applicability of Predict Metrics

All metrics documented in [evaluation_predict.md](evaluation_predict.md) apply equally to Transfer (ControlNet) models. Use them alongside the ControlNet‑specific metrics below for a comprehensive evaluation.

## Core Metrics (Control Fidelity & Technical Quality)

### Blur SSIM (Structural Similarity Index Measure)

Perceptual similarity after applying identical blur to predicted and ground‑truth videos; robust to minor misalignment and can be reported per region.

How it works:

- Apply the same blur strength to both predicted and ground‑truth frames
- Compute SSIM considering luminance, contrast, and structure on blurred frames
- Average per‑pixel SSIM over frames; optionally compute FG/BG using masks

### Canny‑F1 Score

Edge preservation accuracy; treats edge detection as a binary classification task and reports F1 with precision/recall.

How it works:

- Extract Canny edge maps for predicted and ground‑truth frames
- Define positives as “edge” and negatives as “non‑edge”
- Compute: TP (edge in both), FP (edge only in pred), FN (edge only in GT)
- F1 = 2 × (Precision × Recall) / (Precision + Recall); also report precision/recall
- Optional FG/BG evaluation via region masks

### Depth RMSE (Root Mean Square Error)

Scale‑invariant depth error after median scaling; supports log‑space computation and masking invalid values.

How it works:

- Use Scale‑Invariant RMSE (SI‑RMSE) for robustness to outliers
- Median scaling: ratio = median(GT) / median(pred)
- Compute RMSE after scaling: RMSE = sqrt(mean((GT − scaled_pred)²))
- Optional: compute in log‑space; mask zeros/invalid depth

### Seg mIOU (Mean Intersection Over Union)

Segmentation fidelity between predicted and ground‑truth masks with flexible matching strategies.

How it works:

- For each object/segment: IOU = Intersection / Union
- Matching strategies: max‑IOU per GT segment or Hungarian (1‑to‑1 optimal assignment)
- Report mean IOU across matches and recall (GT segments detected above threshold)

### Dover Score (Video Quality Assessment)

Technical video quality score focusing on clarity, compression artifacts, and motion smoothness (not aesthetics).

How it works:

- Use DOVER (Disentangled Objective Video Quality Evaluator) on full videos
- Assess clarity/sharpness, compression artifacts, motion smoothness, overall technical quality
- Return a single quality score; compute for both predicted and ground‑truth videos for comparison
