# Evaluation Overview

Evaluation is a critical component of any post-training workflow. It serves as the standard by which progress is measured. Establishing robust evaluation methodology and benchmarks *before* post-training begins is essential to ensure meaningful and reproducible results.

This section provides evaluation methods for different types of video generation models. Evaluation strategies differ depending on the type of model being trained, and this guide helps you choose the appropriate approach based on your model type and use case.

## Metric Families

- **Qualitative video quality (Predict)**
    - **FID** — Image realism/diversity via Fréchet distance in Inception feature space (lower is better)
    - **FVD** — Spatio‑temporal quality via Fréchet distance on video features (appearance + motion)
- **Geometric consistency (Predict)**
    - **Sampson Error** — First‑order point‑to‑epipolar‑line distance
    - **TSE/CSE** — Temporal and cross‑view consistency for multi‑view videos
- **VLM‑based assessment**
    - **Cosmos Reason** — JPhysical plausibility, causal/temporal reasoning; usable as critic or reward model.
- **Transfer/Control quality**
    - **Blur SSIM, Canny‑F1, Depth RMSE, Seg mIOU, Dover** — Fidelity to control signals and technical quality.

## Recommended Workflow

1. Specify evaluation split(s) and frame/clip sampling policy.
2. Select metrics per model type (Predict vs Transfer/Control) and goals.
3. Align preprocessing (resolution, crop, fps) across Pred ↔ GT.
4. Run core metrics (FID/FVD or control metrics), then geometric checks (TSE/CSE) as needed.
5. Add VLM analysis (critic/reward) for physics and reasoning quality.
6. Report means, variance/CI, and exact configs to ensure reproducibility.

## Model→Metric Map

- Predict (Generative) → [evaluation_predict.md](evaluation_predict.md)
- Transfer/Control (ControlNet) → [evaluation_transfer.md](evaluation_transfer.md)
- Reason Reward/Critic (VLM) → [reason_as_reward.md](reason_as_reward.md)
