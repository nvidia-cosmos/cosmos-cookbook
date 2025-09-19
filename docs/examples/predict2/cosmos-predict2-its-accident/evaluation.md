# Evaluation

This guide covers the evaluation methodology for comparing the original Cosmos-Predict2 model with the LoRA post-trained version on single-view CCTV video generation.

## Evaluation Metrics

### Quantitative Metrics

We employ two primary metrics for objective evaluation of video generation quality:

#### 1. **FID (Fréchet Inception Distance)**

FID ([Heusel et al., 2017](https://arxiv.org/abs/1706.08500)) measures the similarity between the distribution of generated videos and real videos by comparing features extracted from a pre-trained Inception network.

- **Lower is better**: Values closer to 0 indicate better quality
- **Typical ranges**:
    - Excellent: < 10
    - Good: 10-30
    - Acceptable: 30-50
    - Poor: > 50
- **What it measures**: Visual quality and realism at the frame level

#### 2. **FVD (Fréchet Video Distance)**

FVD ([Unterthiner et al., 2018](https://arxiv.org/abs/1812.01717)) extends FID to the temporal dimension, evaluating both visual quality and temporal consistency using an I3D network.

- **Lower is better**: Values closer to 0 indicate better quality
- **Typical ranges**:
    - Excellent: < 100
    - Good: 100-200
    - Acceptable: 200-400
    - Poor: > 400
- **What it measures**: Visual quality AND temporal coherence

### Why These Metrics Matter for ITS

- **FID**: Validates visual realism of individual frames from single camera view
- **FVD**: Ensures temporal consistency and realistic motion dynamics
- Together, they quantify improvements in single-view traffic video generation

### Limitations of FID/FVD Metrics

While FID and FVD effectively measure visual quality and temporal consistency, they have notable limitations for safety-critical ITS applications. These metrics primarily evaluate statistical distributions of visual features but cannot assess **physical plausibility** - a crucial aspect for collision scenarios. For comprehensive evaluation of physical plausibility in generated accidents, additional assessment using physics-aware models like [Cosmos-Reason1](https://github.com/nvidia-cosmos/cosmos-reason1) would be beneficial, as such models can judge whether collision dynamics follow real-world physics principles and spatial-temporal constraints.

## Evaluation Pipeline

### Step 1: Prepare Evaluation Dataset

Creating a standardized evaluation dataset for single-view CCTV footage:

#### Dataset Requirements

1. **Scenario Coverage**
Different type of scenerios should be covered e.g.:

   - Intersection collisions (T-bone, rear-end)
   - Highway incidents (pile-ups, lane changes)
   - Normal traffic flows (baseline quality testing)
   - Various conditions (day/night, weather)

2. **Standardization**
   - Fixed resolution: 1280x720 (single camera view)
   - 50-100 samples per category
   - Consistent single-view perspective
   - Clear prompt descriptions

3. **Ground Truth**
   - Real single-view CCTV footage for metric computation
   - Matching perspective and quality to target deployment

### Step 2: Generate Videos with Both Models

Generate single-view videos from both models for comparison:

#### Baseline Model

- Use original Cosmos-Predict2 without adaptations
- Fixed seeds and parameters for all test scenarios
- Output to baseline directory

#### LoRA Post-Trained Model

- Load LoRA checkpoint with rank=16, alpha=16
- Same test scenarios, seeds, and parameters as baseline
- Output to LoRA directory

**Requirements**: Identical inputs, prompts, seeds, and generation parameters for fair comparison

### Step 3: Compute Metrics

#### FID Score Computation

FID measures visual quality by comparing frame distributions:

- Extract frames from real and generated single-view videos
- Compute Inception network features for each frame
- Calculate Fréchet distance between feature distributions
- **Lower scores = better visual quality** (target: < 25 for good quality)

#### FVD Score Computation

FVD evaluates temporal consistency in single-view videos:

- Process complete video sequences using I3D network
- Capture spatiotemporal features across frames
- Compare distributions between real and generated videos
- **Lower scores = better temporal coherence** (target: < 200 for good quality)

Both metrics require 50-100 single-view videos for statistical reliability.

### Step 4: Comparative Analysis

Evaluate performance across:

- **Checkpoints**: Test iterations 1000, 2000, 5000, 10000 to find optimal model
- **Scenarios**: Compare improvements in collisions vs. normal traffic
- **Conditions**: Assess day/night and weather variations
- Select best checkpoint based on lowest FID/FVD scores

## Expected Results

### Typical Improvements from LoRA Post-Training

| Metric | Baseline Model | LoRA Post-Trained | Improvement |
|--------|---------------|-------------------|-------------|
| **FID Score** | ~35-40 | ~20-25 | 35-40% ↓ |
| **FVD Score** | ~250-300 | ~150-180 | 35-40% ↓ |
