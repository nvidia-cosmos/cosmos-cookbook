# Cosmos Reason as Reward Model

## Overview

NVIDIA Cosmos Reason is an open, customizable, 7B-parameter reasoning vision language model (VLM) for physical AI and robotics. This document covers how to use Cosmos Reason models for video evaluation in two primary modes:

1. **Reward Model**: Scoring videos for RL training and model selection
2. **Video Critic**: Detailed analysis and structured feedback

## Model Capabilities

- **Physics Understanding**: Gravity, collision, fluid dynamics, object permanence
- **Spatial-Temporal Reasoning**: 3D relationships and motion consistency
- **Embodied AI Assessment**: Agent behavior and environmental interaction
- **Chain-of-Thought Analysis**: Step-by-step reasoning without human annotations
- **Zero-Shot Evaluation**: Works across diverse domains and scenarios

## Installation and Setup

### Step 1: Install Dependencies

```bash
# Install core dependencies
pip install torch torchvision transformers
pip install mediapy numpy pillow qwen-vl-utils huggingface-hub

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 2: Download Model

```bash
# Download Cosmos-Reason1-7B-Reward model
huggingface-cli download nvidia/Cosmos-Reason1-7B-Reward --local-dir ./checkpoints --token <YOUR_HF_TOKEN>
```

> **Note**: Requires HuggingFace account and token with access to nvidia/Cosmos-Reason1-7B-Reward model.

## Reward Model Usage

### Single Video Evaluation

```bash
python inference.py --video path/to/video.mp4 --checkpoint ./checkpoints
```

**Output:**

```
Video: sample_video.mp4
Physical accuracy: No
Score (high is good): 0.2341
```

### Batch Processing

```bash
# Process directory of videos
python batch_inference.py --checkpoint ./checkpoints --video-dir ./test_videos --output-dir ./results
```

### Scoring System

- **Score Range**: 0.0 to 1.0 (higher = better physical accuracy)
- **Binary Classification**: "Yes" = anomalies detected, "No" = no anomalies
- **Thresholds**: High (0.7-1.0), Medium (0.3-0.7), Low (0.0-0.3)

### Physics Evaluation Framework

**What the Model Evaluates:**

- **Gravity**: Realistic gravitational behavior
- **Collision**: Proper object interaction physics
- **Object Interaction**: Logical cause-and-effect relationships
- **Fluid Dynamics**: Realistic liquid and gas behavior
- **Object Permanence**: Consistent object existence
- **Human Motion**: Natural body movement and joint constraints

**What the Model Ignores:**

- Animation style (cartoons not automatically anomalous)
- Audio content
- Lighting, shadows, camera effects
- Artistic style and background elements

## Video Critic Usage

### Detailed Analysis Mode

The [Cosmos-Reason1 Video Critic Example](https://github.com/nvidia-cosmos/cosmos-reason1/blob/main/examples/benchmark/README.md) provides structured video analysis:

```bash
# Run video critic evaluation
python video_critic.py \
    --video path/to/video.mp4 \
    --checkpoint ./checkpoints \
    --critique_mode detailed \
    --output_format structured
```

### Critic Output Format

```json
{
  "video_analysis": {
    "physical_accuracy": {
      "score": 0.85,
      "violations": ["gravity_inconsistency_at_12s"],
      "explanation": "Object falls upward at timestamp 12 seconds"
    },
    "reasoning_chain": {
      "logical_consistency": 0.92,
      "causal_relationships": "strong",
      "temporal_coherence": "maintained"
    },
    "content_quality": {
      "visual_coherence": 0.88,
      "object_permanence": 0.95,
      "scene_understanding": "excellent"
    }
  }
}
```

### Video Critic Capabilities

- **Physical Plausibility Assessment**: Detailed physics violation analysis
- **Reasoning Chain Analysis**: Step-by-step logical consistency breakdown
- **Content Quality Critique**: Visual coherence and temporal consistency assessment
- **Contextual Understanding**: Scene context and object relationship evaluation

## Advanced Configuration

### Custom Prompt Engineering

Customize evaluation focus by modifying prompts in `inference.py`:

```python
# Focus on specific physics
USER_PROMPT = "Focus on fluid dynamics and ignore human motion artifacts"

# Domain-specific evaluation
SYSTEM_PROMPT = "Evaluate this medical procedure video for physical plausibility"

# Multi-step reasoning
USER_PROMPT = "Provide step-by-step analysis of physical anomalies with explanations"
```

### Domain Adaptation

**Medical Videos**: Anatomical accuracy and medical procedure realism
**Robotics**: Mechanical constraints and robot behavior
**Synthetic Data**: Simulation physics and rendering accuracy
**Gaming**: Game physics and character movement realism

## Integration with Other Metrics

### Comprehensive Evaluation Pipeline

```bash
# Combine multiple evaluation approaches
python comprehensive_evaluation.py \
    --videos ./test_videos/*.mp4 \
    --metrics fid,fvd,sampson \
    --cosmos_reward ./checkpoints \
    --cosmos_critic ./checkpoints \
    --output_report ./evaluation_report.json
```

### Use Cases

**Reward Model Applications:**

- Reinforcement learning training signals
- Model selection and checkpoint ranking
- Quality filtering for generated content
- Large-scale automated evaluation

**Video Critic Applications:**

- Generated video quality control
- Training data curation and filtering
- Benchmark evaluation and model comparison
- Research analysis and ablation studies

## Visualization and Analysis

### Streamlit Interface

```bash
cd experimental/afasale/visualization

# Launch interactive results browser
./run_streamlit.sh ./results/video ./results/text
```

**Features:**

- Interactive video browsing with scores
- Statistical analysis of evaluation results
- Anomaly detection and problematic video identification
- Comparative analysis across different models

## Performance Optimization

### System Requirements

- **GPU**: CUDA-enabled GPUs recommended
- **Memory**: ~15GB VRAM for batch processing
- **Processing Time**: 10-30 seconds per video
- **Storage**: Sufficient space for output files and logs

### Optimization Tips

```bash
# Large datasets
python batch_inference.py --checkpoint ./checkpoints --video-dir ./dataset --batch-size 4

# Memory-constrained environments
python inference.py --video input.mp4 --checkpoint ./checkpoints --low-memory-mode

# Multi-GPU processing
python distributed_inference.py --checkpoint ./checkpoints --video-dir ./dataset --num-gpus 4
```

## Best Practices

### Evaluation Workflow

1. **Preprocessing**: Ensure videos are in supported formats (MP4, AVI, MOV)
2. **Validation**: Test on known good/bad examples first
3. **Batch Processing**: Use batch inference for large datasets
4. **Custom Prompts**: Adapt evaluation criteria for specific domains
5. **Result Analysis**: Review both scores and detailed critiques
6. **Integration**: Combine with other metrics for comprehensive assessment

### Quality Assurance

- **Prompt Engineering**: Record and version control custom prompts
- **Result Verification**: Manually verify subset of results for calibration
- **Reproducibility**: Use consistent checkpoints and prompts
- **Documentation**: Track evaluation configurations and results

## Resources

### Official Documentation

- [Cosmos-Reason1 GitHub Repository](https://github.com/nvidia-cosmos/cosmos-reason1)
- [Cosmos-Reason1-7B-Reward Model](https://huggingface.co/nvidia/Cosmos-Reason1-7B-Reward)

### Examples and Tutorials

- [Benchmark Example](https://github.com/nvidia-cosmos/cosmos-reason1/blob/main/examples/benchmark/README.md)
- [Video Critic Example](https://github.com/nvidia-cosmos/cosmos-reason1/blob/main/examples/video_critic/README.md)

### Additional Resources

- [Physical AI and Robotics Documentation](https://research.nvidia.com/labs/dir/)
- [VLM Evaluation Best Practices](https://research.nvidia.com/vlm-evaluation)

## Citation

When using Cosmos Reason models in research, please cite the appropriate papers and acknowledge the NVIDIA Cosmos project.
