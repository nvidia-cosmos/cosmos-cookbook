# Traffic Anomaly Generation with Cosmos Predict2

> **Authors:** [Arslan Ali](https://www.linkedin.com/in/arslan-ali-ph-d-5b314239/), [Grace Lam](https://www.linkedin.com/in/grace-lam/), [Amol Fasale](https://www.linkedin.com/in/amolfasale/), [Jingyi Jin](https://www.linkedin.com/in/jingyi-jin/)
> **Organization:** NVIDIA

## Overview

| **Model** | **Workload** | **Use Case** |
|-----------|--------------|--------------|
| [Cosmos Predict 2](https://github.com/nvidia-cosmos/cosmos-predict2) | Post-training | Traffic anomaly generation with improved realism and prompt alignment |

In Intelligent Transportation Systems (ITS), collecting real-world data for rare events like traffic accidents, jaywalking, or blocked intersections includes significant challenges:

- **Privacy concerns**: Recording and using real accident footage raises ethical and legal issues.
- **Infrequent occurrence**: Critical safety events are rare by nature, making data collection expensive and time-consuming.
- **High annotation costs**: Expert annotation of traffic incidents requires specialized knowledge.
- **Safety risks**: Staging real accidents for data collection is dangerous and impractical.

Synthetic data generation (SDG) offers a practical way to augment existing datasets, enabling teams to create targeted scenarios at scale while maintaining control over scenario parameters and data quality.

## The Challenge: Rare Event Data in ITS

Initial evaluations of the pre-trained Cosmos Predict 2 model revealed gaps in generating vehicle collision scenes:

- Unrealistic motion dynamics
- Oversized vehicles (likely due to dash cam bias in pre-training)
- Lack of incident-specific behavior
- Limited ability to maintain a fixed-camera perspective

While the pre-trained model excelled at routine traffic scenes, it struggled with collision scenarios when tested on ITS-specific prompts. This confirmed the need for targeted post-training using anomaly-rich data featuring accidents from fixed CCTV perspectives.

## Prerequisites

Before running training:

1. **Environment setup**: Follow the [Setup guide](https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/docs/setup.md) for installation instructions.
2. **Model checkpoints**: Download required model weights following the *Downloading Checkpoints* sectiopn in the [Setup guide](https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/docs/setup.md).

### Key Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | >=2.5.0 | Deep learning framework |
| `megatron-core` | >=0.11.0 | Distributed training utilities |
| `transformers` | >=4.45.0 | Hugging Face model loading |
| `peft` | >=0.13.0 | LoRA implementation |
| `einops` | >=0.8.0 | Tensor operations |
| `hydra-core` | >=1.3.0 | Configuration management |

## Our Approach: LoRA-Based Domain Adaptation

This case study documents a detailed post-training workflow using Cosmos Predict 2 Video2World with **Low-Rank Adaptation (LoRA)**, with a focus on enhancing model capabilities for generating traffic anomaly videos from a fixed CCTV perspective. Rather than fine-tuning the entire model, we employ LoRA to efficiently adapt the pre-trained foundation model for ITS-specific requirements.

### Why LoRA for ITS Applications?

LoRA (Low-Rank Adaptation) is particularly well-suited for ITS domain adaptation for the following reasons:

#### Critical Advantage: Data Efficiency for Rare Events

Unlike general video datasets with millions of samples, traffic accident datasets typically contain only hundreds to thousands of examples. This data scarcity makes LoRA the optimal choice:

- **Effective with Limited Data**: LoRA can achieve meaningful adaptation with as few as 1,000-2,000 training samples.
- **Reduced Overfitting Risk**: Fewer parameters (45M vs 2B) means less tendency to memorize limited training data.
- **Better Generalization**: The constrained parameter space forces the model to learn generalizable patterns rather than specific examples.
- **Leverages Pre-training**: LoRA builds upon the base model's existing knowledge, requiring only minimal accident-specific data to adapt.

In our case study, with very limited clips, LoRA enabled successful adaptation where full fine-tuning would likely fail or severely overfit.

#### Parameter Efficiency

- **Minimal Storage**: LoRA adds only ~45M trainable parameters to a 2B parameter model (≈2% increase).
- **Quick Deployment**: LoRA adapters are small (10-100MB) compared to full model checkpoints (5-50GB).
- **Multiple Domains**: A different LoRA adapter can be used for each traffic scenario (highways, intersections, parking lots).

#### Resource Optimization

- **Reduced Training Time**: It can take 1-2 hours to train a 2B model, versus 2-4 hours for full fine-tuning.
- **Lower GPU Memory**: 20GB of GPU memory is required for LoRA, versus 50GB for full model training.
- **Faster Iteration**: It's possible to perform rapid experimentation with different training configurations.

#### Preservation of Base Capabilities

- **No Catastrophic Forgetting**: The base model's general video generation capabilities remain intact
- **Additive Learning**: ITS-specific knowledge is added without degrading general performance.
- **Fallback Option**: LoRA can be disabled to access the behavior of the original model when needed.

### LoRA Configuration

Based on the [LoRA paper (Hu et al., 2021)](https://arxiv.org/abs/2106.09685), our configuration includes the following:

- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `output_proj`, `mlp.layer1`, `mlp.layer2`
- **Rank**: 16 (This value determines the dimensionality of the low-rank decomposition--a higher rank allows more expressiveness but increases parameters.)
- **Alpha**: 16 (This scaling hyperparameter controls the magnitude of LoRA updates--it is typically set equal to rank for balanced learning.)
- **Training Data**: A 1:1 mixture of normal traffic scenes and incident scenarios.

This configuration focuses on attention mechanisms and feed-forward layers, which are crucial for the following conditions:

- Understanding spatial relationships between vehicles.
- Capturing temporal dynamics of collisions.
- Maintaining consistent camera perspective.
- Generating physically plausible motion.

## Data Preparation

To address the model limitations, we developed a multi-source data pipeline that combines the following:

- ITS normal traffic scenes: 100 hours of traffic surveillance footage from different intersections at various times of the day, all captured from fixed CCTV viewpoints (no dash cam or moving camera perspectives)
- ITS accident scenes: A compilation of accident scenes from different intersections at various times of the day, all captured from fixed CCTV viewpoints (totaling approximately 3.5 hours of video).

> **Disclaimer**: All data collected for this case study is for research proof of concept and demonstration purposes only. This data has not been merged into the pre-training dataset. This example serves solely to illustrate the data curation methodology and post-training workflow.

### Splitting and Captioning

**ITS accident scenes**: The original 5-10 minute compilations were split into individual clips using `cosmos-curate` with `transnetv2` scene detection and objective captioning.

```json
{
    "pipeline": "split",
    "args": {
        "input_video_path": "s3://your_bucket/raw_data/its_accident_scenes",
        "output_clip_path": "s3://your_bucket/processed_data/its_accident_scenes/v0",
        "generate_embeddings": true,
        "generate_previews": true,
        "generate_captions": true,
        "splitting_algorithm": "transnetv2",
        "captioning_algorithm": "qwen",
        "captioning_prompt_variant": "default",
        "captioning_prompt_text": "You are a video captioning expert trained to describe short CCTV footage of traffic collisions and abnormalities. Every input video contains either a visible traffic collision or a clear traffic abnormality such as a near miss, illegal turn, jaywalking, sudden braking, or swerving. Your task is to generate one concise and factual English paragraph that describes both the static environment and the dynamic physical event. For collision events, clearly describe how the collision unfolds — including the objects involved, their directions and relative speeds, the point of contact, and what happens immediately after. Begin every caption with: 'A traffic CCTV camera' Then describe: Environment: weather, Visible elements: vehicles, pedestrians, traffic lights, signs, road markings, Dynamic event: What vehicles or people are involved, How they move before the event, Where the impact occurs (e.g., front-left bumper hits right side of motorcycle), What happens afterward (e.g., rider falls, car swerves, vehicle spins, traffic halts). Use clear, physics-based verbs such as: collides, hits, swerves, brakes, accelerates, turns, merges, falls, flips, spins, crosses. Output Rules: Output must be one concise paragraph (1-3 small sentences), Focus on visible, physical actions - no speculation or emotional inference, Do not include: driver intentions, license plates, timestamps, brand names, street/building names, or text overlays, Assume all videos contain either a collision or an abnormal traffic event. Output Style Examples: A traffic CCTV camera shows a dry four-way intersection during the day. A red hatchback runs a red light and enters the intersection at moderate speed. From the right, a white SUV proceeds legally and collides into the hatchback's passenger-side door. The hatchback comes to rest near the opposite curb. A traffic CCTV camera captures a multi-lane road during daytime. Vehicles are moving slowly in moderate traffic. A black sedan abruptly slows down, and a silver pickup behind it fails to brake in time, crashing into the sedan's rear bumper. The front of the pickup crumples slightly while the sedan is pushed forward by a few meters. A traffic CCTV camera captures an intersection under clear skies. A motorcyclist enters the intersection diagonally from the left, crossing through oncoming traffic. A silver SUV traveling straight at moderate speed strikes the motorcycle's front wheel with its front-left bumper. The rider is thrown off and skids several feet across the road surface.",
        "limit": 0,
        "limit_clips": 0,
        "perf_profile": true
   }
}
```

**ITS normal traffic scenes**: 100 hours of continuous surveillance footage split into 10-second clips using the `fixed-stride` algorithm. Captioning focused on general scene description since no incidents were detected.

### Dataset Composition

The final curated dataset composition is summarized as follows:

| Dataset | Quality | Incident Coverage | Artifacts | Clips |
|---------|---------|-------------------|-----------|-------|
| ITS normal traffic scenes (10 sec clips) | High | No | None | 44,000 |
| ITS accident scenes (5-15 sec clips) | Medium | Yes | None | 1,200 |

### Post-Training Dataset Sampling

For post-training, we selected 1,000 samples from each dataset (1:1 ratio):

- **Normal traffic scenes**: A diverse selection across intersections and times of day
- **Accident scenes**: 1,000 clips from the available 1,200 to balance normal and anomaly learning

### Video Resolution Requirements

The following are supported resolutions:

- **16:9**: 1280x720
- **1:1**: 960x960
- **4:3**: 960x720
- **3:4**: 720x960
- **9:16**: 720x1280

> **Important**: Resize all videos to supported resolutions before training to avoid errors.

## Post-Training

We performed post-training with a 1:1 mixture of datasets between ITS normal traffic scenes and ITS accident scenes. We selected 1,000 annotated video clips from each of the two datasets that were curated from the previous session.

### Training Setup

- Model: Cosmos Predict 2 Video2World (2B)
- Hardware: Single node with 8 GPUs (8x H100)
- Training Duration: 10k iterations
- Batch Size: 1 per GPU (8 total with data parallelism)
- Learning Rate: ~3.05e-5 (2^-14.5)
- Context Parallel Size: 2
- Loss Monitoring: Visual inspection with convergence curves

An overfitting test on four samples verified pipeline correctness before training. Additional experiments confirmed that including low-quality data degraded results, reinforcing the principle that data quality cannot be traded for volume.

We also experimented with both Full Model post-training and PEFT post-training. The process is detailed below:

### LoRA Post-Training Workflow

#### Dataset and Dataloader Configuration

First, define the dataset and dataloader for ITS training data:

```python
from megatron.core import parallel_state
from torch.utils.data import DataLoader, DistributedSampler

from cosmos_predict2.data.dataset_video import Dataset
from imaginaire.lazy_config import LazyCall as L


def get_sampler(dataset) -> DistributedSampler:
    """Create a distributed sampler for multi-GPU training."""
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


# ITS dataset configuration (1:1 mixture of normal and accident scenes)
example_video_dataset_its = L(Dataset)(
    dataset_dir="/path/to/its/combined_dataset/",  # Update with your dataset path
    num_frames=93,
    video_size=(720, 1280),
)

# Dataloader for ITS training
dataloader_train_its = L(DataLoader)(
    dataset=example_video_dataset_its,
    sampler=L(get_sampler)(dataset=example_video_dataset_its),
    batch_size=1,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
)
```

#### LoRA Configuration Setup (2B Model)

```python
predict2_video2world_lora_training_2b_its = dict(
    defaults=[
        {"override /model": "predict2_video2world_fsdp_2b"},
        {"override /optimizer": "fusedadamw"},
        {"override /scheduler": "lambdalinear"},
        {"override /ckpt_type": "standard"},
        {"override /data_val": "mock"},
        "_self_",
    ],
    job=dict(
        project="posttraining",
        group="video2world_lora",
        name="2b_its_lora",
    ),
    model=dict(
        config=dict(
            train_architecture="lora",                     # Enable LoRA training
            lora_rank=16,                                  # Low-rank decomposition dimension
            lora_alpha=16,                                 # LoRA scaling factor
            lora_target_modules="q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2",
            init_lora_weights=True,
            pipe_config=dict(
                ema=dict(enabled=True),                    # Enable EMA for stability
                guardrail_config=dict(enabled=False),      # Disable during training
            ),
        )
    ),
    model_parallel=dict(
        context_parallel_size=2,                          # For video sequences
    ),
    dataloader_train=dataloader_train_its,
    trainer=dict(
        distributed_parallelism="fsdp",
        callbacks=dict(
            iter_speed=dict(hit_thres=10),                # Report speed every 10 iterations
        ),
        max_iter=10000,                                   # Total training iterations
    ),
    checkpoint=dict(
        save_iter=500,                                    # Save checkpoint every 500 iterations
    ),
    optimizer=dict(
        lr=2 ** (-14.5),                                 # Learning rate: ~3.05e-5
    ),
    scheduler=dict(
        warm_up_steps=[2_000],                           # Warmup period
        cycle_lengths=[400_000],                         # Scheduler cycle length
        f_max=[0.6],                                     # Maximum factor
        f_min=[0.3],                                     # Minimum factor
    ),
)
```

#### LoRA Training Execution

##### Single Node with 8 GPUs

```bash
# Set experiment name for LoRA training
EXP=predict2_video2world_lora_training_2b_its

# Run LoRA training on single node with 8 GPUs
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
  --config=cosmos_predict2/configs/base/config.py \
  --experiment=${EXP} \
  model.config.train_architecture=lora
```

##### Expected log output

```
Adding LoRA adapters: rank=16, alpha=16, targets=['q_proj', 'k_proj', 'v_proj', 'output_proj', 'mlp.layer1', 'mlp.layer2']
Total parameters: 3.96B, Frozen parameters: 3,912,826,880, Trainable parameters: 45,875,200
```

## Inference with Post-Trained ITS Model

After post-training the Cosmos Predict 2 Video2World model on ITS-specific data using LoRA (Low-Rank Adaptation), we can perform efficient inference to generate realistic traffic incident videos.

### Prerequisites

1. **Post-trained checkpoint**: A LoRA checkpoint from the post-training process (e.g., `iter_000001000.pt`)
2. **Input image**: A CCTV traffic camera frame as the starting point (1280x720 recommended)
3. **Environment setup**: A properly configured Cosmos Predict 2 environment with required dependencies

### Basic Command

```bash
export NUM_GPUS=8
export PYTHONPATH=$(pwd)

torchrun --nproc_per_node=${NUM_GPUS} examples/video2world_lora.py \
    --model_size 2B \
    --dit_path "checkpoints/posttraining/video2world/2b_metropolis/checkpoints/model/iter_000001000.pt" \
    --input_path "path/to/input_frame.jpg" \
    --prompt "Your accident scenario description" \
    --save_path "output/generated_accident.mp4" \
    --num_gpus ${NUM_GPUS} \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_target_modules "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2" \
    --offload_guardrail \
    --offload_prompt_refiner
```

### Example: Generating Traffic Collision Scenario

```bash
export NUM_GPUS=8
torchrun --nproc_per_node=${NUM_GPUS} examples/video2world_lora.py \
    --model_size 2B \
    --dit_path "checkpoints/posttraining/video2world/2b_metropolis/checkpoints/model/iter_000001000.pt" \
    --input_path "benchmark/frames/intersection_view.jpg" \
    --prompt 'A static traffic CCTV camera captures an urban street scene, where two cars are speeding down the road. Suddenly, a white sedan abruptly enters from an intersection, cutting across traffic and colliding with one of the vehicles. The impact causes significant damage. Both vehicles come to a halt following the crash.' \
    --save_path output/collision_scenario.mp4 \
    --num_gpus ${NUM_GPUS} \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_target_modules "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2" \
    --offload_guardrail \
    --offload_prompt_refiner
```

### Key Parameters

#### LoRA-Specific Parameters

| Parameter | Description | Required Value |
|-----------|-------------|----------------|
| `--use_lora` | Enable LoRA inference mode | **Must be set** |
| `--lora_rank` | Rank of LoRA adaptation | 16 (match training) |
| `--lora_alpha` | LoRA scaling parameter | 16 (match training) |
| `--lora_target_modules` | Target modules for LoRA | "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2" |

### Prompt Engineering for ITS Scenarios

Effective prompts are crucial for generating realistic traffic incidents. Follow these guidelines:

#### Structure

1. **Start with camera perspective**: "A static traffic CCTV camera..."
2. **Describe the scene**: Location, weather, traffic conditions
3. **Detail the incident**: Vehicle types, movements, collision dynamics
4. **Include aftermath**: Post-collision behavior

#### Example Prompts

##### Intersection Collision

```
A static traffic CCTV camera captures a busy four-way intersection during daytime.
A red sedan runs a red light and enters the intersection at high speed. From the
right, a white SUV proceeds legally and collides with the sedan's passenger side.
The impact causes the sedan to spin and both vehicles come to rest blocking traffic.
```

##### Rear-End Collision

```
A traffic CCTV camera shows a multi-lane highway with moderate traffic flow. A silver
pickup truck suddenly brakes hard, and a black sedan following too closely crashes
into its rear bumper. The sedan's front crumples while the pickup is pushed forward
several meters.
```

## Evaluation

This section covers the evaluation methodology for comparing the original Cosmos Predict 2 model with the LoRA post-trained version on single-view CCTV video generation.

### Evaluation Metrics

#### Quantitative Metrics

We employ two primary metrics for objective evaluation of video generation quality:

##### 1. FID (Fréchet Inception Distance)

FID ([Heusel et al., 2017](https://arxiv.org/abs/1706.08500)) measures the similarity between the distribution of generated videos and real videos by comparing features extracted from a pre-trained Inception network.

- **What values indicate**: Values closer to 0 indicate better quality.
- **Typical ranges**: Excellent (< 10), Good (10-30), Acceptable (30-50), Poor (> 50
- **What it measures**: Visual quality and realism at the frame level

##### 2. FVD (Fréchet Video Distance)

FVD ([Unterthiner et al., 2018](https://arxiv.org/abs/1812.01717)) extends FID to the temporal dimension, evaluating both visual quality and temporal consistency using an I3D network.

- **What values indicate**: Values closer to 0 indicate better quality.
- **Typical ranges**: Excellent (< 100), Good (100-200), Acceptable (200-400), Poor (> 400)
- **What it measures**: Visual quality and temporal coherence

#### Why These Metrics Matter for ITS

- **FID**: Validates visual realism of individual frames from a single camera view,
- **FVD**: Ensures temporal consistency and realistic motion dynamics,

Together, these metrics quantify improvements in single-view traffic video generation.

#### Limitations of FID/FVD Metrics

While FID and FVD effectively measure visual quality and temporal consistency, they have notable limitations for safety-critical ITS applications. These metrics primarily evaluate statistical distributions of visual features but cannot assess **physical plausibility**--a crucial aspect of collision scenarios. For comprehensive evaluation of physical plausibility in generated accidents, additional assessment using physics-aware models like [Cosmos Reason 1](https://github.com/nvidia-cosmos/cosmos-reason1) can be beneficial.

### Expected Results

#### Typical Improvements from LoRA Post-Training

| Metric | Baseline Model | LoRA Post-Trained | Improvement |
|--------|---------------|-------------------|-------------|
| **FID Score** | ~35-40 | ~20-25 | 35-40% ↓ |
| **FVD Score** | ~250-300 | ~150-180 | 35-40% ↓ |

## Expected Outcomes

With LoRA-based post-training, we can achieve the following:

### Quality Improvements

- **Enhanced Physical Realism**: More accurate collision dynamics and vehicle behavior
- **Consistent Perspective**: A fixed CCTV camera viewpoint that is maintained throughout generation
- **Reduced Artifacts**: Fewer unrealistic elements like floating vehicles or impossible physics

### Data Efficiency

- **Successful Training with Minimal Data**: Domain adaptation is achieved with only ~1,000 accident examples.
- **No Data Waste**: Every accident clip from the limited training dataset contributes meaningfully to the model.
- **Synthetic Data Amplification**: The adapted model can now generate unlimited variations of accidents, effectively solving the data scarcity problem.

### Operational Benefits

- **Rapid Adaptation**: New scenarios can be learned in hours rather than days.
- **Cost Efficiency**: Reduced computational requirements enable broader experimentation.
- **Scalable Deployment**: Multiple domain-specific models can coexist efficiently.

## Use Cases

This LoRA-adapted model enables several critical ITS applications:

1. **Safety System Training**: Generate diverse accident scenarios for computer vision model training.
2. **Traffic Simulation**: Create realistic traffic flow videos for urban planning.
3. **Incident Analysis**: Reconstruct and visualize potential accident scenarios.
4. **Emergency Response Planning**: Simulate various incident types for preparedness training.
5. **Infrastructure Assessment**: Evaluate intersection designs with synthetic traffic scenarios.

## Conclusion

Combining the powerful video generation capabilities of Cosmos Predict 2 with the efficient adaptation mechanism of LoRA provides an ideal solution for ITS-specific synthetic data generation. *Most critically, LoRA enables successful domain adaptation despite the severe scarcity of real accident data*--a fundamental constraint in traffic safety applications.

While traditional fine-tuning would require tens of thousands of examples and risk catastrophic overfitting with limited data, LoRA achieved meaningful adaptation with just over 1,000 incident clips. This data efficiency, combined with reduced computational requirements and deployment flexibility, makes LoRA not just a good choice but arguably the only viable approach for adapting large video models to rare-event domains like traffic accidents.

The result is a system capable of generating unlimited high-quality, physically realistic traffic incident videos from minimal real examples--effectively transforming data scarcity from a blocking constraint into a solved problem. This breakthrough can significantly enhance safety system development, emergency response training, and urban planning initiatives worldwide.
