# Post-Training

We decided to perform post-training with a 1:1 mixture of datasets between ITS normal traffic scenes and ITS accident scenes. We selected 1k annotated video clips from each of those two datasets that were curated from the previous session.

## Training Setup

- Model: Cosmos-Predict2 Video2World (2B)
- Hardware: Single node with 8 GPUs (e.g., 8 × H100)
- Training Duration: 10k iterations
- Batch Size: 1 per GPU (8 total with data parallel)
- Learning Rate: ~3.05e-5 (2^-14.5)
- Context Parallel Size: 2
- Loss Monitoring: Visual inspection + convergence curves

An overfitting test on four samples verified pipeline correctness before training. Additional experiments confirmed that including low-quality data degraded results, reinforcing that data quality cannot be traded for volume.

We also experimented both the Full Model post-training and PEFT post-training. Detailing the process below:

## Data Configuration

```python
from cosmos_predict2.data.dataset_video import Dataset
from torch.utils.data import DataLoader, DistributedSampler
from megatron.core import parallel_state

# Dataset configuration for ITS traffic videos
example_video_dataset_its = Dataset(
    dataset_dir="datasets/its_traffic/train",        # Your ITS dataset path
    num_frames=93,                                   # Standard frame count
    video_size=(720, 1280),                         # Height, Width (720p resolution)
)

# DataLoader for single node training
dataloader_train_its = DataLoader(
    dataset=example_video_dataset_its,
    sampler=DistributedSampler(
        example_video_dataset_its,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    ),
    batch_size=1,                                   # Batch size per GPU
    drop_last=True,
    num_workers=8,                                   # Parallel data loading
    pin_memory=True,
)
```

## Full Post-Training Workflow

### Configuration Setup (2B Model)

```python
predict2_video2world_training_2b_its = dict(
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
        group="video2world",
        name="2b_its",
    ),
    model=dict(
        config=dict(
            pipe_config=dict(
                ema=dict(enabled=True),
                prompt_refiner_config=dict(enabled=False),
                guardrail_config=dict(enabled=False),
            ),
        )
    ),
    model_parallel=dict(
        context_parallel_size=2,                           # For video sequences
    ),
    dataloader_train=dataloader_train_its,
    trainer=dict(
        distributed_parallelism="fsdp",
        max_iter=2000,
    ),
    checkpoint=dict(
        save_iter=500,
    ),
    optimizer=dict(
        lr=2 ** (-14.5),                                  # Learning rate: ~3.05e-5
    ),
    scheduler=dict(
        warm_up_steps=[2_000],                            # Warmup period
        cycle_lengths=[400_000],                          # Scheduler cycle length
        f_max=[0.6],                                      # Maximum factor
        f_min=[0.3],                                      # Minimum factor
    ),
)
```

### Training Execution

**2B Model (Single-node, 8 GPUs):**

```bash
# Set experiment name
EXP=predict2_video2world_training_2b_its

# Run training on single node with 8 GPUs
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
  --config=cosmos_predict2/configs/base/config.py \
  --experiment=${EXP}
```

### Training Monitoring

Monitor the following key metrics during single node training:

- Loss convergence over 10k iterations
- GPU memory usage across 8 GPUs
- Training speed (reported every 10 iterations)
- Checkpoint saves (every 500 iterations)

Example training log output:

```bash
[INFO] Starting training on single node with 8 GPUs...
[INFO] Context parallel size: 2, Data parallel size: 4
[INFO] Iteration 500/2000, Loss: 0.245, Speed: 2.3 it/s
[INFO] Checkpoint saved: checkpoints/posttraining/video2world/2b_its/checkpoints/model/iter_000000500.pt
```

## LoRA Post-Training Workflow

### LoRA Configuration Setup (2B Model)

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
        max_iter=2000,                                   # Total training iterations
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

### LoRA Parameter Guidelines

**2B Model (Single Node - 8 GPUs):**

- **Conservative**: rank=8, alpha=8, lr=2^(-15)
- **Recommended**: rank=16, alpha=16, lr=2^(-14.5)
- **Aggressive**: rank=32, alpha=32, lr=2^(-14)

### LoRA Training Execution

**Single Node with 8 GPUs:**

```bash
# Set experiment name for LoRA training
EXP=predict2_video2world_lora_training_2b_its

# Run LoRA training on single node with 8 GPUs
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
  --config=cosmos_predict2/configs/base/config.py \
  --experiment=${EXP} \
  model.config.train_architecture=lora
```

**Expected log output:**

```
Adding LoRA adapters: rank=16, alpha=16, targets=['q_proj', 'k_proj', 'v_proj', 'output_proj', 'mlp.layer1', 'mlp.layer2']
Total parameters: 3.96B, Frozen parameters: 3,912,826,880, Trainable parameters: 45,875,200
```

## Training Optimization Tips

For single node training with 8 GPUs:

1. **Memory Management**: Use gradient checkpointing if encountering OOM errors
2. **Data Loading**: Set `num_workers=8` in dataloader for optimal performance
3. **Context Parallelism**: Set to 2 for efficient video sequence processing
4. **Checkpoint Frequency**: Save every 200-500 iterations to balance I/O and recovery points
5. **Learning Rate**: Start with 2^(-14.5) for stable convergence
