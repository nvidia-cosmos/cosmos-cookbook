# Post-training for Action-Controlled Surgical Robotics

> **Author:** [Lukas Zbinden](https://www.linkedin.com/in/lukas-zbinden-49667316b/) • [Nigel Nelson](https://www.linkedin.com/in/nigel-nelson-nvidia/)
> **Organization:** NVIDIA

| **Model** | **Workload** | **Use Case**      |
|-----------|--------------|-------------------|
| [Cosmos Predict 2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) | Post-training | Surgical Robotics |

This guide provides instructions on running post-training with the action conditioned Cosmos Predict 2.5 model for surgical robotics control and soft tissue simulation.

## Abstract

This recipe demonstrates how to finetune Cosmos Predict 2.5 for action-conditioned video generation in surgical robotics, enabling the model to predict future visual frames based on robot control inputs. We leverage the publicly available [SutureBot dataset](https://huggingface.co/datasets/jchen396/SutureBot), which provides high-quality surgical video data paired with robot state and action sequences from a da Vinci Research Kit (dVRK) platform. The resulting model can generate realistic soft tissue deformation and surgical tool interactions, providing a foundation for simulation-based training and policy learning in autonomous surgical systems.

## Motivation

TODO While the base Cosmos Predict 2.5 model excels at general video generation, sports content demands specialized understanding of athletic dynamics and game rules. Post-training addresses critical gaps in **player kinematic realism and physics**, ensuring natural body movements and accurate ball trajectories. The adapted model achieves higher **rule-coherence scores** by respecting sport-specific constraints like offside lines, field boundaries, and valid player positions. Additionally, post-training significantly improves **identity consistency**, maintaining stable player appearances, jersey numbers, and team colors throughout generated sequences—essential for realistic sports simulation and analysis applications.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Preparing Data](#1-preparing-data)
TODO COMPLETE TOC

## Prerequisites

### 1. Environment Setup

Follow the [Setup guide](./setup.md) for general environment setup instructions, including installing dependencies.

### 2. Hugging Face Configuration

Model checkpoints are automatically downloaded during post-training if they are not present. Configure Hugging Face as follows:

```bash
# Login with your Hugging Face token (required for downloading models)
hf auth login

# Set custom cache directory for HF models
# Default: ~/.cache/huggingface
export HF_HOME=/path/to/your/hf/cache
```

> **💡 Tip**: Ensure you have sufficient disk space in `HF_HOME`.

### 3. Training Output Directory

Configure where training checkpoints and artifacts will be saved:

```bash
# Set output directory for training checkpoints and artifacts
# Default: /tmp/imaginaire4-output
export IMAGINAIRE_OUTPUT_ROOT=/path/to/your/output/directory
```

> **💡 Tip**: Ensure you have sufficient disk space in `IMAGINAIRE_OUTPUT_ROOT`.

## Weights & Biases (W&B) Logging

By default, training will attempt to log metrics to Weights & Biases. You have several options:

### Option 1: Enable W&B

To enable full experiment tracking with W&B:

1. Create a free account at [wandb.ai](https://wandb.ai)
2. Get your API key from [https://wandb.ai/authorize](https://wandb.ai/authorize)
3. Set the environment variable:

    ```bash
    export WANDB_API_KEY=your_api_key_here
    ```

> ⚠️ **Security Warning:** Store API keys in environment variables or secure vaults. Never commit API keys to source control.

### Option 2: Disable W&B

Add `job.wandb_mode=disabled` to your training command to disable wandb logging.

## 1. Preparing Data

### 1.1 Dataset Location

The SutureBot dataset should be organized in a directory structure that you'll specify in the configuration. Set your dataset path to point to your dataset root folder:

```
/path/to/dataset/SutureBot
```

Replace this path with the actual location of the SutureBot dataset. The dataset should contain sports video clips in **MP4 format** at 720p resolution (704x1280). For the current training, we prepared approximately **4,350 training clips** and **50 clips for validation/inference** to support both Image2World and Video2World generation tasks.

### 1.2 Dataset Downloads
In your environment, install the HuggingFace library:
```python
python -m pip install --upgrade huggingface_hub
```
then download the dataset as follows:
```python
python - << 'EOF'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="jchen396/SutureBot",
    repo_type="dataset",
    local_dir="/path/to/dataset/SutureBot",
    local_dir_use_symlinks=False,
)
EOF
```

Unpack zip files:

```bash
cd /path/to/dataset/SutureBot
ls -1 *.zip | parallel 'echo "Unzipping {}"; unzip -q -o "{}"'
```

Run the following script to convert the SutureBot dataset to LeRobot format:
```bash
python3 -u convert_suturebot_to_lerobot.py --input-path /path/to/dataset/SutureBot --output-path /path/to/dataset/SutureBot-LeRobot

TODO: replace script with convert_suturebot_to_lerobot_v2.py
TODO: add script compute_rel_action_stats.py
```
The script creates a training and a test split.

### 1.3 Cosmos-predict 2.5 finetuning
The finetuning wil be performed at 720x960 resolution with 12 frames prediction horizon.

Before executing the training script, the following source code changes must be implemented:

```python
# file: cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/embodiment_tags.py
    DVRK = "dvrk"
```
next:
```python
# file: cosmos_predict2/experiments/base/action.py
import copy
ac_predict2p5_video2world_2b_suturebot_training = copy.deepcopy(ac_reason_embeddings_rectified_flow_2b_256_320)
ac_predict2p5_video2world_2b_suturebot_training['job']['name'] = 'def_ac_predict2p5_video2world_2b_suturebot_training'
ac_predict2p5_video2world_2b_suturebot_training['defaults'] = [
    DEFAULT_CHECKPOINT.experiment,
    {"override /model": "action_conditioned_video2world_fsdp_rectified_flow"},
    {"override /net": "cosmos_v1_2B_action_conditioned"},
    {"override /conditioner": "action_conditioned_video_conditioner"},
    {"override /data_train": "suturebot_train"},
    {"override /data_val": "suturebot_val"},
    "_self_",
]
ac_predict2p5_video2world_2b_suturebot_training['model']['config']['net']['action_dim'] = 20
ac_predict2p5_video2world_2b_suturebot_training['dataloader_train'] = {'batch_size': 4}
ac_predict2p5_video2world_2b_suturebot_training['optimizer']['lr'] = 7.5e-6 # assuming 4 nodes

cs = ConfigStore.instance()

for _item in [ac_reason_embeddings_rectified_flow_2b_256_320, ac_predict2p5_video2world_2b_suturebot_training]:
```
next:
```python
# file: cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py

# experiment for action-sequence video prediction
base_path_suturebot_ds = "/SutureBot"
# Construct modality configs and transforms
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset import LeRobotDataset
from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.groot_configs import (
    construct_modality_config_and_transforms,
)
modality_configs, train_transform, test_transform = construct_modality_config_and_transforms(
    num_frames=13, embodiment="dvrk", downscaled_res=False
)

suturebot_train_dataset = L(LeRobotDataset)(
    num_frames=13,
    time_division_factor=4,
    time_division_remainder=1,
    max_pixels=1920 * 1080,
    data_file_keys=("video",),
    image_file_extension=("jpg", "jpeg", "png", "webp"),
    video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
    repeat=1,
    args=None,
    dataset_path=base_path_suturebot_ds,
    data_split="train",
    embodiment="dvrk",
    downscaled_res=False,
)

suturebot_val_dataset = L(LeRobotDataset)(
    num_frames=13,
    time_division_factor=4,
    time_division_remainder=1,
    max_pixels=1920 * 1080,
    data_file_keys=("video",),
    image_file_extension=("jpg", "jpeg", "png", "webp"),
    video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
    repeat=1,
    args=None,
    dataset_path=base_path_suturebot_ds,
    data_split="test",
    embodiment="dvrk",
    downscaled_res=False,
)

...

suturebot_train_dataloader = L(DataLoader)(
    dataset=suturebot_train_dataset,
    sampler=L(get_sampler)(dataset=suturebot_train_dataset),
    batch_size=1,
    drop_last=True,
)
suturebot_val_dataloader = L(DataLoader)(
    dataset=suturebot_val_dataset,
    sampler=L(get_sampler)(dataset=suturebot_val_dataset),
    batch_size=1,
    drop_last=True,
)

...

    cs.store(
        group="data_train",
        package="dataloader_train",
        name="suturebot_train",
        node=suturebot_train_dataloader,
    )
    cs.store(
        group="data_val",
        package="dataloader_val",
        name="suturebot_val",
        node=suturebot_val_dataloader,
    )
```
next:
```python
# file: cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/groot_configs.py
    elif embodiment == "dvrk":
        timestep_interval = 3  # LZ: downsampling rate
        delta_indices = list(range(0, num_frames * timestep_interval, timestep_interval))
        config = {
            "video": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=["video.observation.images.main"],
            ),
            "state": ModalityConfig(
                delta_indices=[0],
                modality_keys=["state.observation.state"],
            ),
            "action": ModalityConfig(
                delta_indices=delta_indices,
                modality_keys=['action.action']
            ),
        }

...

    elif embodiment == "dvrk":
        # width = 512 if not downscaled_res else 256
        # height = 320 if not downscaled_res else 256
        width = 960 if not downscaled_res else 256
        height = 720 if not downscaled_res else 256

...

# further, replace "min_max" by "mean_std" (w.r.t. normalization_modes, 4x times):
    normalization_modes={key: "mean_std" for key in action_modality.modality_keys},
```
next:
```python
# file: cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/video.py
# replace line
split_keys = key.split(".")
# with
split_keys = key.split(".", 1)
```
next:
```python
# file: cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/concat.py
# replace line
split_keys = key.split(".")
# with
split_keys = key.split(".", 1)
```
next:
```python
# file: cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/state_action.py
# replace lines 
         # Check that all state keys specified in apply_to have their modality_metadata
         for key in self.apply_to:
-            split_key = key.split(".")
+            split_key = key.split(".", 1)
             assert len(split_key) == 2, "State keys should have two parts: 'modality.key'"
             if key not in self.modality_metadata:
                 modality, state_key = split_key
@@ -389,7 +389,7 @@ class StateActionTransform(InvertibleModalityTransform):

         # Check that all state keys specified in normalization_modes have their statistics in state_statistics
         for key in self.normalization_modes:
-            split_key = key.split(".")
+            split_key = key.split(".", 1)

         for key in self.normalization_modes:
-            modality, state_key = key.split(".")
+            modality, state_key = key.split(".", 1)
```
next (3 files in total for relative action computation):
```python
diff --git a/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/groot_configs.py b/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/groot_configs.py
index 24a2c1d..6f14d54 100644
--- a/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/groot_configs.py
+++ b/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/groot_configs.py
@@ -23,6 +23,7 @@ from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset imp
 from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.transform.base import ComposedModalityTransform
 from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.transform.concat import ConcatTransform
 from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.transform.state_action import (
+    RelativeActionTransform,
     StateActionToTensor,
     StateActionTransform,
 )
@@ -170,6 +171,7 @@ def construct_modality_config_and_transforms(num_frames, embodiment, downscaled_
                 normalization_modes={key: "mean_std" for key in state_modality.modality_keys},
             ),
             StateActionToTensor(apply_to=action_modality.modality_keys),
+            RelativeActionTransform(apply_to=action_modality.modality_keys),
             StateActionTransform(
                 apply_to=action_modality.modality_keys,
                 normalization_modes={key: "mean_std" for key in action_modality.modality_keys},
@@ -191,6 +193,7 @@ def construct_modality_config_and_transforms(num_frames, embodiment, downscaled_
                 normalization_modes={key: "mean_std" for key in state_modality.modality_keys},
             ),
             StateActionToTensor(apply_to=action_modality.modality_keys),
+            RelativeActionTransform(apply_to=action_modality.modality_keys),
             StateActionTransform(
                 apply_to=action_modality.modality_keys,
                 normalization_modes={key: "mean_std" for key in action_modality.modality_keys},
```
next:
```python
    # file: cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/state_action.py 
```

next:
```python
  # file: cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/dataset.py
```
next:
```python
  # add bugfix in dataset.py
```

Now start the finetuning, using 4 nodes (32 GPUs):
```bash
mkdir logs
sbatch run_finetuning.sh
```
Notabene: some checkpoint downloads will occur from nvidia/Cosmos-Experimental on HF. TODO: no public access yet? Ask Jingyi.

Run the finetuning for 20,000 steps.

The ckeckpoints will be saved in
```bash
cd ${IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/def_ac_predict2p5_video2world_2b_suturebot_training/checkpoints
```

TODO AT WORK 29/12 ****************************************

Convert the distributed checkpoint to compact format as follows:
```bash
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/ac_predict2p5_video2world_2b_suturebot_training/checkpoints
CHECKPOINT_ITER=iter_000020000
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER
python ./scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR
```

### 1.4 Generate Self-Forcing Teacher Trajectory Cache
Using one GPU, generate the teacher trajectory cache. In the following command, specify the teacher checkpoint from the previous finetuning run.
The teacher trajectory cache will be written to the directory `trajectory_cache/warmup_regenerated_4step`. 
```bash
mkdir trajectory_cache
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python cosmos_predict2/_src/predict2/action/inference/inference_gr00t_warmup.py \
    --experiment=ac_predict2p5_video2world_2b_jhu_training   \
    --ckpt_path ${IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/ac_predict2p5_video2world_2b_suturebot_training/checkpoints/iter_000020000/model_ema_bf16.pt \
    --input_video_root /path/to/dataset/SutureBot-LeRobot \
    --save_root trajectory_cache/warmup_regenerated_4step \
    --resolution 720,960 \
    --guidance 0 \
    --chunk_size 12 \
    --start 0 \
    --end 1000 \
    --query_steps 0,9,18,27,34
```


### 1.5 Launch Self-Forcing Warmup Training


### 1.7 Run Inference and Evaluation Script

The `inference_dvrk.py` script runs autoregressive video generation for policy evaluation. It:

1. Loads only the **first frame** from the dataset as initial conditioning
2. Generates frames using ground-truth actions from the dataset
3. Uses each chunk's **last predicted frame** as conditioning for the next chunk
4. Stitches all chunks into a full episode video

This demonstrates policy evaluation: the GT actions serve as a proxy for any action source. Replace them with policy-predicted actions to evaluate a learned policy.

#### Convert Checkpoint

Training produces distributed checkpoints (DCP). Convert to PyTorch format:

```bash
CHECKPOINTS_DIR=/your/checkpoint/dir
CHECKPOINT_ITER=iter_000020000

python scripts/convert_distcp_to_pt.py \
    $CHECKPOINTS_DIR/$CHECKPOINT_ITER \
    $CHECKPOINTS_DIR/$CHECKPOINT_ITER
```

This outputs `model.pt`, `model_ema_fp32.pt`, and `model_ema_bf16.pt`. Use `model_ema_bf16.pt` for inference.

#### Run Inference

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/inference_dvrk.py \
    --experiment=ac_predict2p5_video2world_2b_suturebot_training \
    --ckpt_path $CHECKPOINTS_DIR/$CHECKPOINT_ITER/model_ema_bf16.pt \
    --dataset_path /path/to/SutureBot-LeRobot \
    --save_root results/dvrk_eval \
    --data_split test \
    --episode_ids 0,1,2 \
    --save_comparison
```

The `--save_comparison` flag generates side-by-side videos (GT left, predicted right).

#### Swapping in a Policy

To evaluate a policy instead of GT actions, modify the inference loop in `inference_dvrk.py`:

```python
# Current (GT actions from dataset):
actions = data["action"].numpy()

# With a policy:
actions = policy.predict(current_frame)  # Returns (12, action_dim)
```

The model accepts **normalized** action sequences matching the expected shape `(chunk_size, action_dim)` and following the **relative action formulation** used in this recipe.

Checkpoints are saved to:

- Text format: `${IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict_v2p5/lora/2b_cosmos_sports_assets_lora/checkpoints`
- JSON format: `${IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict_v2p5/lora/2b_cosmos_sports_assets_json_lora/checkpoints`

**Note**: By default, `IMAGINAIRE_OUTPUT_ROOT` is `/tmp/imaginaire4-output`. We strongly recommend setting `IMAGINAIRE_OUTPUT_ROOT` to a location with sufficient storage space for your checkpoints.

## Checkpointing

Training uses two checkpoint formats:

### 1. Distributed Checkpoint (DCP) Format

**Primary format for training checkpoints.**

- **Structure**: Multi-file directory with sharded model weights
- **Used for**: Saving checkpoints during training, resuming training
- **Advantages**:
  - Efficient parallel I/O for multi-GPU training
  - Supports FSDP (Fully Sharded Data Parallel)
  - Optimized for distributed workloads

**Example directory structure:**

```
checkpoints/
├── iter_{NUMBER}/
│   ├── model/
│   │   ├── .metadata
│   │   └── __0_0.distcp
│   ├── optim/
│   ├── scheduler/
│   └── trainer/
└── latest_checkpoint.txt
```

### 2. Consolidated PyTorch (.pt) Format

**Single-file format for inference and distribution.**

- **Structure**: Single `.pt` file containing the complete model state
- **Used for**: Inference, model sharing, initial post-training
- **Advantages**:
  - Easy to distribute and version control
  - Standard PyTorch format
  - Simpler for single-GPU workflows

## 3. Inference with LoRA Post-trained checkpoint

### 3.1 Converting DCP Checkpoint to Consolidated PyTorch Format

Since the checkpoints are saved in DCP format during training, you need to convert them to consolidated PyTorch format (.pt) for inference. Use the `convert_distcp_to_pt.py` script:

```bash
# For text format checkpoint
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}/cosmos_predict_v2p5/lora/2b_cosmos_sports_assets_lora/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

# Convert DCP checkpoint to PyTorch format
python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR
```

This conversion will create three files:

- `model.pt`: Full checkpoint containing both regular and EMA weights
- `model_ema_fp32.pt`: EMA weights only in float32 precision
- `model_ema_bf16.pt`: EMA weights only in bfloat16 precision (recommended for inference)

### 3.2 Running Inference

After converting the checkpoint, you can run inference with your post-trained model using the command-line interface.

The model can be used for any generation mode. Simply use the appropriate JSON configuration with the corresponding experiment:

#### Text2World Generation

```bash
# Using Text format checkpoint
torchrun --nproc_per_node=8 examples/inference.py \
  -i assets/sports_text2world_prompts.json \
  -o outputs/sports_text2world \
  --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
  --experiment predict2_lora_training_2b_cosmos_sports_assets_txt

# Using JSON format checkpoint
torchrun --nproc_per_node=8 examples/inference.py \
  -i assets/sports_text2world_prompts.json \
  -o outputs/sports_text2world \
  --checkpoint-path $JSON_CHECKPOINT_DIR/model_ema_bf16.pt \
  --experiment predict2_lora_training_2b_cosmos_sports_assets_json_rank32
```

#### Image2World Generation

```bash
torchrun --nproc_per_node=8 examples/inference.py \
  -i assets/sports_image2world_inputs.json \
  -o outputs/sports_image2world \
  --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
  --experiment predict2_lora_training_2b_cosmos_sports_assets_txt
```

#### Video2World Generation

```bash
torchrun --nproc_per_node=8 examples/inference.py \
  -i assets/sports_video2world_inputs.json \
  -o outputs/sports_video2world \
  --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
  --experiment predict2_lora_training_2b_cosmos_sports_assets_txt
```

The model automatically detects the generation mode based on the input:

- Provide text only → Text2World generation
- Provide 1 image frame → Image2World generation
- Provide 2+ video frames → Video2World generation

Generated videos will be saved to the output directory.

### Example Prompts for Soccer Video Generation

#### Image2World/Video2World Generation Example

```json
{
  "inference_type": "image2world",
  "name": "soccer_action_sequence",
  "prompt": "A soccer player in a red and black uniform dribbles the ball past an opponent in an orange and white uniform. The player in red sprints towards the goal, evading the defender. As he approaches the goalpost, the goalkeeper dives to make a save but fails to stop the ball from entering the net. The camera follows the ball as it flies into the goal, capturing the excitement of the moment.",
  "input_path": "first_frame.mp4",
  "seed": 0,
  "guidance": 3,
  "num_output_frames": 93
}
```

## Evaluation and Results

### Comparison: Base Model vs Post-Trained Model

The LoRA post-training significantly improves the quality and realism of generated soccer videos. Below is a comparison of videos generated by the base model versus the post-trained model:

| Sample | Base Model | Post-Trained Model |
|--------|------------|-------------------|
| **Sample 1** | <video width="320" controls autoplay loop muted><source src="assets/base/0.mp4" type="video/mp4"></video> | <video width="320" controls autoplay loop muted><source src="assets/post_trained/0.mp4" type="video/mp4"></video> |
| **Sample 2** | <video width="320" controls autoplay loop muted><source src="assets/base/12.mp4" type="video/mp4"></video> | <video width="320" controls autoplay loop muted><source src="assets/post_trained/12.mp4" type="video/mp4"></video> |
| **Sample 3** | <video width="320" controls autoplay loop muted><source src="assets/base/5.mp4" type="video/mp4"></video> | <video width="320" controls autoplay loop muted><source src="assets/post_trained/5.mp4" type="video/mp4"></video> |

### Key Improvements After Post-Training

**The post-training experiment demonstrated model improvements with limited data, despite not solving all physics artifacts.**

The post-trained model demonstrates substantial enhancements in several critical areas:

#### 1. **Rule Coherence and Field Semantics**

The post-trained model shows significantly better understanding of soccer rules and field layout:

- Players stay within field boundaries and respect offside lines
- Goal areas and penalty boxes are rendered with proper dimensions
- Ball physics follow realistic trajectories during passes and shots

#### 2. **Identity and Team Preservation**

Team identities and player characteristics are better maintained throughout the generated sequences:

- Consistent jersey colors and numbers across all frames
- Individual player features remain stable during complex movements
- Team formations and tactical positioning are more realistic

#### 3. **Reduced Broadcast Artifacts**

The post-trained model produces cleaner, broadcast-quality videos:

- Minimized motion blur during fast-paced action
- Reduced ghosting effects around players and the ball
- Cleaner rendering of stadium elements and crowd backgrounds
- Improved temporal consistency across frame sequences

#### 4. **Sport-Specific Motion Dynamics**

The model better captures soccer-specific movements:

- Realistic dribbling patterns and ball control
- Natural goalkeeper diving and saving motions
- Accurate representation of tackles, passes, and shots
- Proper player acceleration and deceleration patterns

These improvements make the post-trained model particularly suitable for:

- Training computer vision systems for sports analytics
- Generating synthetic data for referee training
- Creating realistic game simulations for tactical analysis
- Producing content for sports broadcasting and entertainment

For more inference options and advanced usage, see the Cosmos Predict 2 [inference documentation](https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/docs/inference.md).

## Resources

1. [Cosmos Predict 2.5 Model](https://github.com/nvidia-cosmos/cosmos-predict2.5) - Model weights and documentation.
2. [SutureBot](https://huggingface.co/datasets/jchen396/SutureBot) - A Precision Framework & Benchmark For Autonomous End-to-End Suturing.