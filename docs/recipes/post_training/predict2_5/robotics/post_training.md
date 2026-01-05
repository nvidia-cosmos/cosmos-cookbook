# Post-training for Action-Controlled Surgical Robotics

> **Author:** [Lukas Zbinden](https://www.linkedin.com/in/lukas-zbinden-49667316b/) • [Nigel Nelson](https://www.linkedin.com/in/nigel-nelson-nvidia/)
> **Organization:** NVIDIA

## Overview

| **Model** | **Workload** | **Use Case**      |
|-----------|--------------|-------------------|
| [Cosmos Predict 2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) | Post-training | Surgical Robotics |

This recipe details how to post‑train the Cosmos Predict world foundation model (WFM) to function as a learned simulator for 
policy evaluation. Developers are guided on how to finetune an action‑conditioned variant of Cosmos Predict 2.5 using domain‑specific 
surgical robotic data, leveraging the public [SutureBot](https://huggingface.co/datasets/jchen396/SutureBot) dataset, which contains endoscopic video paired with 
kinematic action sequences from the da Vinci Research Kit (dVRK). The resulting model implicitly captures both robot kinematics 
and task‑relevant environment dynamics, including realistic deformation and tool–deformable object interactions. This 
learned model forms the basis for simulation‑based policy evaluation, executed via a software‑in‑the‑loop rollout loop for 
autonomous surgical systems. While demonstrated on a surgical robotic use case, this recipe generalizes to other robotic 
systems and broader embodied AI applications.

## Table of Contents

- [Prerequisites](#1-prerequisites)
- [Preparing Data](#2-preparing-data)
- [Model Configuration](#3-model-configuration)
- [Finetuning](#4-finetuning)
- [Inference](#5-run-inference-and-evaluation-script)
- [Results](#6-results)


## 1. Prerequisites

### 1.1. Environment Setup

Follow the [Setup guide](./setup.md) for general environment setup instructions, including installing dependencies.

### 1.2. Hugging Face Configuration

Model checkpoints are automatically downloaded during post-training if they are not present. Configure Hugging Face as follows:

```bash
# Login with your Hugging Face token (required for downloading models)
hf auth login

# Set custom cache directory for HF models
# Default: ~/.cache/huggingface
export HF_HOME=/path/to/your/hf/cache
```

> **💡 Tip**: Ensure you have sufficient disk space in `HF_HOME`.

### 1.3. Training Output Directory

Configure where training checkpoints and artifacts will be saved:

```bash
# Set output directory for training checkpoints and artifacts
# Default: /tmp/imaginaire4-output
export IMAGINAIRE_OUTPUT_ROOT=/path/to/your/output/directory
```

> **💡 Tip**: By default, `IMAGINAIRE_OUTPUT_ROOT` is `/tmp/imaginaire4-output`. We strongly recommend setting `IMAGINAIRE_OUTPUT_ROOT` to a location with sufficient storage space for your checkpoints.

### 1.4. Weights & Biases (W&B) Logging

By default, training will attempt to log metrics to Weights & Biases. You have several options:

#### Option 1: Enable W&B

To enable full experiment tracking with W&B:

1. Create a free account at [wandb.ai](https://wandb.ai)
2. Get your API key from [https://wandb.ai/authorize](https://wandb.ai/authorize)
3. Set the environment variable:

    ```bash
    export WANDB_API_KEY=your_api_key_here
    ```

> ⚠️ **Security Warning:** Store API keys in environment variables or secure vaults. Never commit API keys to source control.

#### Option 2: Disable W&B

Add `job.wandb_mode=disabled` to your training command to disable wandb logging.

## 2. Preparing Data

### 2.1 Dataset Location

The SutureBot dataset should be organized in a directory structure that you'll specify in the configuration. Set your dataset path to point to your dataset root folder:

```
/path/to/dataset/SutureBot
```

Replace this path with the actual download location of the SutureBot dataset. The dataset should contain da Vinci robot 
video clips stored as individual JPG files at 640x480 resolution.

### 2.2 Dataset Downloads
In your environment (conda, docker, etc.), install the HuggingFace library:
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

Then run the following script to convert the SutureBot dataset to LeRobot format (notice lerobot==0.3.3 is expected). The output path is defined by the env variable \$HF_LEROBOT_HOME. Override \$HF_LEROBOT_HOME to change the location of the output.  
```bash
# optional: export HF_LEROBOT_HOME=/path/to/dataset/SutureBot/LeRobot
python3 -u convert_suturebot_to_lerobot_v3.py --data-path /path/to/dataset/SutureBot 
```
The script will save the SutureBot dataset in LeRobot format at whatever was specified in HF_LEROBOT_HOME.

## 3. Model Configuration
The finetuning wil be performed at 720x960 resolution (to match 720p pre-training) with 12 frames prediction horizon.

Before executing the training script, the following source code changes must be applied to the cloned repository (as described in [Setup guide](./setup.md)), which cover both model configuration and data processing:

### 3.1 embodiment_tags.py
```python
    # file: cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/embodiment_tags.py
    DVRK = "dvrk"
```

### 3.2 action.py
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

### 3.3 data.py
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

### 3.4 groot_configs.py
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

### 3.5 video.py
```python
# file: cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/video.py
# replace line
split_keys = key.split(".")
# with
split_keys = key.split(".", 1)
```

### 3.6 concat.py
```python
# file: cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/concat.py
# replace line
split_keys = key.split(".")
# with
split_keys = key.split(".", 1)
```

### 3.7 state_action.py
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

### 3.8 groot_configs.py (1/3 files in total for relative action computation):
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

### 3.9 state_action.py (2/3 files in total for relative action computation):
```python
    # file: cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/state_action.py 
```

### 3.10 dataset.py (3/3 files in total for relative action computation):
```python
  # file: cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/dataset.py
```
next:
```python
  # add bugfix in dataset.py
```
next:
```python
  # add fix_torchvision_av_timestamp_matching.patch
```

## 4. Finetuning
Now start the finetuning, using 4 nodes (32 GPUs):
```bash
mkdir logs
sbatch run_finetuning.sh
```
Notice: some checkpoint downloads will occur from nvidia/Cosmos-Experimental on HF. TODO: no public access yet? Ask Jingyi.

Run the finetuning for 20,000 steps.

The checkpoints in distributed format (DCP) will be saved in:
```bash
cd ${IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/def_ac_predict2p5_video2world_2b_suturebot_training/checkpoints
```

## 5 Run Inference and Evaluation Script

The `inference_dvrk.py` script runs autoregressive video generation for policy evaluation. It:

1. Loads only the **first frame** from the dataset as initial conditioning
2. Generates frames using ground-truth actions from the dataset
3. Uses each chunk's **last predicted frame** as conditioning for the next chunk
4. Stitches all chunks into a full episode video

This demonstrates policy evaluation: the GT actions serve as a proxy for any action source. Replace them with policy-predicted actions to evaluate a learned policy.

### 5.1 Convert Checkpoint

Training produces distributed checkpoints (DCP). Convert to PyTorch format:

```bash
CHECKPOINTS_DIR=/your/checkpoint/dir
CHECKPOINT_ITER=iter_000020000

python scripts/convert_distcp_to_pt.py \
    $CHECKPOINTS_DIR/$CHECKPOINT_ITER/model \
    $CHECKPOINTS_DIR/$CHECKPOINT_ITER
```

This conversion will create three files:

- `model.pt`: Full checkpoint containing both regular and EMA weights
- `model_ema_fp32.pt`: EMA weights only in float32 precision
- `model_ema_bf16.pt`: EMA weights only in bfloat16 precision (recommended for inference)


### 5.2 Run Inference

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python scripts/inference_dvrk.py \
    --experiment=ac_predict2p5_video2world_2b_suturebot_training \
    --ckpt_path $CHECKPOINTS_DIR/$CHECKPOINT_ITER/model_ema_bf16.pt \
    --dataset_path /path/to/dataset/SutureBot/LeRobot \
    --save_root results/dvrk_eval \
    --data_split test \
    --episode_ids 0,1,2 \
    --save_comparison
```

The `--save_comparison` flag generates side-by-side videos (GT left, predicted right).

For more inference options and advanced usage, see the Cosmos Predict 2 [inference documentation](https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/docs/inference.md).

### 5.3 Swapping in a Policy

To evaluate a policy instead of GT actions, modify the inference loop in `inference_dvrk.py`:

```python
# Current (GT actions from dataset):
actions = data["action"].numpy()

# With a policy:
actions = policy.predict(current_frame)  # Returns (12, action_dim)
```

The model accepts **normalized** action sequences matching the expected shape `(chunk_size, action_dim)` and following the **relative action formulation** used in this recipe.



## 6. Results

### Comparison: Base Model vs Post-Trained Model

The LoRA post-training significantly improves the quality and realism of generated soccer videos. Below is a comparison of videos generated by the base model versus the post-trained model:

| Sample | Ground Truth                                                                                               | Post-Trained Model |
|--------|------------------------------------------------------------------------------------------------------------|-------------------|
| **Sample 1** | <video width="320" controls autoplay loop muted><source src="assets/base/0.mp4" type="video/mp4"></video>  | <video width="320" controls autoplay loop muted><source src="assets/post_trained/0.mp4" type="video/mp4"></video> |
| **Sample 2** | <video width="320" controls autoplay loop muted><source src="assets/base/12.mp4" type="video/mp4"></video> | <video width="320" controls autoplay loop muted><source src="assets/post_trained/12.mp4" type="video/mp4"></video> |
| **Sample 3** | <video width="320" controls autoplay loop muted><source src="assets/base/5.mp4" type="video/mp4"></video>  | <video width="320" controls autoplay loop muted><source src="assets/post_trained/5.mp4" type="video/mp4"></video> |




## Resources

1. [Cosmos Predict 2.5 Model](https://github.com/nvidia-cosmos/cosmos-predict2.5) - Model weights and documentation.
2. [SutureBot](https://huggingface.co/datasets/jchen396/SutureBot) - A Precision Framework & Benchmark For Autonomous End-to-End Suturing.