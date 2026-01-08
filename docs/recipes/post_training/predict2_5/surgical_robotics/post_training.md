# Post-training for Action-Controlled Surgical Robotics

> **Author:** [Lukas Zbinden](https://www.linkedin.com/in/lukas-zbinden-49667316b/) • [Nigel Nelson](https://www.linkedin.com/in/nigel-nelson-nvidia/)
> **Organization:** NVIDIA

## Overview

| **Model** | **Workload** | **Use Case**      |
|-----------|--------------|-------------------|
| [Cosmos Predict 2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) | Post-training | Surgical Robotics |

This recipe builds on [Cosmos-Surg-dVRK](https://cosmos-surg-dvrk.github.io/) by reproducing its core methodology with the improved Cosmos Predict 2.5 model, replacing the original Cosmos Predict 2 backbone while preserving the overall training approach.

Building on this foundation, we post‑train the Cosmos Predict world foundation model (WFM) to function as a learned simulator for policy evaluation. Developers are guided on how to finetune an action‑conditioned variant of Cosmos Predict 2.5 using domain‑specific surgical robotic data, leveraging the public [SutureBot](https://huggingface.co/datasets/jchen396/SutureBot) dataset, which contains endoscopic video paired with kinematic action sequences from the da Vinci Research Kit (dVRK). The resulting model implicitly captures both robot kinematics and task‑relevant environment dynamics, including realistic deformation and tool–deformable object interactions. This learned model forms the basis for simulation‑based policy evaluation, executed via a software‑in‑the‑loop rollout loop for autonomous surgical systems. While demonstrated on a surgical robotic use case, this recipe generalizes to other robotic systems and broader embodied AI applications.

TODO RECIPE FEEDBACK:
- recipes are encouraged to be more visual:
- in the overview, illustrate the painpoint/motivation for the work through a failure case

## Table of Contents

- [Prerequisites](#1-prerequisites)
- [Preparing Data](#2-preparing-data)
- [Model Configuration](#3-model-configuration)
- [Finetuning](#4-finetuning)
- [Inference & Evaluation](#5-inference--evaluation)
- [Results](#6-results)
- [Conclusion](#7-conclusion)


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

### 2.1 Exploration
SutureBot is dataset for autonomous end-to-end suturing on the da Vinci Research Kit (dVRK), covering subtasks 
like needle pickup, needle insertion, and knot tying. It provides multi-camera surgical video paired 
with robot kinematics to support imitation learning and evaluation of VLA/robotic policies. 
SutureBot contains about 1,890 demonstrations, amounting to 6 hours of video or 629,183 samples. 
Public access is via the [project 
page](https://suturebot.github.io/) and a [Hugging Face release](https://huggingface.co/datasets/jchen396/SutureBot).

Following is an example for each surgical task:

| Needle pickup                                                                                                              | Needle insertion                                                                                                          | Knot tying                                                                                                              |
|----------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|
| <video width="320" controls autoplay loop muted><source src="assets/suturebot_needle_pickup.mp4" type="video/mp4"></video> | <video width="320" controls autoplay loop muted><source src="assets/suturebot_needle_throw.mp4" type="video/mp4"></video> | <video width="320" controls autoplay loop muted><source src="assets/suturebot_knot_tying.mp4" type="video/mp4"></video> |



### 2.2 Location

The [SutureBot](https://huggingface.co/datasets/jchen396/SutureBot) dataset should be organized in a directory structure that you'll specify in the configuration. Set your dataset path to point to your dataset root folder:

```
/path/to/dataset/SutureBot
```

Replace this path with the actual download location of the SutureBot dataset. The dataset should contain da Vinci robot 
video clips stored as individual JPG files at 640x480 resolution.

### 2.3 Download
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

### 2.4 Convert to LeRobot Dataset format
To be compatible with Cosmos data processing, we need to convert the raw SutureBot data to the LeRobot Dataset format. 

Run the following script to convert the [SutureBot](https://huggingface.co/datasets/jchen396/SutureBot) dataset to the LeRobot format (notice lerobot==0.3.3 is expected). Notice that the output path is retrieved from the env variable \$HF_LEROBOT_HOME. Override \$HF_LEROBOT_HOME to change the location of the output.  
```bash
# optional: export HF_LEROBOT_HOME=/path/to/dataset/SutureBot/LeRobot
python3 -u convert_suturebot_to_lerobot_v3.py --data-path /path/to/dataset/SutureBot 
```
The script will save the SutureBot dataset in LeRobot format at the location as specified by $HF_LEROBOT_HOME.

## 3. Model Configuration
The finetuning wil be performed at 720x960 resolution (to match 720p pre-training) with 12 frames prediction horizon.

Before executing the finetuning script, the following source code changes must be applied to the cloned repository (as described in [Setup guide](./setup.md)). Those changes will address both model configuration and [SutureBot](https://huggingface.co/datasets/jchen396/SutureBot) data processing:

TODO RECIPE FEEDBACK:
- make presentation of code changes more intuitive, i.e. for tutorial purposes only thanks to fork

### 3.1 cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/embodiment_tags.py
Rationale: Register the embodiment 'dvrk'.
```python
diff --git a/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/embodiment_tags.py b/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/embodiment_tags.py
index e31586f..9133347 100644
--- a/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/embodiment_tags.py
+++ b/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/embodiment_tags.py
@@ -48,3 +48,5 @@ class EmbodimentTag(Enum):
     """
     AGIBOT = "agibot"
+
+    DVRK = "dvrk"
```

### 3.2 cosmos_predict2/experiments/base/action.py
Rationale: Configure the 2B Cosmos-predict 2.5 model for the [SutureBot](https://huggingface.co/datasets/jchen396/SutureBot) dataset.
```python
diff --git a/cosmos_predict2/experiments/base/action.py b/cosmos_predict2/experiments/base/action.py
index 30dbc91..c219ba0 100644
--- a/cosmos_predict2/experiments/base/action.py
+++ b/cosmos_predict2/experiments/base/action.py
@@ -125,9 +125,30 @@ ac_reason_embeddings_rectified_flow_2b_256_320 = LazyDict(
     flags={"allow_objects": True},
 )

+import copy
+ac_predict2p5_video2world_2b_suturebot_training = copy.deepcopy(ac_reason_embeddings_rectified_flow_2b_256_320)
+ac_predict2p5_video2world_2b_suturebot_training['job']['name'] = 'def_ac_predict2p5_video2world_2b_suturebot_training'
+ac_predict2p5_video2world_2b_suturebot_training['defaults'] = [
+    DEFAULT_CHECKPOINT.experiment,
+    {"override /model": "action_conditioned_video2world_fsdp_rectified_flow"},
+    {"override /net": "cosmos_v1_2B_action_conditioned"},
+    {"override /conditioner": "action_conditioned_video_conditioner"},
+    {"override /data_train": "suturebot_train"},
+    {"override /data_val": "suturebot_val"},
+    "_self_",
+]
+ac_predict2p5_video2world_2b_suturebot_training['model']['config']['net']['action_dim'] = 20
+ac_predict2p5_video2world_2b_suturebot_training['dataloader_train'] = {'batch_size': 4}
+ac_predict2p5_video2world_2b_suturebot_training['optimizer']['lr'] = 7.5e-6 # assuming 4 nodes
+
+print(f"+++++++++++++++++++++++++++++")
+print(ac_predict2p5_video2world_2b_suturebot_training)
+print(f"+++++++++++++++++++++++++++++")
+
+
 cs = ConfigStore.instance()

-for _item in [ac_reason_embeddings_rectified_flow_2b_256_320]:
+for _item in [ac_reason_embeddings_rectified_flow_2b_256_320, ac_predict2p5_video2world_2b_suturebot_training]:
     # Get the experiment name from the global variable
     experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015
```

### 3.3 cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py
Rationale: Define the data loading for the [SutureBot](https://huggingface.co/datasets/jchen396/SutureBot) dataset.
```python
diff --git a/cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py b/cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py
index 6b45363..f7316ed 100644
--- a/cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py
+++ b/cosmos_predict2/_src/predict2/action/configs/action_conditioned/data.py
@@ -93,6 +93,48 @@ bridge_13frame_480_640_val_dataset = L(Dataset_3D)(
     mode="val",
 )

+# experiment for action-sequence video prediction
+base_path_suturebot_ds = "/SutureBot"
+# Construct modality configs and transforms
+from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.dataset import LeRobotDataset
+from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.groot_configs import (
+    construct_modality_config_and_transforms,
+)
+modality_configs, train_transform, test_transform = construct_modality_config_and_transforms(
+    num_frames=13, embodiment="dvrk", downscaled_res=False
+)
+
+suturebot_train_dataset = L(LeRobotDataset)(
+    num_frames=13,
+    time_division_factor=4,
+    time_division_remainder=1,
+    max_pixels=1920 * 1080,
+    data_file_keys=("video",),
+    image_file_extension=("jpg", "jpeg", "png", "webp"),
+    video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
+    repeat=1,
+    args=None,
+    dataset_path=base_path_suturebot_ds,
+    data_split="train",
+    embodiment="dvrk",
+    downscaled_res=False,
+)
+
+suturebot_val_dataset = L(LeRobotDataset)(
+    num_frames=13,
+    time_division_factor=4,
+    time_division_remainder=1,
+    max_pixels=1920 * 1080,
+    data_file_keys=("video",),
+    image_file_extension=("jpg", "jpeg", "png", "webp"),
+    video_file_extension=("mp4", "avi", "mov", "wmv", "mkv", "flv", "webm"),
+    repeat=1,
+    args=None,
+    dataset_path=base_path_suturebot_ds,
+    data_split="test",
+    embodiment="dvrk",
+    downscaled_res=False,
+)

 # ------------------------------------------------------------

@@ -153,6 +195,19 @@ bridge_13frame_480_640_val_dataloader = L(DataLoader)(
     drop_last=True,
 )

+suturebot_train_dataloader = L(DataLoader)(
+    dataset=suturebot_train_dataset,
+    sampler=L(get_sampler)(dataset=suturebot_train_dataset),
+    batch_size=1,
+    drop_last=True,
+)
+suturebot_val_dataloader = L(DataLoader)(
+    dataset=suturebot_val_dataset,
+    sampler=L(get_sampler)(dataset=suturebot_val_dataset),
+    batch_size=1,
+    drop_last=True,
+)
+

 def register_training_and_val_data():
     cs = ConfigStore.instance()
@@ -199,6 +254,19 @@ def register_training_and_val_data():
         node=bridge_13frame_480_640_val_dataloader,
     )

+    cs.store(
+        group="data_train",
+        package="dataloader_train",
+        name="suturebot_train",
+        node=suturebot_train_dataloader,
+    )
+    cs.store(
+        group="data_val",
+        package="dataloader_val",
+        name="suturebot_val",
+        node=suturebot_val_dataloader,
+    )
+
     # Register gr00t_customized_gr1 data
     if register_gr00t_customized_gr1_data is not None:
         register_gr00t_customized_gr1_data()
```

### 3.4 cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/groot_configs.py
Rationale: Add more configuration required for the [SutureBot](https://huggingface.co/datasets/jchen396/SutureBot) dataset (e.g., resolution, delta action computation, normalization).
```python
diff --git a/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/groot_configs.py b/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/groot_configs.py
index 9932214..6f14d54 100644
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
@@ -127,6 +128,23 @@ def construct_modality_config_and_transforms(num_frames, embodiment, downscaled_
                 ],
             ),
         }
+    elif embodiment == "dvrk":
+        timestep_interval = 3  # LZ: downsampling rate
+        delta_indices = list(range(0, num_frames * timestep_interval, timestep_interval))
+        config = {
+            "video": ModalityConfig(
+                delta_indices=delta_indices,
+                modality_keys=["video.observation.images.main"],
+            ),
+            "state": ModalityConfig(
+                delta_indices=[0],
+                modality_keys=["state.observation.state"],
+            ),
+            "action": ModalityConfig(
+                delta_indices=delta_indices,
+                modality_keys=['action.action']
+            ),
+        }

     video_modality, state_modality, action_modality = config["video"], config["state"], config["action"]
     if embodiment == "gr1" or embodiment == "gr1_video_only":
@@ -135,6 +153,11 @@ def construct_modality_config_and_transforms(num_frames, embodiment, downscaled_
     elif embodiment == "agibot":
         width = 640 if not downscaled_res else 256
         height = 480 if not downscaled_res else 256
+    elif embodiment == "dvrk":
+        # width = 512 if not downscaled_res else 256
+        # height = 320 if not downscaled_res else 256
+        width = 960 if not downscaled_res else 256
+        height = 720 if not downscaled_res else 256

     train_transform = ComposedModalityTransform(
         transforms=[
@@ -145,12 +168,13 @@ def construct_modality_config_and_transforms(num_frames, embodiment, downscaled_
             StateActionToTensor(apply_to=state_modality.modality_keys),
             StateActionTransform(
                 apply_to=state_modality.modality_keys,
-                normalization_modes={key: "min_max" for key in state_modality.modality_keys},
+                normalization_modes={key: "mean_std" for key in state_modality.modality_keys},
             ),
             StateActionToTensor(apply_to=action_modality.modality_keys),
+            RelativeActionTransform(apply_to=action_modality.modality_keys),
             StateActionTransform(
                 apply_to=action_modality.modality_keys,
-                normalization_modes={key: "min_max" for key in action_modality.modality_keys},
+                normalization_modes={key: "mean_std" for key in action_modality.modality_keys},
             ),
             ConcatTransform(
                 video_concat_order=video_modality.modality_keys,
@@ -166,12 +190,13 @@ def construct_modality_config_and_transforms(num_frames, embodiment, downscaled_
             StateActionToTensor(apply_to=state_modality.modality_keys),
             StateActionTransform(
                 apply_to=state_modality.modality_keys,
-                normalization_modes={key: "min_max" for key in state_modality.modality_keys},
+                normalization_modes={key: "mean_std" for key in state_modality.modality_keys},
             ),
             StateActionToTensor(apply_to=action_modality.modality_keys),
+            RelativeActionTransform(apply_to=action_modality.modality_keys),
             StateActionTransform(
                 apply_to=action_modality.modality_keys,
-                normalization_modes={key: "min_max" for key in action_modality.modality_keys},
+                normalization_modes={key: "mean_std" for key in action_modality.modality_keys},
             ),
             ConcatTransform(
                 video_concat_order=video_modality.modality_keys,
```

### 3.5 cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/video.py
Rationale: A small bugfix on the Cosmos OSS code.
```python
diff --git a/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/video.py b/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/video.py
index 5eb4a32..5b022a8 100644
--- a/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/video.py
+++ b/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/video.py
@@ -127,7 +127,7 @@ class VideoTransform(ModalityTransform):
         super().set_metadata(dataset_metadata)
         self.original_resolutions = {}
         for key in self.apply_to:
-            split_keys = key.split(".")
+            split_keys = key.split(".", 1)
             assert len(split_keys) == 2, f"Invalid key: {key}. Expected format: modality.key"
             sub_key = split_keys[1]
             if sub_key in dataset_metadata.modalities.video:
```

### 3.6 cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/concat.py
Rationale: A small bugfix on the Cosmos OSS code.
```python
diff --git a/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/concat.py b/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/concat.py
index 9f9b537..9804503 100644
--- a/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/concat.py
+++ b/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/concat.py
@@ -73,7 +73,7 @@ class ConcatTransform(InvertibleModalityTransform):
         grouped_keys = {}
         for key in data.keys():
             try:
-                modality, _ = key.split(".")
+                modality, _ = key.split(".", 1)
             except:  # noqa: E722
                 ### Handle language annotation special case
                 if "annotation" in key:
@@ -173,7 +173,7 @@ class ConcatTransform(InvertibleModalityTransform):
         return self.apply(data)

     def get_modality_metadata(self, key: str) -> StateActionMetadata:
-        modality, subkey = key.split(".")
+        modality, subkey = key.split(".", 1)
         assert self.dataset_metadata is not None, "Metadata not set"
         modality_config = getattr(self.dataset_metadata.modalities, modality)
         assert subkey in modality_config, f"{subkey=} not found in {modality_config=}"
```

### 3.7 cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/state_action.py
Rationale: Functions to compute the kinematic delta action representation following [Stanford's UMI implementation](https://github.com/real-stanford/universal_manipulation_interface). 
```python
 diff --git a/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/state_action.py b/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/state_action.py
index 06c82d9..6572892 100644
--- a/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/state_action.py
+++ b/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/transform/state_action.py
@@ -18,6 +18,7 @@ import random
 from typing import Any, ClassVar

 import numpy as np
+from scipy.spatial.transform import Rotation

 # import pytorch3d.transforms as pt
 import torch
@@ -34,6 +35,131 @@ from cosmos_predict2._src.predict2.action.datasets.gr00t_dreams.data.transform.b
 )


+def rotation_6d_to_matrix(rot6d):
+    """
+    Convert 6D rotation representation to rotation matrix.
+    6D rotation is the first two ROWS of a rotation matrix (row-major format),
+    orthonormalized via Gram-Schmidt.
+
+    This matches the incoming dVRK/SutureBot data format:
+        [r11, r12, r13, r21, r22, r23] = [row1, row2]
+
+    Args:
+        rot6d: Array of shape (..., 6) containing [row1 (3), row2 (3)]
+
+    Returns:
+        Rotation matrices of shape (..., 3, 3)
+    """
+    shape = rot6d.shape[:-1]
+    rot6d = rot6d.reshape(*shape, 2, 3)
+
+    # First row (normalized)
+    row1 = rot6d[..., 0, :]
+    row1 = row1 / (np.linalg.norm(row1, axis=-1, keepdims=True) + 1e-8)
+
+    # Second row (orthogonalized and normalized)
+    row2 = rot6d[..., 1, :]
+    row2 = row2 - np.sum(row1 * row2, axis=-1, keepdims=True) * row1
+    row2 = row2 / (np.linalg.norm(row2, axis=-1, keepdims=True) + 1e-8)
+
+    # Third row (cross product)
+    row3 = np.cross(row1, row2)
+
+    # Stack into rotation matrix (as rows)
+    R = np.stack([row1, row2, row3], axis=-2)
+    return R
+
+
+def compute_rel_actions(actions):
+    """
+    Computes relative actions for a dual-arm robot.
+    Global translation delta, local (tooltip frame) rotation delta in 6D format.
+
+    Reference: https://github.com/real-stanford/universal_manipulation_interface
+
+    actions[0] is used as the base pose, actions[1:] are the targets.
+
+    Input per-arm: [xyz (3), 6D_rotation (6), gripper (1)] = 10
+    Dual-arm input: [n_actions, arm1 (10) + arm2 (10)] = [n_actions, 20]
+    Output per-arm: [delta_xyz (3), delta_rot6d (6), gripper (1)] = 10
+    Dual-arm output: [n_actions-1, arm1 (10) + arm2 (10)] = [n_actions-1, 20]
+
+    The relative rotation R_rel = R_base.T @ R_target is represented in 6D format
+    (first two rows of the rotation matrix, flattened).
+    """
+    if isinstance(actions, torch.Tensor):
+        actions = actions.numpy()
+
+    base = actions[0]
+    targets = actions[1:]
+    n_targets = targets.shape[0]
+    rel_actions = np.zeros((n_targets, 20))
+
+    for arm in range(2):
+        i = arm * 10  # Both input and output use same stride
+        R_base = rotation_6d_to_matrix(base[i + 3 : i + 9])
+        R_tgt = rotation_6d_to_matrix(targets[:, i + 3 : i + 9])
+
+        # Global translation delta
+        rel_actions[:, i : i + 3] = targets[:, i : i + 3] - base[i : i + 3]
+        # Relative rotation in 6D format (first 2 rows of R_rel, flattened)
+        R_rel = R_base.T @ R_tgt  # [n_targets, 3, 3]
+        rel_actions[:, i + 3 : i + 9] = R_rel[:, :2, :].reshape(n_targets, 6)
+        # Gripper (absolute value, not delta)
+        rel_actions[:, i + 9] = targets[:, i + 9]
+
+    return rel_actions
+
+
+def compute_rel_actions_local(actions):
+    """
+    Computes relative actions for a dual-arm robot using SE(3) transformation.
+    Both translation and rotation deltas are in the local (tooltip) frame.
+
+    Follows UMI 'relative' mode: T_rel = T_base^(-1) @ T_action
+    Reference: https://github.com/real-stanford/universal_manipulation_interface
+
+    actions[0] is used as the base pose, actions[1:] are the targets.
+
+    Input per-arm: [xyz (3), 6D_rotation (6), gripper (1)] = 10
+    Dual-arm input: [n_actions, arm1 (10) + arm2 (10)] = [n_actions, 20]
+    Output per-arm: [delta_xyz (3), delta_rotvec (3), gripper (1)] = 7
+    Dual-arm output: [n_actions-1, arm1 (7) + arm2 (7)] = [n_actions-1, 14]
+    """
+    if isinstance(actions, torch.Tensor):
+        actions = actions.numpy()
+
+    base = actions[0]
+    targets = actions[1:]
+    n_targets = targets.shape[0]
+    rel_actions = np.zeros((n_targets, 14))
+
+    for arm in range(2):
+        i, o = arm * 10, arm * 7
+
+        # Build 4x4 base pose matrix
+        T_base = np.eye(4)
+        T_base[:3, :3] = rotation_6d_to_matrix(base[i + 3 : i + 9])
+        T_base[:3, 3] = base[i : i + 3]
+
+        # Build 4x4 target pose matrices
+        T_targets = np.zeros((n_targets, 4, 4))
+        T_targets[:, :3, :3] = rotation_6d_to_matrix(targets[:, i + 3 : i + 9])
+        T_targets[:, :3, 3] = targets[:, i : i + 3]
+        T_targets[:, 3, 3] = 1.0
+
+        # SE(3) relative: T_rel = T_base^(-1) @ T_target
+        T_base_inv = np.linalg.inv(T_base)
+        T_rel = T_base_inv @ T_targets
+
+        # Extract components
+        rel_actions[:, o : o + 3] = T_rel[:, :3, 3]
+        rel_actions[:, o + 3 : o + 6] = Rotation.from_matrix(T_rel[:, :3, :3]).as_rotvec()
+        rel_actions[:, o + 6] = targets[:, i + 9]
+
+    return rel_actions
+
+
 class RotationTransform:
     """Adapted from https://github.com/real-stanford/diffusion_policy/blob/548a52bbb105518058e27bf34dcf90bf6f73681a/diffusion_policy/model/common/rotation_transformer.py"""

@@ -379,7 +505,7 @@ class StateActionTransform(InvertibleModalityTransform):

         # Check that all state keys specified in apply_to have their modality_metadata
         for key in self.apply_to:
-            split_key = key.split(".")
+            split_key = key.split(".", 1)
             assert len(split_key) == 2, "State keys should have two parts: 'modality.key'"
             if key not in self.modality_metadata:
                 modality, state_key = split_key
@ -389,7 +515,7 @@ class StateActionTransform(InvertibleModalityTransform):

         # Check that all state keys specified in normalization_modes have their statistics in state_statistics
         for key in self.normalization_modes:
-            split_key = key.split(".")
+            split_key = key.split(".", 1)
             assert len(split_key) == 2, "State keys should have two parts: 'modality.key'"
             modality, state_key = split_key
             assert hasattr(dataset_statistics, modality), f"{modality} statistics not found"
@@ -414,7 +540,7 @@ class StateActionTransform(InvertibleModalityTransform):

         # Initialize the normalizers
         for key in self.normalization_modes:
-            modality, state_key = key.split(".")
+            modality, state_key = key.split(".", 1)
             # If the state has a nontrivial rotation, we need to handle it more carefully
             # For absolute rotations, we need to convert them to the target representation and normalize them using min_max mode,
             # since we can infer the bounds by the representation
@@ -501,6 +627,36 @@ class StateActionTransform(InvertibleModalityTransform):
         return data


+class RelativeActionTransform(ModalityTransform):
+    """
+    Converts absolute actions to relative actions using compute_rel_actions.
+
+    This transform is used for dVRK (da Vinci Research Kit) datasets where:
+    - Input: 20D absolute actions [T, 20] (xyz + 6D_rot + gripper per arm)
+    - Output: 20D relative actions [T-1, 20] (delta_xyz + delta_rot6d + gripper per arm)
+
+    The relative actions are computed using global translation delta and local
+    (tooltip frame) rotation delta. The rotation delta is represented in 6D format
+    (first two rows of the relative rotation matrix).
+    """
+
+    apply_to: list[str] = Field(..., description="The action keys to transform to relative actions.")
+
+    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
+        for key in self.apply_to:
+            if key not in data:
+                continue
+            actions = data[key]
+            # Convert to numpy if tensor
+            is_tensor = isinstance(actions, torch.Tensor)
+            actions_np = actions.numpy() if is_tensor else actions
+            # Compute relative actions: [T, 20] -> [T-1, 20]
+            rel_actions = compute_rel_actions(actions_np)
+            # Convert back to tensor if input was tensor
+            data[key] = torch.from_numpy(rel_actions).to(actions.dtype) if is_tensor else rel_actions
+        return data
+
+
 class StateActionPerturbation(ModalityTransform):
     """
     Class for state or action perturbation.
```

### 3.8 cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/utils/video.py
Rationale: A bugfix on the Cosmos OSS code (occurs in case dataset videos in mp4 format use AV1 codec).
```python
diff --git a/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/utils/video.py b/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/utils/video.py
index 58a777f..9996f36 100644
--- a/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/utils/video.py
+++ b/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/utils/video.py
@@ -107,7 +107,7 @@ def get_frames_by_timestamps(
         # Note: closest key frame timestamp is usally smaller than `first_ts` (e.g. key frame can be the first frame of the video)
         # for details on what `seek` is doing see: https://pyav.basswood-io.com/docs/stable/api/container.html?highlight=inputcontainer#av.container.InputContainer.seek
         reader.seek(first_ts, keyframes_only=True)
-        # load all frames until last requested frame
+        # load all frames from first to last requested timestamp
         loaded_frames = []
         loaded_ts = []
         for frame in reader:
@@ -116,11 +116,18 @@ def get_frames_by_timestamps(
             loaded_ts.append(current_ts)
             if current_ts >= last_ts:
                 break
-            if len(loaded_frames) >= len(timestamps):
-                break
         reader.container.close()
         reader = None
-        frames = np.array(loaded_frames)
+
+        if len(loaded_frames) == 0:
+            raise ValueError(f"No frames loaded from {video_path} for timestamps {timestamps[0]:.3f} to {timestamps[-1]:.3f}")
+
+        # Match requested timestamps to closest loaded frames (like decord/opencv backends do)
+        loaded_ts = np.array(loaded_ts).reshape(-1, 1)  # (num_loaded, 1)
+        requested_ts = np.array(timestamps)  # (num_requested,)
+        # Find closest loaded frame for each requested timestamp
+        indices = np.abs(loaded_ts - requested_ts).argmin(axis=0)
+        frames = np.array([loaded_frames[i] for i in indices])
         return frames.transpose(0, 2, 3, 1)
     else:
         raise NotImplementedError
```


### 3.9 cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/dataset.py
Rationale: Minor code changes to facilitate kinematic delta action representation following [Stanford's UMI implementation](https://github.com/real-stanford/universal_manipulation_interface).
```python
diff --git a/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/dataset.py b/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/dataset.py
index 0aed5bb..7d0c1f0 100644
--- a/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/dataset.py
+++ b/cosmos_predict2/_src/predict2/action/datasets/gr00t_dreams/data/dataset.py
@@ -1066,8 +1066,8 @@ class LeRobotDataset(torch.utils.data.Dataset):
         self.lerobot_datasets = []
         for p in self.dataset_path:
             config, train_transform, test_transform = construct_modality_config_and_transforms(
-                num_frames=(num_frames + 1), embodiment=embodiment, downscaled_res=downscaled_res
-            )  # Add an additional prefix frame as baseline to compute delta actions
+                num_frames=num_frames, embodiment=embodiment, downscaled_res=downscaled_res
+            )
             self.lerobot_datasets.append(
                 WrappedLeRobotSingleDataset(
                     dataset_path=p,
@@ -1098,7 +1098,7 @@ class LeRobotDataset(torch.utils.data.Dataset):

         video = lerobot_data["video"]
         video_frames = []
-        for i in range(1, video.shape[1]):  # Skip first frame (used only as action baseline)
+        for i in range(video.shape[1]):
             frame = video[:, i, :, :]
             frame = Image.fromarray(frame.permute(1, 2, 0).numpy())
             video_frames.append(frame)
@@ -1106,24 +1106,17 @@ class LeRobotDataset(torch.utils.data.Dataset):
             print(
                 f"Warning: Expected {self.num_frames} frames, but got {len(video_frames)} frames. Randomly sampling an item instead."
             )
-            return self.__getitem__(random.randint(0, len(self) - 1))  # noqa: F821
+            return self.__getitem__(randint(0, len(self) - 1))  # noqa: F821
         video_frames = np.stack([np.array(frame, dtype=np.uint8) for frame in video_frames])

-        # Cumulative baselined delta actions (old version)
-        # NOTE: Need to tweak this after (num_frames + 1) change
-        # delta_actions = lerobot_data["action"][1:] - lerobot_data["action"][[0]]
-        # Chunked cumulative baselined delta actions (for chunked action architecture)
+        # Actions are now relative after RelativeActionTransform in the pipeline
+        # The transform converts 20D absolute actions to 20D relative actions
         actions = lerobot_data["action"]
-        delta_actions = []
-        for t in range(1, len(actions) - 1, self.time_division_factor):
-            delta_actions.append(actions[t : t + self.time_division_factor] - actions[t - 1])
-        delta_actions = torch.cat(delta_actions, dim=0)

         data = {
             "prompt": prompt,
             "video": torch.from_numpy(video_frames).permute(3, 0, 1, 2),
-            # "action": torch.from_numpy(delta_actions),
-            "action": (delta_actions),
+            "action": actions,
             "ai_caption": "",
             "text": prompt,
             "t5_text_embeddings": torch.zeros(512, 1024, dtype=torch.bfloat16).cuda(),
@@ -1143,3 +1136,4 @@ class LeRobotDataset(torch.utils.data.Dataset):

     def __len__(self):
         return sum([len(d) for d in self.lerobot_datasets]) * self.repeat
+
```

## 4. Finetuning
With the code changes applied, we are set to start the finetuning, using 4 nodes (32 GPUs). 
The batch size was configured to be 4, resulting in a global batch size of 128. 
We recommend using at least 1 node with 8 GPUs for finetuning the 2B Cosmos model (global batch size of 32).
```bash
mkdir logs
sbatch run_finetuning.sh
```

Run the finetuning for 20,000 steps.

The checkpoints in distributed format (DCP) will be saved in:
```bash
cd ${IMAGINAIRE_OUTPUT_ROOT}/cosmos_predict2_action_conditioned/cosmos_predict_v2p5/def_ac_predict2p5_video2world_2b_suturebot_training/checkpoints
```
TODO RECIPE FEEDBACK:
- please include more information, such as training log (you can include wandb log screen shots), 
and also give an approximate training elapsed time.
- (optional) have you done experiments with different parameter setups? 
provide intuition on the choice of hyperparameters or data blending


## 5 Inference & Evaluation

This recipe is grounded in the methodology of [Cosmos-Surg-dVRK](https://cosmos-surg-dvrk.github.io/), which validated the world model by comparing policy success rates in Cosmos simulation against real-world robot execution. Across three SutureBot tasks and six VLA models, that approach achieved a strong positive correlation with the real-world dVRK rollouts (Pearson r = 0.718, p < 0.001).

Notice that the public SutureBot dataset lacks the failure trajectories used in the [Cosmos-Surg-dVRK](https://cosmos-surg-dvrk.github.io/) to balance training. Without this negative data, WFMs will tend to hallucinate success (false positives). Therefore, instead of replicating the full policy rollout validation, this recipe demonstrates how to run open-loop inference on held-out test data. This generates video outputs that users can visually compare against ground truth to assess the model's kinematic faithfulness and physical realism.

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
Run model inference using the latest checkpoint from the finetuning step above. The script will generate several rollouts given the ground truth kinematic action trajectories and an initial frame, both selected from the dataset test split.
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

> For more inference options and advanced usage, see the Cosmos Predict 2 [inference documentation](https://github.com/nvidia-cosmos/cosmos-predict2/blob/main/docs/inference.md).

### 5.3 Swapping in a Surgical Policy

To evaluate a surgical policy (a VLA model) instead of GT actions, modify the inference loop in `inference_dvrk.py` as follows:

```python
# Current (GT actions from dataset):
actions = data["action"].numpy()

# With a policy:
actions = policy.predict(current_frame)  # Returns (12, action_dim)
```

The finetuned Cosmos model expects **normalized** action sequences matching the expected shape `(chunk_size, action_dim)` and following the **relative action formulation** used above in this recipe.

> Note: Running Cosmos with a policy's output actions generates video rollouts (MP4 files) for manual review. To automate this evaluation process, [Cosmos-Reason2](https://github.com/nvidia-cosmos/cosmos-reason2) can be post-trained to serve as a judge, automatically detecting task successes, failures, and physics anomalies.



## 6. Results

### Comparison: Base Model vs Post-Trained Model
The post-trained Cosmos-predict 2.5 model generates faithful and highly realistic rollouts as compared to the ground truth video. Below is a comparison of videos generated by the post-trained model versus the real-world ground truth videos:

| Sample | Ground Truth                                                                                               | Post-Trained Model |
|--------|------------------------------------------------------------------------------------------------------------|-------------------|
| **Sample 1** | <video width="320" controls autoplay loop muted><source src="assets/base/0.mp4" type="video/mp4"></video>  | <video width="320" controls autoplay loop muted><source src="assets/post_trained/0.mp4" type="video/mp4"></video> |
| **Sample 2** | <video width="320" controls autoplay loop muted><source src="assets/base/12.mp4" type="video/mp4"></video> | <video width="320" controls autoplay loop muted><source src="assets/post_trained/12.mp4" type="video/mp4"></video> |
| **Sample 3** | <video width="320" controls autoplay loop muted><source src="assets/base/5.mp4" type="video/mp4"></video>  | <video width="320" controls autoplay loop muted><source src="assets/post_trained/5.mp4" type="video/mp4"></video> |


## 7. Conclusion
TODO RECIPE FEEDBACK: 
- add one section for conclusion, with your message for key takeaways. You can include insights, 
or limitations that you which can be aimed for the future.


## Further Reading

1. [Cosmos-Surg-dVRK](https://cosmos-surg-dvrk.github.io/) - World foundation model-based automated online evaluation of surgical robot policy learning
2. [Cosmos Predict 2.5 Model](https://github.com/nvidia-cosmos/cosmos-predict2.5) - Model weights and documentation.
3. [SutureBot](https://suturebot.github.io/) - A Precision Framework & Benchmark For Autonomous End-to-End Suturing.
4. [The da Vinci Research Kit](https://www.intuitive-foundation.org/dvrk/) - A community effort supporting research in the field of telerobotic surgery