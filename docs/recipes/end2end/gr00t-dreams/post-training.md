# Leveraging World Foundation Models for Synthetic Trajectory Generation in Robot Learning

> **Author:** [Rucha Apte](https://www.linkedin.com/in/ruchaa-apte/), [Jingyi Jin](https://www.linkedin.com/in/jingyi-jin/), [Saurav Nanda](https://www.linkedin.com/in/sauravnanda/)
> **Organization:** NVIDIA


| **Model**                                                                | **Workload**             | **Use Case**                                   |
| ------------------------------------------------------------------------ | ------------------------ | ---------------------------------------------- |
| [Cosmos Predict 2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) | Post-training, Inference | Synthetic Trajectory Generation                |
| [Cosmos Reason 2](https://github.com/nvidia-cosmos/cosmos-reason2)       | Inference                | Reasoning and filtering synthetic trajectories |


This guide walks you through post-training the Cosmos Predict 2.5 model on the [PhysicalAI-Robotics-GR00T-GR1](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-GR1) open dataset to generate synthetic robot trajectories for robot learning applications. After post-training, we'll use the fine-tuned model to generate trajectory predictions on the [PhysicalAI-Robotics-GR00T-Eval](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-Eval) dataset. Finally, Cosmos Reason 2 is leveraged to evaluate these generated trajectories by assessing their physical plausibility, helping to quantify and filter for valid, realistic, and successful robot motions.

## Motivation

Generalist robotics is emerging, driven by advances in mechatronics and robot foundation models, but scaling skill learning remains limited by the need for massive training data. [NVIDIA Isaac GR00T-Dreams](https://github.com/nvidia/gr00t-dreams), built on [NVIDIA Cosmos](https://www.nvidia.com/en-us/ai/cosmos/?sortBy=developer_learning_library%2Fsort%2Ffeatured_in.cosmos%3Adesc%2Ctitle%3Aasc&hitsPerPage=6), addresses this by generating large-scale synthetic trajectory data from a single image and language prompt. This enables efficient training of models such as [NVIDIA Isaac GR00T N1.5](https://developer.nvidia.com/isaac/gr00t) for reasoning and skill learning.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Preparing Data](#1-preparing-data)
- [Post-training Cosmos Predict 2.5](#2-post-training-cosmos-predict)
- [Inference with Post-trained Predict 2.5](#3-inference-with-post-trained-cosmos-predict)
  - [Converting DCP Checkpoint to Consolidated PyTorch Format](#31-converting-dcp-checkpoint-to-consolidated-pytorch-format)
  - [Running Inference](#32-running-inference)
- [Policy Evaluation using Cosmos Predict 2.5](#4-Policy-evaluation-using-cosmos-predict)
- [Using Cosmos-Reason 2 as Video Critic for Rejection Sampling](#4-using-cosmos-reason-as-video-critic-for-rejection-sampling)


## Prerequisites

Follow the [Setup guide](./setup.md) for general environment setup instructions, including installing dependencies for Cosmos Predict 2.5 and Cosmos Reason 2.

## 1. Preparing Data

First, we will download the [GR1 training dataset](https://huggingface.co/datasets/nvidia/GR1-100) and then preprocess it to create text prompt txt files for each video.

Download DreamGen Bench Training Dataset 

```bash
cd cosmos-predict2.5 
```

```bash
hf download nvidia/GR1-100 --repo-type dataset --local-dir datasets/benchmark_train/hf_gr1/ && \
mkdir -p datasets/benchmark_train/gr1/videos && \
mv datasets/benchmark_train/hf_gr1/gr1/*mp4 datasets/benchmark_train/gr1/videos && \
mv datasets/benchmark_train/hf_gr1/metadata.csv datasets/benchmark_train/gr1/
```

Preprocess DreamGen Bench Training Dataset

```bash
python -m scripts.create_prompts_for_gr1_dataset --dataset_path datasets/benchmark_train/gr1
```

Upon running the above preprocessing, the dataset folder format should look like this:

```bash
datasets/benchmark_train/gr1/
├── metas/
│   ├── *.txt
├── videos/
│   ├── *.mp4
├── metadata.csv
```

Preview of the Training Dataset

| Input Prompt File | Video File |
| ----------------- | ---------- |
| The robot arm is performing a task. Use the right hand to pick up green bok choy from tan table right side to bottom level of wire basket. | <video width="320" controls autoplay loop muted><source src="assets/1.mp4" type="video/mp4"></video> |
| The robot arm is performing a task. Use the right hand to pick up rubik's cube from top level of the shelf to bottom level of the shelf. | <video width="320" controls autoplay loop muted><source src="assets/2.mp4" type="video/mp4"></video> |
| The robot arm is performing a task. Use the right hand to pick up banana from teal plate to wooden table. |  <video width="320" controls autoplay loop muted><source src="assets/3.mp4" type="video/mp4"></video> |

## 2. Post Training Cosmos Predict 2.5

For this tutorial we will post train Cosmos-Predict2.5 2B model. The 14B post-training is very similar to the 2B example below.
Run the following command to execute an example post-training job with GR1 data.

```bash
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_predict2/_src/predict2/configs/video2world/config.py -- experiment=predict2_video2world_training_2b_groot_gr1_480
```
> **Note**: To disabling W&B Logging, add job.wandb_mode=disabled to disable wandb

This script makes use of `predict2_video2world_training_2b_groot_gr1_480` config. See the job config belwo to understand how they are determined.

```bash
predict2_video2world_training_2b_groot_gr1_480 = dict(
    ...,
    dataloader_train=dataloader_train_gr1,
    ...,
    job=dict(
        project="cosmos_predict_v2p5",
        group="video2world",
        name="2b_groot_gr1_480",
    ),
    ...,
)
```

> Checkpoints are saved to ${IMAGINAIRE_OUTPUT_ROOT}/PROJECT/GROUP/NAME/checkpoints. By default, IMAGINAIRE_OUTPUT_ROOT is /tmp/imaginaire4-output. In the above example, PROJECT: cosmos_predict_v2p5, GROUP: video2world, NAME: 2b_groot_gr1_480. Checkpoints will be saved in Distributed Checkpoint (DCP) Format.

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

Upon completing the post-training you should see similar loss curve in your W&B tracker.
train/loss is the best single indicator of convergence. train/video_loss measures the video generation quality and train/video_edm_loss and train/edm_loss meeasures the diffusion training quality.

<img src="assets/loss.png" width="320"/> 

## 3. Inference with Post Trained Cosmos Predict 2.5

### 3.1 Converting DCP Checkpoint to Consolidated PyTorch Format

Since the checkpoints are saved in DCP format during training, you need to convert them to consolidated PyTorch format (.pt) for inference. Use the `convert_distcp_to_pt.py` script:

```bash
# Get path to the latest checkpoint
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}/cosmos_predict_v2p5/video2world/2b_groot_gr1_480/checkpoints
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

Single Video Generation

```bash
torchrun --nproc_per_node=8 examples/inference.py \
  -i assets/sample_gr00t_dreams_gr1/gr00t_image2world.json \
  -o outputs/gr00t_gr1_sample \
  --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
  --experiment predict2_video2world_training_2b_groot_gr1_480
```

Below is an example visualizing the batch inference output.

| Prompt | Input Image | Generated Video |
|--------|-------------|----------------|
| Use the right hand to pick up red apple from brown tray to beige placemat. | <img src="assets/40_Use the right hand to pick up red apple from brown tray to beige placemat..png" width="128"/> | <video src="assets/40_Use the right hand to pick up red apple from brown tray to beige placemat..mp4" width="160" controls></video> |

## 4. Policy Evaluation using Cosmos Predict 2.5 

Lastly, we will download the [GR00T Eval Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-GR1) and then preprocess it to create batch input.

Download the DreamGen Benchmark dataset

```bash
huggingface-cli download nvidia/EVAL-175 --repo-type dataset --local-dir dream_gen_benchmark
```

Prepare batch input json

```bash
python -m scripts.prepare_batch_input_json \
  --dataset_path dream_gen_benchmark/gr1_object/ \
  --save_path output/dream_gen_benchmark/cosmos_predict2_14b_gr1_object/ \
  --output_path dream_gen_benchmark/gr1_object/batch_input.json
```

Preprocess batch input for inference input

For generating multiple videos with different inputs and prompts, you can use a JSONL file with batch inputs. The JSONL file should contain an array of objects, where each object has:

```bash
# Adjust paths based on where you cloned the repositories
cp -r cosmos-cookbook/scripts/examples/predict2.5/gr00t-dreams/gr1_batch_to_jsonl.py cosmos-predict2.5/scripts/
```

```bash
python scripts/gr1_batch_to_jsonl.py
```

After running the above script, the jsonl file will have following structure:

```jsonl
{"inference_type": "image2world", "name": "000_11_Use_the_right_hand_to_pick_up_green_pepper_from_black_shelf_to_inside_brown_p", "prompt": "Use the right hand to pick up green pepper from black shelf to inside brown paper bag.", "input_path": "11_Use the right hand to pick up green pepper from black shelf to inside brown paper bag..png", "num_output_frames": 93, "resolution": "432,768", "seed": 0, "guidance": 7}
```
Multiple Video Rollouts Generation

Using the same input, the Cosmos Predict2.5 generates multiple video rollouts. Among these, some exhibit greater physical plausibility than others.

```bash
# Adjust paths based on where you cloned the repositories
cp -r cosmos-cookbook/scripts/examples/predict2.5/gr00t-dreams/inference.py cosmos-predict2.5/
cp -r cosmos-cookbook/scripts/examples/predict2.5/gr00t-dreams/config.py cosmos-predict2.5/
```

```bash
torchrun --nproc_per_node=1 examples/inference.py \
  -i dream_gen_benchmark/gr1_object/gr1_batch.jsonl \
  -o outputs/gr1_object_run_ng5 \
  --num-generations 5 \
  --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
  --experiment predict2_video2world_training_2b_groot_gr1_480
```

| Generation 1 | Generation 2 | Generation 3 | Generation 4 | Generation 5 |
|---------|---------|---------|---------|---------|
| <video src="assets/002_28_Use_the_right_hand_to_pick_up_tall_red_glass_from_center_of_tan_table_to_brig_seed0.mp4" controls width="200"></video> | <video src="assets/002_28_Use_the_right_hand_to_pick_up_tall_red_glass_from_center_of_tan_table_to_brig_seed1.mp4" controls width="200"></video> | <video src="assets/002_28_Use_the_right_hand_to_pick_up_tall_red_glass_from_center_of_tan_table_to_brig_seed2.mp4" controls width="200"></video> | <video src="assets/002_28_Use_the_right_hand_to_pick_up_tall_red_glass_from_center_of_tan_table_to_brig_seed3.mp4" controls width="200"></video> | <video src="assets/002_28_Use_the_right_hand_to_pick_up_tall_red_glass_from_center_of_tan_table_to_brig_seed4.mp4" controls width="200"></video> |

## Using Cosmos Reason 2 as Video Critic for Rejection Sampling
Cosmos Reason2 is capable of evaluating if a video adheres to fundamental physical laws such as Gravity, Object Permanency, Collision dynamics, and Cause-and-effect relationships. When paired with a world model such as Cosmos Predict2.5, it enables best-of-N sampling by generating multiple video candidates and selecting the most physically accurate ones, thereby improving generation quality.

### Evaluation Criteria

Each generated video receives human evaluations based on **adherence to physical laws** using a standardized 5-point scale:

| **Score** | **Description** | **Physics Adherence** |
|-----------|-----------------|----------------------|
| **1** | No adherence to physical laws | Completely implausible |
| **2** | Poor adherence to physical laws | Mostly unrealistic |
| **3** | Moderate adherence to physical laws | Mixed realistic/unrealistic |
| **4** | Good adherence to physical laws | Mostly realistic |
| **5** | Perfect adherence to physical laws | Completely plausible |

### Zero Shot Inference 

???+ code "Prompt for Scoring Physical Plausibility"

    ```yaml
    --8<-- "recipes/end2end/gr00t-dreams/assets/video_reward.yaml"
    ```

To run zero-shot inference, you need to clone both repositories and copy the necessary files:

```bash
# Adjust paths based on where you cloned the repositories
cp -r cosmos-cookbook/scripts/examples/predict2.5/gr00t-dreams/inference_videophy2.py cosmos-reason2/examples/gr00t-dreams
# Adjust paths based on where you cloned the repositories
cp cosmos-cookbook/docs/recipes/post_training/reason2/physical-plausibility-check/assets/video_reward.yaml cosmos-reason2/prompts/video_reward.yaml
```

Run inference on the generated videos using Cosmos Reason 2. From the Cosmos Reason 2 project root directory:

```bash
uv run examples/gr00t-dreams/inference_videophy2.py \
  --video-dir ./cosmos-predict2.5/outputs/gr1_object_run \
  --output-dir outputs/gr1_object_run_critic
```

The above script will generate a json with scores and explanation for each generated video.

| **Score** | **Video Link** | **Reason2 Explanation** |
|-----------|---------------|--------------------|
| 4.0 | <video src="assets/000_11_Use_the_right_hand_to_pick_up_green_pepper_from_black_shelf_to_inside_brown_p_seed3.mp4" controls width="200"></video> | The video shows two robotic arms working together to pack items into a paper bag. The left arm holds the bag open while the right arm places a green bell pepper into it. Both arms move smoothly and deliberately, demonstrating coordination and precision. The person in the background observes the process without interfering. All actions are consistent with real-world physics, showing no signs of unrealistic behavior or violations of physical laws. Objects interact naturally, and motions are fluid and logical. |
| 1.0 | <video src="assets/000_11_Use_the_right_hand_to_pick_up_green_pepper_from_black_shelf_to_inside_brown_p_seed1.mp4" controls width="200"></video> | The robot arm fails to grasp the green bell pepper, which contradicts its intended function. The lack of successful grasping suggests a malfunction or error in the programming controlling the robot's movements. Additionally, the person's hand remains stationary throughout the video, showing no interaction with the objects on the table, which is inconsistent with typical human behavior during such tasks. |

Computing Evaluation Metrics

<img src="assets/critic_metrics.png" alt="Critic Evaluation Metrics" width="600">


### Citation

If you use this recipe or reference this work, please cite it as:

```bibtex
@misc{cosmos_cookbook_gr00t_2026,
  title={Leveraging World Foundation Models for Synthetic Trajectory Generation in Robot Learning},
  author={Apte, Rucha and Jin, Jingyi and Nanda, Saurav},
  organization={NVIDIA},
  year={2026},
  month={March},
  howpublished={\url{https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/end2end/gr00t-dreams/post-training.html}},
  note={NVIDIA Cosmos Cookbook}
}
```

**Suggested text citation:**

> Rucha Apte, Jingyi Jin, Saurav Nanda (2026). Leveraging World Foundation Models for Synthetic Trajectory Generation in Robot Learning. In *NVIDIA Cosmos Cookbook*. NVIDIA. Accessible at <https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/end2end/gr00t-dreams/post-training.html>

