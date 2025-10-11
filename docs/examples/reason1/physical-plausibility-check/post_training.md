# Physical Plausibility Prediction with Cosmos-Reason1

> **Authors:** [Shun Zhang](https://www.linkedin.com/in/shun-zhang-1b154437/) • [Jingyi Jin](https://www.linkedin.com/in/jingyi-jin/)
> **Organization:** NVIDIA

## Overview

| **Model** | **Workload** | **Use Case** |
|-----------|--------------|--------------|
| Cosmos-Reason1 | Post-training | Physical plausibility prediction |

In synthetic video generation, it is crucial to determine the quality of generated videos and filter out videos that don't meet quality standards. In this case study, we demonstrate using the Cosmos-Reason1 model for physical plausibility prediction. Physics plausibility assessment involves evaluating whether the physical interactions and behaviors observed in videos are consistent with real-world laws of physics and constraints.

- [Setup and System Requirement](setup.md)

We first evaluate the model's ability to predict physical plausibility on an open-source dataset. We then fine-tune the model and evaluate its performance.

## Dataset: VideoPhy-2

We use the [VideoPhy-2 dataset](https://github.com/Hritikbansal/videophy/tree/main/VIDEOPHY2) for this case study, which is designed as an action-centric benchmark for evaluating physical common sense in generated videos.

### Dataset Overview

VideoPhy-2 provides a comprehensive evaluation framework for testing how well models understand and predict physical plausibility in video content. The dataset features **human evaluations on physics adherence** using a **standardized 1-5 point scale**.

| **Dataset Split** | **Size** | **Access** |
|-------------------|----------|------------|
| **Training Set** | 3.4k videos | [videophysics/videophy2_train](https://huggingface.co/datasets/videophysics/videophy2_train/) |
| **Evaluation Set** | 3.3k videos | [videophysics/videophy2_test](https://huggingface.co/datasets/videophysics/videophy2_test) |

### Evaluation Criteria

Each video receives human evaluations based on **adherence to physical laws** using a standardized 5-point scale:

| **Score** | **Description** | **Physics Adherence** |
|-----------|-----------------|----------------------|
| **1** | No adherence to physical laws | Completely implausible |
| **2** | Poor adherence to physical laws | Mostly unrealistic |
| **3** | Moderate adherence to physical laws | Mixed realistic/unrealistic |
| **4** | Good adherence to physical laws | Mostly realistic |
| **5** | Perfect adherence to physical laws | Completely plausible |

### Key Physics Challenges

The dataset highlights critical challenges for generative models in understanding fundamental physical rules:

- **Conservation Laws**: Mass, energy, and momentum conservation
- **Gravitational Effects**: Realistic falling and weight behavior
- **Collision Dynamics**: Object interaction physics
- **Temporal Causality**: Cause-and-effect relationships
- **Spatial Constraints**: Object boundaries and spatial logic

### Example Videos from the Dataset

#### Low Physical Plausibility (Score: 2/5)

<video controls width="480">
  <source src="assets/A_robotic_arm_gently_pokes_a_stack_of_plastic_cups,_making_the_bottom_cups_slide_out_and_the_whole_stack_fall.mp4" type="video/mp4">
</video>

- **Scene**: A robotic arm gently pokes a stack of plastic cups.

- **Physics Issue**: The stack of cups does not maintain its shape when the robotic arm interacts with it.

- **Key Problems**: Conservation of mass and elasticity.

#### High Physical Plausibility (Score: 4/5)

<video controls width="480">
  <source src="assets/A_robotic_arm_pushes_a_metal_cube_off_a_steel_table.mp4" type="video/mp4">
</video>

- **Scene**: A robotic arm pushes a metal cube off a steel table.

- **Physics Strengths**: The robotic arm moves the cube from one position to another. The cube maintains its shape and volume throughout the interaction.

- **Key Success**: Conservation of mass and gravity.

## Zero-Shot Inference

We first evaluate the model's ability to predict physical plausibility on the VideoPhy-2 evaluation set without any fine-tuning. We use the same prompt from the [VideoPhy-2 paper](https://arxiv.org/abs/2503.06800), which provides detailed instructions on what aspects of the video to evaluate and scoring criteria.

???+ code "Prompt for Scoring Physical Plausibility"

    ```yaml
    --8<-- "docs/examples/reason1/physical-plausibility-check/assets/score_prompt.yaml"
    ```

The following command runs zero-shot inference on the VideoPhy-2 evaluation set using 8 GPUs:

    # In the cosmos-reason root directory
    uv run video_reward_videophy.py \
        --dataset videophysics/videophy2_test \
        --split test \
        --model nvidia/Cosmos-Reason1-7B \
        --prompt_path prompts/video_reward_v1_no_thinking.yaml \
        --num_gpus 8

### Evaluation Metrics

We evaluate the model performance using two key metrics:

- **Accuracy**: The percentage of videos where predicted scores match ground truth scores
- **Correlation**: The correlation between predicted and ground truth scores

### Results

We compared Cosmos-Reason1 with Gemini-2.0-Flash-Exp (the baseline from the paper). Even without fine-tuning, Cosmos-Reason1 demonstrates superior performance.

| Model                    | Accuracy | Correlation |
|--------------------------|----------|-------------|
| **Gemini-2.0-Flash-Exp** | -        | 0.11        |
| **Cosmos-Reason1**       | 0.196    | 0.293       |

### Example Predictions

The following examples demonstrate zero-shot predictions from the Cosmos-Reason1 model:

#### Car Crashes into Stack of Cardboard Boxes

<video controls width="480">
  <source src="assets/A_car_crashes_into_a_stack_of_cardboard_boxes,_sending_the_boxes_flying_in_all_directions.mp4" type="video/mp4">
</video>

- **Model prediction**: 1
- **Ground truth**: 2 (poor adherence to physical laws)

#### Robotic Arm Operates on Circuit Board

<video controls width="480">
  <source src="assets/A_robotic_arm_uses_a_slender_tool_to_carefully_reposition_a_circuit_board,_nudging_it_a_fraction_of_an_inch_2265.mp4" type="video/mp4">
</video>

- **Model prediction**: 5
- **Ground truth**: 5 (perfect adherence to physical laws)

## Post-Training

Having demonstrated that Cosmos-Reason1 can predict physical plausibility and outperform baseline models in zero-shot evaluation, we now apply supervised fine-tuning (SFT) using the VideoPhy-2 training set to further improve the model's performance.

### Training Data Format

The fine-tuning process uses the following data structure:

- **Input**: Video + language instruction (from the evaluation prompt)
- **Output**: Physical plausibility score (1-5 scale)

### Setup

We use the [cosmos-rl](https://github.com/nvidia-cosmos/cosmos-rl) library for fine-tuning. First, download and prepare the VideoPhy-2 training data:

    # From the cosmos-reason root directory
    cd examples/post_training_hf/
    uv run scripts/download_videophy2.py \
        --output data/videophy2_train \
        --dataset videophysics/videophy2_train \
        --split train

### Training Configuration

We use the following configuration optimized for 8 GPUs:

???+ code "Training Configuration"

    ```toml
    --8<-- "docs/examples/reason1/physical-plausibility-check/assets/sft_config.toml"
    ```

### Running Training

We run training with the following steps:

1. Save the configuration as `examples/post_training_hf/configs/videophy2_sft.toml`.
2. Execute the training script:

        # In [cosmos-reason root directory]/examples/post_training_hf
        cosmos-rl --config configs/videophy2_sft.toml scripts/custom_sft.py

> **Note**: The training process uses the [custom SFT script](https://github.com/nvidia-cosmos/cosmos-reason1/blob/main/examples/post_training_hf/scripts/custom_sft.py), which includes a data loader that works with the **Hugging Face datasets format**.

### Results

After fine-tuning, we evaluate the model on the VideoPhy-2 evaluation set using the same metrics. The results demonstrate significant performance improvements:

| **Model Configuration** | **Accuracy** | **Correlation** |
|-------------------------|--------------|-----------------|
| Cosmos-Reason1 (Zero-shot) | 0.196 | 0.293 |
| + SFT (20 steps) | 0.219 | 0.280 |
| + SFT (40 steps) | 0.311 | 0.375 |
| + SFT (60 steps) | 0.324 | **0.395** |
| + SFT (80 steps) | 0.259 | 0.388 |
| + SFT (100 steps) | **0.340** | 0.383 |
| + SFT (120 steps) | 0.308 | 0.386 |
| **VideoPhy-AutoEval** | - | 0.37 |

**Key observations**

- Performance improves significantly after fine-tuning
- Best correlation achieved at 60 steps (0.395)
- Best accuracy achieved at 100 steps (0.340)
- Outperforms VideoPhy-AutoEval baseline after 40-60 training steps

### Comparison Examples

The following examples show prediction improvements from fine-tuning:

#### Robotic Arm Operates on Circuit Board

<video controls width="480">
    <source src="assets/A_robotic_arm_uses_a_slender_tool_to_carefully_reposition_a_circuit_board,_nudging_it_a_fraction_of_an_inch.mp4" type="video/mp4">
</video>

- **Before SFT**: 3
- **After SFT (60 steps)**: 5
- **Ground truth**: 5

#### Robot Shovels Snow

<video controls width="480">
    <source src="assets/ed9bc667-9b64-4224-8860-8108a42c8823_result.mp4" type="video/mp4">
</video>

- **Before SFT**: 4
- **After SFT (60 steps)**: 5
- **Ground truth**: 5

For both examples, the fine-tuned model correctly predicts the physical plausibility scores.
