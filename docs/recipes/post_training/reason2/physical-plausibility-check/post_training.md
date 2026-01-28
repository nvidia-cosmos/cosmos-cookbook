# Physical Plausibility Prediction with Cosmos Reason 2

> **Authors:** [Shun Zhang](https://www.linkedin.com/in/shun-zhang-1b154437/) • [Zekun Hao](https://www.linkedin.com/in/zekunhao/) • [Jingyi Jin](https://www.linkedin.com/in/jingyi-jin/)
> **Organization:** NVIDIA

| **Model** | **Workload** | **Use Case** |
|-----------|--------------|--------------|
| [Cosmos Reason 2](https://github.com/nvidia-cosmos/cosmos-reason2) | Post-training | Physical plausibility prediction |

> **Note**: For experiments using Cosmos Reason 1, please refer to the [Physical Plausibility Prediction with Cosmos Reason 1](../../reason1/physical-plausibility-check/post_training.md) recipe.

## Overview

In synthetic video generation, it is crucial to determine the quality of the generated videos and filter out videos of bad quality.
In this case study, we demonstrate using the Cosmos Reason 2 model for physical plausibility prediction. Physics plausibility assessment involves evaluating whether the physical interactions and behaviors observed in videos are consistent with real-world physics laws and constraints.

<p align="center">
  <img src="assets/data_filter_pipeline.png" alt="Data filtering pipeline with Cosmos Reason 2" style="width: 100%; max-width: 600px;">
</p>

When generating synthetic videos using generative models (e.g., [Cosmos Predict](https://github.com/nvidia-cosmos/cosmos-predict2.5) or [Cosmos Transfer](https://github.com/nvidia-cosmos/cosmos-transfer2.5)), we filter out videos that are not physically plausible before including them in downstream datasets or tasks (illustrated in the figure above).

- [Setup and System Requirements](setup.md)

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
  <source src="https://videophysics2trainvideos.s3.us-east-2.amazonaws.com/hunyuan_xedit_train/A_robotic_arm_gently_pokes_a_stack_of_plastic_cups,_making_the_bottom_cups_slide_out_and_the_whole_stack_fall.mp4" type="video/mp4">
</video>

- **Scene**: A robotic arm gently pokes a stack of plastic cups.

- **Physics Issue**: The stack of cups does not maintain its shape when the robotic arm interacts with it.

- **Key Problems**: Conservation of mass and elasticity.

#### High Physical Plausibility (Score: 4/5)

<video controls width="480">
  <source src="https://videophysics2trainvideos.s3.us-east-2.amazonaws.com/cosmos_videophy2_train_challenging/A_robotic_arm_pushes_a_metal_cube_off_a_steel_table;_the_cube_lands_precisely_on_a_marked_spot.mp4" type="video/mp4">
</video>

- **Scene**: A robotic arm pushes a metal cube off a steel table.

- **Physics Strengths**: The robotic arm moves the cube from one position to another. The cube maintains its shape and volume throughout the interaction.

- **Key Success**: Conservation of mass and gravity.

## Zero-Shot Inference

We first evaluate the model's ability to predict physical plausibility on the VideoPhy-2 evaluation set without any fine-tuning. We modified the prompt from the [VideoPhy-2 paper](https://arxiv.org/abs/2503.06800) to fit the Cosmos Reason 2 prompt format.

???+ code "Prompt for Scoring Physical Plausibility"

    ```yaml
    --8<-- "docs/recipes/post_training/reason2/physical-plausibility-check/assets/video_reward.yaml"
    ```

**Note**: Since Cosmos Reason 2 is fine-tuned from Qwen3-VL, we follow their prompt guidelines: The system prompt should be set to "You are a helpful assistant", and the model response does not have a `<answer>` tag (see the [Qwen3-VL GitHub repository](https://github.com/QwenLM/Qwen3-VL) for more details).

### Evaluation Metrics

We evaluate the model performance using two key metrics:

- **Accuracy**: The percentage of videos where predicted scores match ground truth scores (exact integer match)
- **Correlation**: The Pearson correlation between predicted and ground truth scores

### Setup

To run zero-shot inference, you need to clone both repositories and copy the necessary files:

1. Clone the Cosmos Reason 2 repository:
   ```bash
   git clone https://github.com/nvidia-cosmos/cosmos-reason2.git
   ```

2. Clone the cosmos-cookbook repository (this repository):
   ```bash
   git clone https://github.com/nvidia-cosmos/cosmos-cookbook.git
   ```

3. Copy the `video_critic` folder from the cosmos-cookbook repository to your Cosmos Reason 2 clone:
   ```bash
   # Adjust paths based on where you cloned the repositories
   cp -r cosmos-cookbook/scripts/examples/reason2/physical-plausibility-check/video_critic cosmos-reason2/examples/
   ```

4. Copy the prompt file from the cosmos-cookbook repository to your Cosmos Reason 2 repository (shown in the "Prompt for Scoring Physical Plausibility" section above):
   ```bash
   # Adjust paths based on where you cloned the repositories
   cp cosmos-cookbook/docs/recipes/post_training/reason2/physical-plausibility-check/assets/video_reward.yaml cosmos-reason2/prompts/video_reward.yaml
   ```

### Running Zero-Shot Inference

Run inference on the VideoPhy-2 test set using Cosmos Reason 2. From the Cosmos Reason 2 project root directory:

```bash
uv run examples/video_critic/inference_videophy2.py \
    --model nvidia/Cosmos-Reason2-8B \
    --output-dir results/videophy2_test \
    --dataset videophysics/videophy2_test \
    --split test \
    --input-file prompts/video_reward.yaml
```

**Arguments:**

- `--model`: Model name or path
- `--output-dir`: Output directory for JSON results
- `--dataset`: HuggingFace dataset name
- `--split`: Dataset split (the dataset only has a "test" split)
- `--input-file`: Path to prompt YAML file
- `--revision`: Optional model revision/branch

**Output:**

- Each video gets a JSON file containing:
  - `video_url`: Source video URL
  - `ground_truth`: Ground truth physics score
  - `output_text`: Model output text
  - `pred_score`: Parsed predicted score

### Computing Evaluation Metrics

After running inference, compute accuracy and correlation metrics from the inference results. From the Cosmos Reason 2 project root directory:

```bash
python3 examples/video_critic/compute_metrics.py results/videophy2_test
```

**Output:**

- Prints accuracy (exact integer match percentage), Pearson correlation, and number of samples
- Generates `summary.json` in the output directory with `accuracy`, `pearson_correlation`, and `num_samples` metrics

### Results

We compare Cosmos Reason 2 with Gemini-2.0-Flash-Exp (the baseline from the paper) and Cosmos Reason 1. Even without fine-tuning, Cosmos Reason 2 demonstrates better performance than both baselines on both accuracy and correlation metrics.

<img src="assets/correlation_bar_graph.png" alt="Correlation comparison between Gemini-2.0-Flash-Exp, Cosmos Reason 1, and Cosmos Reason 2" style="max-width: 800px; width: 100%;">

## Supervised Fine-Tuning (SFT)

Having demonstrated that Cosmos Reason 2 can predict physical plausibility, we now apply supervised fine-tuning (SFT) using the VideoPhy-2 training set to further improve the model's performance.

### Training Data Format

The fine-tuning process uses the following data structure:

- **Input**: Video + language instruction (from the evaluation prompt)
- **Output**: Physical plausibility score (1-5 scale)

### Data Pre-processing

Before fine-tuning, prepare the VideoPhy-2 training dataset. From the Cosmos Reason 2 project root directory:

```bash
uv run examples/video_critic/download_videophy2_train.py \
    --dataset videophysics/videophy2_train \
    --split train \
    --output data/videophy2_train \
    --prompt_path prompts/video_reward.yaml
```

**Arguments:**

- `--output`: Output directory for the prepared dataset (required)
- `--dataset`: HuggingFace dataset name
- `--split`: Dataset split to download (the dataset only has a "train" split)
- `--prompt_path`: Path to prompt YAML file

The script will:

1. Download videos from URLs in the dataset
2. Create conversations using the prompt template
3. Save the dataset in HuggingFace format for training

### Training Configuration

We use the following configuration optimized for 8 GPUs:

???+ code "Training Configuration"

    ```toml
    --8<-- "scripts/examples/reason2/physical-plausibility-check/video_critic/configs/videophy2_sft.toml"
    ```

> **Note**: Set `dp_shard_size` to the number of GPUs you are using. We tested on A100/H100 GPUs where the model fits in the memory of a GPU, so we only use data parallelism. If you use GPUs with less memory, you may increase `tp_size` to enable tensor parallelism.

We performed hyperparameter search across different learning rates (1e-5, 2e-7, and 1e-6) and found that 1e-6 performs best overall, which is used in the configuration file above.

### Running Training

Fine-tune the model on the prepared dataset. From the Cosmos Reason 2 project root directory:

```bash
cd examples/cosmos_rl
uv run cosmos-rl --config ../video_critic/configs/videophy2_sft.toml scripts/hf_sft.py
```

**Output:**

- Checkpoints saved to `outputs/videophy2_sft/{timestamp}/safetensors/step_*`
- Training logs in WandB (if configured)

### Evaluating Fine-tuned Checkpoints

After fine-tuning, evaluate checkpoints by running inference and then computing metrics:

1. Run inference on the test set using a checkpoint. Replace `{timestamp}` and `{number}` with the actual timestamp and step number of the checkpoint.

   ```bash
   uv run examples/video_critic/inference_videophy2.py \
       --model outputs/videophy2_sft/{timestamp}/safetensors/step_{number} \
       --output-dir results/videophy2_test_sft_step_{number} \
       --dataset videophysics/videophy2_test \
       --split test \
       --input-file prompts/video_reward.yaml
   ```

2. Compute metrics for the checkpoint. From the Cosmos Reason 2 project root directory:

   ```bash
   python3 examples/video_critic/compute_metrics.py results/videophy2_test_sft_step_{number}
   ```

This will generate a `summary.json` file with accuracy, correlation, and sample count metrics, allowing you to compare different checkpoints and select the best one.

### Results

After fine-tuning, we evaluate the model on the VideoPhy-2 evaluation set using the same metrics. The results demonstrate significant performance improvements. VideoPhy-AutoEval is the baseline model from the VideoPhy-2 paper.

<img src="assets/videophy2_accuracy_vs_step_summary.png" alt="SFT Accuracy over Training Steps" style="width: 100%; max-width: 600px; display: block; margin-bottom: 36px;">
<img src="assets/videophy2_correlation_vs_step_summary.png" alt="SFT Correlation over Training Steps" style="width: 100%; max-width: 600px; display: block;">

**Observations:**

- **Accuracy Trajectory**: Cosmos Reason 2 achieves its best accuracy of 0.401 at step 40, outperforming Cosmos Reason 1's peak of 0.395 at step 60.
However, the accuracy of both Reason 1 and Reason 2 declines after step 40, showing signs of overfitting.

- **Correlation Performance**: Cosmos Reason 2 consistently outperforms Cosmos Reason 1 in correlation at all training steps, achieving its best correlation of 0.419 at step 80, compared to Reason 1's peak of 0.395 at step 60. Additionally, Reason 2 reaches a correlation score similar to VideoPhy-AutoEval at step 20, which requires a smaller number of training steps compared to Reason 1.

- **Improvement Trend**: The improvement trend is more consistent across both metrics compared to Reason 1.

#### Example: Model Prediction Before and After Fine-tuning

<video controls width="480">
  <source src="https://videophysics2testvideos.s3.us-east-2.amazonaws.com/hunyuan_xdit/A_speeding_car_crashes_into_a_brick_wall,_crumpling_the_front_end_and_stopping_abruptly.mp4" type="video/mp4">
</video>

- **Model prediction (before fine-tuning):** 3  
- **Model prediction (after fine-tuning):** 2
- **Ground truth:** 2 (poor adherence to physical laws)

<video controls width="480">
  <source src="https://videophysics2testvideos.s3.us-east-2.amazonaws.com/videocrafter_videophy2_hard/A_speeding_car_crashes_into_a_brick_wall__crumpling_the_front_end_and_stopping_abruptly_.mp4" type="video/mp4">
</video>

- **Model prediction (before fine-tuning):** 3  
- **Model prediction (after fine-tuning):** 2  
- **Ground truth:** 1 (no adherence to physical laws; completely implausible)

After fine-tuning, Cosmos Reason 2 correctly identifies the low physical plausibility of the videos, matching human judgment. In contrast, the base model, prior to fine-tuning, overestimated plausibility.

## Conclusion

Supervised fine-tuning on the VideoPhy-2 dataset significantly improves physical plausibility prediction for Cosmos Reason 2, progressing from zero-shot to best SFT performance. Key insights:

- **Strong Baseline**: Cosmos Reason 2 demonstrates better zero-shot performance compared to Cosmos Reason 1, achieving 25% higher accuracy and 12% higher correlation before any fine-tuning, indicating improved physics reasoning capabilities in the base model.

- **Metric-Specific Optimization**: Different metrics peak at different training steps (accuracy at step 40, correlation at step 80), suggesting that practitioners should select checkpoints based on their primary evaluation metric or use ensemble approaches.

- **Consistent Improvements**: Fine-tuning delivers measurable gains in both metrics, with correlation showing more sustained improvement (reaching 0.419) compared to accuracy, which peaks earlier and then stabilizes.

- **Flexibility**: This methodology can be adapted to other video quality assessment tasks by substituting the datasets and defining appropriate metrics. The better zero-shot performance of Cosmos Reason 2 also suggests it may require less fine-tuning data to achieve target performance levels.
