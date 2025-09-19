# Inference with Post-Trained ITS Model

This guide explains how to run inference with the LoRA post-trained Cosmos-Predict2 model for generating ITS accident scenarios.

## Overview

After post-training the Cosmos-Predict2 Video2World model on ITS-specific data using LoRA (Low-Rank Adaptation), we can perform efficient inference to generate realistic traffic incident videos. The inference process leverages the base model's general capabilities while applying the domain-specific adaptations learned during post-training.

## Prerequisites

1. **Post-trained checkpoint**: A LoRA checkpoint from the post-training process (e.g., `iter_000001000.pt`)
2. **Input image**: A CCTV traffic camera frame as the starting point (1280x720 recommended)
3. **Environment setup**: Properly configured Cosmos-Predict2 environment with required dependencies

## Inference Command Structure

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
    --input_path "benchmark/frames_1280x704/intersection_view.jpg" \
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

## Key Parameters

### Model Configuration

| Parameter | Description | ITS Recommendation |
|-----------|-------------|-------------------|
| `--model_size` | Model size (2B or 14B) | 2B for faster inference, 14B for higher quality |
| `--dit_path` | Path to LoRA checkpoint | Use latest checkpoint from post-training |
| `--num_gpus` | Number of GPUs for parallel inference | 8 for optimal speed |

### LoRA-Specific Parameters

| Parameter | Description | Required Value |
|-----------|-------------|----------------|
| `--use_lora` | Enable LoRA inference mode | **Must be set** |
| `--lora_rank` | Rank of LoRA adaptation | 16 (match training) |
| `--lora_alpha` | LoRA scaling parameter | 16 (match training) |
| `--lora_target_modules` | Target modules for LoRA | "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2" |

### Input/Output Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--input_path` | Path to input CCTV frame | "frames/intersection_01.jpg" |
| `--prompt` | Accident scenario description | See prompt examples below |
| `--save_path` | Output video path | "output/accident_01.mp4" |

### Optimization Parameters

| Parameter | Description | Recommendation |
|-----------|-------------|----------------|
| `--offload_guardrail` | Move guardrail to CPU | Use to save GPU memory |
| `--offload_prompt_refiner` | Move prompt refiner to CPU | Use to save GPU memory |
| `--guidance` | Guidance scale | 7 (default) |
| `--seed` | Random seed for reproducibility | Set for consistent results |

## Prompt Engineering for ITS Scenarios

Effective prompts are crucial for generating realistic traffic incidents. Follow these guidelines:

### Structure

1. **Start with camera perspective**: "A static traffic CCTV camera..."
2. **Describe the scene**: Location, weather, traffic conditions
3. **Detail the incident**: Vehicle types, movements, collision dynamics
4. **Include aftermath**: Post-collision behavior

### Example Prompts

#### Intersection Collision

```
A static traffic CCTV camera captures a busy four-way intersection during daytime.
A red sedan runs a red light and enters the intersection at high speed. From the
right, a white SUV proceeds legally and collides with the sedan's passenger side.
The impact causes the sedan to spin and both vehicles come to rest blocking traffic.
```

#### Rear-End Collision

```
A traffic CCTV camera shows a multi-lane highway with moderate traffic flow. A silver
pickup truck suddenly brakes hard, and a black sedan following too closely crashes
into its rear bumper. The sedan's front crumples while the pickup is pushed forward
several meters.
```

#### Pedestrian Near-Miss

```
A static traffic camera captures an urban crosswalk where pedestrians are crossing.
A delivery van takes a sharp right turn without yielding, causing pedestrians to
jump back quickly. The van swerves to avoid contact and continues through the
intersection.
```

## Batch Processing

For processing multiple scenarios, create a JSON file with input-output pairs:

```json
[
    {
        "input_video": "frames/intersection_01.jpg",
        "prompt": "A traffic CCTV camera captures a T-bone collision...",
        "output_video": "output/collision_01.mp4"
    },
    {
        "input_video": "frames/highway_02.jpg",
        "prompt": "A static camera shows a multi-vehicle pileup...",
        "output_video": "output/pileup_02.mp4"
    }
]
```

Run batch inference:

```bash
torchrun --nproc_per_node=${NUM_GPUS} examples/video2world_lora.py \
    --model_size 2B \
    --dit_path "checkpoints/posttraining/video2world/2b_metropolis/checkpoints/model/iter_000001000.pt" \
    --batch_input_json "batch_scenarios.json" \
    --num_gpus ${NUM_GPUS} \
    --use_lora \
    --lora_rank 16 \
    --lora_alpha 16 \
    --lora_target_modules "q_proj,k_proj,v_proj,output_proj,mlp.layer1,mlp.layer2"
```

## Performance Considerations

### Resolution and FPS

- Default: 720p at 16 FPS
- Supported: 480p (10/16 FPS) or 720p (10/16 FPS)
- Higher resolution/FPS requires more memory and computation
