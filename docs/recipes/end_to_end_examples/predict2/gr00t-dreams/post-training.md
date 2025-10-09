# Isaac GR00T-Dreams for Synthetic Trajectory Data Generation

> **Authors:** NVIDIA Isaac Team
>
> **Organization:** NVIDIA

## Overview

| **Model** | **Workload** | **Use Case** |
|-----------|--------------|--------------|
| Cosmos Predict2 | Post-training | Synthetic trajectory data generation for humanoid robots |

Isaac GR00T-Dreams leverages **Cosmos Predict2** to generate synthetic trajectory data for teaching humanoid robots new actions in novel environments. By using world foundation models, a small team can create training data that would otherwise require thousands of demonstrators.

## Key Features

- **Scalable Generation**: Produce large-scale synthetic trajectories from minimal human demonstrations
- **Environment Generalization**: Adapt to new environments without extensive retraining
- **Diverse Behaviors**: Cover wide-ranging scenarios and edge cases
- **Cost-Effective**: Dramatically reduce manual data collection effort

## How It Works

1. **Start with Demonstrations**: Use a small set of human demonstration videos
2. **Generate Variations**: Apply Cosmos Predict2 to create synthetic trajectories with environmental variations
3. **Scale Training Data**: Produce thousands of variations from each demonstration
4. **Train Policies**: Use synthetic data to train robust robot control policies

## Applications

- Humanoid locomotion (walking, running, navigation)
- Object manipulation and interaction
- Multi-terrain adaptation
- Rare scenario and edge case coverage

## Resources

- **[GR00T-Dreams GitHub](https://github.com/nvidia/gr00t-dreams)** - Source code and documentation
- **[Technical Blog](https://developer.nvidia.com/blog/enhance-robot-learning-with-synthetic-trajectory-data-generated-by-world-foundation-models/)** - In-depth overview and results
- **[NVIDIA Isaac Platform](https://developer.nvidia.com/isaac)** - Robotics development platform
- **[Cosmos Predict2](https://github.com/nvidia-cosmos/cosmos-predict2)** - World foundation model
- **[Isaac GR00T](https://developer.nvidia.com/isaac/gr00t)** - Humanoid robot foundation model

## Related Recipes

- [Isaac GR00T-Mimic](../../inference/transfer1/gr00t-mimic/inference.md) - Synthetic manipulation motion generation
