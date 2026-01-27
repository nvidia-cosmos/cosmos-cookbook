# [Domain/Task] with [Cosmos Model]

> **Authors:** [Name](LinkedIn URL) • [Name](LinkedIn URL)
> **Organization:** [Your Organization]

## Overview

| **Model**      | **Workload**  | **Use Case**        | **Category**                               | **Date**    |
|----------------|---------------|---------------------|--------------------------------------------|-------------|
| Cosmos [Model] | Post-training | [Brief description] | Robotics / Autonomous Vehicles / Vision AI | Jan 1, 2026 |

[Introduce your company/organization, project, and application domain. Describe the specific challenge or pain point you're addressing, why post-training is needed for your use case, and your approach (e.g., LoRA, full fine-tuning, RL).]

- [Setup and System Requirements](setup.md) *(optional)*

> **Recommended Structure:** For post-training contributions, we recommend organizing your content to include the following elements:
>
> 1. **Problem Statement & Benchmarks** - Define the task and target benchmarks
> 2. **Zero-shot Evaluation** - Establish baseline performance and identify gaps
> 3. **Data Sourcing & Curation** - Describe targeted data collection and high-quality annotation
> 4. **Post-Training** - Detail your approach (SFT, RL, etc.) and configuration
> 5. **Re-evaluation** - Compare before/after results to demonstrate improvement
>
> This structure helps readers understand the full journey from problem to solution. Feel free to adapt this format to best suit your use case.

## Problem

[Describe the specific problem or task you're addressing. What are the target benchmarks? What makes this challenging for the base model?]

[Include initial zero-shot evaluation results if available to establish a baseline. This helps readers understand the starting point and the gaps you identified.]

## Methodology

[Explain your training approach (e.g., Supervised Fine-Tuning, Reinforcement Learning) and why it suits this use case. Include key benefits and trade-offs.]

### Training Configuration

[Key parameters, model selection, and configuration choices]

- **Parameter 1**: Value and rationale
- **Parameter 2**: Value and rationale

## Data Preparation

[Describe training data sources, preprocessing steps, and expected format. If possible, explain how your data addresses the specific gaps identified in the baseline evaluation.]

[Consider including:

- Data sources and their relevance to the domain
- Data size and statistics
- Quality control and annotation procedures
- Any targeted curation to address model weaknesses]

```bash
# Data preparation commands
```

## Training

[Step-by-step instructions with commands]

```[language]
# Configuration file
```

```bash
# Training commands
```

[Include monitoring, checkpointing, and debugging information]

## Results

[Describe evaluation methodology, metrics, and qualitative analysis. Show before/after comparisons demonstrating impact.]

[**Recommended:** Include a comparison table showing baseline (zero-shot) vs. post-trained performance to clearly demonstrate improvement. Explain how specific gaps identified earlier were addressed.]

**Example comparison:**

| Metric | Baseline | Post-Trained | Improvement |
|--------|----------|--------------|-------------|
| [Metric 1] | [Score] | [Score] | [+X%] |
| [Metric 2] | [Score] | [Score] | [+X%] |

## Conclusion

[Key achievements and lessons learned]

**Next Steps:**

- Suggestion 1
- Suggestion 2

## Resources

- **[Resource Name](URL)** - Description
- **[Resource Name](URL)** - Description
