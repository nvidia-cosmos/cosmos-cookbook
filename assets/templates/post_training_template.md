# [Domain/Task] with [Cosmos Model]

> **Authors:** [Name](LinkedIn URL) • [Name](LinkedIn URL)
> **Organization:** [Your Organization]

## Overview

| **Model** | **Workload** | **Use Case** |
|-----------|--------------|--------------|
| Cosmos [Model] | Post-training | [Brief description] |

> **Recommended Structure:** We recommend following this comprehensive workflow for the post-training recipe:
>
> - [Setup and System Requirements](setup.md) *(optional)*
>
> 1. **Benchmark Selection** - Pick and document your target benchmark
> 2. **Zero-shot Evaluation** - Establish baseline performance using VLMEvalKit
> 3. **Problem Analysis** - Identify gaps and define improvement goals
> 4. **Data Curation** - Source, annotate, and prepare high-quality training data
> 5. **Post-Training** - Fine-tune the model using Cosmos Cookbook examples
> 6. **Re-evaluation** - Compare before/after results to demonstrate improvement

### Overview of the Capability

[Describe the capability, the challenge, why post-training is needed, and your approach (e.g., LoRA, full fine-tuning, RL).]

**Example:** Robots do not know the left and right relationship of objects in a warehouse. It's an important capability because we want robots to follow the language instruction of left and right to manipulate objects.

## Step 1: Benchmark Selection

### Pick Your Benchmark

[Describe the benchmark you've selected and why it's relevant for your use case. The benchmark can be existing or self-built, and it can be internal or external. When adopting a self-built benchmark, please actively seek review and feedback from other domain experts to avoid the pitfall of making the mission too easy to accomplish. Our goal is to build something meaningful for the Cosmos models in the long run.]

**Example Benchmarks:**

We source N images showing cluttered objects and annotate QA to benchmark the capability. The question is in the form of MCQ (multiple choice questions). Here is an example question:

```
User prompt: What is the position of orange relative to apple in the image?
A. left.
B. right.
```

## Step 2: Zero-shot Evaluation

### Run Baseline Evaluation

[Describe which zero-shot checkpoint used; how the evaluation was done. If using customized scripts, preferably also include the script into the report (source code saved in a capability directory).]

### Baseline Results

| Metric | Baseline Score | Notes |
|--------|---------------|-------|
| [Metric 1] | [Score] | [What does this tell us?] |
| [Metric 2] | [Score] | [What does this tell us?] |

[Document the key findings from your zero-shot evaluation. What are the specific gaps or weaknesses you identified?]

## Step 3: Problem Analysis

[Describe the specific problem or task you're addressing.]

### Key Gaps Identified

- **Gap 1:** [Description]
- **Gap 2:** [Description]
- **Gap 3:** [Description]

### Success Criteria

- **Target 1:** [Goal]
- **Target 2:** [Goal]

## Step 4: Data Curation

### Domain Data Sourcing

[Describe the main source of the data for the post-training, and major challenges encountered.]

### Data Curation

[Describe the data curation pipeline, which tool used to set that up; and end results of the curation.]

Provide information about the final training dataset such as:

- **Input type:** [image/video]
- **Resolution:** [e.g., 1920x1080]
- **Encoding:** [e.g., H.264, JPEG]
- **Length:** [e.g., 10 seconds average for videos, N images]
- **FPS:** [for videos]
- **Caption/Annotation:** [Description of annotation format]
- **Sample contents:** [Describe representative samples or include visualizations]

## Step 5: Post-Training

### Methodology

- **Approach:** [SFT, RL, LoRA, full fine-tuning, etc.]
- **Rationale:** [Why this method; what did you instruct the model to learn?]

### Training Configuration

- **Learning Rate:** [Value]
- **Batch Size:** [Value]
- **Epochs/Steps:** [Value]
- **Other Parameters:** [List]

```toml
# Include configuration file
```

### Data Loader

```python
# Include data loader script
```

### Command

```bash
# Include training commands
```

### Monitoring & Debugging

**Training Logs:**

```
[Sample of training logs]
```

**Common Issues:**

- [Issue 1 and solution]
- [Issue 2 and solution]

**Troubleshooting:**

- [Tip 1]
- [Tip 2]

## Step 6: Re-evaluation

### Results Comparison

| Metric | Baseline | Post-Trained | Improvement |
|--------|----------|--------------|-------------|
| [Metric 1] | [Score] | [Score] | [+X%] |
| [Metric 2] | [Score] | [Score] | [+X%] |

### Qualitative Analysis

**Example:**

- **Input:** [Description]
- **Baseline Output:** [Result]
- **Post-Trained Output:** [Improved Result]
- **Analysis:** [Why it improved]

**Example:** By fine-tuning the curated dataset, we improve the accuracy from 55% to 98% showing the curated data improves capability for robots to understand left and right!

## Conclusion

### Key Achievements

- **Achievement 1:** [Improvement]
- **Achievement 2:** [Insight]

### Lessons Learned

- **Lesson 1:** [What worked]
- **Lesson 2:** [What to improve]

### Next Steps

- Extend to related benchmarks
- Scale up training data
- Explore additional techniques

## References

- **[Resource Name](URL)** - Description
- **[Resource Name](URL)** - Description
