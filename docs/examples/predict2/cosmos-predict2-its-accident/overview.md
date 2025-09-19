# Overview

## The Challenge: Rare Event Data in ITS

In Intelligent Transportation Systems (ITS), collecting real-world data for rare events like traffic accidents, jaywalking, or blocked intersections faces significant challenges:

- **Privacy concerns**: Recording and using real accident footage raises ethical and legal issues
- **Infrequent occurrence**: Critical safety events are rare by nature, making data collection expensive and time-consuming
- **High annotation costs**: Expert annotation of traffic incidents requires specialized knowledge
- **Safety risks**: Staging real accidents for data collection is dangerous and impractical

Synthetic data generation (SDG) offers a practical way to augment existing datasets, enabling teams to create targeted scenarios at scale while maintaining control over scenario parameters and data quality.

## Our Approach: LoRA-Based Domain Adaptation

This case study documents a detailed post-training workflow using Cosmos-Predict2 Video2World with **Low-Rank Adaptation (LoRA)**, focusing on enhancing model capabilities for generating traffic anomaly videos from a fixed CCTV perspective. Rather than fine-tuning the entire model, we employ LoRA to efficiently adapt the pretrained foundation model for ITS-specific requirements.

### Why LoRA for ITS Applications?

LoRA (Low-Rank Adaptation) is particularly well-suited for the ITS domain adaptation challenge for several compelling reasons:

#### 1. **Critical Advantage: Data Efficiency for Rare Events**

**The core challenge in ITS**: Real accident data is inherently scarce and difficult to obtain. Unlike general video datasets with millions of samples, traffic accident datasets typically contain only hundreds to thousands of examples. This data scarcity makes LoRA the optimal choice:

- **Effective with Limited Data**: LoRA can achieve meaningful adaptation with as few as 1,000-2,000 training samples
- **Reduced Overfitting Risk**: Fewer parameters (45M vs 2B) means less tendency to memorize limited training data
- **Better Generalization**: The constrained parameter space forces the model to learn generalizable patterns rather than specific examples
- **Leverages Pre-training**: LoRA builds upon the base model's existing knowledge, requiring only minimal accident-specific data to adapt

In our case study, with very limited clips, LoRA enabled successful adaptation where full fine-tuning would likely fail or severely overfit.

#### 2. **Parameter Efficiency**

- **Minimal Storage**: LoRA adds only ~45M trainable parameters to a 2B parameter model (≈2% increase)
- **Quick Deployment**: LoRA adapters are small (10-100MB) compared to full model checkpoints (5-50GB)
- **Multiple Domains**: Different traffic scenarios (highways, intersections, parking lots) can each have their own LoRA adapter

#### 3. **Resource Optimization**

- **Reduced Training Time**: 1-2 hours for 2B model vs 2-4 hours for full fine-tuning
- **Lower GPU Memory**: 20GB for LoRA vs 50GB for full model training
- **Faster Iteration**: Enables rapid experimentation with different training configurations

#### 4. **Preservation of Base Capabilities**

- **No Catastrophic Forgetting**: Base model's general video generation capabilities remain intact
- **Additive Learning**: ITS-specific knowledge is added without degrading general performance
- **Fallback Option**: Can disable LoRA to access original model behavior when needed

#### 5. **Domain-Specific Advantages for ITS**

**Targeted Adaptation**: LoRA excels at learning specific visual patterns critical for traffic scenarios:

- Fixed camera perspectives (CCTV viewpoints)
- Consistent vehicle physics and collision dynamics
- Traffic flow patterns and intersection behavior
- Lighting conditions typical of street surveillance

**Modular Architecture**: Different LoRA adapters can be trained for:

- Different camera angles (intersection view, highway view, parking lot view)
- Various weather conditions (rain, fog, snow)
- Time of day variations (day, night, dawn/dusk)
- Specific incident types (collisions, near-misses, violations)

**Production Flexibility**: In deployment, organizations can:

- Switch between adapters based on the required scenario
- Update specific capabilities without retraining the entire model

## Technical Implementation

Our approach targets specific model components for adaptation:

### LoRA Configuration

Based on the [LoRA paper (Hu et al., 2021)](https://arxiv.org/abs/2106.09685), our configuration includes:

- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `output_proj`, `mlp.layer1`, `mlp.layer2`
- **Rank**: 16 (determines the dimensionality of the low-rank decomposition - higher rank allows more expressiveness but increases parameters)
- **Alpha**: 16 (scaling hyperparameter that controls the magnitude of LoRA updates - typically set equal to rank for balanced learning)
- **Training Data**: 1:1 mixture of normal traffic scenes and incident scenarios

This configuration focuses on attention mechanisms and feed-forward layers, which are crucial for:

- Understanding spatial relationships between vehicles
- Capturing temporal dynamics of collisions
- Maintaining consistent camera perspective
- Generating physically plausible motion

## Workflow Overview

The complete pipeline consists of four main stages:

1. **Data Curation**: Collecting and processing CCTV footage and incident videos
   - Resolution standardization to 1280x720
   - Scene splitting and captioning
   - Quality filtering and artifact removal

2. **LoRA Post-Training**: Efficient domain adaptation
   - 6k-10k training iterations
   - Mixed dataset of normal and anomaly scenes
   - Continuous validation on held-out scenarios

3. **Inference**: Generating synthetic accident scenarios
   - LoRA-enhanced generation with domain-specific knowledge
   - Prompt-guided scenario control
   - Batch processing for dataset creation

4. **Evaluation**: Assessing generation quality
   - Physical realism metrics
   - Scenario diversity analysis
   - Domain expert validation

## Expected Outcomes

By using LoRA-based post-training, we achieve:

### Quality Improvements

- **Enhanced Physical Realism**: More accurate collision dynamics and vehicle behavior
- **Consistent Perspective**: Maintains fixed CCTV camera viewpoint throughout generation
- **Reduced Artifacts**: Fewer unrealistic elements like floating vehicles or impossible physics

### Data Efficiency Benefits

- **Successful Training with Minimal Data**: Achieved domain adaptation with only ~1,000 accident examples
- **No Data Waste**: Every precious accident clip contributes meaningfully to the model
- **Synthetic Data Amplification**: The adapted model can now generate unlimited variations of accidents, effectively solving the data scarcity problem

### Operational Benefits

- **Rapid Adaptation**: New scenarios can be learned in hours rather than days
- **Cost Efficiency**: Reduced computational requirements enable broader experimentation
- **Scalable Deployment**: Multiple domain-specific models can coexist efficiently

### Research Advantages

- **Controlled Experimentation**: Isolate and study specific adaptation strategies
- **Reproducible Results**: Smaller models and faster training enable better reproducibility
- **Iterative Improvement**: Quick feedback loops accelerate research progress

## Use Cases Enabled

This LoRA-adapted model enables several critical ITS applications:

1. **Safety System Training**: Generate diverse accident scenarios for computer vision model training
2. **Traffic Simulation**: Create realistic traffic flow videos for urban planning
3. **Incident Analysis**: Reconstruct and visualize potential accident scenarios
4. **Emergency Response Planning**: Simulate various incident types for preparedness training
5. **Infrastructure Assessment**: Evaluate intersection designs with synthetic traffic scenarios

## Conclusion

The combination of Cosmos-Predict2's powerful video generation capabilities with LoRA's efficient adaptation mechanism provides an ideal solution for ITS-specific synthetic data generation. **Most critically, LoRA enables successful domain adaptation despite the severe scarcity of real accident data** - a fundamental constraint in traffic safety applications.

Where traditional fine-tuning would require tens of thousands of examples and risk catastrophic overfitting with limited data, LoRA achieved meaningful adaptation with just over 1,000 incident clips. This data efficiency, combined with reduced computational requirements and deployment flexibility, makes LoRA not just a good choice but arguably the only viable approach for adapting large video models to rare-event domains like traffic accidents.

The result is a system capable of generating unlimited high-quality, physically realistic traffic incident videos from minimal real examples - effectively transforming data scarcity from a blocking constraint into a solved problem. This breakthrough can significantly enhance safety system development, emergency response training, and urban planning initiatives worldwide.
