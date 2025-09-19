# Post-Training

Post-training is a critical step for adapting foundation models to domain-specific tasks and improving performance beyond the capabilities of general-purpose checkpoints. While zero-shot models provide strong baselines, many applications require improved accuracy, physical realism, or domain alignment that can only be achieved through targeted fine-tuning.

In a typical post-training workflow, several variables can be adjusted to optimize model outcomes:

- **Data Mixture**: The composition of the training dataset plays a central role in post-training effectiveness. Mixing data from multiple sources—each with different levels of quality, camera perspective, or content relevance—requires careful balancing. Often, a two-stage approach is used: broader exposure during early training followed by focused fine-tuning on high-quality subsets.

- **Training Strategy**: The choice of training strategy depends on data availability, task complexity, and computational budget. Three supported approaches are:

    - **Full Post-Training**: Updates all model parameters. Recommended when large amounts of training data are available and the goal requires significant adaptation or full control over model behavior.

    - **LoRA Post-Training (for Diffusion models only)**: A parameter-efficient fine-tuning method that requires fewer resources and enables faster iteration. Ideal when data is limited but sufficient, and when the learning objectives are relatively simple or when maintaining base model capabilities is important.

    - **Reinforcement Learning (for VLMs only)**: Applicable when data is scarce but of high quality, and especially useful for learning complex reasoning behavior. RL is not currently supported for diffusion-based models such as Cosmos-Predict or Cosmos-Transfer.

- **Hyperparameter Tuning**: Fine-tuning hyperparameters such as learning rate, batch size, and optimizer settings is essential to achieve convergence without overfitting. Small-scale experiments or overfitting tests on a few samples are often used to validate pipeline correctness and identify promising configurations before scaling up.

- **Data Augmentation and Filtering**: In some cases, synthetic data augmentation or filtering pipelines are introduced to increase training signal density or improve realism by removing noisy samples and aligning content with the target use case.

The post-training process is iterative, with evaluation playing a key role at each stage to verify quality improvements, generalization ability, and alignment with intended deployment conditions.
