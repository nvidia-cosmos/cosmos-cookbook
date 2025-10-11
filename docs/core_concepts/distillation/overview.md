# Distillation

Model distillation is a powerful technique for creating efficient "Student" models that maintain the quality and capabilities of "Teacher" models while dramatically reducing computational requirements. This process enables deployment of high-performance models in resource-constrained environments while preserving output quality and diversity.

There are two primary distillation approaches widely used today:

- **Size Distillation**: Compresses a large model into a smaller architecture while maintaining performance.

- **Step Distillation**: Reduces the number of inference steps required.

This guide concentrates on step distillation–where you train Student models to achieve comparable results in a single diffusion step–compared to Teacher models, which require multiple steps (e.g. 36 diffusion steps in Cosmos-Transfer1).

In a typical distillation workflow, several variables can be adjusted to optimize model outcomes:

- **Data Mixture**: The composition of the distillation training dataset is critical for distillation effectiveness. Both sufficient scale and high quality of video data in the target domain are essential for achieving optimal distilled model performance. The dataset should be diverse, representative of the intended use cases, and free from artifacts that could degrade the Student model capabilities.

- **Training Strategy**: The choice of distillation training strategy depends on data availability, task complexity, and computational budget. Distillation typically employs multi-stage curriculum training, with each stage utilizing different algorithms optimized for specific learning objectives. The following are supported approaches:

    - **Knowledge Distillation (KD)**: A lightweight yet effective distillation method that aligns the Student model with the Teacher model using regression loss on model outputs. This approach requires generating a comprehensive synthetic dataset of input-output pairs using the Teacher model, making it ideal as an initial training stage to establish baseline alignment between models.

    - **Improved Distribution Matching Distillation (DMD2)**: A distribution-matching approach that combines adversarial training with variational score distillation to preserve output diversity and quality. This method requires a diverse dataset of ground-truth videos and involves training multiple models simultaneously (Student, Teacher, fake score network, and discriminator), resulting in higher memory requirements.

- **Hyperparameter Tuning**: Careful optimization of hyperparameters such as learning rate and batch size is essential for achieving optimal distillation results. The relative importance of specific hyperparameters varies by distillation strategy. Systematic small-scale experimentation is recommended to identify optimal configurations before full-scale training.

The distillation process is iterative, with evaluation playing a key role at each stage to verify quality improvements, generalization ability, and alignment with intended deployment conditions.
