# Distilling Cosmos Predict 2.5

> **Authors:** [Qianli Ma](https://qianlim.github.io/)
> **Organization:** NVIDIA

## Overview

The distillation process compresses a pre-trained "teacher" video diffusion model, which requires many inference steps, into a "student" model capable of few-step inference. Both models typically share the same architecture. A key benefit is that the teacher's classifier-free guidance (CFG) knowledge is distilled into the student, eliminating the need for CFG during the student's inference and providing an additional 2x speedup.

This cookbook presents a case study of how we use the [DMD2 algorithm](https://arxiv.org/abs/2405.14867) to distill the Cosmos Predict2.5 Video2World model into a 4-step student model. The reference code can be found [here](https://github.com/nvidia-cosmos/cosmos-predict2.5/tree/main/cosmos_predict2/_src/predict2/distill).

During the distillation training, an auxiliary critic network (often called the 'fake score net' in literature and code) is trained alongside the student. The training process alternates between updating the student and the critic networks.
The process involves the following steps:

- Initialization: Load the pre-trained teacher network. Initialize the student and critic networks using the teacher network's weights.
- Optional Supervised Warm-up: Generate a synthetic dataset (usually thousands of input-output pairs) using the teacher model. This data can be used to perform a supervised warm-up training for the student model. While common in DMD2-like methods, we empirically found this step unnecessary when distilling a 4-step Text/Video2World model.
- Alternating Training: Alternate between $K$ critic steps and $1$ student step. We set $K=4$. Both the student and critic training steps include their respective loss functions. Note that we observed no noticeable improvement from adding the GAN loss described in the DMD2 paper, so it is omitted here for simplicity.

## Difference between Distillation and Regular Model Training

Below are the key code differences compared to a standard Cosmos video model:

- Distillation training uses a dedicated trainer and checkpointer (see `cosmos_predict2/_src/predict2/distill/checkpointer/` and `cosmos_predict2/_src/predict2/distill/trainer/`) to handle saving and loading both the student and critic networks.
- The training step (see `cosmos_predict2/_src/predict2/distill/models/`) alternates between student and critic updates. The student loss also differs from the standard diffusion / flow-matching loss: we construct a distribution-matching objective so that samples from the student follow the same distribution as the teacher.
- For the math formulation, we use TrigFlow as introduced in the [sCM paper](https://arxiv.org/abs/2410.11081) as a shared parameterization between DMD2 and consistency distillation (coming soon). This can be converted to and from both EDM and RectifiedFlow, and is compatible with teacher models trained under either formulation.

The following aspects remain the same as standard Cosmos model training:

- Data loading pipeline.
- Conditioning via the `Conditioner` object, including text embeddings and first-few-frame conditioning in the Video2World setting.
- Student and critic architectures, which typically mirror and are initialized from the teacher network.

## A Quick Peek into the Code

To understand how to add DMD2 distillation support to your custom Cosmos model, here’s a quick peek into the code using the Predict2.5 Video2World model as an example [[code](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/cosmos_predict2/_src/predict2/distill/models/video2world_model_distill_dmd2.py)].

```python
class Video2WorldModelDistillDMD2TrigFlow(DistillationCoreMixin, TrigFlowMixin, Video2WorldModel):
    ...
```

The distillation model inherits common distillation-related codes from the `DistillationCoreMixin`. The `TrigFlowMixin` provides handy training-time timestep sampling functions since we use Trigflow as a unified parameterization for both distillation methods. Then the model then inherits the teacher model class -- in this case the Predict2.5 `Video2WorldModel`, to reuse most of its functions including tokenizer, data handling, conditioner, etc. Note that the order of inheritance matters here.

The key of implementation is to rewrite the training step. The high-level training_step that alternates student and critic phases is in `DistillationCoreMixin`. For DMD2, implement two methods in your model:

Student phase (`training_step_generator`):

- Freeze critic (and discriminator if enabled); unfreeze student.
- Sample time and noise; generate few-step student samples from noise.
- Re-noise the student-generated samples to the sampled time, then feed this re-noised state to the teacher twice (cond/uncond) to form the CFG target; also feed the same re-noised state to the critic if enabled.
- Compute DMD2 losses from teacher and critic predictions; backprop into the student only. Optionally include GAN terms if configured.

Critic phase (`training_step_critic`):

- Freeze student; unfreeze critic (and discriminator if enabled).
- Generate student samples via a short backward simulation (few reverse steps); re-noise to the sampled time.
- Train the critic on these student samples to fit the denoising target; if a discriminator head is used, also run the real/noisy-real path and apply GAN loss.

## Explanation of Key Hyperparameters

- `scaling`: controls how the time (noise level) coefficients are mapped into the TrigFlow parameterization. Set this according to how the teacher model was trained (`'edm'` or `'rectified_flow'`).
- `optimizer_fake_score_config`: configuration for the critic (fake score) network optimizer; in particular, the `lr` field specifies the critic’s learning rate.
- `student_update_freq`: controls how often the student training step runs. The default is 5, meaning every 5th training step updates the student, while the remaining steps update only the critic.
- `tangent_warmup`: number of initial steps during which we only train the student (without alternating with the critic). In our DMD2 experiments this warmup did not provide a clear benefit.

## Example Training Progress

The DMD2 distillation process usually achieves quick convergence. For instance, in the given example, satisfactory video quality is obtained from the 4-step student after 1500 iterations, which corresponds to 300 student steps and 1200 critic steps.
![DMD2 Predict2.5 vis 2k](../../assets/images/distillation/dmd2_predict2.5_step2k.png)
