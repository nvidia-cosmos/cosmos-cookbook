# Data Curation

This section details the data curation process for the Autonomous Vehicle VQA use case using the Nexar collision prediction dataset.

## Dataset Preparation

### Download and Preprocessing

Data curation is performed on the [Nexar collision prediction dataset](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction). The preprocessing pipeline involves:

1. **Download**: Obtain the complete Nexar collision prediction dataset
2. **Video Segmentation**: Split the original videos (typically 40 seconds) into shorter clips of less than 20 seconds
3. **Dataset Generation**: This process generates a total of approximately 2,000 video clips
4. **Data Splitting**: Divide the dataset as follows:
   - **80%** for training (1,600 videos)
   - **10%** for testing (200 videos)
   - **10%** for evaluation (200 videos)

### Annotation Workflow

The data curation follows a multi-stage annotation process:

1. **Dense Captioning**: Generate detailed video descriptions using a Vision-Language Model (VLM), with human validation
2. **Chain of Thought (CoT) Reasoning**: Use a reasoning LLM to generate logical reasoning traces from the captions
3. **MCQ Annotation**: For Reinforcement Learning training, annotate videos with multiple-choice questions (optionally with Chain of Thought reasoning)

## Setup

This tutorial requires the following hosted models:

- Reasoning LLM (e.g. DeepSeek-R1)
- VLM (e.g. Cosmos-Reason1)

You can host your own models with [vLLM](https://docs.vllm.ai/en/v0.9.2/getting_started/quickstart.html#openai-compatible-server).

The models can be queried using the [`openai` API library](https://platform.openai.com/docs/guides/structured-outputs).

## Captioning

As part of the dense captioning workflow, use a VLM to generate detailed captions of the video clips. These captions are then validated by human annotators to ensure accuracy and completeness for the autonomous vehicle domain.

???+ code "Prompt for VLM"

    ```yaml
    --8<-- "docs/examples/reason1/cosmos-reason1-av-vqa/assets/captioning_prompt.yaml"
    ```

??? code "Sample Output"

    ```json
    {
        "description": "The video shows a view from inside a vehicle at an intersection during what appears to be a cloudy day. The road is wet, suggesting recent rain. There are residential houses on both sides of the street, and the traffic light is red for the direction the vehicle is facing. Several cars are waiting at the intersection, and one car is seen turning right. The pedestrian crossing is visible, but no pedestrians are crossing at the moment. The environment suggests a suburban area with moderate traffic.",
        "driving_difficulity_explanation":  "The driving difficulty in this scenario is relatively low due to the clear visibility of the traffic light and the lack of immediate obstacles. However, the wet road conditions may require cautious driving to avoid hydroplaning. The presence of multiple vehicles waiting at the intersection adds a slight level of complexity, as drivers need to be attentive to the movements of other vehicles.",
        "notice": [
            "The traffic light is red for the direction the vehicle is facing.",
            "Several cars are waiting at the intersection, including a car turning right.",
            "The road is wet, indicating recent rain."
        ]
    }
    ```

## SFT Dataset

For the Supervised Fine-Tuning (SFT) phase, use a reasoning LLM to generate Chain of Thought (CoT) reasoning traces from the dense captions. This step transforms the descriptive captions into structured reasoning patterns that help the model understand the logical flow of autonomous vehicle decision-making.

???+ code "Prompt for LLM"

    ```yaml
    --8<-- "docs/examples/reason1/cosmos-reason1-av-vqa/assets/reasoning_prompt.yaml"
    ```

For the SFT dataset, we exclude the caption from the prompt.

???+ code "Prompt for SFT dataset"

    ```yaml
    --8<-- "docs/examples/reason1/cosmos-reason1-av-vqa/assets/reasoning_training.yaml"
    ```

Create a SFT dataset where each sample has the following format ([example](https://huggingface.co/datasets/nvidia/Cosmos-Reason1-SFT-Dataset)):

- `video`: Video URL.
- `question`: Use above `Prompt for SFT dataset`.
- `reasoning`: Reasoning trace extracted from the LLM output.
- `prediction`: Predicted action extracted from the LLM output.

## RL Dataset

For the Reinforcement Learning (RL) phase, use a LLM (e.g. DeepSeek-R1) to generate Visual Question Answer (VQA) pairs from the captions. This process focuses on creating multiple-choice questions (MCQs) that target specific aspects of autonomous vehicle behavior and scene understanding. The MCQs can optionally include Chain of Thought reasoning to provide additional context for the model's decision-making process.

???+ code "Prompt for LLM"

    ```yaml
    --8<-- "docs/examples/reason1/cosmos-reason1-av-vqa/assets/qa_prompt.yaml"
    ```

??? code "Sample Generated VQA"

    ```json
    {
        "Q1 (Space: Relationship)": {
            "question": "Is the robotic arm positioned correctly to reach the laundry basket from the table?",
            "answer": "YES"
        },
        "Q2 (Space: Relationship)": {
            "question": "Are the colorful fabrics visibly located in the laundry basket before the robotic arm picks them up?",
            "answer": "YES"
        },
        "Q3 (Space: Interaction)": {
            "question": "Does the robotic arm successfully grasp a piece of cloth from the basket?",
            "answer": "YES"
        },
    ...
    }
    ```

Create a RL dataset where each sample has the following format ([example](https://huggingface.co/datasets/nvidia/Cosmos-Reason1-RL-Dataset)):

- `video`: Video URL.
- `question`: Question extracted from the LLM output.
- `answer`: Answer extracted from the LLM output.

## Post-Training

[Cosmos-Reason1 SFT/RL Example](https://github.com/nvidia-cosmos/cosmos-reason1/blob/main/examples/post_training/README.md)
