# Worker Safety in a Classical Warehouse with Cosmos Reason 2

> **Authors:** [Paula Ramos, PhD](https://www.linkedin.com/in/paula-ramos-phd/)
> **Organization:** [NVIDIA](https://nvidia.com/)

## Overview

This recipe demonstrates a complete video reasoning pipeline for automating industrial safety inspections in challenging **"brownfield"** environments using [**Cosmos Reason 2**](https://github.com/nvidia-cosmos/cosmos-reason2) and [FiftyOne](https://github.com/voxel51/fiftyone).

It shows how to prompt a Video Language Model (VLM) to ignore environmental noise (like faded paint or old machinery) and strictly classify worker behaviors based on visual ground truths defined in the Video Dataset for Safe and Unsafe Behaviours.

Main notebook: [Worker Safety notebook](../../../../../scripts/examples/reason2/worker_safety.ipynb)  
Setup guide: [Setup and system requirements](setup.md)


| **Model** | **Workload** | **Use Case** |
|-----------|--------------|--------------|
| [**Cosmos Reason 2**](https://github.com/nvidia-cosmos/cosmos-reason2) | Inference | Zero-shot safety compliance and hazard detection |

<video src="assets/overload_forklift.webm" controls width="720"></video>

*Forklift overload example: the model should classify this as “Carrying Overload with Forklift” when 3+ blocks are visible.*

---

## Setup

Before running this recipe, ensure you have the necessary environment configuration.

Prerequisites:
- A GPU environment (e.g., NVIDIA RTX PRO 5000 Blackwell) with CUDA 13.0 available.
- `fiftyone`, `transformers`, `torch`, and `qwen-vl-utils` installed.
- The [`pjramg/Safe_Unsafe_Test`](https://huggingface.co/datasets/pjramg/Safe_Unsafe_Test-Understanding) dataset available on Hugging Face.

---

## Motivation: Safety in "Classical" Warehouses

Modern factories are often pristine, but many real-world facilities are "classical" warehouses - older buildings with irregular layouts, poor illumination, and worn infrastructure. In these environments, standard computer vision models often fail because they confuse faded floor markings with active safety zones.
The Safe/Unsafe Behaviours Dataset, collected from a metal manufacturing plant, captures this reality perfectly. It contains 8 specific behavior classes (4 safe, 4 unsafe) recorded over 39 days. The dataset we are using in this recipe is just a sample of the original dataset ([Papaer](https://www.sciencedirect.com/science/article/pii/S235234092400756X), [Dataset](https://data.mendeley.com/datasets/xjmtb22pff/1))

The challenges defined in the dataset:
- Environmental noise: faded yellow lines, unpainted areas, and complex backgrounds.
- 2D projection limits: camera angles make workers appear "on" a path when they are actually walking alongside it.
- Strict compliance: safety rules are binary (e.g., a vest is either worn or not), but visual data is messy.

The solution: Cosmos Reason 2 allows us to bypass training a custom classifier. Instead, we use **prompt engineering** to act as an **expert inspector**. We instruct the model to ignore the **"old warehouse"** aesthetic and focus strictly on the visual definitions provided in the dataset paper - specifically Green Paths, Green Vests, and Block Counts.

This recipe demonstrates a reproducible pipeline that:
1. Loads the safety dataset into FiftyOne.
2. Constructs a context-aware prompt based on the dataset's ground truth.
3. Runs Cosmos Reason 2 inference on video clips.
4. Parses structured JSON output for hazard detection.
5. Visualizes the results and compare with ground truth.

---

## Pipeline Overview

This is the end-to-end flow:
1. Load data: ingest the [`Safe_Unsafe_Test`](https://huggingface.co/datasets/pjramg/Safe_Unsafe_Test-Understanding) dataset from Hugging Face.
2. Define prompts: encode the dataset's specific visual rules into system and user prompts.
3. Run inference: process videos using [`Cosmos-Reason2-2B`](https://github.com/nvidia-cosmos/cosmos-reason2) via Hugging Face Transformers.
4. Parse output: extract JSON predictions (Class ID, Label, Rationale).
5. Visualize: explore the results in FiftyOne to verify accuracy against the "old warehouse" constraints.

---

## 1. Loading the Dataset

We start by loading the dataset from the Hugging Face Hub into FiftyOne. This dataset contains Full HD (1920x1080) clips at 24 fps, ranging from 1 to 20 seconds in length. 

```python
import fiftyone as fo
import fiftyone.utils.huggingface as fouh

# Load the dataset (persistent=True ensures we don't re-download on every run)
dataset = fo.load_dataset("pjramg/Safe_Unsafe_Test")

# Verify the first sample is a video
sample = dataset.first()
print(f"Loaded dataset with {len(dataset)} samples. Media type: {sample.media_type}")
```

---

## 2. The "Expert Inspector" Prompt Strategy

This is the most critical step. Because the facility is old, we cannot ask the model to look for "safe areas" generically. We must map the system instructions to the dataset's specific constraints defined in the paper.

System prompt (the persona and constraints): we instruct the model to act as an industrial safety inspector. Crucially, we add negative constraints to handle the dataset's limitations:
- Ignore background: do not flag hazards based on faded paint or disrepair.
- Ignore sitting workers: "intervention" classes only apply to standing workers at machine boards; sitting workers (or drivers) are background noise.
- Priority: prioritize unsafe behaviors over safe ones.

User prompt (the strict classification table): we provide the model with the exact definitions from Table 1 of the dataset paper:

| ID | Label | Visual Definition (Ground Truth) |
| :--- | :--- | :--- |
| 0 | Safe Walkway Violation | Walking OUTSIDE the Green Path. |
| 4 | Safe Walkway | Walking INSIDE the Green Path. |
| 1 | Unauthorized Intervention | Interacting with machine board WITHOUT a green vest. |
| 3 | Forklift Overload | Carrying 3 or more blocks. |

```python
SYSTEM_INSTRUCTIONS = """
You are an expert Industrial Safety Inspector monitoring a manufacturing facility.
Your goal is to classify the video into EXACTLY ONE of the 8 classes defined below.

CRITICAL NEGATIVE CONSTRAINTS (What to IGNORE):
1. IGNORE SITTING WORKERS:
   - If a person is SITTING at a machine board working, this is NOT an intervention class. Ignore them.
   - If a person is SITTING driving a forklift, the driver is NOT the class. Focus only on the LOAD carried.
2. IGNORE BACKGROUND:
   - The facility is old. Do not report hazards based on faded floor markings or unpainted areas.
3. SINGLE OUTPUT:
   - Even if multiple things happen, choose the MOST PROMINENT behavior.
   - Prioritize UNSAFE behaviors over SAFE behaviors if both are present.
"""

USER_PROMPT_CONTENT = """
Analyze the video and output a JSON object. You MUST select the class ID and Label EXACTLY from the table below.

STRICT CLASSIFICATION TABLE (Use these exact IDs and Labels):

| ID | Label | Definition (Ground Truth) | Hazard Status |
| :--- | :--- | :--- | :--- |
| 0 | Safe Walkway Violation | Worker walks OUTSIDE the designated Green Path. | TRUE (Unsafe) |
| 4 | Safe Walkway | Worker walks INSIDE the designated Green Path. | FALSE (Safe) |
| 1 | Unauthorized Intervention | Worker interacts with machine board WITHOUT a green vest. | TRUE (Unsafe) |
| 5 | Authorized Intervention | Worker interacts with machine board WITH a green vest. | FALSE (Safe) |
| 2 | Opened Panel Cover | Machine panel cover is left OPEN after intervention. | TRUE (Unsafe) |
| 6 | Closed Panel Cover | Machine panel cover is CLOSED after intervention. | FALSE (Safe) |
| 3 | Carrying Overload with Forklift | Forklift carries 3 OR MORE blocks. | TRUE (Unsafe) |
| 7 | Safe Carrying | Forklift carries 2 OR FEWER blocks. | FALSE (Safe) |

INSTRUCTIONS:
1. Identify the behavior in the video.
2. Match it to one row in the table above.
3. Output the exact "ID" and "Label" from that row. Do not invent new labels like "safe and compliant".

OUTPUT FORMAT:
{
  "prediction_class_id": [Integer from Table],
  "prediction_label": "[Exact String from Table]",
  "video_description": "[Concise description of the observed action]",
  "hazard_detection": {
    "is_hazardous": [true/false based on the Hazard Status column],
    "temporal_segment": "[Start Time - End Time] or null"
  }
}
"""
```

---

## 3. Running Cosmos Reason 2 Inference

We use the transformers library to run the model. Note that we define a conversation structure where the video input precedes the text prompt, matching the model's training inputs.

```python
import torch
import transformers

def run_inference(model, processor, video_path):
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_INSTRUCTIONS}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": USER_PROMPT_CONTENT},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        fps=4,
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    # ... decode output ...
```

---

## 4. Parsing and Storing Results

The model returns a structured JSON object containing the `prediction_class_id`, `prediction_label`, and a `hazard_detection` flag. We parse this string and store it directly into the FiftyOne sample.
This allows us to leverage FiftyOne's powerful filtering capabilities later - for example, isolating all "Class 3" (Forklift Overload) predictions to verify if the model correctly counted the blocks.

```python
try:
    # Clean and parse the JSON output
    clean_json = output_text.strip().replace("```json", "").replace("```", "")
    json_data = json.loads(clean_json)

    # Store in FiftyOne
    sample["cosmos_analysis"] = json_data
    sample["safety_label"] = fo.Classification(label=json_data.get("prediction_label"))
    sample.save()

except Exception as e:
    print(f"JSON Parsing failed: {e}")
```

---

## 5. Results and Visualization

Launch the FiftyOne App to audit the inspector's performance.

```python
session = fo.launch_app(dataset)
session.wait()
```

What to look for:
- Green path compliance: check if the model correctly distinguishes between Class 0 and Class 4 based on the green markings, even when the floor paint is faded.
- Vest detection: verify that Class 1 (Unauthorized) is triggered only when the worker at the board is standing and missing the green vest.
- Forklift counting: ensure Class 3 (Overload) is strictly applied to loads of 3+ blocks, while 2 blocks remain Class 7 (Safe).

Observation: you may notice that the model correctly ignores workers driving forklifts (sitting), focusing only on the load, thanks to the negative constraints in the prompt.

Example result clips:

<video src="assets/unauthorized_intervention.webm" controls width="720"></video>

*Unauthorized intervention: standing worker at the machine board without a green vest.*

<video src="assets/safe_walkway_violation.webm" controls width="720"></video>

*Safe walkway violation: worker walks outside the designated green path.*

<video src="assets/overload_forklift.webm" controls width="720"></video>

*Forklift overload: forklift carries 3 or more blocks.*

---

## Conclusion

This recipe demonstrates:
- How to adapt Cosmos Reason 2 for specialized industrial domains without fine-tuning.
- The importance of prompt engineering in overcoming "brownfield" environmental noise.
- How to map academic dataset definitions (like those in the Safe and Unsafe Behaviours paper) into executable model constraints.

This approach can be generalized to:
- Construction site monitoring (PPE detection).
- Retail loss prevention.
- Logistics and inventory auditing.

For the full code and to run this analysis yourself, verify you have the `pjramg/Safe_Unsafe_Test` dataset and the Cosmos-Reason2 model accessible in your environment.

Run the full workflow in the main notebook: [Worker Safety notebook](../../../../../scripts/examples/reason2/worker_safety.ipynb).
