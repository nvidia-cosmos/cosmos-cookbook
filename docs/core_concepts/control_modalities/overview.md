# Cosmos Transfer 2.5 Control Modalities: Core Concepts

> **Authors:** [Aiden Chang](https://www.linkedin.com/in/aiden-chang/) • [Akul Santhosh](https://www.linkedin.com/in/akulsanthosh/)
> **Organization:** NVIDIA

## Overview: The Control Challenge

This document serves as a comprehensive guide to the **core concepts** required for using the **Transfer 2.5** video generation model. Success depends on understanding the balance between the **Guidance Scale** (your text prompt) and the influence of the four primary **Control Modalities (Edge, Depth, Segmentation, and Vis)**.

**Key Takeaway:** Multi-control tuning, using a strategic combination of Edge, Vis, Depth, and Seg is required for achieving high-fidelity, structurally consistent video results.

---

## 1. Key Concepts: Governing Strength

### 1.1. Guidance Scale (Prompt Strength)

This principle dictates how strictly the model adheres to your text prompt versus the visual controls.

- **What it is:** Controls the influence of the text prompt.
- **Good Starting Point:** Guidance = 3.
- **When to Increase:** Increase to **5+** if the visual output fails to incorporate the changes described in your prompt (e.g., trying to change a shirt into a specific texture).

### 1.2. Control Weight Normalization (Very Important)

This principle governs how the model balances the influence of multiple control modalities (e.g., Edge + Seg + Vis) against each other.

- **Rule 1: Weights WILL NOT Normalize** if the total sum of all control weights is **$1.0$ or less**. The weights are applied as-is.
  - *Example:* {seg: 0.2, edge: 0.2} (sum is 0.4) will be used as-is.
- **Rule 2: Weights WILL NORMALIZE** if the total sum is **greater than $1.0$**. The weights are re-scaled proportionally so the new total sum equals 1.0.
  - *Example:* {seg: 4.0, edge: 1.0} (sum is 5.0) will be normalized and run as {seg: 0.8, edge: 0.2}.

---

## 2. Technical Details: The Control Modalities

The system uses four primary modalities to inject structural, semantic, relative, and visual consistency into the video.

### 2.1. Edge Control (Structure Preservation)

- **Function:** Preserves the **original structure, shape, and layout** of the video.
- **Best For:** Changing textures, clothing, or lighting where the underlying shape must be maintained.
- **Limitation:** Performs poorly when attempting to drastically change an object's shape (e.g., turning a shirt into a banana).

Edge control is natively built on the CT2.5 (Cosmos Transfer 2.5) [repository](https://github.com/nvidia-cosmos/cosmos-transfer2.5). If no `control_path` is specified, CT2.5 will automatically generate the edge control modality on the fly. When object and background contours are too similar, edges may not be detected reliably. In these cases, increasing the brightness and contrast of the video before running Canny edge detection can help produce a more detailed and stable edge map.

An example implementation of this preprocessing step can be found [here](https://github.com/aiden200/cosmos_transfer_2.5_data_preprocessing/blob/main/control_net_generation/get_object_edges.py).

### 2.2. Segmentation (Seg) Control (Structural Change & Semantic Replacement)

- **Function:** Facilitates **large, structural changes** and semantic replacement. Used to completely transform or replace objects, people, or backgrounds.
- **Best For:** Generating realistic *new* objects/scenes where the prompt requires a large change.
- **Limitation:** High weights can lead to **"hallucinations"** (unrealistic or physically incorrect objects).
- **Recommended Usage:** **Always** use Seg with a **mask** of the parts you want to change, and **always** use it as part of a **multi-control** configuration (e.g., with Edge).

There are two ways to generate segmentation masks:

1. **Specify objects manually**: You can provide the list of objects you want to segment and run the SAM2 endpoint in the CT2.5 repository.
The implementation is available in the [SAM2 pipeline code](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_pipeline.py).
2. **Automatic object detection (recommended for scale)**: We also provide an automated workflow using [RAM++](https://github.com/xinyu1205/recognize-anything) to detect objects, which are then passed into the Cosmos pipeline to generate segmentation masks.
The preprocessing [code is available here](https://github.com/aiden200/cosmos_transfer_2.5_data_preprocessing).

### 2.3. Vis Control (Lighting & Background Feel)

- **Function:** Preserves the **original video's background, lighting, and general feel**. It has a default effect of softening or blurring the background.
- **Best For:** Acting as a *supplement* to Edge or Seg, typically with a **lower weight**, to fine-tune visual consistency.
- **Intuition:**
  - **Increase Vis weight** to keep more of the original video's look.
  - **Decrease Vis weight** to allow more changes from the original (though too low can *increase* background hallucinations).
- **Limitation:** If the weight is too high, it will just return your original video. Masking Vis control is known to cause hallucinations.

Vis control is natively built on the CT2.5 (Cosmos Transfer 2.5) [repository](https://github.com/nvidia-cosmos/cosmos-transfer2.5). No need to specify a `control_path`

### 2.4. Depth Control

- **Function:** Maintains **3D realism** and **spatial consistency** by respecting distance and perspective.
- **Potential Use:** Helps when placing new objects into a scene or maintaining camera movement integrity.
- *Documentation in progress.*

<div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: space-between;">
  <div style="flex: 1 1 45%; min-width: 300px;">
    <strong>RGB Video</strong>: Original Video
    <video controls width="100%" aria-label="Original Video">
      <source src="assets/wave.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>
  <div style="flex: 1 1 45%; min-width: 300px;">
    <strong>Segmentation Map</strong>: Semantic segmentation of the original video.
    <video controls width="100%" aria-label="Semantic segmentation video of the original video">
      <source src="./assets/seg.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>
</div>

<div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: space-between; margin-top: 20px;">
  <div style="flex: 1 1 45%; min-width: 300px;">
    <strong>Edge Map</strong>: Geometric boundaries of all objects and road infrastructure.
    <video controls width="100%" aria-label="Edge detection video showing geometric boundaries of objects and infrastructure">
      <source src="./assets/edge.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>
  <div style="flex: 1 1 45%; min-width: 300px;">
    <strong>Vis Map</strong>: Blurred out version of the original video.
    <video controls width="100%" aria-label="Semantic segmentation video showing labeled scene elements including vehicles and roads">
      <source src="./assets/vis.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
  </div>
</div>

---

## 2.5. Binary Masking: Localizing Control

Masking is the technique used to apply a control modality to specific areas of the video frame.

- **Mechanism:** A binary mask (a black and white image/video) is used. The control modality is applied **only to the white pixels** in the mask.
  - **White Pixels:** The area of **change/control application**.
  - **Black Pixels:** The area that should **remain unchanged** or where the control is suppressed.
- **Seg Masking (Standard):** This is an **effective** and standard use of masking. You supply a mask to the Seg control input to tell it exactly where to perform the semantic replacement.
- **Vis Masking (Avoid):** Masking Vis control is known to cause visual **hallucinations** and is generally discouraged. Use Vis globally with a low weight instead.

Binary masking is not natively supported in CT2.5. We have some starter code to generate [the masks using DINO + SAM2](https://github.com/aiden200/cosmos_transfer_2.5_data_preprocessing).

<video width="720" controls>
  <source src="assets/mask.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

---

## 3. Building Intuition: Multi-Control Tuning

To achieve complex goals like background replacement, you must strategically combine the modalities.

| If Your Goal Is... | Increase This Setting | Decrease This Setting |
| :---- | :---- | :---- |
| **Reduce hallucinations / weird objects** | | Seg weight or Guidance |
| **Preserve original background/lighting** | **Vis weight** | |
| **Keep the original video structure** | **Edge weight** | |
| **Make more realistic drastic changes** | **Seg weight** | Vis weight |
| **Keep object boundaries consistent** | **Edge weight** | Vis weight |

### The Background Replacement Walkthrough

The following steps illustrate how each control builds upon the last to achieve a high-fidelity result, starting from a base video. The objective is to change the original video's background to an outside street type environment.

#### **Step 1: Edge Only (Base Structure)**

The first step is often applying **Edge control** to keep the core structure (e.g., the human's gesture).

- **Action:** Apply Edge control using a **filtered edge map** (edges of only the human).
- **Result Intuition:** The human's motion is preserved, but the background still looks unrealistic and distorted. This shows that **Edge only controls shape, not *visual fidelity***.

<video width="500" controls>
<strong>Mask Video</strong>
  <source src="assets/only_edge.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

#### **Step 2: Edge + Vis (Adding Lighting Consistency)**

To fix the unrealistic look and preserve camera effects, **Vis control** is added.

- **Action:** Add **Vis control** with a medium weight (e.g., 0.6).
- **Result Intuition:** The fisheye distortion is more accurate, and the background is less blurry. This confirms **Vis**'s role in **preserving overall *visual feel* and camera properties**. However, overall realism is still lacking.

<video width="500" controls>
<strong>Mask Video</strong>
  <source src="assets/edge_with_vis.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

#### **Step 3: Edge + Vis + Seg (Injecting Realism)**

To generate a *completely new and realistic* background, **Segmentation control** is used.

- **Action:** Add **Seg control** with a moderate weight (e.g., 0.4), using an **mask** (white on the background) to direct the semantic replacement only to the background.
- **Result Intuition:** The final output is visually sharp and consistent. **Seg** provides the **semantic information** necessary to generate a plausible, new environment, while **Edge** and **Vis** ensure the subject and lighting remain consistent.

<video width="500" controls>
<strong>Mask Video</strong>
  <source src="assets/street_background.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

Full receipes can be found at **[Real World Video Manipulation Guidelines with Cosmos Transfer 2.5](../../recipes/inference/transfer2_5/inference-real-augmentation/)**.

---

## Best Practices

**Do:**

- ✅ **Use Multi-Control:** Combine modalities (especially Seg + Edge) for complex tasks.
- ✅ **Use a Mask with Seg:** Provide a mask when using Seg control to isolate the area of change.
- ✅ **Start with Lower Weights for Vis:** Use Vis as a supplement with a lower weight (e.g., 0.4-0.6) to maintain visual feel.

**Don't:**

- ❌ **Use Seg Control Alone:** **Do not use Segmentation control by itself**; it leads to highly unrealistic results.
- ❌ **Use Seg + Vis (without Edge):** This combination is **not recommended** as it can lead to unpredictable outcomes.
- ❌ **Mask Vis Control:** **Avoid masking Vis control** as it is known to cause hallucinations.
- ❌ **Use Vis with a High Weight:** A very high Vis weight will just return your original video—Use [lower weights] instead.

## Use Cases

- **[Real World Video Manipulation Guidelines with Cosmos Transfer 2.5](../../recipes/inference/transfer2_5/inference-real-augmentation/)**
