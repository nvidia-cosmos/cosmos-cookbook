# Multi-Control Recipes with Cosmos Transfer 2.5

> **Authors:** [Aiden Chang](https://www.linkedin.com/in/aiden-chang/) • [Akul Santhosh](https://www.linkedin.com/in/akulsanthosh/)
> **Organization:** NVIDIA

## Overview

| **Model** | **Workload** | **Use Case** |
|-----------|--------------|--------------|
| Cosmos Transfer 2.5 | Inference | Multi-control video editing for background replacement, lighting adjustment, object transformation, and color/texture changes |

Please make sure to read the core concepts on control modalities in the
[Control Modalities Summary](../../../../core_concepts/control_modalities/overview.md) before proceeding. It is important to understand each control modality before proceeding.

Cosmos Transfer 2.5 enables precise video manipulation using multiple control modalities (Edge, Segmentation, Vis) with masking capabilities. This cookbook provides four key recipes for common video editing tasks, each optimized for predictable, high-quality results.

## Control Modalities Reference

Before diving into recipes, here are the basic control modalities we'll be using. Please make sure to read the core concepts on control modalities in the
[Control Modalities Summary](../../../../core_concepts/control_modalities/overview.md) before proceeding.:

| **Control Type** | **Description** | **Example** |
|-----------------|-----------------|-------------|
| **Original Video** | Source video input | <video src="assets/wave.mp4" controls width="300"></video> |
| [**Edge**](#generating-a-more-detailed-edge-control-modality) | Canny edge detection output | <video src="assets/edge.mp4" controls width="300"></video> |
| [**Filtered Edge**](#generating-a-filtered-edge) | Edge with mask applied (keeping only desired edges) | <video src="assets/filtered.mp4" controls width="300"></video> |
| **Segmentation** | Semantic segmentation map | <video src="assets/seg.mp4" controls width="300"></video> |
| **Vis** | Visual features from original | <video src="assets/vis.mp4" controls width="300"></video> |
| **Mask** | Binary mask (white = change allowed) | <video src="assets/mask.mp4" controls width="300"></video> |
| [**Inverted Mask**](#generating-an-inverted-mask) | Inverse of mask (white = background) | <video src="assets/mask_inverted.mp4" controls width="300"></video> |

Note: Edge and Vis can be automatically computed on the fly. To compute all other modalities, check out the [Control Modalities Summary](../../../../core_concepts/control_modalities/overview.md) and [some starter code if necessary](https://github.com/aiden200/cosmos_transfer_2.5_data_preprocessing).

## **Quick Recipes: Common Use Cases**

Use this table as a starting point for your projects.

| Task | Suggested Controls & Settings | Original Video | Augmented Video |
| :---- | :---- | :---- | :---- |
| **Change clothing or textures** | Edge: 1, Guidance: 3 | <video src="assets/wave.mp4" controls width="300"></video> | <video src="assets/color.mp4" controls width="300"></video> |
| **Change lighting** | Guidance: 3, Edge: 1 + Vis: 0.2 | <video src="assets/wave.mp4" controls width="300"></video> | <video src="assets/lighting.mp4" controls width="300"></video> |
| **Change background, keep subject** | Guidance: 3, Edge Filtered: 1.0 \+ Seg (Mask Inverted): 0.4 + Vis: (medium weight, e.g., 0.6) | <video src="assets/wave.mp4" controls width="300"></video> | <video src="assets/ocean.mp4" controls width="300"></video> |
| **Make object changes, but keep it realistic** | Guidance: 3, Edge: 0.2 \+ Seg (Mask): 1.0 + Vis: (medium weight, e.g., 0.5) | <video src="assets/humanoid.mp4" controls width="300"></video> | <video src="assets/object_change.mp4" controls width="300"></video> |

---

## Recipe 1: Background Change

<!-- TODO Simulation augmentation -->

### Overview

Replace video backgrounds while preserving foreground subjects and their motion. This recipe is ideal for placing subjects in new environments without reshooting.

<img src="./assets/background_change_recipe.png" controls width="1300"></img>

### Example Results

| Original | Background Changed |
|----------|----------|
| <video src="assets/wave.mp4" controls width="300"></video> | <video src="assets/street_background.mp4" controls width="300"></video>  |

### Pipeline Configuration

```json
{
  "name": "change_background",
  "prompt_path": "prompt.txt",
  "video_path": "original.mp4",
  "guidance": 3,
  "edge": {
    "control_weight": 1.0,
    "control_path": "filtered_edge.mp4"
  },
  "seg": {
    "control_weight": 0.4,
    "control_path": "segmentation.mp4",
    "mask_path": "mask_inverted.mp4"
  },
  "vis": {
    "control_weight": 0.6
  }
}
```

### Step-by-Step Process

#### 1. Generate Filtered Edge

Extract edges only for the objects you want to preserve (e.g., human, table). In this example, we only want to keep the person waving and modify everything else in the scene. Therefore, we generate a *filtered edge map* that isolates the human’s edges. For more details on generating filtered edges, see [this section](#generating-a-filtered-edge).

Extract edges only for objects to preserve (human, table, etc.)

<img src="./assets/filtered_edge_recipe.png" controls width="1300"></img>

#### 2. Create Inverted Mask

Reference [this section](#generating-an-inverted-mask) for inverting the mask.

<video src="assets/mask_inverted.mp4" controls width="300"></video>

#### 3. Configure Controls

1. **Edge (1.0)**: Preserve subject structure completely
2. **Seg (0.4) + Inverted Mask**: Allow realistic background generation
3. **Vis (0.6)**: Maintain lighting consistency and fisheye distortion

#### 4. Results

- Original background completely replaced
- Subject motion and structure preserved
- Realistic lighting and perspective maintained
- Fisheye lens distortion preserved

#### 5. Debugging

- Turn down `vis` if there exist some artifacts of the original video.
- Turn up `seg` to generate a more complicated background.
- Incorrect scaling / proportion: Depth control may solve this issue, but this seems like an edge case.

Here's another result generated with "ocean" as the background. I had to turn down the `vis` due to the background artifacts.

| Original Video | High Vis | Low Vis |
|----------|----------|----------|
| <video src="assets/wave.mp4" controls width="300"></video> | <video src="assets/ocean_high_vis.mp4" controls width="300"></video> | <video src="assets/ocean.mp4" controls width="300"></video>  |

For reference, this is the ocean prompt:

```txt
A realistic, static full-body shot of a young man standing outdoors near the coast. He has short dark hair and is dressed casually in a dark grey t-shirt, loose black pants, and white sneakers, with an ID badge clipped to his waistband. He faces the camera directly and waves his right hand continuously in a friendly greeting. The surrounding environment is bright and open. In the background, a vast ocean stretches out toward the horizon, with gentle waves, shimmering reflections, and a clear blue sky above. A coastal walkway with railings and scattered pedestrians lines the foreground, replacing the busy city street elements. Soft natural lighting from the sun enhances the calm, breezy seaside atmosphere.
```

## Recipe 2: Lighting Change

<!-- TODO Simulation augmentation -->

### Overview

Modify scene lighting conditions (e.g., day to night, indoor to outdoor lighting) while maintaining object structure and composition.

<img src="./assets/lighting_change_recipe.png" controls width="600"></img>

### Example Results

| Original | Lighting Changed |
|----------|----------|
| <video src="assets/wave.mp4" controls width="300"></video> | <video src="assets/lighting.mp4" controls width="300"></video>  |

### Pipeline Configuration

```json
{
  "name": "change_lighting",
  "prompt_path": "lighting_prompt.txt",
  "video_path": "original.mp4",
  "guidance": 3,
  "edge": {
    "control_weight": 1.0
  },
  "vis": {
    "control_weight": 0.2
  }
}
```

### Comparison: With and Without vis

| Original Video | Edge Only | Edge + Vis |
|----------|----------|----------|
| <video src="assets/wave.mp4" controls width="300"></video> | <video src="assets/lighting_no_vis.mp4" controls width="300"></video> | <video src="assets/lighting.mp4" controls width="300"></video> |

### Configure Controls

1. **Edge (1.0):** Full structure preservation
2. **Vis (0.2):** Small amount for realistic lighting physics
3. **No Segmentation:** Not needed for global lighting changes

This is the prompt that was used:

```
A realistic, static full-body shot of a young man standing in the center of a spacious, modern atrium. He has short dark hair and is dressed casually in a dark grey t-shirt, loose black pants, and white sneakers, with an ID badge clipped to his waistband. He faces the camera directly and waves his right hand continuously in a friendly greeting. The surrounding space is bright and open, featuring a high industrial-style ceiling with exposed white beams and large, angular black structural supports. The floor is polished light grey concrete, subtly reflecting the warm, soft afternoon sunlight that pours in from large windows above. The overall lighting has a gentle golden tint, with natural shadows stretching slightly to the side in the way they do during late afternoon. In the background, a mezzanine level with glass railings is visible, along with several modern wooden benches and tables scattered throughout the area.
```

## Recipe 3: Color/Texture Change

<!-- TODO Simulation augmentation -->

### Overview

Modify colors or textures of specific objects without altering structure. Simplest recipe, ideal for product variations.

<img src="./assets/color_change_recipe.png" controls width="400"></img>

### Example Results

| Original | Color/Texture Changed |
|----------|-------------------|
| <video src="assets/wave.mp4" controls width="300"></video> | <video src="assets/color.mp4" controls width="300"></video>  |

### Pipeline Configuration

```json
{
  "name": "color_change",
  "prompt_path": "color_prompt.txt",
  "video_path": "original.mp4",
  "guidance": 3,
  "edge": {
    "control_weight": 1.0
  }
}

```

### Why No Vis Control?

1. Vis would preserve original colors/textures
2. Pure edge control allows color changes while maintaining structure
3. Trade-off: May see minor color shifts in other areas

### Configure Controls

1. **Edge (1.0):** Full structure preservation.
2. Specify in the prompt exactly what to change.

This is the prompt that was used:

```
A realistic, static full-body shot of a young man standing in the center of a spacious, modern atrium. He has short dark hair and is dressed casually in a red t-shirt, loose black pants, and white sneakers, with an ID badge clipped to his waistband. He faces the camera directly and waves his right hand continuously in a friendly greeting. The surrounding space is bright and open, featuring a high industrial-style ceiling with exposed white beams and large, angular black structural supports. The floor is polished light grey concrete, reflecting the artificial overhead lighting. In the background, a mezzanine level with glass railings is visible, along with several modern wooden benches and tables scattered throughout the area.
```

**White color generation issues:** We've seen some issues with the generation of white colors, but this is also an edge case.

## Recipe 4: Object Change

<!-- TODO Simulation augmentation -->

### Overview

Transform specific objects while maintaining realistic interaction and physics. Perfect for product variations or creative editing.

<img src="./assets/object_change_recipe.png" controls width="1300"></img>

### Example Results

| Original | Object Changed |
|----------|-------------------|
| <video src="assets/humanoid.mp4" controls width="300"></video> | <video src="assets/object_change.mp4" controls width="300"></video>  |

### Pipeline Configuration

```json
{
  "name": "object_change",
  "prompt_path": "object_prompt.txt",
  "video_path": "original.mp4",
  "guidance": 3,
  "edge": {
    "control_weight": 0.2
  },
  "seg": {
    "control_weight": 1.0,
    "control_path": "segmentation.mp4",
    "mask_path": "object_mask.mp4"
  },
  "vis": {
    "control_weight": 0.5
  }
}

```

### Step-by-Step Process

#### 1. Generate Standard Edge Map

Unlike background replacement, we use the *full edge map* with a low weight (0.2). This allows the model to deviate from the original object structure while maintaining scene coherence.

<video src="assets/object_edge_full.mp4" controls width="300"></video>

**Why low edge weight?** If you're changing a chip bag into a watermelon, the shape must change dramatically. High edge weight would force the model to keep the bag's shape, creating unrealistic results.

#### 2. Create Object Mask

Generate a mask that includes both the object AND any interacting elements (e.g., robotic gripper). White pixels indicate areas the model can modify. The mask should include:

1. Primary object (e.g., vegatable) - white pixels
2. Interacting elements (e.g., robot hand) - white pixels
3. Everything else - black pixels

<video src="assets/object_mask.mp4" controls width="300"></video>

#### 3. Generate Segmentation

Use the full segmentation map to provide semantic understanding of the scene:

<video src="assets/object_seg.mp4" controls width="300"></video>

#### 4. Configure Controls

1. **Edge (0.2):** Low weight allows shape changes
2. **Seg (1.0) + Object Mask:** Strong segmentation focused on masked regions
3. **Vis (0.5):** Medium weight balances realism with transformation ability

#### 5. Debugging

- Grasp looks unrealistic → Expand mask around contact points
- Object shape too constrained → Reduce edge weight to 0.1
- Background changing → Increase vis weight to 0.7

For reference, this is the object change prompt:

```txt
A first-person point-of-view video from a dual-arm robotic system operating in a research lab. The camera is positioned between the robot's two black, multi-fingered hands, which are visible in the lower corners. The left robotic hand is stationary and is already holding a green bell pepper. On the wooden table in front of the robot, there is an assortment of artificial fruits and vegetables, including two yellow bananas, a bunch of green okra, purple grapes, a green avocado-like object, a grey metal rod, and an orange pear-shaped fruit. The primary action of the video follows the right robotic hand. It starts by moving towards the grey metal rod on the table. The hand's fingers then actuate, closing around the metal rod to grasp it securely. The hand lifts the metal rod off the table and begins moving it towards the right, presumably to place it in a multi-tiered black wire basket that is visible on the right side of the frame and already contains a red apple. The background shows a large, open-plan workshop or lab setting with a gray floor marked by hazard tape, a person sitting at a desk (possibly an operator), other computer workstations, and various pieces of equipment.
```

---

## Additional Modality Generation

### Generating a Filtered Edge

In practice, we’ve found that pre-filtering the edge modality with a mask produces more reliable results than passing a mask_path directly to the model. Performing this as a preprocessing step gives you cleaner, more controllable edge signals.

To generate a filtered edge video, we simply combine the mask video with the raw edge video. The process is:

<img src="./assets/filtered_edge_recipe.png" controls width="1300"></img>

Some example code of how to create this can be found [here](https://github.com/aiden200/cosmos_transfer_2.5_data_preprocessing/blob/main/utils/filter_out_edges.py).

### Generating an Inverted Mask

To create an inverted mask (where white becomes black and vice-versa), we simply apply a color inversion to the original mask video. This can be done quickly with ffmpeg:

```bash
ffmpeg -y -i mask.mp4 \
  -vf "negate" \
  -c:v libx264 -pix_fmt yuv420p mask_inverted.mp4
```

### Generating a more detailed edge control modality

When object and background contours are too similar, edges may not be detected reliably. In these cases, increasing the brightness and contrast of the video before running Canny edge detection can help produce a more detailed and stable edge map.

An example implementation of this preprocessing step can be found [here](https://github.com/aiden200/cosmos_transfer_2.5_data_preprocessing/blob/main/control_net_generation/get_object_edges.py).

## Resources

1. [Cosmos Transfer 2.5 Model](https://github.com/nvidia-cosmos/cosmos-transfer2.5) - Model weights and documentation.
2. [Control Extraction Tools](https://github.com/aiden200/cosmos_transfer_2.5_data_preprocessing) - Scripts for generating control modalities.
3. [Control Modalities Summary](../../../../core_concepts/control_modalities/overview.md) - Summary of the role of each control modality.
