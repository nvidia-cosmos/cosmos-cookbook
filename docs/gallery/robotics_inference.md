# Robotics Domain Adaptation Gallery

> **Authors:**  [Raju Wagwani](https://www.linkedin.com/in/raju-wagwani-a4746027/) • [Jathavan Sriram](https://www.linkedin.com/in/jathavansriram) • [Richard Yarlett](https://www.linkedin.com/in/richardyarlett/) • [Joshua Bapst](https://www.linkedin.com/in/joshbapst/) • [Jinwei Gu](https://www.linkedin.com/in/jinweigu/)

> **Organization:** NVIDIA

## Overview

This page showcases results from Cosmos Transfer 2.5 for robotics applications. The examples demonstrate sim-to-real transfer for robotic manipulation tasks in kitchen environments, showing how synthetic simulation videos can be transformed into photorealistic scenes with varied materials, lighting, and environmental conditions. These results enable domain adaptation and data augmentation for robotic training and validation.

**Use Case**: Robotics engineers can use these techniques to generate diverse training data from a single simulation, creating variations in kitchen styles, materials, and lighting conditions without re-running expensive simulations or capturing real-world data.

## Example 1: Edge-Only Control for Environment Variation

This example demonstrates how to transform synthetic robotic simulation videos into photorealistic scenes with different kitchen styles and materials using **edge control**. Edge control preserves the original structure, motion, and geometry of the robot and scene while allowing the visual appearance to change dramatically based on the text prompt.

- **Edge control**: Maintains the structure and layout of objects, robot poses, and camera motion from the simulation while transforming the visual appearance (materials, lighting, colors) according to the prompt.
- **Why use edge-only**: Ideal when you want to preserve exact robot motions and object positions from simulation while varying environmental aesthetics.

For detailed explanations of control modalities, see [Control Modalities Overview](../core_concepts/control_modalities/overview.md).

<style>
table td {
  vertical-align: top;
}
table {
  border: none;
}
table td, table th {
  border: none;
}
</style>

### Scene 1a: Kitchen Stove - Cooking Task

This scene shows a humanoid robot performing a cooking task at a stove. The examples demonstrate how different kitchen cabinet styles (white, red, wood tones) and robot materials (plastic, metal, gold) can be generated from the same simulation.

#### Input Video

<video width="600" controls>
  <source src="assets/kitchen_stove_input.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

#### Parameters

```json
{
    "seed": 1,
    "prompt_path": "assets/prompt_robot.json",
    "output_dir": "outputs/robot",
    "video_path": "assets/kitchen_stove_input.mp4",
    "guidance": 7,
    "edge": {
        "control_weight": 1
    }
}
```

#### Examples

<table>
<colgroup>
<col style="width: 20%;">
<col style="width: 80%;">
</colgroup>
<tbody>
<tr>
<td></td>
<td><video width="600" controls>  <source src="assets/kitchen_stove_white.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>This scene depicts a photo realistic luxury kitchen with high end professional finishes and lighting. ALL The kitchen cabinets are all highly polished bright white panels with chrome accents and pulls. The kitchen counters are stainless steel. The kitchen walls and backsplash are all white subway tile. The kitchen contains an expensive double door stainless steel refrigerator, a stainless steel microwave, a stainless steel oven, a stainless steel coffee machine, a stainless steel toaster, a stainless steel stove top, a stainless steel sink, and stainless steel pots. Standing in the kitchen is a humanoid robot. The robot is made of orange polished plastic panels with chrome accents. The camera is fixed and steady. The robot is at a kitchen stainless steel stove picking up a glass cooking pot lid with his left hand and lifting it in the air. The robot is picking up two tomatoes with his right hand and putting them inside the stainless steel pot. There is steam coming out of the pot.</td>
</tr>
<tr>
<td><strong>Parameters</strong></td>
<td>seed: 1, guidance: 7, edge: 1.0</td>
</tr>
<tr>
<td></td>
<td><video width="600" controls>  <source src="assets/kitchen_stove_red.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>This scene depicts a photo realistic luxury kitchen with high end professional finishes and lighting. ALL The kitchen cabinets are all highly polished bright red panels with stainless steel accents and pulls. The kitchen counters are stainless steel. The kitchen walls and backsplash are all white subway tile. The kitchen contains an expensive double door stainless steel refrigerator, a stainless steel microwave, a stainless steel oven, a stainless steel coffee machine, a stainless steel toaster, a stainless steel stove top, a stainless steel sink, and stainless steel pots. Standing in the kitchen is a humanoid robot. The robot is made of white polished panels with black accents. The camera is fixed and steady. The robot is at a kitchen stainless steel stove picking up a red cooking pot lid with his left hand and lifting it in the air. The robot is picking up two tomatoes with his right hand and putting them inside the red pot. There is steam coming out of the pot.</td>
</tr>
<tr>
<td><strong>Parameters</strong></td>
<td>seed: 1, guidance: 7, edge: 1.0</td>
</tr>
<tr>
<td></td>
<td><video width="600" controls>  <source src="assets/kitchen_stove_light_wood.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>This scene depicts a photo realistic luxury kitchen with high end professional finishes and lighting. ALL The kitchen cabinets are all light wood, with stainless steel accents and pulls. The kitchen counters, kitchen walls and backsplash are all expensive black veined marble. The kitchen contains an expensive double door stainless steel refrigerator, a stainless steel microwave, a stainless steel oven, a stainless steel coffee machine, a stainless steel toaster, a stainless steel stove top, a stainless steel sink, and stainless steel pots. Standing in the kitchen is a humanoid robot. The robot is made of stainless steel polished panels with chrome accents. The camera is fixed and steady. The robot is at a kitchen stainless steel stove picking up a stainless steel cooking pot lid with his left hand and lifting it in the air. The robot is picking up two tomatoes with his right hand and putting them inside the stainless steel pot. There is steam coming out of the pot.</td>
</tr>
<tr>
<td><strong>Parameters</strong></td>
<td>seed: 1, guidance: 7, edge: 1.0</td>
</tr>
<tr>
<td></td>
<td><video width="600" controls>  <source src="assets/kitchen_stove_dark_wood.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>This scene depicts a photo realistic luxury kitchen with high end professional finishes and lighting. ALL The kitchen cabinets are all dark wood, with chrome accents and pulls. The kitchen counters, kitchen walls and backsplash are all expensive beige veined marble. The kitchen contains an expensive double door stainless steel refrigerator, a stainless steel microwave, a stainless steel oven, a stainless steel coffee machine, a stainless steel toaster, a stainless steel sink, and stainless steel pots. Standing in the kitchen is a humanoid robot. The robot is made of gold polished reflective panels with shiny black accents. The camera is fixed and steady. The robot is at a kitchen gold stove picking up a gold cooking pot lid with his left hand and lifting it in the air. The robot is picking up two tomatoes with his right hand and putting them inside the gold pot. There is steam coming out of the pot.</td>
</tr>
<tr>
<td><strong>Parameters</strong></td>
<td>seed: 1, guidance: 7, edge: 1.0</td>
</tr>
<tr>
<td></td>
<td><video width="600" controls>  <source src="assets/kitchen_stove.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>The video captures a stunning, photorealistic scene with remarkable attention to detail, giving it a lifelike appearance that is almost indistinguishable from reality. It appears to be from a high-budget 4K movie, showcasing ultra-high-definition quality with impeccable resolution</td>
</tr>
<tr>
<td><strong>Parameters</strong></td>
<td>seed: 1, guidance: 7, edge: 1.0</td>
</tr>
</tbody>
</table>

### Scene 1b: Kitchen Island - Object Manipulation

This scene shows a robot performing precise object manipulation at a kitchen island, picking up and placing items. The examples demonstrate material variations (different fruit/objects) coordinated with kitchen style changes.

#### Input Video

<video width="600" controls>
  <source src="assets/kitchen_oranges_input.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

#### Parameters

```json
{
    "seed": 1,
    "prompt_path": "assets/prompt_robot.json",
    "output_dir": "outputs/robot",
    "video_path": "assets/kitchen_oranges_input.mp4",
    "guidance": 7,
    "edge": {
        "control_weight": 1
    }
}
```

#### Examples

<table>
<colgroup>
<col style="width: 20%;">
<col style="width: 80%;">
</colgroup>
<tbody>
<tr>
<td></td>
<td><video width="600" controls>  <source src="assets/kitchen_oranges_white.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>This scene depicts a photo realistic luxury kitchen with high end professional finishes and lighting. ALL The kitchen cabinets are all highly polished bright white panels with chrome accents and pulls. The kitchen counters are stainless steel. The kitchen walls and backsplash are all white subway tile. The kitchen contains an expensive double door stainless steel refrigerator, a stainless steel microwave, a stainless steel oven, a stainless steel coffee machine, a stainless steel toaster, a stainless steel stove top, a stainless steel sink, and stainless steel pots. In the center of the room is a kitchen island. This is also finished with highly polished bright white panels cabinets and an stainless steel countertop. In the middle of the island counter is a large glass bowl of oranges. Standing in the kitchen is a humanoid robot. The robot is made of orange polished plastic panels with chrome accents. The camera is fixed and steady. The robot is picking up two oranges from either side of a small glass plate, and placing them on the plate.</td>
</tr>
<tr>
<td><strong>Parameters</strong></td>
<td>seed: 1, guidance: 7, edge: 1.0</td>
</tr>
<tr>
<td></td>
<td><video width="600" controls>  <source src="assets/kitchen_oranges_red.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>This scene depicts a photo realistic luxury kitchen with high end professional finishes and lighting. ALL The kitchen cabinets are all highly polished bright red panels with stainless steel accents and pulls. The kitchen counters are stainless steel. The kitchen walls and backsplash are all white subway tile. The kitchen contains an expensive double door stainless steel refrigerator, a stainless steel microwave, a stainless steel oven, a stainless steel coffee machine, a stainless steel toaster, a stainless steel stove top, a stainless steel sink, and stainless steel pots. In the center of the room is a kitchen island. This is also finished with highly polished bright red panels cabinets and an stainless steel countertop. In the middle of the island counter is a large white bowl of eggs. Standing in the kitchen is a humanoid robot. The robot is made of white polished panels with black accents. The camera is fixed and steady. The robot is picking up two eggs from either side of a small white plate, and placing them on the plate.</td>
</tr>
<tr>
<td><strong>Parameters</strong></td>
<td>seed: 1, guidance: 7, edge: 1.0</td>
</tr>
<tr>
<td></td>
<td><video width="600" controls>  <source src="assets/kitchen_oranges_light_wood.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>This scene depicts a photo realistic luxury kitchen with high end professional finishes and lighting. ALL The kitchen cabinets are all light wood, with stainless steel accents and pulls. The kitchen counters, kitchen walls and backsplash are all expensive black veined marble. The kitchen contains an expensive double door stainless steel refrigerator, a stainless steel microwave, a stainless steel oven, a stainless steel coffee machine, a stainless steel toaster, a stainless steel stove top, a stainless steel sink, and stainless steel pots. In the center of the room is a kitchen island. This is also finished with light wood cabinets and an expensive black veined marble countertop. In the middle of the island counter is a large white bowl of lemons. Standing in the kitchen is a humanoid robot. The robot is made of stainless steel polished panels with chrome accents. The camera is fixed and steady. The robot is picking up two lemons from either side of a small white plate, and placing them on the plate.</td>
</tr>
<tr>
<td><strong>Parameters</strong></td>
<td>seed: 1, guidance: 7, edge: 1.0</td>
</tr>
<tr>
<td></td>
<td><video width="600" controls>  <source src="assets/kitchen_oranges_dark_wood.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>This scene depicts a photo realistic luxury kitchen with high end professional finishes and lighting. ALL The kitchen cabinets are all dark wood, with chrome accents and pulls. The kitchen counters, kitchen walls and backsplash are all expensive beige veined marble. The kitchen contains an expensive double door stainless steel refrigerator, a stainless steel microwave, a stainless steel oven, a stainless steel coffee machine, a stainless steel toaster, a stainless steel sink, and stainless steel pots. In the center of the room is a kitchen island. This is also finished with dark wood cabinets and an expensive beige veined marble countertop. In the middle of the island counter is a large white bowl of apples. Standing in the kitchen is a humanoid robot. The robot is made of gold polished reflective panels with shiny black accents. The camera is fixed and steady. The robot is picking up two apples from either side of a small white plate, and placing them on the plate.</td>
</tr>
<tr>
<td><strong>Parameters</strong></td>
<td>seed: 1, guidance: 7, edge: 1.0</td>
</tr>
<tr>
<td></td>
<td><video width="600" controls>  <source src="assets/kitchen_oranges.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>The video captures a stunning, photorealistic scene with remarkable attention to detail, giving it a lifelike appearance that is almost indistinguishable from reality. It appears to be from a high-budget 4K movie, showcasing ultra-high-definition quality with impeccable resolution.</td>
</tr>
<tr>
<td><strong>Parameters</strong></td>
<td>seed: 1, guidance: 7, edge: 1.0</td>
</tr>
</tbody>
</table>

### Scene 1c: Kitchen Refrigerator - Appliance Interaction

This scene demonstrates robot interaction with appliances, showing the robot opening a refrigerator. The examples maintain the lighting dynamics (fridge interior light) while varying kitchen aesthetics.

#### Input Video

<video width="600" controls>
  <source src="assets/kitchen_fridge_input.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

#### Parameters

```json
{
    "seed": 1,
    "prompt_path": "assets/prompt_robot.json",
    "output_dir": "outputs/robot",
    "video_path": "assets/kitchen_fridge_input.mp4",
    "guidance": 7,
    "edge": {
        "control_weight": 1
    }
}
```

#### Examples

<table>
<colgroup>
<col style="width: 20%;">
<col style="width: 80%;">
</colgroup>
<tbody>
<tr>
<td></td>
<td><video width="600" controls>  <source src="assets/kitchen_fridge_white.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>This scene depicts a photo realistic luxury kitchen with high end professional finishes and lighting. ALL The kitchen cabinets are all highly polished bright white panels with chrome accents and pulls. The kitchen counters are stainless steel. The kitchen walls and backsplash are all white subway tile. The kitchen contains an expensive double door stainless steel refrigerator, a stainless steel microwave, a stainless steel oven, a stainless steel coffee machine, a stainless steel toaster, a stainless steel stove top, a stainless steel sink, and stainless steel pots. In the center of the room is a kitchen island. This is also finished with highly polished bright white panels cabinets and an stainless steel countertop. In the middle of the island counter is a large glass bowl of oranges. Standing in the kitchen is a humanoid robot. The robot is made of orange polished plastic panels with chrome accents. The camera is fixed and steady. The robot is opening the fridge with his right hand and looking inside. The fridge light turns on and it very bright, showing the inside of the fridge filled with food and drink.</td>
</tr>
<tr>
<td><strong>Parameters</strong></td>
<td>seed: 1, guidance: 7, edge: 1.0</td>
</tr>
<tr>
<td></td>
<td><video width="600" controls>  <source src="assets/kitchen_fridge_red.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>This scene depicts a photo realistic luxury kitchen with high end professional finishes and lighting. ALL The kitchen cabinets are all highly polished bright red panels with stainless steel accents and pulls. The kitchen counters are stainless steel. The kitchen walls and backsplash are all white subway tile. The kitchen contains an expensive double door stainless steel refrigerator, a stainless steel microwave, a stainless steel oven, a stainless steel coffee machine, a stainless steel toaster, a stainless steel stove top, a stainless steel sink, and stainless steel pots. In the center of the room is a kitchen island. This is also finished with highly polished bright red panels cabinets and an stainless steel countertop. In the middle of the island counter is a large white bowl of eggs. Standing in the kitchen is a humanoid robot. The robot is made of white polished panels with black accents. The camera is fixed and steady. The robot is opening the fridge with his right hand and looking inside. The fridge light turns on and it very bright, showing the inside of the fridge filled with food and drink.</td>
</tr>
<tr>
<td><strong>Parameters</strong></td>
<td>seed: 1, guidance: 7, edge: 1.0</td>
</tr>
<tr>
<td></td>
<td><video width="600" controls>  <source src="assets/kitchen_fridge_light_wood.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>This scene depicts a photo realistic luxury kitchen with high end professional finishes and lighting. ALL The kitchen cabinets are all light wood, with stainless steel accents and pulls. The kitchen counters, kitchen walls and backsplash are all expensive black veined marble. The kitchen contains an expensive double door stainless steel refrigerator, a stainless steel microwave, a stainless steel oven, a stainless steel coffee machine, a stainless steel toaster, a stainless steel stove top, a stainless steel sink, and stainless steel pots. In the center of the room is a kitchen island. This is also finished with light wood cabinets and an expensive black veined marble countertop. In the middle of the island counter is a large white bowl of lemons. Standing in the kitchen is a humanoid robot. The robot is made of stainless steel polished panels with chrome accents. The camera is fixed and steady. The robot is opening the fridge with his right hand and looking inside. The fridge light turns on and it very bright, showing the inside of the fridge filled with food and drink.</td>
</tr>
<tr>
<td><strong>Parameters</strong></td>
<td>seed: 1, guidance: 7, edge: 1.0</td>
</tr>
<tr>
<td></td>
<td><video width="600" controls>  <source src="assets/kitchen_fridge_dark_wood.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>This scene depicts a photo realistic luxury kitchen with high end professional finishes and lighting. ALL The kitchen cabinets are all dark wood, with chrome accents and pulls. The kitchen counters, kitchen walls and backsplash are all expensive beige veined marble. The kitchen contains an expensive double door stainless steel refrigerator, a stainless steel microwave, a stainless steel oven, a stainless steel coffee machine, a stainless steel toaster, a stainless steel sink, and stainless steel pots. In the center of the room is a kitchen island. This is also finished with dark wood cabinets and an expensive beige veined marble countertop. In the middle of the island counter is a large white bowl of apples. Standing in the kitchen is a humanoid robot. The robot is made of gold polished reflective panels with shiny black accents. The camera is fixed and steady. The robot is opening the fridge with his right hand and looking inside. The fridge light turns on and it very bright, showing the inside of the fridge filled with food and drink.</td>
</tr>
<tr>
<td><strong>Parameters</strong></td>
<td>seed: 1, guidance: 7, edge: 1.0</td>
</tr>
<tr>
<td></td>
<td><video width="600" controls>  <source src="assets/kitchen_fridge.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>The video captures a stunning, photorealistic scene with remarkable attention to detail, giving it a lifelike appearance that is almost indistinguishable from reality. It appears to be from a high-budget 4K movie, showcasing ultra-high-definition quality with impeccable resolution.</td>
</tr>
<tr>
<td><strong>Parameters</strong></td>
<td>seed: 1, guidance: 7, edge: 1.0</td>
</tr>
</tbody>
</table>

## Example 2: Multi-Control with Custom Control Videos

These examples demonstrate advanced usage where you provide **custom pre-computed control videos** (depth, edge, segmentation) alongside the input video. Multi-control gives you fine-grained control over different aspects of the transformation:

- **depth**: Controls 3D spatial relationships and perspective
- **edge**: Maintains structural boundaries and object shapes
- **seg**: Enables semantic-level changes and object replacement
- **vis**: Preserves lighting and camera properties (set to 0 in this example)

**When to use multi-control**: Use this approach when you need precise control over the transformation by pre-generating and fine-tuning specific control signals, especially for complex scene manipulations or when edge-only control is insufficient.

### Scene 2a

### Input Video

<video width="600" controls>
  <source src="assets/kitchen2_cg.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Parameters

```json
{
    "seed": 2000,
    "prompt_path": "assets/prompt_kitchen2.json",
    "video_path": "assets/kitchen2_cg.mp4",
    "guidance": 2,
    "depth": {
        "control_path": "assets/kitchen2_depth.mp4",
        "control_weight": 0.6
    },
    "edge": {
        "control_path": "assets/kitchen2_edge.mp4",
        "control_weight": 0.2
    },
    "seg": {
        "control_path": "assets/kitchen2_seg.mp4",
        "control_weight": 0.4
    },
    "vis": {
        "control_weight": 0
    }
}
```

### Control Videos

#### Depth Control

<video width="600" controls>
  <source src="assets/kitchen2_depth.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

#### Edge Control

<video width="600" controls>
  <source src="assets/kitchen2_edge.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

#### Segmentation Control

<video width="600" controls>
  <source src="assets/kitchen2_seg.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Output Video

<video width="600" controls>
  <source src="assets/kitchen2_output.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Scene 2b

### Input Video

<video width="600" controls>
  <source src="assets/robotic_arm_input.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Parameters

```json
{
    "name": "robot_multicontrol",
    "video_path": "assets/kitchen2_cg.mp4",
    "guidance": 3,
    "depth": {
        "control_path": "assets/robotic_arm_input_depth.mp4",
        "control_weight": 0.6
    },
    "edge": {
        "control_path": "assets/robotic_arm_input_edge.mp4",
        "control_weight": 1
    },
    "seg": {
        "control_path": "assets/robotic_arm_input_seg.mp4",
        "control_weight": 0.4
    },
    "vis": {
       "control_path": "assets/robotic_arm_input_vis.mp4",
        "control_weight": 0
    },
    "prompt": "The video features two robotic arms with brushed matte black bodies, and contrasting black joints, manipulating a small red glass cube. They are positioned on a plastic table, with minimalistic office in the background, illuminated by artificial white light."
}
```

### Control Videos

#### Depth Control

<video width="600" controls>
  <source src="assets/robotic_arm_input_depth.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

#### Edge Control

<video width="600" controls>
  <source src="assets/robotic_arm_input_edge.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

#### Segmentation Control

<video width="600" controls>
  <source src="assets/robotic_arm_input_seg.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

#### Vis Control

<video width="600" controls>
  <source src="assets/robotic_arm_input_vis.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

#### Examples

<table>
<colgroup>
<col style="width: 20%;">
<col style="width: 80%;">
</colgroup>
<tbody>

<tr>
<td><strong>Output Video</strong></td>
<td><video width="600" controls>  <source src="assets/robotic_arm_1.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>The video features two robotic arms with brushed matte black bodies, and contrasting black joints, manipulating a small red glass cube. They are positioned on a plastic table, with minimalistic office in the background, illuminated by artificial white light.</td>
</tr>

<tr>
<td><strong>Output Video</strong></td>
<td><video width="600" controls>  <source src="assets/robotic_arm_2.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>The video features two robotic arms with brushed bronze bodies, and contrasting yellow joints, manipulating a small purple plastic cube. They are positioned on a granite table, with urban rooftop in the background, illuminated by natural light</td>
</tr>

<tr>
<td><strong>Output Video</strong></td>
<td><video width="600" controls>  <source src="assets/robotic_arm_3.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>The video features two robotic arms with matte matte black bodies, and contrasting blue joints, manipulating a small white plastic cube. They are positioned on a marble table, with industrial warehouse in the background, illuminated by colored ambient light (blue).</td>
</tr>

<tr>
<td><strong>Output Video</strong></td>
<td><video width="600" controls>  <source src="assets/robotic_arm_4.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>The video features two robotic arms with matte white bodies, and contrasting black joints, manipulating a small green glass cube. They are positioned on a marble table, with closed room in the background, illuminated by artificial white light.</td>
</tr>
</tbody>
</table>

## Quality Enhancements: Transfer 2.5 vs Transfer 1

Compared to Cosmos Transfer 1, Cosmos Transfer 2.5 offers significant improvements in both **video quality** and **inference speed**. The examples below show side-by-side comparisons where each video transitions between Transfer 1 results and Transfer 2.5 results, illustrating the quality improvements achieved in the latest version.

### Example A

<video width="800" controls>
  <source src="assets/robot1_t1_t2.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Example B

<video width="800" controls>
  <source src="assets/robot2_t1_t2.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
