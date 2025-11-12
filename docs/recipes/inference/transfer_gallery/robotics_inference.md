# Robotics Sim-to-Real Transfer with Cosmos Transfer 2.5

> **Authors:**  [Raju Wagwani](https://www.linkedin.com/in/raju-wagwani-a4746027/) • [Jathavan Sriram](https://www.linkedin.com/in/jathavansriram) • [Richard Yarlett](https://www.linkedin.com/in/richardyarlett/) • [Jinwei Gu](https://www.linkedin.com/in/jinweigu/)

> **Organization:** NVIDIA

## Overview

This page showcases a collection of results generated using Cosmos Transfer 2.5 for robotics applications, specifically focusing on sim-to-real transfer scenarios. The examples demonstrate how to transform synthetic robotic simulation videos into realistic environments, enabling domain adaptation for robotic training and validation. These results are intended to serve as inspiration for users exploring how to leverage the model for bridging the gap between simulated and real-world robotic environments, similar to the autonomous vehicle domain adaptation shown in the [AV inference gallery](av_inference.md).


## Example 1

- **Edge control**: The model extracts the edges from the input video and creates an edge video from the user input. It then uses this control video along with the prompt/text to generate the final output.

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

### Scene 1a: Kitchen Stove

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

### Scene 1b: Kitchen Island with Oranges

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

### Scene 1c: Kitchen Refrigerator

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

## Example 2

- **Multi control**: The model uses different control nets, each with different control weights to produce an output. This gives the user more control over how the output would look like.

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

**Depth Control**

<video width="600" controls>
  <source src="assets/kitchen2_depth.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

**Edge Control**

<video width="600" controls>
  <source src="assets/kitchen2_edge.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

**Segmentation Control**

<video width="600" controls>
  <source src="assets/kitchen2_seg.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Output Video

<video width="600" controls>
  <source src="assets/kitchen2_output.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Quality Enhancements: Transfer 2.5 vs Transfer 1

Compared to Cosmos Transfer 1, Cosmos Transfer 2.5 offers significant improvements in both **video quality** and **inference speed**. The examples below show side-by-side comparisons where each video transitions between Transfer 1 results and Transfer 2.5 results, illustrating the quality improvements achieved in the latest version.

**Example A**

<video width="800" controls>
  <source src="assets/robot1_t1_t2.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

**Example B**

<video width="800" controls>
  <source src="assets/robot2_t1_t2.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
