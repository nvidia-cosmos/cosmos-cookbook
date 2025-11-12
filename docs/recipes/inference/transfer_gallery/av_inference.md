# Autonomous Vehicle Domain Adaptation Gallery

> **Authors:**  [Raju Wagwani](https://www.linkedin.com/in/raju-wagwani-a4746027/) • [Nikolay Matveiev](https://www.linkedin.com/in/nikolay-markovskiy-273443b/) • [Jinwei Gu](https://www.linkedin.com/in/jinweigu/)

> **Organization:** NVIDIA

## Overview

This page showcases a collection of results generated using Cosmos Transfer 2.5 for autonomous vehicle (AV) applications. The examples demonstrate how to transform real-world or simulation-based driving videos across various environmental conditions such as different weather, lighting, and time of day. These results are intended to serve as inspiration for users exploring how to leverage the model for domain adaptation and synthetic data augmentation in autonomous driving use cases.

## Driving Scene 1

- **Multi control**: The model uses different controls, each with different control weights to produce an output. This gives the user more control over how the output would look like.
  - **depth**: Maintains 3D realism and spatial consistency
  - **edge**: Preserves original structure, shape, and layout
  - **seg**: Enables structural changes and semantic replacement
  - **vis**: Preserves background, lighting, and overall visual appearance

For detailed explanations of control modalities, see [Control Modalities Overview](../../../core_concepts/control_modalities/overview.md).

### Input Video

<video width="600" controls>
  <source src="assets/av_car_input.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Base Parameters

```json
{
    // Update the paramater values for control weights, seed, guidance in below json file
    "seed": 5000,
    "prompt_path": "assets/prompt_av.json",           // Update the prompt in the json file accordingly
    "video_path": "assets/av_car_input.mp4",
    "guidance": 3,
    "depth": {
        "control_weight": 0.4
    },
    "edge": {
        "control_weight": 0.1
    },
    "seg": {
        "control_weight": 0.5
    },
    "vis": {
        "control_weight": 0.1
    }
}
```

### Examples

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

<table>
<colgroup>
<col style="width: 20%;">
<col style="width: 80%;">
</colgroup>
<tbody>
<tr>
<td></td>
<td><video width="600" controls>  <source src="assets/av_car_output_1.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>The video is a driving scene through a modern urban environment, likely captured from a dashcam or a similar fixed camera setup inside a vehicle. The scene unfolds on a wide, multi-lane road flanked by tall, modern buildings with glass facades. The road is relatively empty, with only a few cars visible, including a black car directly ahead of the camera, maintaining a steady pace. The camera remains static, providing a consistent view of the road and surroundings as the vehicle moves forward.On the left side of the road, there are several trees lining the sidewalk, providing a touch of greenery amidst the urban setting. Pedestrians are visible on the sidewalks, some walking leisurely, while others stand near the buildings. The buildings are a mix of architectural styles, with some featuring large glass windows and others having more traditional concrete exteriors. A few commercial signs and logos are visible on the buildings, indicating the presence of businesses and offices.Traffic cones are placed on the road ahead, suggesting some form of roadwork or lane closure, guiding the vehicles to merge or change lanes. The road markings are clear, with white arrows indicating the direction of travel. Throughout the video, the vehicle maintains a steady speed, and the camera captures the gradual approach towards the intersection, where the road splits into different directions. The overall atmosphere is calm and orderly, typical of a city during non-peak hours.  heavy rain, wet road with puddles</td>
</tr>
<tr>
<td><strong>Parameters</strong></td>
<td>seed: 4000, guidance: 3, depth: 0.5, edge: 0.1, seg: 0.35, vis: 0.0</td>
</tr>
<tr>
<td></td>
<td><video width="600" controls>  <source src="assets/av_car_output_2.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>The video is a driving scene through a modern urban environment, likely captured from a dashcam or a similar fixed camera setup inside a vehicle. The scene unfolds on a wide, multi-lane road flanked by tall, modern buildings with glass facades. The road is relatively empty, with only a few cars visible, including a black car directly ahead of the camera, maintaining a steady pace. The camera remains static, providing a consistent view of the road and surroundings as the vehicle moves forward.On the left side of the road, there are several trees lining the sidewalk, providing a touch of greenery amidst the urban setting. Pedestrians are visible on the sidewalks, some walking leisurely, while others stand near the buildings. The buildings are a mix of architectural styles, with some featuring large glass windows and others having more traditional concrete exteriors. A few commercial signs and logos are visible on the buildings, indicating the presence of businesses and offices.Traffic cones are placed on the road ahead, suggesting some form of roadwork or lane closure, guiding the vehicles to merge or change lanes. The road markings are clear, with white arrows indicating the direction of travel. Throughout the video, the vehicle maintains a steady speed, and the camera captures the gradual approach towards the intersection, where the road splits into different directions. The overall atmosphere is calm and orderly, typical of a city during non-peak hours.  night time, bright street lamps and colorful neon lights on buildings</td>
</tr>
<tr>
<td><strong>Parameters</strong></td>
<td>seed: 4000, guidance: 7, depth: 0.5, edge: 0.1, seg: 0.35, vis: 0.0</td>
</tr>
<tr>
<td></td>
<td><video width="600" controls>  <source src="assets/av_car_output_3.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>Dashcam video, driving through a modern urban environment, sunset with beautiful clouds</td>
</tr>
<tr>
<td><strong>Parameters</strong></td>
<td>seed: 5000, guidance: 3, depth: 0.35, edge: 0.1, seg: 0.35, vis: 0.0</td>
</tr>
<tr>
<td></td>
<td><video width="600" controls>  <source src="assets/av_car_output_4.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>Dashcam video, driving through a modern urban environment, heavy rain, wet road with puddles</td>
</tr>
<tr>
<td><strong>Parameters</strong></td>
<td>seed: 5000, guidance: 3, depth: 0.35, edge: 0.1, seg: 0.35, vis: 0.0</td>
</tr>
<tr>
<td></td>
<td><video width="600" controls>  <source src="assets/av_car_output_5.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>Dashcam video, driving through a modern urban environment, winter with heavy snow storm, trees and sidewalks covered in snow</td>
</tr>
<tr>
<td><strong>Parameters</strong></td>
<td>seed: 8001, guidance: 3, depth: 0.35, edge: 0.15, seg: 0.35, vis: 0.15</td>
</tr>
<tr>
<td></td>
<td><video width="600" controls>  <source src="assets/av_car_output_6.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>Dashcam video, driving through a modern urban environment, heavy rain, wet road with puddles</td>
</tr>
<tr>
<td><strong>Parameters</strong></td>
<td>seed: 5000, guidance: 6, depth: 0.6, edge: 0.1, seg: 0.4, vis: 0.1</td>
</tr>
<tr>
<td></td>
<td><video width="600" controls>  <source src="assets/av_car_output_7.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>Dashcam video, driving through a modern urban environment, sunset with beautiful clouds</td>
</tr>
<tr>
<td><strong>Parameters</strong></td>
<td>seed: 7000, guidance: 3, depth: 0.35, edge: 0.1, seg: 0.35, vis: 0.05</td>
</tr>
<tr>
<td></td>
<td><video width="600" controls>  <source src="assets/av_car_output_8.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>Dashcam video, driving through a modern urban environment, winter with heavy snow storm, trees and sidewalks covered in snow</td>
</tr>
<tr>
<td><strong>Parameters</strong></td>
<td>seed: 7001, guidance: 3, depth: 0.35, edge: 0.1, seg: 0.35, vis: 0.05</td>
</tr>
<tr>
<td></td>
<td><video width="600" controls>  <source src="assets/av_car_output_9.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>Dashcam video, driving through a modern urban environment, twilight or early morning, partly cloudy</td>
</tr>
<tr>
<td><strong>Parameters</strong></td>
<td>seed: 5000, guidance: 3, depth: 0.4, edge: 0.1, seg: 0.5, vis: 0.1</td>
</tr>
<tr>
<td></td>
<td><video width="600" controls>  <source src="assets/av_car_output_10.mp4" type="video/mp4">  Your browser does not support the video tag.</video></td>
</tr>
<tr>
<td><strong>Input Prompt</strong></td>
<td>Dashcam video, driving through a modern urban environment, night time, bright street lamps lighting up the fog</td>
</tr>
<tr>
<td><strong>Parameters</strong></td>
<td>seed: 5000, guidance: 3, depth: 0.4, edge: 0.1, seg: 0.5, vis: 0.1</td>
</tr>
</tbody>
</table>

## Quality Enhancements: Transfer 2.5 vs Transfer 1

Compared to Cosmos Transfer 1, Cosmos Transfer 2.5 offers significant improvements in both **video quality** and **inference speed**. The examples below show side-by-side comparisons where each video transitions between Transfer 1 results and Transfer 2.5 results, illustrating the quality improvements achieved in the latest version.

### Example A

<video width="800" controls>
  <source src="assets/av1_t1_t2.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Example B

<video width="800" controls>
  <source src="assets/av2_t1_t2.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Example C

<video width="800" controls>
  <source src="assets/av3_t1_t2.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Example D

<video width="800" controls>
  <source src="assets/av4_t1_t2.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
