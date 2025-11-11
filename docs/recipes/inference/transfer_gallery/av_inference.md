# Conversion from simulation to real for AV with Cosmos Transfer 2.5

> **Authors:**  [Nikolay Matveiev]() • [Jinwei Gu](https://www.linkedin.com/in/jinweigu/)

> **Organization:** NVIDIA

## Overview

| **Model** | **Workload** | **Use Case** |
|-----------|--------------|--------------|
| Cosmos Transfer 2.5| Inference | Sim to real using different controlnets |

## Example 1

- **Multi control**: The model uses different control nets, each with different control weights to produce an output. This gives the user more control over how the output would look like.

**Inference command**

```Json
{
    // Update the paramater values for control weights, seed, guidance in below json file as per the values given in the column named "Parameter values" in table below
    "seed": 5000,
    "prompt_path": "assets/prompt_av.json",                 // Update the prompt in the json file as per the input selected from below table
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

**Input and Output**

<style>
table td {
  vertical-align: top;
}
</style>

| **Input prompt** | **Parameter values** | **Input Video** | **Output Video** |
|:---|:---|:---|:---|
|The video is a driving scene through a modern urban environment, likely captured from a dashcam or a similar fixed camera setup inside a vehicle. The scene unfolds on a wide, multi-lane road flanked by tall, modern buildings with glass facades. The road is relatively empty, with only a few cars visible, including a black car directly ahead of the camera, maintaining a steady pace. The camera remains static, providing a consistent view of the road and surroundings as the vehicle moves forward.On the left side of the road, there are several trees lining the sidewalk, providing a touch of greenery amidst the urban setting. Pedestrians are visible on the sidewalks, some walking leisurely, while others stand near the buildings. The buildings are a mix of architectural styles, with some featuring large glass windows and others having more traditional concrete exteriors. A few commercial signs and logos are visible on the buildings, indicating the presence of businesses and offices.Traffic cones are placed on the road ahead, suggesting some form of roadwork or lane closure, guiding the vehicles to merge or change lanes. The road markings are clear, with white arrows indicating the direction of travel. Throughout the video, the vehicle maintains a steady speed, and the camera captures the gradual approach towards the intersection, where the road splits into different directions. The overall atmosphere is calm and orderly, typical of a city during non-peak hours.  heavy rain, wet road with puddles | seed: 4000 <br> guidance: 3 <br><br> Control weights: <br> depth: 0.5 <br> edge: 0.1 <br> seg: 0.35 <br> vis: 0.0 | <video width="4096" controls>  <source src="assets/av_car_input.mp4" type="video/mp4">  Your browser does not support the video tag.</video> | <video width="4096" controls>  <source src="assets/av_car_output_1.mp4" type="video/mp4">  Your browser does not support the video tag.</video> |
|The video is a driving scene through a modern urban environment, likely captured from a dashcam or a similar fixed camera setup inside a vehicle. The scene unfolds on a wide, multi-lane road flanked by tall, modern buildings with glass facades. The road is relatively empty, with only a few cars visible, including a black car directly ahead of the camera, maintaining a steady pace. The camera remains static, providing a consistent view of the road and surroundings as the vehicle moves forward.On the left side of the road, there are several trees lining the sidewalk, providing a touch of greenery amidst the urban setting. Pedestrians are visible on the sidewalks, some walking leisurely, while others stand near the buildings. The buildings are a mix of architectural styles, with some featuring large glass windows and others having more traditional concrete exteriors. A few commercial signs and logos are visible on the buildings, indicating the presence of businesses and offices.Traffic cones are placed on the road ahead, suggesting some form of roadwork or lane closure, guiding the vehicles to merge or change lanes. The road markings are clear, with white arrows indicating the direction of travel. Throughout the video, the vehicle maintains a steady speed, and the camera captures the gradual approach towards the intersection, where the road splits into different directions. The overall atmosphere is calm and orderly, typical of a city during non-peak hours.  night time, bright street lamps and colorful neon lights on buildings | seed: 4000 <br> guidance: 7 <br><br> Control weights: <br> depth: 0.5 <br> edge: 0.1 <br> seg: 0.35 <br> vis: 0.0 | <video width="4096" controls>  <source src="assets/av_car_input.mp4" type="video/mp4">  Your browser does not support the video tag.</video> | <video width="4096" controls>  <source src="assets/av_car_output_2.mp4" type="video/mp4">  Your browser does not support the video tag.</video> |
|Dashcam video, driving through a modern urban environment, sunset with beautiful clouds | seed: 5000 <br> guidance: 3 <br><br> Control weights: <br> depth: 0.35 <br> edge: 0.1 <br> seg: 0.35 <br> vis: 0.0 | <video width="4096" controls>  <source src="assets/av_car_input.mp4" type="video/mp4">  Your browser does not support the video tag.</video> | <video width="4096" controls>  <source src="assets/av_car_output_3.mp4" type="video/mp4">  Your browser does not support the video tag.</video> |
|Dashcam video, driving through a modern urban environment, heavy rain, wet road with puddles | seed: 5000 <br> guidance: 3 <br><br> Control weights: <br> depth: 0.35 <br> edge: 0.1 <br> seg: 0.35 <br> vis: 0.0 | <video width="4096" controls>  <source src="assets/av_car_input.mp4" type="video/mp4">  Your browser does not support the video tag.</video> | <video width="4096" controls>  <source src="assets/av_car_output_4.mp4" type="video/mp4">  Your browser does not support the video tag.</video> |
|Dashcam video, driving through a modern urban environment, winter with heavy snow storm, trees and sidewalks covered in snow | seed: 8001 <br> guidance: 3 <br><br> Control weights: <br> depth: 0.35 <br> edge: 0.15 <br> seg: 0.35 <br> vis: 0.15 | <video width="4096" controls>  <source src="assets/av_car_input.mp4" type="video/mp4">  Your browser does not support the video tag.</video> | <video width="4096" controls>  <source src="assets/av_car_output_5.mp4" type="video/mp4">  Your browser does not support the video tag.</video> |
|Dashcam video, driving through a modern urban environment, heavy rain, wet road with puddles | seed: 5000 <br> guidance: 6 <br><br> Control weights: <br> depth: 0.6 <br> edge: 0.1 <br> seg: 0.4 <br> vis: 0.1 | <video width="4096" controls>  <source src="assets/av_car_input.mp4" type="video/mp4">  Your browser does not support the video tag.</video> | <video width="4096" controls>  <source src="assets/av_car_output_6.mp4" type="video/mp4">  Your browser does not support the video tag.</video> |
|Dashcam video, driving through a modern urban environment, sunset with beautiful clouds | seed: 7000 <br> guidance: 3 <br><br> Control weights: <br> depth: 0.35 <br> edge: 0.1 <br> seg: 0.35 <br> vis: 0.05 | <video width="4096" controls>  <source src="assets/av_car_input.mp4" type="video/mp4">  Your browser does not support the video tag.</video> | <video width="4096" controls>  <source src="assets/av_car_output_7.mp4" type="video/mp4">  Your browser does not support the video tag.</video> |
|Dashcam video, driving through a modern urban environment, winter with heavy snow storm, trees and sidewalks covered in snow | seed: 7001 <br> guidance: 3 <br><br> Control weights: <br> depth: 0.35 <br> edge: 0.1 <br> seg: 0.35 <br> vis: 0.05 | <video width="4096" controls>  <source src="assets/av_car_input.mp4" type="video/mp4">  Your browser does not support the video tag.</video> | <video width="4096" controls>  <source src="assets/av_car_output_8.mp4" type="video/mp4">  Your browser does not support the video tag.</video> |
|Dashcam video, driving through a modern urban environment, twilight or early morning, partly cloudy | seed: 5000 <br> guidance: 3 <br><br> Control weights: <br> depth: 0.4 <br> edge: 0.1 <br> seg: 0.5 <br> vis: 0.1 | <video width="4096" controls>  <source src="assets/av_car_input.mp4" type="video/mp4">  Your browser does not support the video tag.</video> | <video width="4096" controls>  <source src="assets/av_car_output_9.mp4" type="video/mp4">  Your browser does not support the video tag.</video> |
|Dashcam video, driving through a modern urban environment, night time, bright street lamps lighting up the fog | seed: 5000 <br> guidance: 3 <br><br> Control weights: <br> depth: 0.4 <br> edge: 0.1 <br> seg: 0.5 <br> vis: 0.1 | <video width="4096" controls>  <source src="assets/av_car_input.mp4" type="video/mp4">  Your browser does not support the video tag.</video> | <video width="4096" controls>  <source src="assets/av_car_output_10.mp4" type="video/mp4">  Your browser does not support the video tag.</video> |

## Examples showing quality enhancements in Transfer 2.5 vs Transfer 1

| **Example A** | **Example B** | **Example C** | **Example D** |
|-----------|--------------|-----------|--------------|
|<video width="1024" controls>  <source src="assets/av1_t1_t2.mp4" type="video/mp4">  Your browser does not support the video tag.</video> |<video width="1024" controls>  <source src="assets/av2_t1_t2.mp4" type="video/mp4">  Your browser does not support the video tag.</video>|<video width="1024" controls>  <source src="assets/av3_t1_t2.mp4" type="video/mp4">  Your browser does not support the video tag.</video>|<video width="1024" controls>  <source src="assets/av4_t1_t2.mp4" type="video/mp4">  Your browser does not support the video tag.</video>
