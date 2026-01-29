# Prompt Guide Cosmos Reason 2

> **Authors:** [Tsung-Yi Lin](https://www.linkedin.com/in/tsung-yi-lin-48a4b541/) • [Xuan Li](https://www.linkedin.com/in/xuanli1030/) • [Diego Garzon](https://www.linkedin.com/in/dgarzon-engineering-art/)
> **Organization:** NVIDIA

| **Model** | **Workload** | **Use Case** |
|-----------|--------------|--------------|
| [Cosmos Reason 2](https://github.com/nvidia-cosmos/cosmos-reason2) | Inference | Prompt Guide|
## Overview

Cosmos Reason 2 is an open, customizable, reasoning vision language model designed to operate across vision, robotics, autonomous driving, and physical-world understanding tasks. While many workflows in the Cosmos Cookbook focus on *what* to run, such as recipes and end-to-end pipelines, this document focuses on *how* to interact with the model effectively through prompting. It serves as a conceptual reference for prompting Cosmos Reason 2, consolidating best practices, common patterns, and illustrative examples that explain how system prompts, user instructions, sampling parameters, and multimodal message structure influence model behavior. Rather than prescribing a fixed workflow, this guide helps users build a correct mental model of how Cosmos Reason 2 interprets inputs and produces structured outputs across different task types. It is intended for developers, researchers, and practitioners who are designing new recipes, evaluating model outputs, or integrating Cosmos Reason 2 into real-world systems, and it is not a benchmark, API reference, or end-to-end workflow. By centralizing these prompting conventions in one place, this document provides a shared vocabulary that allows the rest of the Cosmos Cookbook to remain consistent, easier to extend, and easier to maintain as new domains and workflows are added.

## Key Takeaways

- **Prompting is foundational**: Prompt structure, sampling parameters, and multimodal message ordering have a direct impact on Cosmos Reason 2 behavior across all tasks.
- **Minimal system prompts work best**: Cosmos Reason 2 generally performs well with lightweight system prompts, relying primarily on clear user instructions and structured output requests.
- **Media-first message ordering matters**: When using images or video, media inputs should appear before user text to align with the model’s training conventions.
- **Sampling controls trade-offs**: Adjusting parameters such as temperature, top-p, and presence penalty allows users to balance determinism, exploration, and verbosity depending on the task.
- **Structured outputs are prompt-driven**: Temporal localization, JSON outputs, trajectories, and reasoning-style responses are best achieved by explicitly requesting structure in the prompt.
- **Concepts over copy-paste**: The examples in this guide are illustrative; they are intended to teach prompting patterns rather than serve as fixed or deterministic templates.

## How to Use This Guide

This document is organized to support both first-time users of Cosmos Reason 2 and experienced practitioners looking for specific prompting patterns.

If you are new to Cosmos Reason 2, we recommend reading the sections in the following order:

1. **Message Structure and Media Ordering** – Understand how system prompts, user prompts, and multimodal inputs are interpreted by the model.
2. **Sampling Parameters** – Learn how to tune determinism, exploration, and verbosity based on task requirements.
3. **Structured Output Patterns** – See how to request captions, timestamps, JSON outputs, and action predictions.
4. **Task-Specific Examples** – Explore how these prompting patterns apply to embodied reasoning, autonomous driving, grounding, safety, and evaluation tasks.
## Sampling Parameters ([Following Qwen3 recommended params](https://github.com/QwenLM/Qwen3-VL?tab=readme-ov-file#evaluation-reproduction))


| **Parameter** | **Default** | **With reasoning** | **Practical effect** |
|--------------|------------:|-------------------:|----------------------|
| top_p | 0.8 | 0.95 | Smaller vs larger nucleus (tighter vs broader token pool) |
| top_k | 20 | 20 | Same cap on candidate tokens |
| repetition_penalty | 1.0 | 1.0 | Same (neutral repetition handling) |
| presence_penalty | 1.5 | 0.0 | Strong novelty push vs none |
| temperature | 0.7 | 0.6 | Slightly more varied vs more deterministic |


## System Prompt

In Cosmos Reason 2, we are more aligned with Qwen's use of system 
prompt. In the examples given, we simply use 'You are a helpful
assistant.' You may note this is different from Reason 1, where the
system prompt was heavily used.

```text
You are a helpful assistant.
```
## Turn on/off reasoning

**This is important.** If you want to use reasoning, you must append the
following instructions after any **user prompt** you've written**:**

```text
Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag.
```

## Media/Text Order

In the payload, make sure you follow the training we took, and ensure
that Media (video_url or image_url) is listed before the user text
prompt:

```text
messages=[
	{    "role": "system",
 	     "content": [{"type": "text", "text": "You are a helpful assistant."}]
},
        {
            "role": "user",
            "content": [
               {
                    "type": "video_url",
                    "video_url": "https://example.com/video.mp4"
                },
   {"type": "text", "text": "What’s happening in this video?"},
            ]
        }
    ]
```

## Caption and Temporal Localization

**Media Input**

1280x720 @ 30.00 fps

<video controls muted loop width="720">
  <source src="assets/caption.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

### Caption

**User Prompt**

```text
Caption the video in detail.
```

**Model Answer**

```text
<think>
Okay, let’s see. The user wants me to describe the video shown based on several paragraphs of detailed text. First, I need to parse all the information given and make sure I understand each scene.

Starting with the first part, there’s a red race car labeled “IndyCar” moving slowly towards the camera among other cars at a racetrack. The setting is sunny with green fields and empty stands. Then inside the cockpit, two drivers—one smiling, wearing sunglasses and a helmet, another focused on driving. They’re talking while navigating turns. Next shot shows an orange/red car drifting, creating smoke as it maneuvers through curves. Another angle from behind shows the same car drifting again, emphasizing speed and control.
</think>
The video begins with an exterior shot of several race cars lined up on a track, ready for action. The focus is particularly on a red car prominently displaying "Indy Lights" and other sponsor logos such as "ENEOS." This scene sets the stage for what appears to be a high-energy motorsport event.

Cutting to the interior of one of these vehicles, we see two individuals inside: a driver wearing gloves and a helmet, gripping the steering wheel tightly, and another person seated beside him, possibly a co-pilot or navigator. Both appear focused and engaged, suggesting they're preparing for intense driving maneuvers ahead.

As the camera shifts back outside, it captures dynamic footage from various angles showcasing the cars speeding along the racetrack at high speeds. One notable sequence features a red car executing sharp turns while emitting thick plumes of smoke from its tires due to drifting techniques. These scenes highlight the skillful control required by drivers during competitive racing conditions.

Inside yet again, this time focusing more closely on the occupants' expressions and actions within their cockpit environment. They seem exhilarated but also concentrated, emphasizing the adrenaline-fueled nature of professional racing events.

Finally, the perspective broadens outwards once more, offering aerial views that reveal expansive landscapes surrounding the track. We observe additional elements like grassy areas, water bodies, and distant structures, providing context about the location's layout and setting. Throughout all sequences, there’s a consistent theme of speed, precision, teamwork, and excitement inherent in modern motorsports competitions.
```

### Temporal localization

**User Prompt**

```text
Describe the video. Add timestamps in mm:ss format.

Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag and include the timestamps.
```

**Model Answer**

```text
<think>
Okay, let’s see. The user wants me to describe the video content shown across multiple frames with specific time stamps. First, I need to parse each frame description carefully and note all relevant details such as settings, actions, characters, vehicles, and any other elements like logos or text.
</think>

The video showcases various dynamic moments from a motorsport event, blending interior cockpit perspectives with exhilarating exterior action sequences. Here's a breakdown of the key scenes and timestamps:

Opening Scene (00:00-00:02)
An outdoor racetrack setting introduces a line-up of sleek racing cars poised for competition. A prominent red car featuring "Indy Lights" branding leads the pack, surrounded by other competitors. Spectators fill tiered seating in the background, emphasizing the scale of the event.

Cockpit Interaction (00:02-00:06)
Inside a moving vehicle, two individuals engage in conversation—one donning a dark helmet marked "CONN," sunglasses, and a harness, while the second sports a white helmet and vibrant blue gloves. Their dialogue appears lively, possibly strategizing or celebrating as the car accelerates forward.

Driver Focus (00:06-00:08)
From within another vehicle, a hand grips the steering wheel firmly, navigating a curved road. The dashboard prominently displays "ENEOS," suggesting sponsorship ties. Smoke trails hint at aggressive maneuvering or drift techniques.

Dynamic Driving Shots (00:08-00:13)
Exterior angles capture intense driving action:

Black/Orange Car: A driver navigates a sharp turn, showcasing precision control.
Pink/Red Drift Car: Emits thick tire smoke as it slides sideways around a bend, highlighting skilled drifting.
Yellow/Black Vehicle: Follows suit, executing controlled drifts with dramatic plumes of exhaust fumes.
Aerial Perspective (00:13-00:17)
Bird’s-eye views reveal multiple cars drifting simultaneously along a winding track bordered by lush greenery and safety barriers. The fluid motion emphasizes teamwork and coordination among participants.

Close-Up Action (00:17-00:19)
The pink/red car continues its drift, kicking up dense clouds of smoke against a backdrop of empty bleachers, underscoring the raw power and spectacle of the sport.

Spectator Viewpoint (00:19-00:21)
An elevated perspective highlights the track layout, including orange traffic cones guiding paths and blurred figures of spectators lining the course.

Interior Adjustments (00:21-00:23)
Inside yet another cockpit, a driver fastens their seatbelt securely before shifting gears, preparing for further action.

Drifting Mastery (00:23-00:25)
The iconic pink/red car executes precise drifts, leaving behind swirling patterns of smoke as it glides over asphalt curves.

High-Speed Thrills (00:25-00:28)
Accelerated footage captures the car hurtling down straightaways, tires screeching as it approaches turns with practiced ease.

Co-Pilot Dynamics (00:28-00:32)
In this intimate cockpit moment, the driver exchanges banter with their co-pilot, who gestures enthusiastically. Both wear full-face helmets and protective gear, reflecting professionalism and camaraderie.

Steering Precision (00:32-00:35)
Hands clad in gloves grip the wheel tightly during a tight corner, demonstrating expert handling amid roaring engines and shifting scenery outside.

Panoramic Circuit Tour (00:35-00:44)
Final aerial sweeps offer sweeping vistas of the expansive racetrack nestled amidst rolling hills, dotted with water bodies and sparse vegetation. Distant structures and faint smoke trails suggest ongoing activity beyond the immediate frame, concluding the immersive experience of speed, skill, and adrenaline-fueled motorsports.
```
### Temporal localization Json File

**User Prompt**

```text
Describe the video. Add timestamps in mm:ss format. Provide the result in json format with 'mm:ss.ff' format for time depiction for each event. Use keywords 'start', 'end' and 'caption' in the json output.
Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag and include the timestamps.
```


**Model Answer**


```json
[
  {
    "start": "00:00.10",
    "end": "00:02.80",
    "caption": "Cars line up on a racetrack; a red car moves forward slowly toward the camera."
  },
  {
    "start": "00:02.80",
    "end": "00:06.10",
    "caption": "Two helmeted individuals sit in a racecar\u2019s cockpit, conversing animatedly while one drives."
  },
  {
    "start": "00:06.10",
    "end": "00:08.10",
    "caption": "An interior view shows a vehicle accelerating quickly down a straightaway."
  },
  {
    "start": "00:08.10",
    "end": "00:09.80",
    "caption": "A driver sharpens a turn, causing the car to slide sideways and emit smoke."
  },
  {
    "start": "00:10.10",
    "end": "00:11.80",
    "caption": "A pink-red car drifts around a curve, producing white exhaust fumes."
  },
  {
    "start": "00:11.80",
    "end": "00:13.80",
    "caption": "A yellow-black car drifts, leaving dark tire marks on the pavement."
  },
  {
    "start": "00:13.80",
    "end": "00:17.10",
    "caption": "Aerial footage tracks a red car drifting precisely along a curving track."
  },
  {
    "start": "00:17.50",
    "end": "00:19.80",
    "caption": "A red car drifts over uneven terrain near safety barriers, releasing dense smoke."
  },
  {
    "start": "00:19.80",
    "end": "00:21.50",
    "caption": "A red car drifts tightly within a coned-off section of a paved circuit."
  },
  {
    "start": "00:21.50",
    "end": "00:23.10",
    "caption": "Two occupants converse urgently inside a cockpit, gesturing emphatically."
  },
  {
    "start": "00:23.50",
    "end": "00:25.10",
    "caption": "A red car drifts aggressively on a track bordered by spectators seated in tiered stands."
  },
  {
    "start": "00:25.10",
    "end": "00:28.90",
    "caption": "A red car drifts perilously close to a trackside barrier, kicking up dust and debris."
  },
  {
    "start": "00:28.90",
    "end": "00:32.20",
    "caption": "Two occupants engage in lively conversation inside a cockpit, displaying expressive gestures."
  },
  {
    "start": "00:32.20",
    "end": "00:35.20",
    "caption": "A driver executes a controlled turn, navigating a winding course."
  }
]

```

## Embodied Reasoning

**Media Input**

644x476 @ 4.00 fps

<video controls muted loop width="720">
  <source src="assets/agibot.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>


### Robotics Next Action Prediction

**User Prompt**

```text
What can be the next immediate action? 

Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag.

```

**Model Answer**

```text
<think\>Okay, let's see. The video is a lab or workshop with two robot arms. The camera is top-down, showing the table with a cardboard box in the center, open and filled with air column wraps. Robot Arm 1 ison the left, not moving. Robot Arm 2 is on the right, holding a white object and moving it upwards and away from the table.

The task is to predict the next immediate action. Since Robot Arm 2 is lifting the white object, the next step would likely be moving it to a specific location. The box is open and has air column wraps, which are used for packaging. The white object might be an item that needs to be placed into the box for packaging. Since the arm is moving it upwards and away, the next action would probably be positioning it over the box and then placing it inside. Alternatively, maybe placing it into another container or wrapping it with the air column wraps. But given the setup, the most logical next step after lifting is to move it towards the box and release it there. So the predicted action is placing the white object into the cardboard box.</think\>

Robot Arm 2 places the white object into the cardboard box.
```

### Goal Verification

**User Prompt**

```text
Has the robot successfully completed the intended packaging subtask? Explain why not.
Answer the question using the following format:

<think> Your reasoning. </think>

Write your final answer immediately after the </think> tag.
```

**Model Answer**

```text
<think>Okay, let's see. The user is asking if the robot has successfully completed the intended packaging subtask. The video shows that Robot Arm 2 is picking up a samrt charger. The task is about packaging, so maybe the next step is placing the cable into the box with the air column wraps. But the current action is just lifting the cable. The video does not show placing it into the box yet. So, even though the arm has the cable, the subtask of packaging isn't fully done until it's placed inside. Therefore, the answer would be no because the cable is still being lifted and not yet placed.
</think>
no
No, the robot has not yet completed the intended packaging subtask. While Robot Arm 2 successfully grasps and lifts the smart charger, the action shown does not include placing it into the cardboard box or securing it with the air column wraps. Packaging typically involves both retrieving the item and properly positioning it within protective materials, which has not been executed in the observed sequence.
```
### AV Next Action Prediction

**Media Input**

1280x720 @ 30.00 fps

<video controls muted loop width="720">
  <source src="assets/ego_v.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

**User Prompt**

```text
What's the next immediate action for the Ego vehicle?

Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag.
```

**Model Answer**

```text
<think>
Okay, let's break down this scenario step by step. The driver is stopped at an intersection with a stop sign in front of them. They've checked for oncoming traffic and pedestrians but didn't mention seeing any yet. Now they're slowly moving forward into the intersection.

First, since there's no visible traffic or pedestrians, the logical next step would be to proceed through the intersection. But wait, there are construction cones ahead blocking part of the road. So even though it's clear now, they need to stay cautious as they move forward. Maybe check mirrors again while accelerating gently to avoid sudden movements that could startle workers nearby. Also, keeping an eye out for any unexpected obstacles like debris from the construction zone.

Since they just started moving, maintaining low speed makes sense until they pass the coned-off area. No signs shown about turning left or right, so probably going straight. Double-checking surroundings before fully entering the intersection ensures safety. Yeah, proceeding cautiously through the intersection seems right here.
</think>

Proceed through the intersection cautiously while continuing to monitor for pedestrians, cyclists, and potential hazards (e.g., construction zones, uneven pavement) beyond the initial view. Maintain reduced speed due to ongoing street work and prepare to yield if unexpected obstacles arise.
```

## Video Critic

**Media Input**

1280x704 @ 16.00 fps

<video controls muted loop width="720">
  <source src="assets/sdg.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

**User Prompt**

```text
Approve or reject this generated video for inclusion in a dataset for physical world model ai training. It must perfectly adhere to physics, object permanence, and have no anomalies. Any issue or concern causes rejection.
Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag. Answer with Approve or Reject only.
```

**Model Answer**

```text
<think>
Okay, let's see. The user wants me to approve or reject the generated video based on whether it adheres to physics, object permanence, and has no anomalies.

First, I need to visualize the scenario shown. There's an orange robotic arm interacting with blocks: blue, yellow, red. The setup is on a wooden table against a light blue wall. The camera angle is high and static.

The key actions here involve the robotic arm manipulating the blue block. Initially, the blue block is held above the others. Then it tilts downward, causing instability. When released, another blue block falls off-screen. Later, the arm tries again but fails as all three blocks fall over together.

Now, considering physics principles. Gravity should act consistently. If the blue block is tilted beyond its center of gravity, it would naturally topple. That makes sense when it slides down and causes the stack to collapse. Object permanence means objects shouldn't vanish unexpectedly. However, one blue block disappears from view when released. That's a problem because it violates object permanence — it just vanishes without explanation. Also, if multiple blocks fall at once due to imbalance, that could be realistic depending on how they're stacked. But the disappearance of the second blue block seems like an anomaly since there's no indication where it went. Maybe it fell out of frame, but the video shows "another blue block also falls away," implying it's gone entirely. Without seeing it land or being moved elsewhere, that's inconsistent.

So the main issues are the disappearing block and possibly unrealistic stacking behavior leading to simultaneous collapse. Therefore, the video likely contains violations of these rules, so I should reject it.

</think>
Reject
```


## 2D Grounding

**Media Input**
<img src="assets/load.png" alt="Load" width="720" />

**User Prompt**

```text
Locate the bounding box of the load and determine if its size and weight of load within the forklift's limits. Estimate weights. Return all as json. Include json location, estimated weight of the load, and if it's in the limit. Return a json. 
```

**Model Answer**

```json
{
  "location": [
    {
      "box_2d": [180, 457, 639, 1000],
      "estimated_weight": "The palletized water bottles likely weigh between 1,000 to 1,500 pounds (450-680 kg), depending on their quantity and packaging."
    }
  ],
  "in_limit": true
}

```
**Visualization**

<img src="assets/load_bbox.png" alt="Load" width="720" />



### Action CoT

Note about trajectory coordinates
We now support point coordinates on images and videos. The pixel coordinate is normalized to 0-1000. The origin is left top corner. X is to the right (horizontal axis), and Y is to the bottom (vertical axis).  It doesn't matter what the aspect ratio is, each axis has independent 0-1000 normalization. E.g 1920x1080:

  1. Width 1000px / 2 = 500px is  500
  2. Height 600px / 2 = 300px is also 500

<figure>
  <img src="assets/trajectory.png" alt="Trajectory verification visualization" width="720">
  <figcaption>
    An example where the normalized values are converted to pixel values to verify trajectory.
  </figcaption>
</figure>

**Media Input**

<img src="assets/trajectory_clean.png" alt="Load" width="720" />

**User Prompt**

```text
You are given the task "Move the tape into the basket". Specify the 2D trajectory your end effector should follow in pixel space. Return the trajectory coordinates in JSON format like this: {"point_2d": [x, y], "label": "gripper trajectory"}.

Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag.
```

**Model Answer**

```text
<think> I will move my left gripper from its current position at [195, 630] to the blue painter’s tape located at [487, 721] so I can grasp it securely. After picking up the tape, I’ll lift it and carry it over to the gray woven basket positioned at [460, 410]. Then I’ll carefully lower the tape into the open space inside the basket at [469, 400], making sure it’s fully placed within the container. This sequence gives me precise control while avoiding any collisions with the table or other objects. </think>
```

```json
[
  {"point_2d": [195, 630], "label": "gripper trajectory"},
  {"point_2d": [485, 657], "label": "gripper trajectory"},
  {"point_2d": [496, 650], "label": "gripper trajectory"},
  {"point_2d": [496, 423], "label": "gripper trajectory"},
  {"point_2d": [496, 388], "label": "gripper trajectory"}
]

```

## Safety

**Media Input**

1842x1080 @ 19.99 fps

<video controls muted loop width="720">
  <source src="assets/warehouse_1.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

**User Prompt**

```text
were all the workers working safety equipment
Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag.
```

**Model Answer**

```text

<think>
Okay, let's see. The user is asking if all the workers in the video were wearing safety equipment.

First, I need to recall the details provided about each worker.

Worker 0: Enters from frame right with two boxes. They're wearing a yellow hard hat and reflective vest. So that's definitely safety gear. Then they drop one box and place another on shelf D. Still has the hard hat and vest when placing it. Then walks away down the aisle. No mention of removing any gear here. So Worker 0 is equipped with PPE (personal protective equipment).

Worker 2 enters later carrying a single box. He drops it but then picks it up again. His description mentions he's wearing casual attire—shirt and jeans. Wait, no mention of a hard hat or reflective vest for him. That's important. If he's not shown as having safety gear, maybe he wasn't required? Or perhaps the video didn't note it because he was handling only one item. But according to the given info, his clothing isn't specified beyond shirt and jeans. So unless there's an assumption that all workers have basic PPE, which might be standard, but the text doesn't say so explicitly. However, since Worker 0 had specific items like hard hat and vest, while Worker 2 does not, this suggests inconsistency. Therefore, not all workers had visible safety equipment based on the video.
</think>

No, not all workers were shown wearing safety equipment. While Worker 0 (the first person) is clearly depicted wearing a yellow hard hat and a reflective vest, Worker 2 (the second person entering later) is only noted to be wearing casual attire ("shirt and jeans") without explicit mention of safety gear like a helmet or vest. This indicates at least one worker did not visibly adhere to full safety protocols in the observed scene.
```

## Security

**Media Input**

1842x1080 @ 19.99 fps

<video controls muted loop width="720">
  <source src="assets/package.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

**User Prompt**

```text
Is the person allowed to pick up the packages?
Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag.
```

**Model Answer**

```text

<think>
Okay, let’s determine whether the person is allowed to pick up the packages. The video shows two cardboard boxes outside a door. A man enters, picks up the smaller box first, then returns for the larger one and leaves carrying both. A blue device labeled “Ring” is visible, suggesting the area may be monitored.

Taking unattended packages is generally not allowed unless the person is authorized by the owner. There’s no indication in the video that the packages belong to him or that they were abandoned or marked for pickup. Since the boxes were outside the home and not inside, and no permission or signage is shown, taking them could be considered unauthorized.

Based only on the visual information provided, there’s no evidence he has permission to take the packages.
</think>

Based on the visual information provided, there is no explicit evidence that the individual has been granted permission to remove the packages. Key observations include:

Ownership/Authorization:

The packages are placed near the entrance of what appears to be an occupied residence (evidenced by the visible interior through windows).
No labels, notices, or markings indicate the packages were intended to be picked up by anyone other than their rightful recipient(s).
Contextual Clues:

The presence of a "Ring"-branded device (likely a security system) suggests the property is monitored, reinforcing the likelihood that unauthorized removal would violate privacy or legal norms.
The individual's casual attire and lack of identifiers (e.g., uniforms, delivery badges) imply he may not be an employee or agent permitted to handle deliveries.
Behavioral Cues:

While the act itself involves physical effort (lifting both boxes), this does not inherently justify legality. Theft laws typically prohibit removing unattended items without consent.
Conclusion: Without clear indicators of ownership transfer (e.g., labels, instructions, or verbal/written approval), it is reasonable to infer the individual acted unlawfully or improperly by removing the packages. Legal permissibility hinges on context not fully revealed in the clip.
```


## 3 OpenAI API Samples for NIM Docs

**Temporal Localization**

```python
client = openai.OpenAI(api_key=token, base_url=url)
model = client.models.list().data[0].id

_USER_PROMPT = "Describe the notable events in the provided video. Provide the result in json format with 'mm:ss' format for time depiction for each event. Use keywords 'start', 'end' and 'caption' in the json output."
message = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": "assets.ngc.nvidia.com/products/api-catalog/cosmos-reason1-7b/car_curb.mp4"}},
        {"type": "text", "text": _USER_PROMPT},
    ]},
]

response = client.chat.completions.create(
    model=model,
    messages=message,
    max_tokens=2048,
    temperature=0.2,
    top_p=0.95,
    extra_body={"mm_processor_kwargs": {"fps": 4, "do_sample_frames": True}},
)

print(response.choices[0].message.content)
```

**SDG Critic**

```python
client = openai.OpenAI(api_key=token, base_url=url)
model = client.models.list().data[0].id

_USER_PROMPT = "Approve or reject this generated video for inclusion in a dataset for physical world model ai training. It must perfectly adhere to physics, object permanence, and have no anomalies. Any issue or concern causes rejection.
Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag. Answer with Approve or Reject only."
message = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": "https://assets.ngc.nvidia.com/products/api-catalog/cosmos-reason1-7b/car_curb.mp4"}},
        {"type": "text", "text": _USER_PROMPT},
    ]},
]

response = client.chat.completions.create(
    model=model,
    messages=message,
    max_tokens=4096,
    temperature=0.3,
    top_p=0.3,
    extra_body={"mm_processor_kwargs": {"fps": 4, "do_sample_frames": True}},
)

print(response.choices[0].message.content)

```

**2D Trajectory Creation**

```python
client = openai.OpenAI(api_key=token, base_url=url)
model = client.models.list().data[0].id

_USER_PROMPT = "You are given the task "Move the left bottle to far right". Specify the 2D trajectory your end effector should follow in pixel space. Return the trajectory coordinates in JSON format like this: {"point_2d": [x, y], "label": "gripper trajectory"}.
Answer the question using the following format:

<think>
Your reasoning.
</think>

Write your final answer immediately after the </think> tag."
message = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": [
        {"type": "video_url", "video_url": {"url": "https://assets.ngc.nvidia.com/products/api-catalog/cosmos-reason1-7b/critic_rejection_sampling.jpg"}},
        {"type": "text", "text": _USER_PROMPT},
    ]},
]

response = client.chat.completions.create(
    model=model,
    messages=message,
    max_tokens=4096,
    temperature=0.3,
    top_p=0.3,
    extra_body={"mm_processor_kwargs": {"fps": 2, "do_sample_frames": True}},
)

print(response.choices[0].message.content)


```
