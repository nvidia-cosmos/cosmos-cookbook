# Egocentric Social and Physical Reasoning with Cosmos-Reason2-8B

> **Authors:** [Tingyu Zhang](https://www.linkedin.com/in/tingyu-zhang-b3357314a) • [Sharon Yang](http://linkedin.com/in/sharonxueyangintel)
> **Organization:** [IntBot Inc.](https://www.intbot.ai/)

| **Model** | **Workload** | **Use Case** |
|-----------|--------------|--------------|
| [Cosmos-Reason2-8B](https://github.com/nvidia-cosmos/cosmos-reason2) | Inference | Egocentric social and physical reasoning for robotics |

- [Setup and System Requirements](https://github.com/nvidia-cosmos/cosmos-reason2#setup)

## 1. Overview

[IntBot](https://www.intbot.ai/) is a robotic startup company building social intelligent humanoid robots to address the structural labor shortage across service industries including hotels, airports, etc. IntBot robots elevate guest experiences by handling routine inquiries with warm, human-like presence and expressive body language, engaging in natural conversation across 50+ languages, and providing accurate answers, directions, and local recommendations. Available 24/7 and customizable to each brand, IntBot robots reliably cover repetitive and always-on service needs, enable human staff to focus on more important tasks.

<table>
  <tr>
    <td align="center"><img src="./assets/IntBot-GTC.jpg" width="400"><br><em>IntBot at NVIDIA GTC</em></td>
    <td align="center"><img src="./assets/IntBot-GTC-reception.jpg" width="400"><br><em>IntBot at NVIDIA booth reception</em></td>
  </tr>
</table>

To be effective in real-world service settings, IntBot’s robots must continuously perceive, interpret, and respond to a wide range of human **actions, intentions, and social cues**, including:

- **Greetings and gestures** — detecting and appropriately responding to waves, fist bumps, handshakes, and other invitation signals
- **Shared attention management** — maintaining engagement with multiple people simultaneously and determining who the robot should address at any given moment
- **Social context boundaries** — understanding when *not* to engage, such as when two guests are already conversing with each other or when interruption would be socially inappropriate
- **Object handoffs** — recognizing when a person is offering an item (e.g., documents, badges, promotional materials) *(future work)*
- **Emotional state awareness** — inferring when a guest appears confused, frustrated, disengaged, or in need of assistance *(future work)*

Together, these capabilities enable socially aware, context-sensitive interaction that goes beyond scripted dialogue and is essential for deploying robots at scale in live public environments, which is the key motivation behind this evaluation. The tests below focus on greetings/gestures, object motion, shared attention, and social boundaries; object handoffs and emotional state awareness are planned for future evaluation.

This recipe evaluates **[Cosmos-Reason2-8B](https://github.com/nvidia-cosmos/cosmos-reason2)** on a set of targeted egocentric video tests designed to probe robot-relevant reasoning, including:

- Human intent directed toward the robot
- Object motion relative to the robot
- Spatial relationships among multiple people
- Social context awareness (when not to engage)

Performance is compared directly against **Qwen3-VL-8B-Instruct**, the base vision-language model underlying Cosmos-Reason2-8B, using identical inputs and prompts. The goal is to isolate the impact of Cosmos-specific physical and social reasoning alignment, rather than differences in model scale or raw visual recognition capability.

## 2. Experimental Setup

### Inputs

- Short RGB videos (2–10 seconds)
- Egocentric camera perspective ("the camera view is my view")
- Natural-language prompts explicitly framed from the robot's point of view

### Models Evaluated

- [Cosmos-Reason2-8B](https://github.com/nvidia-cosmos/cosmos-reason2)
- [Qwen3-VL-8B-Instruct](https://github.com/QwenLM/Qwen3-VL)

### Controls

Both models were evaluated using:

- The same video clips
- Identical prompts
- The same inference pipeline

This controlled setup ensures that observed differences reflect reasoning behavior, not experimental variance.

## 3. Evaluation Results

> **Note:** All test images are extracted from egocentric video recordings. Faces have been blurred to protect privacy.

### 3.1 Human Intent Toward the Robot (Fist Bump)

#### Scenario

A person performs a fist-bump gesture directed at the robot.

<img src="./assets/Test1-fistbump.jpeg" width="400">

Sample frame: Person initiating fist bump toward the robot's egocentric view

#### Prompt

> "Is this person doing a fist bump at me? Can you predict what is the most likely position her fist will be in the next 2 seconds?"

#### Ground Truth

The person is intentionally fist-bumping the robot.

#### Observed Outputs

| Model | Outcome |
|-------|---------|
| Cosmos-Reason2-8B | Correctly identifies a fist bump directed at the robot and predicts persistence of the fist-bump pose |
| Qwen3-VL-8B-Instruct | Misclassifies the gesture as pointing and predicts a static, non-interactive posture |

<details>
<summary><strong>Raw Model Outputs</strong></summary>

**Cosmos-Reason2-8B:**
> Yes, the person is doing a fistbump with her right hand. In the next 2 seconds, her fist will likely remain in a fistbump position, maintaining contact with the viewer's fist as she holds the gesture.

**Qwen3-VL-8B-Instruct:**
> No, the person is not doing a fistbump. She is pointing her right index finger forward toward the camera, with her fist clenched but not extended for a fistbump. Her gesture appears to be a direct, assertive, or commanding motion rather than an interactive or celebratory one.
>
> **Most likely position of her fist in the next 2 seconds:**
> Her fist will likely remain clenched and extended forward, continuing to point at the camera. She may slightly adjust her arm's angle or tighten her grip, but the overall position—pointing forward with a clenched fist—will likely persist. There's no indication she will retract her arm or move her fist toward her own body or toward the camera for a fistbump.
>
> In short: **Her fist will stay pointed forward, likely with minor adjustments in angle or tension, but not retracting or moving toward a fistbump.**

</details>

#### Interpretation

Cosmos-Reason2-8B successfully infers interactive social intent toward the robot, whereas the base model focuses primarily on gesture appearance without recognizing engagement intent.

### 3.2 Object Motion, Distance, and Risk (Hat Scenarios)

Seven hat-related videos test diverse object motion patterns:

- Side throw away from the robot
- Spin-and-throw toward the robot
- Lateral throw across the robot's field of view
- Toss upward and fall
- Spinning prior to a throw
- Throw preparation posture
- Post-throw recovery

<table>
  <tr>
    <td align="center"><img src="./assets/Test2-side-throw-away.jpeg" height="180"><br><em>Side throw away</em></td>
    <td align="center"><img src="./assets/Test2-Spin-and-Throw.jpeg" height="180"><br><em>Spin and throw</em></td>
    <td align="center"><img src="./assets/Test2-Laterial-Throw.jpeg" height="180"><br><em>Lateral throw</em></td>
  </tr>
  <tr>
    <td align="center"><img src="./assets/Test2-Toss-Up.jpeg" height="180"><br><em>Toss upward</em></td>
    <td align="center"><img src="./assets/Test2-Spin.jpeg" height="180"><br><em>Spinning</em></td>
    <td align="center"><img src="./assets/Test2-Throw-Prep.jpeg" height="180"><br><em>Throw preparation</em></td>
  </tr>
  <tr>
    <td align="center" colspan="3"><img src="./assets/Test2-Post-throw-recovery.jpeg" height="180"><br><em>Post-throw recovery</em></td>
  </tr>
</table>

#### 3.2.1 Robot-Relative Trajectory and Risk Reasoning

**Prompt:**

> "Please estimate the landing location of the hat. Is the hat getting closer to me or further away from me? Is there danger for the hat to hit me?"

**Observed Patterns:**

| Scenario Type | Cosmos-Reason2-8B | Qwen3-VL-8B-Instruct |
|---------------|-------------------|----------------------|
| Side throw away | Correctly identifies motion away; no danger | Incorrectly flags approaching danger |
| Spin + throw toward | Correctly flags approach and potential risk | Mixed interpretation; risk downplayed |
| Lateral throw | Correctly identifies safe distance | Verbose but inconsistent distance framing |
| Toss upward | Flags possible approach during descent | Treats as generally safe; less decisive |

<details>
<summary><strong>Raw Model Outputs</strong></summary>

**Side throw away:**

| Model | Output |
|-------|--------|
| Cosmos-Reason2-8B | "The hat lands on the desk near the person working at the computer. It is getting further away from you. There is no danger for the hat to hit you." |
| Qwen3-VL-8B-Instruct | "The hat is getting closer to you. It is falling downward and will likely land near your head or shoulders. There is a potential danger for the hat to hit you, so you should move your head away or duck to avoid it." |

**Spin + throw toward:**

| Model | Output |
|-------|--------|
| Cosmos-Reason2-8B | "The hat will land at the center of the frame, near the person. The hat is getting closer to me. Yes, there is a danger for the hat to hit me." |
| Qwen3-VL-8B-Instruct | "The hat is currently being tossed upward and will land on the floor in front of you, slightly to your left. It is getting closer to you as it descends, and there is a low risk of it hitting you since it will land on the floor before reaching you." |

**Lateral throw:**

| Model | Output |
|-------|--------|
| Cosmos-Reason2-8B | "The hat is moving upward and to the left, away from your position. It is getting further away from you, and there is no immediate danger of it hitting you." |
| Qwen3-VL-8B-Instruct | "The hat is currently in mid-air, moving upward and slightly forward from the woman's hand. Based on its trajectory, it will likely descend toward the floor or a nearby surface, not directly toward you. It is getting closer to the ceiling and then will begin to fall downward, so relative to your position, it is moving away from you as it ascends, then closer as it descends—but not in a direct path toward you. There is no immediate danger of the hat hitting you..." |

**Toss upward:**

| Model | Output |
|-------|--------|
| Cosmos-Reason2-8B | "The hat will land at the center of the frame, closer to the camera. It is moving closer to you, and there is a potential risk of it hitting you." |
| Qwen3-VL-8B-Instruct | "The hat is currently being thrown upward and is moving away from you as it ascends. It will reach a peak and then begin to descend back toward the ground. Since it's moving away from you at the moment, it is not getting closer. There is no immediate danger of the hat hitting you..." |

</details>

**Interpretation:**

Cosmos-Reason2-8B reasons more consistently in robot-relative spatial and risk terms, while Qwen3-VL-8B-Instruct often defaults to descriptive or narrative interpretations that may misjudge proximity.

#### 3.2.2 Action Prediction Limitations

**Prompt:**

> "What is the person doing? Can you predict what is the next action this person likely to take?"

**Observation:**

In several cases (e.g., the spinning scenario), Cosmos-Reason2-8B produces a generic descriptive explanation and does not reliably predict the upcoming throw. Qwen3-VL-8B-Instruct exhibits similar behavior.

**Implication:**

Cosmos-Reason2-8B shows its strongest advantages when prompts are explicitly robot-centric and action-relevant. For neutral third-person prompts, its behavior converges toward the base model.

### 3.3 Multi-Person Spatial Reasoning

#### Scenario

Multiple people appear at different depths around the robot.

<img src="./assets/Test3-MultiPerson-Spatial-Reasoning.jpeg" width="400">

Sample frame: Multiple people at varying distances from the robot's perspective

#### Prompt

> "How many people are there in the view? Please describe the relative position of each person to me. Who is closest to me?"

#### Observed Outputs

| Aspect | Cosmos-Reason2-8B | Qwen3-VL-8B-Instruct |
|--------|-------------------|----------------------|
| Person count | Correct | Correct |
| Relative positions | Explicitly reasoned from robot view | Descriptive |
| Closest person | Explicitly identified | Correctly stated |

<details>
<summary><strong>Raw Model Outputs</strong></summary>

**Cosmos-Reason2-8B:**

> *\<think\>*
> Okay, let's break this down. The user is asking how many people are in the video and who is closest to them. The video shows a man in a black T-shirt and glasses standing in front of the camera. He's the main subject, positioned centrally and closest to the camera. There's also a woman to his left, slightly behind him. She's wearing a dark top and light pants. Another man is in the background on the right, leaning against a desk. There's also a child on the desk near the man in the background. The camera is focused on the man in the black shirt, so he's definitely the closest. The others are behind or to the sides but not as close. So the answer should list the people and note the closest one.
> *\</think\>*
>
> There are four people in the view:
>
> 1. **The man in the black T-shirt and glasses** (closest to the camera, positioned centrally).
> 2. **The woman in a dark top and light pants** (to his left, slightly behind him).
> 3. **The man in the background on the right** (leaning against a desk, further away).
> 4. **The child on the desk** (near the man in the background).
>
> The **man in the black T-shirt** is closest to you.

**Qwen3-VL-8B-Instruct:**

> There are four people in the view.
>
> - The man in the black t-shirt is closest to me, standing in the center of the frame.
> - The woman in the black t-shirt is to my left, slightly behind the man.
> - The man in the blue t-shirt is to my right, standing behind a desk.
> - The child is seated at the desk to the right, behind the man in the blue t-shirt.
>
> The man in the black t-shirt is closest to me.

</details>

#### Interpretation

Both models succeed at counting and basic localization. Cosmos-Reason2-8B provides more explicit reasoning grounded in the robot's body frame, while the base model offers a flatter description.

### 3.4 Social Context Awareness (Handshake Between Humans)

#### Scenario

Two humans shake hands with each other within the robot's field of view.

<img src="./assets/Test4-Two-Person-Handshake.jpeg" width="400">

Sample frame: Two humans shaking hands, observed from the robot's perspective

#### Prompt

> "Does anyone in the view show interest to engage with me? Should I shake hands with anyone?"

#### Observed Outputs

| Model | Outcome |
|-------|---------|
| Cosmos-Reason2-8B | Correctly reasons the interaction is human-to-human and answers "No" |
| Qwen3-VL-8B-Instruct | Also answers "No", with a descriptive explanation |

<details>
<summary><strong>Raw Model Outputs</strong></summary>

**Cosmos-Reason2-8B:**

> *\<think\>*
> Okay, let's break this down. The user is asking if they should shake hands with anyone in the camera view based on the shown interactions. The video shows a man in a black t-shirt and green pants who extends his hand to a woman entering the frame. They shake hands briefly, and then the woman moves out of view. The camera is static, so the user's perspective is from behind the camera.
>
> The key points here are: the man initiates the handshake with the woman, they complete it, and then she leaves. The user is behind the camera, observing this. The question is whether the user (behind the camera) should shake hands with anyone. Since the handshake happens between the man and the woman, and the user isn't shown as participating in that interaction, it seems like the user isn't part of the handshake. The man and woman are the ones interacting, and the user is just observing. So, the answer would be no, because the user isn't involved in the handshake. The interaction is between the man and the woman, not the user. The user's presence is passive, just watching. Therefore, there's no need for the user to shake hands with anyone in the scene.
> *\</think\>*
>
> **No**

**Qwen3-VL-8B-Instruct:**

> No one in the view appears to show interest in engaging with you (behind the camera). There is no indication that anyone is aware of or looking toward you. Additionally, there is no one in the frame who you should shake hands with, as no one is approaching or initiating interaction with you. The people in the scene are either engaged in their own activities or interacting with each other, but not with you.

</details>

#### Interpretation

Both models reach the correct conclusion. Cosmos-Reason2-8B explicitly reasons about social participation boundaries, reinforcing appropriate non-engagement behavior.

## 4. Prompt Design Guidance

Based on observed behavior:

- **Explicit egocentric framing is critical** ("the camera view is my view")
- **Relative, robot-centered questions outperform descriptive ones** ("closer to me", "danger to me")
- **Action-oriented prompts highlight Cosmos advantages**
- **Neutral third-person prompts reduce differentiation** from the base model

## 5. Summary Comparison

| Dimension | Cosmos-Reason2-8B | Qwen3-VL-8B-Instruct |
|-----------|-------------------|----------------------|
| Human intent toward robot | Strong | Weak |
| Robot-relative object motion | More consistent | Inconsistent |
| Collision risk assessment | Conservative and grounded | Frequently over- or under-estimates |
| Multi-person spatial reasoning | Explicit | Descriptive |
| Social boundary awareness | Explicit | Correct but implicit |

## 6. Key Takeaways

Based on the evaluated test results:

- **Cosmos-Reason2-8B demonstrates clear advantages** when reasoning is framed from an embodied robot perspective, particularly for intent recognition, physical risk assessment, and social engagement decisions.
- **For neutral or third-person prompts**, behavior converges toward the base model, underscoring the importance of prompt alignment with embodiment.

These results support Cosmos-Reason2-8B as a strong foundation for social robots and physical AI systems, where physical law constraints and action relevance are critical.

### Limitations and Future Work

This evaluation was conducted within a constrained time frame, which limited the scope of benchmark design. As a result, the current test set represents a preliminary exploration rather than a comprehensive benchmark. Future work should address the following:

- **Expanded test coverage** — Increase the number and diversity of test scenarios to improve statistical robustness
- **Systematic example selection** — Apply more rigorous criteria for scenario selection to ensure balanced representation across interaction types
- **Additional capability testing** — Incorporate tests for object handoffs and emotional state awareness, as noted in Section 1

---

## Document Information

**Publication Date:** January 1, 2026

### Citation

If you use this recipe or reference this work, please cite it as:

```bibtex
@misc{cosmos_cookbook_intbot_2026,
  title={Egocentric Social and Physical Reasoning with Cosmos-Reason2-8B},
  author={Zhang, Tingyu and Yang, Sharon},
  organization={IntBot Inc.},
  year={2026},
  month={January},
  howpublished={\url{https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/inference/reason2/intbot_showcase/inference.html}},
  note={NVIDIA Cosmos Cookbook}
}
```

**Suggested text citation:**

> Tingyu Zhang, & Sharon Yang (2026). Egocentric Social and Physical Reasoning with Cosmos-Reason2-8B. In *NVIDIA Cosmos Cookbook*. IntBot Inc. Accessible at <https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/inference/reason2/intbot_showcase/inference.html>
