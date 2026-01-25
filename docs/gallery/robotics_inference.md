# Robotics Domain Adaptation Gallery

> **Authors:** [Raju Wagwani](https://www.linkedin.com/in/raju-wagwani-a4746027/) • [Jathavan Sriram](https://www.linkedin.com/in/jathavansriram) • [Richard Yarlett](https://www.linkedin.com/in/richardyarlett/) • [Joshua Bapst](https://www.linkedin.com/in/joshbapst/) • [Jinwei Gu](https://www.linkedin.com/in/jinweigu/)

> **Organization:** NVIDIA

## Overview

This page showcases results from Cosmos Transfer 2.5 for robotics applications. The examples demonstrate sim-to-real transfer for robotic manipulation tasks in kitchen environments, showing how synthetic simulation videos can be transformed into photorealistic scenes with varied materials, lighting, and environmental conditions. These results enable domain adaptation and data augmentation for robotic training and validation.

**Use Case**: Robotics engineers can use these techniques to generate diverse training data from a single simulation, creating variations in kitchen styles, materials, and lighting conditions without re-running expensive simulations or capturing real-world data.

## Example 1: Edge-Only Control for Environment Variation

This example demonstrates how to transform synthetic robotic simulation videos into photorealistic scenes with different kitchen styles and materials using **edge control**, which preserves the original structure, motion, and geometry of the robot and scene while allowing the visual appearance to change dramatically based on the text prompt.

- **Edge control**: Maintains the structure and layout of objects, robot poses, and camera motion from the simulation, while transforming the visual appearance (materials, lighting, colors) according to the prompt.
- **Why use edge-only**: To preserve exact robot motions and object positions from simulation while varying environmental aesthetics.

For detailed explanations of control modalities, refer to the [Control Modalities Overview](../core_concepts/control_modalities/overview.md).

<style>
.carousel {
  position: relative;
  margin-top: 1rem;
  overflow: visible;
}

.carousel-track {
  position: relative;
}

.carousel-slide {
  display: none;
  flex-direction: column;
  gap: 1rem;
  padding: 1rem;
  border: 1px solid var(--md-default-fg-color--lightest, #e2e8f0);
  background: var(--md-default-bg-color, #fff);
  box-shadow: 0 10px 34px rgba(0, 0, 0, 0.06);
  border-radius: 0px;
}

.carousel-slide.is-active {
  display: flex;
}

.media-wrap {
  position: relative;
  overflow: visible;
  background: #000;
  margin-bottom: 0.5rem;
  border-radius: 0px;
}

.media-wrap video {
  width: 100%;
  display: block;
  border-radius: 0px;
}

.carousel-btn {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  width: 36px;
  height: 36px;
  border: none;
  background: var(--md-accent-fg-color, #76b900);
  color: #fff;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.16);
  opacity: 1;
  z-index: 3;
  font-size: 18px;
  line-height: 1;
  font-weight: 700;
  border-radius: 50%;
}

.carousel-btn:hover {
  opacity: 1;
}

.carousel-btn.prev {
  left: -18px;
}

.carousel-btn.next {
  right: -18px;
}

.text-stack {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.text-block .label {
  font-weight: 700;
  text-transform: uppercase;
  font-size: 0.75rem;
  letter-spacing: 0.04em;
  color: var(--md-default-fg-color--light, #5b6472);
}

.text-block .preview-text {
  display: block;
}

.text-block .full-text {
  display: none;
  margin-top: 0.25rem;
}

.carousel-slide.expanded .preview-text {
  display: none;
}

.carousel-slide.expanded .full-text {
  display: block;
}

.see-more {
  align-self: flex-start;
  padding: 0.4rem 0.8rem;
  border: 1px solid var(--md-accent-fg-color, #76b900);
  background: transparent;
  cursor: pointer;
  font-weight: 600;
  color: var(--md-accent-fg-color, #76b900);
  border-radius: 0px;
}

/*Format parameters JSON with line breaks*/
.text-block .full-text {
  white-space: pre-wrap;
  font-family: var(--md-code-font, monospace);
  font-size: 0.9em;
}

.text-block .preview-text {
  font-family: var(--md-code-font, monospace);
  font-size: 0.9em;
}
</style>

### Scene 1a: Kitchen Stove - Cooking Task

This scene shows a humanoid robot performing a cooking task at a stove. The examples demonstrate how different kitchen cabinet styles (white, red, wood tones) and robot materials (plastic, metal, gold) can be generated from the same simulation.

#### Input Video

<div class="carousel" data-interval="5000">
  <div class="carousel-track">
    <article class="carousel-slide is-active">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/kitchen_stove_input.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text">{ "seed": 1,</span>
          <span class="full-text">{
    "seed": 1,
    "prompt_path": "assets/prompt_robot.json",
    "output_dir": "outputs/robot",
    "video_path": "assets/kitchen_stove_input.mp4",
    "guidance": 7,
    "edge": {
        "control_weight": 1
    }
}</span>
        </div>
        <button class="see-more" type="button">Show full parameters</button>
      </div>
    </article>
  </div>
</div>

#### Examples

<div class="carousel" data-interval="5000">
  <div class="carousel-track">
    <article class="carousel-slide is-active">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/kitchen_stove_white.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Input Prompt</div>
          <span class="preview-text"></span>
          <span class="full-text">This scene depicts a photo realistic luxury kitchen with high end professional finishes and lighting. ALL The kitchen cabinets are all highly polished bright white panels with chrome accents and pulls. The kitchen counters are stainless steel. The kitchen walls and backsplash are all white subway tile. The kitchen contains an expensive double door stainless steel refrigerator, a stainless steel microwave, a stainless steel oven, a stainless steel coffee machine, a stainless steel toaster, a stainless steel stove top, a stainless steel sink, and stainless steel pots. Standing in the kitchen is a humanoid robot. The robot is made of orange polished plastic panels with chrome accents. The camera is fixed and steady. The robot is at a kitchen stainless steel stove picking up a glass cooking pot lid with his left hand and lifting it in the air. The robot is picking up two tomatoes with his right hand and putting them inside the stainless steel pot. There is steam coming out of the pot.</span>
        </div>
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">seed: 1, guidance: 7, edge: 1.0</span>
        </div>
        <button class="see-more" type="button">Show full prompt</button>
      </div>
    </article>

    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/kitchen_stove_red.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Input Prompt</div>
          <span class="preview-text"></span>
          <span class="full-text">This scene depicts a photo realistic luxury kitchen with high end professional finishes and lighting. ALL The kitchen cabinets are all highly polished bright red panels with stainless steel accents and pulls. The kitchen counters are stainless steel. The kitchen walls and backsplash are all white subway tile. The kitchen contains an expensive double door stainless steel refrigerator, a stainless steel microwave, a stainless steel oven, a stainless steel coffee machine, a stainless steel toaster, a stainless steel stove top, a stainless steel sink, and stainless steel pots. Standing in the kitchen is a humanoid robot. The robot is made of white polished panels with black accents. The camera is fixed and steady. The robot is at a kitchen stainless steel stove picking up a red cooking pot lid with his left hand and lifting it in the air. The robot is picking up two tomatoes with his right hand and putting them inside the red pot. There is steam coming out of the pot.</span>
        </div>
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">seed: 1, guidance: 7, edge: 1.0</span>
        </div>
        <button class="see-more" type="button">Show full prompt</button>
      </div>
    </article>

    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/kitchen_stove_light_wood.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Input Prompt</div>
          <span class="preview-text"></span>
          <span class="full-text">This scene depicts a photo realistic luxury kitchen with high end professional finishes and lighting. ALL The kitchen cabinets are all light wood, with stainless steel accents and pulls. The kitchen counters, kitchen walls and backsplash are all expensive black veined marble. The kitchen contains an expensive double door stainless steel refrigerator, a stainless steel microwave, a stainless steel oven, a stainless steel coffee machine, a stainless steel toaster, a stainless steel stove top, a stainless steel sink, and stainless steel pots. Standing in the kitchen is a humanoid robot. The robot is made of stainless steel polished panels with chrome accents. The camera is fixed and steady. The robot is at a kitchen stainless steel stove picking up a stainless steel cooking pot lid with his left hand and lifting it in the air. The robot is picking up two tomatoes with his right hand and putting them inside the stainless steel pot. There is steam coming out of the pot.</span>
        </div>
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">seed: 1, guidance: 7, edge: 1.0</span>
        </div>
        <button class="see-more" type="button">Show full prompt</button>
      </div>
    </article>

    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/kitchen_stove_dark_wood.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Input Prompt</div>
          <span class="preview-text"></span>
          <span class="full-text">This scene depicts a photo realistic luxury kitchen with high end professional finishes and lighting. ALL The kitchen cabinets are all dark wood, with chrome accents and pulls. The kitchen counters, kitchen walls and backsplash are all expensive beige veined marble. The kitchen contains an expensive double door stainless steel refrigerator, a stainless steel microwave, a stainless steel oven, a stainless steel coffee machine, a stainless steel toaster, a stainless steel stove top, a stainless steel sink, and stainless steel pots. Standing in the kitchen is a humanoid robot. The robot is made of gold polished reflective panels with shiny black accents. The camera is fixed and steady. The robot is at a kitchen gold stove picking up a gold cooking pot lid with his left hand and lifting it in the air. The robot is picking up two tomatoes with his right hand and putting them inside the gold pot. There is steam coming out of the pot.</span>
        </div>
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">seed: 1, guidance: 7, edge: 1.0</span>
        </div>
        <button class="see-more" type="button">Show full prompt</button>
      </div>
    </article>

    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/kitchen_stove.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Input Prompt</div>
          <span class="preview-text"></span>
          <span class="full-text">The video captures a stunning, photorealistic scene with remarkable attention to detail, giving it a lifelike appearance that is almost indistinguishable from reality. It appears to be from a high-budget 4K movie, showcasing ultra-high-definition quality with impeccable resolution.</span>
        </div>
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">seed: 1, guidance: 7, edge: 1.0</span>
        </div>
        <button class="see-more" type="button">Show full prompt</button>
      </div>
    </article>

  </div>
</div>

### Scene 1b: Kitchen Island - Object Manipulation

This scene shows a robot performing precise object manipulation at a kitchen island, picking up and placing items. The examples demonstrate material variations (different fruit/objects) coordinated with kitchen style changes.

#### Input Video

<div class="carousel" data-interval="5000">
  <div class="carousel-track">
    <article class="carousel-slide is-active">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/kitchen_oranges_input.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text">{ "seed": 1,</span>
          <span class="full-text">{
    "seed": 1,
    "prompt_path": "assets/prompt_robot.json",
    "output_dir": "outputs/robot",
    "video_path": "assets/kitchen_oranges_input.mp4",
    "guidance": 7,
    "edge": {
        "control_weight": 1
    }
}</span>
        </div>
        <button class="see-more" type="button">Show full parameters</button>
      </div>
    </article>
  </div>
</div>

#### Examples

<div class="carousel" data-interval="5000">
  <div class="carousel-track">
    <article class="carousel-slide is-active">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/kitchen_oranges_white.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Input Prompt</div>
          <span class="preview-text"></span>
          <span class="full-text">This scene depicts a photo realistic luxury kitchen with high end professional finishes and lighting. ALL The kitchen cabinets are all highly polished bright white panels with chrome accents and pulls. The kitchen counters are stainless steel. The kitchen walls and backsplash are all white subway tile. The kitchen contains an expensive double door stainless steel refrigerator, a stainless steel microwave, a stainless steel oven, a stainless steel coffee machine, a stainless steel toaster, a stainless steel stove top, a stainless steel sink, and stainless steel pots. In the center of the room is a kitchen island. This is also finished with highly polished bright white panels cabinets and an stainless steel countertop. In the middle of the island counter is a large glass bowl of oranges. Standing in the kitchen is a humanoid robot. The robot is made of orange polished plastic panels with chrome accents. The camera is fixed and steady. The robot is picking up two oranges from either side of a small glass plate, and placing them on the plate.</span>
        </div>
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">seed: 1, guidance: 7, edge: 1.0</span>
        </div>
        <button class="see-more" type="button">Show full prompt</button>
      </div>
    </article>

    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/kitchen_oranges_red.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Input Prompt</div>
          <span class="preview-text"></span>
          <span class="full-text">This scene depicts a photo realistic luxury kitchen with high end professional finishes and lighting. ALL The kitchen cabinets are all highly polished bright red panels with stainless steel accents and pulls. The kitchen counters are stainless steel. The kitchen walls and backsplash are all white subway tile. The kitchen contains an expensive double door stainless steel refrigerator, a stainless steel microwave, a stainless steel oven, a stainless steel coffee machine, a stainless steel toaster, a stainless steel stove top, a stainless steel sink, and stainless steel pots. In the center of the room is a kitchen island. This is also finished with highly polished bright red panels cabinets and an stainless steel countertop. In the middle of the island counter is a large white bowl of eggs. Standing in the kitchen is a humanoid robot. The robot is made of white polished panels with black accents. The camera is fixed and steady. The robot is picking up two eggs from either side of a small white plate, and placing them on the plate.</span>
        </div>
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">seed: 1, guidance: 7, edge: 1.0</span>
        </div>
        <button class="see-more" type="button">Show full prompt</button>
      </div>
    </article>

    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/kitchen_oranges_light_wood.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Input Prompt</div>
          <span class="preview-text"></span>
          <span class="full-text">This scene depicts a photo realistic luxury kitchen with high end professional finishes and lighting. ALL The kitchen cabinets are all light wood, with stainless steel accents and pulls. The kitchen counters, kitchen walls and backsplash are all expensive black veined marble. The kitchen contains an expensive double door stainless steel refrigerator, a stainless steel microwave, a stainless steel oven, a stainless steel coffee machine, a stainless steel toaster, a stainless steel stove top, a stainless steel sink, and stainless steel pots. In the center of the room is a kitchen island. This is also finished with light wood cabinets and an expensive black veined marble countertop. In the middle of the island counter is a large white bowl of lemons. Standing in the kitchen is a humanoid robot. The robot is made of stainless steel polished panels with chrome accents. The camera is fixed and steady. The robot is picking up two lemons from either side of a small white plate, and placing them on the plate.</span>
        </div>
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">seed: 1, guidance: 7, edge: 1.0</span>
        </div>
        <button class="see-more" type="button">Show full prompt</button>
      </div>
    </article>

    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/kitchen_oranges_dark_wood.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Input Prompt</div>
          <span class="preview-text"></span>
          <span class="full-text">This scene depicts a photo realistic luxury kitchen with high end professional finishes and lighting. ALL The kitchen cabinets are all dark wood, with chrome accents and pulls. The kitchen counters, kitchen walls and backsplash are all expensive beige veined marble. The kitchen contains an expensive double door stainless steel refrigerator, a stainless steel microwave, a stainless steel oven, a stainless steel coffee machine, a stainless steel toaster, a stainless steel stove top, a stainless steel sink, and stainless steel pots. In the center of the room is a kitchen island. This is also finished with dark wood cabinets and an expensive beige veined marble countertop. In the middle of the island counter is a large white bowl of apples. Standing in the kitchen is a humanoid robot. The robot is made of gold polished reflective panels with shiny black accents. The camera is fixed and steady. The robot is picking up two apples from either side of a small white plate, and placing them on the plate.</span>
        </div>
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">seed: 1, guidance: 7, edge: 1.0</span>
        </div>
        <button class="see-more" type="button">Show full prompt</button>
      </div>
    </article>

    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/kitchen_oranges.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Input Prompt</div>
          <span class="preview-text"></span>
          <span class="full-text">The video captures a stunning, photorealistic scene with remarkable attention to detail, giving it a lifelike appearance that is almost indistinguishable from reality. It appears to be from a high-budget 4K movie, showcasing ultra-high-definition quality with impeccable resolution.</span>
        </div>
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">seed: 1, guidance: 7, edge: 1.0</span>
        </div>
        <button class="see-more" type="button">Show full prompt</button>
      </div>
    </article>

  </div>
</div>

### Scene 1c: Kitchen Refrigerator - Appliance Interaction

This scene demonstrates robot interaction with appliances, showing the robot opening a refrigerator. The examples maintain the lighting dynamics (fridge interior light) while varying kitchen aesthetics.

#### Input Video

<div class="carousel" data-interval="5000">
  <div class="carousel-track">
    <article class="carousel-slide is-active">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/kitchen_fridge_input.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text">{ "seed": 1,</span>
          <span class="full-text">{
    "seed": 1,
    "prompt_path": "assets/prompt_robot.json",
    "output_dir": "outputs/robot",
    "video_path": "assets/kitchen_fridge_input.mp4",
    "guidance": 7,
    "edge": {
        "control_weight": 1
    }
}</span>
        </div>
        <button class="see-more" type="button">Show full parameters</button>
      </div>
    </article>
  </div>
</div>

#### Examples

<div class="carousel" data-interval="5000">
  <div class="carousel-track">
    <article class="carousel-slide is-active">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/kitchen_fridge_white.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Input Prompt</div>
          <span class="preview-text"></span>
          <span class="full-text">This scene depicts a photo realistic luxury kitchen with high end professional finishes and lighting. ALL The kitchen cabinets are all highly polished bright white panels with chrome accents and pulls. The kitchen counters are stainless steel. The kitchen walls and backsplash are all white subway tile. The kitchen contains an expensive double door stainless steel refrigerator, a stainless steel microwave, a stainless steel oven, a stainless steel coffee machine, a stainless steel toaster, a stainless steel stove top, a stainless steel sink, and stainless steel pots. In the center of the room is a kitchen island. This is also finished with highly polished bright white panels cabinets and an stainless steel countertop. In the middle of the island counter is a large glass bowl of oranges. Standing in the kitchen is a humanoid robot. The robot is made of orange polished plastic panels with chrome accents. The camera is fixed and steady. The robot is opening the fridge with his right hand and looking inside. The fridge light turns on and it very bright, showing the inside of the fridge filled with food and drink.</span>
        </div>
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">seed: 1, guidance: 7, edge: 1.0</span>
        </div>
        <button class="see-more" type="button">Show full prompt</button>
      </div>
    </article>

    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/kitchen_fridge_red.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Input Prompt</div>
          <span class="preview-text"></span>
          <span class="full-text">This scene depicts a photo realistic luxury kitchen with high end professional finishes and lighting. ALL The kitchen cabinets are all highly polished bright red panels with stainless steel accents and pulls. The kitchen counters are stainless steel. The kitchen walls and backsplash are all white subway tile. The kitchen contains an expensive double door stainless steel refrigerator, a stainless steel microwave, a stainless steel oven, a stainless steel coffee machine, a stainless steel toaster, a stainless steel stove top, a stainless steel sink, and stainless steel pots. In the center of the room is a kitchen island. This is also finished with highly polished bright red panels cabinets and an stainless steel countertop. In the middle of the island counter is a large white bowl of eggs. Standing in the kitchen is a humanoid robot. The robot is made of white polished panels with black accents. The camera is fixed and steady. The robot is opening the fridge with his right hand and looking inside. The fridge light turns on and it very bright, showing the inside of the fridge filled with food and drink.</span>
        </div>
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">seed: 1, guidance: 7, edge: 1.0</span>
        </div>
        <button class="see-more" type="button">Show full prompt</button>
      </div>
    </article>

    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/kitchen_fridge_light_wood.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Input Prompt</div>
          <span class="preview-text"></span>
          <span class="full-text">This scene depicts a photo realistic luxury kitchen with high end professional finishes and lighting. ALL The kitchen cabinets are all light wood, with stainless steel accents and pulls. The kitchen counters, kitchen walls and backsplash are all expensive black veined marble. The kitchen contains an expensive double door stainless steel refrigerator, a stainless steel microwave, a stainless steel oven, a stainless steel coffee machine, a stainless steel toaster, a stainless steel stove top, a stainless steel sink, and stainless steel pots. In the center of the room is a kitchen island. This is also finished with light wood cabinets and an expensive black veined marble countertop. In the middle of the island counter is a large white bowl of lemons. Standing in the kitchen is a humanoid robot. The robot is made of stainless steel polished panels with chrome accents. The camera is fixed and steady. The robot is opening the fridge with his right hand and looking inside. The fridge light turns on and it very bright, showing the inside of the fridge filled with food and drink.</span>
        </div>
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">seed: 1, guidance: 7, edge: 1.0</span>
        </div>
        <button class="see-more" type="button">Show full prompt</button>
      </div>
    </article>

    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/kitchen_fridge_dark_wood.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Input Prompt</div>
          <span class="preview-text"></span>
          <span class="full-text">This scene depicts a photo realistic luxury kitchen with high end professional finishes and lighting. ALL The kitchen cabinets are all dark wood, with chrome accents and pulls. The kitchen counters, kitchen walls and backsplash are all expensive beige veined marble. The kitchen contains an expensive double door stainless steel refrigerator, a stainless steel microwave, a stainless steel oven, a stainless steel coffee machine, a stainless steel toaster, a stainless steel stove top, a stainless steel sink, and stainless steel pots. In the center of the room is a kitchen island. This is also finished with dark wood cabinets and an expensive beige veined marble countertop. In the middle of the island counter is a large white bowl of apples. Standing in the kitchen is a humanoid robot. The robot is made of gold polished reflective panels with shiny black accents. The camera is fixed and steady. The robot is opening the fridge with his right hand and looking inside. The fridge light turns on and it very bright, showing the inside of the fridge filled with food and drink.</span>
        </div>
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">seed: 1, guidance: 7, edge: 1.0</span>
        </div>
        <button class="see-more" type="button">Show full prompt</button>
      </div>
    </article>

    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/kitchen_fridge.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Input Prompt</div>
          <span class="preview-text"></span>
          <span class="full-text">The video captures a stunning, photorealistic scene with remarkable attention to detail, giving it a lifelike appearance that is almost indistinguishable from reality. It appears to be from a high-budget 4K movie, showcasing ultra-high-definition quality with impeccable resolution.</span>
        </div>
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">seed: 1, guidance: 7, edge: 1.0</span>
        </div>
        <button class="see-more" type="button">Show full prompt</button>
      </div>
    </article>

  </div>
</div>

## Example 2: Multi-Control with Custom Control Videos

These examples demonstrate advanced usage, where you provide **custom pre-computed control videos** (depth, edge, segmentation) alongside the input video. Multi-control gives you fine-grained control over different aspects of the transformation:

- **depth**: Controls 3D spatial relationships and perspective
- **edge**: Maintains structural boundaries and object shapes
- **seg**: Enables semantic-level changes and object replacement
- **vis**: Preserves lighting and camera properties (set to 0 in this example)

**When to use multi-control**: Use this approach when you need precise control over the transformation by pre-generating and fine-tuning specific control signals, especially for complex scene manipulations or when edge-only control is insufficient.

### Scene 2a

### Input and Control Videos

<div class="carousel" data-interval="5000">
  <div class="carousel-track">
    <article class="carousel-slide is-active">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/kitchen2_cg.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Video Type</div>
          <span class="preview-text">Input Video</span>
          <span class="full-text">Input Video</span>
        </div>
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text">{ "seed": 2000,</span>
          <span class="full-text">{
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
}</span>
        </div>
        <button class="see-more" type="button">Show full parameters</button>
      </div>
    </article>

    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/kitchen2_depth.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Video Type</div>
          <span class="preview-text">Depth Control</span>
          <span class="full-text">Depth Control</span>
        </div>
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text">{ "seed": 2000,</span>
          <span class="full-text">{
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

}</span>

</div>
<button class="see-more" type="button">Show full parameters</button>
</div>
</article>

    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/kitchen2_edge.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Video Type</div>
          <span class="preview-text">Edge Control</span>
          <span class="full-text">Edge Control</span>
        </div>
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text">{ "seed": 2000,</span>
          <span class="full-text">{
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

}</span>

</div>
<button class="see-more" type="button">Show full parameters</button>
</div>
</article>

    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/kitchen2_seg.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Video Type</div>
          <span class="preview-text">Segmentation Control</span>
          <span class="full-text">Segmentation Control</span>
        </div>
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text">{ "seed": 2000,</span>
          <span class="full-text">{
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

}</span>

</div>
<button class="see-more" type="button">Show full parameters</button>
</div>
</article>

  </div>
</div>

### Output Video

<div class="media-wrap">
  <video autoplay loop muted playsinline>
    <source src="assets/kitchen2_output.mp4" type="video/mp4">
    Your browser does not support the video tag.
  </video>
</div>

### Scene 2b

### Input and Control Videos

<div class="carousel" data-interval="5000">
  <div class="carousel-track">
    <article class="carousel-slide is-active">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/robotic_arm_input.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Video Type</div>
          <span class="preview-text">Input Video</span>
          <span class="full-text">Input Video</span>
        </div>
        <div class="text-block">
          <div class="label">Input Prompt</div>
          <span class="preview-text">The video features two robotic arms with brushed matte black bodies, and contrasting black joints, manipulating a small red glass cube.</span>
          <span class="full-text">The video features two robotic arms with brushed matte black bodies, and contrasting black joints, manipulating a small red glass cube. They are positioned on a plastic table, with minimalistic office in the background, illuminated by artificial white light.</span>
        </div>
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text">{ "name": "robot_multicontrol",</span>
          <span class="full-text">{
    "name": "robot_multicontrol",
    "video_path": "assets/robotic_arm_input.mp4",
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
}</span>
        </div>
        <button class="see-more" type="button">Show full parameters</button>
      </div>
    </article>

    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/robotic_arm_input_depth.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Video Type</div>
          <span class="preview-text">Depth Control</span>
          <span class="full-text">Depth Control</span>
        </div>
        <div class="text-block">
          <div class="label">Input Prompt</div>
          <span class="preview-text">The video features two robotic arms with brushed matte black bodies, and contrasting black joints, manipulating a small red glass cube.</span>
          <span class="full-text">The video features two robotic arms with brushed matte black bodies, and contrasting black joints, manipulating a small red glass cube. They are positioned on a plastic table, with minimalistic office in the background, illuminated by artificial white light.</span>
        </div>
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text">{ "name": "robot_multicontrol",</span>
          <span class="full-text">{
    "name": "robot_multicontrol",
    "video_path": "assets/robotic_arm_input.mp4",
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

}</span>

</div>
<button class="see-more" type="button">Show full parameters</button>
</div>
</article>

    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/robotic_arm_input_edge.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Video Type</div>
          <span class="preview-text">Edge Control</span>
          <span class="full-text">Edge Control</span>
        </div>
        <div class="text-block">
          <div class="label">Input Prompt</div>
          <span class="preview-text">The video features two robotic arms with brushed matte black bodies, and contrasting black joints, manipulating a small red glass cube.</span>
          <span class="full-text">The video features two robotic arms with brushed matte black bodies, and contrasting black joints, manipulating a small red glass cube. They are positioned on a plastic table, with minimalistic office in the background, illuminated by artificial white light.</span>
        </div>
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text">{ "name": "robot_multicontrol",</span>
          <span class="full-text">{
    "name": "robot_multicontrol",
    "video_path": "assets/robotic_arm_input.mp4",
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

}</span>

</div>
<button class="see-more" type="button">Show full parameters</button>
</div>
</article>

    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/robotic_arm_input_seg.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Video Type</div>
          <span class="preview-text">Segmentation Control</span>
          <span class="full-text">Segmentation Control</span>
        </div>
        <div class="text-block">
          <div class="label">Input Prompt</div>
          <span class="preview-text">The video features two robotic arms with brushed matte black bodies, and contrasting black joints, manipulating a small red glass cube.</span>
          <span class="full-text">The video features two robotic arms with brushed matte black bodies, and contrasting black joints, manipulating a small red glass cube. They are positioned on a plastic table, with minimalistic office in the background, illuminated by artificial white light.</span>
        </div>
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text">{ "name": "robot_multicontrol",</span>
          <span class="full-text">{
    "name": "robot_multicontrol",
    "video_path": "assets/robotic_arm_input.mp4",
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

}</span>

</div>
<button class="see-more" type="button">Show full parameters</button>
</div>
</article>

    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/robotic_arm_input_vis.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Video Type</div>
          <span class="preview-text">Vis Control</span>
          <span class="full-text">Vis Control</span>
        </div>
        <div class="text-block">
          <div class="label">Input Prompt</div>
          <span class="preview-text">The video features two robotic arms with brushed matte black bodies, and contrasting black joints, manipulating a small red glass cube.</span>
          <span class="full-text">The video features two robotic arms with brushed matte black bodies, and contrasting black joints, manipulating a small red glass cube. They are positioned on a plastic table, with minimalistic office in the background, illuminated by artificial white light.</span>
        </div>
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text">{ "name": "robot_multicontrol",</span>
          <span class="full-text">{
    "name": "robot_multicontrol",
    "video_path": "assets/robotic_arm_input.mp4",
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

}</span>

</div>
<button class="see-more" type="button">Show full parameters</button>
</div>
</article>

  </div>
</div>

#### Examples

<div class="carousel" data-interval="5000">
  <div class="carousel-track">
    <article class="carousel-slide is-active">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/robotic_arm_1.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Input Prompt</div>
          <span class="preview-text"></span>
          <span class="full-text">The video features two robotic arms with brushed matte black bodies, and contrasting black joints, manipulating a small red glass cube. They are positioned on a plastic table, with minimalistic office in the background, illuminated by artificial white light.</span>
        </div>
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">guidance: 3, depth: 0.6, edge: 1.0, seg: 0.4, vis: 0</span>
        </div>
        <button class="see-more" type="button">Show full prompt</button>
      </div>
    </article>

    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/robotic_arm_2.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Input Prompt</div>
          <span class="preview-text"></span>
          <span class="full-text">The video features two robotic arms with brushed bronze bodies, and contrasting yellow joints, manipulating a small purple plastic cube. They are positioned on a granite table, with urban rooftop in the background, illuminated by natural light.</span>
        </div>
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">guidance: 3, depth: 0.6, edge: 1.0, seg: 0.4, vis: 0</span>
        </div>
        <button class="see-more" type="button">Show full prompt</button>
      </div>
    </article>

    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/robotic_arm_3.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Input Prompt</div>
          <span class="preview-text"></span>
          <span class="full-text">The video features two robotic arms with matte black bodies, and contrasting blue joints, manipulating a small white plastic cube. They are positioned on a marble table, with industrial warehouse in the background, illuminated by colored ambient light (blue).</span>
        </div>
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">guidance: 3, depth: 0.6, edge: 1.0, seg: 0.4, vis: 0</span>
        </div>
        <button class="see-more" type="button">Show full prompt</button>
      </div>
    </article>

    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/robotic_arm_4.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Input Prompt</div>
          <span class="preview-text"></span>
          <span class="full-text">The video features two robotic arms with matte white bodies, and contrasting black joints, manipulating a small green glass cube. They are positioned on a marble table, with closed room in the background, illuminated by artificial white light.</span>
        </div>
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">guidance: 3, depth: 0.6, edge: 1.0, seg: 0.4, vis: 0</span>
        </div>
        <button class="see-more" type="button">Show full prompt</button>
      </div>
    </article>

  </div>
</div>

<script>
document.addEventListener("DOMContentLoaded", () => {
  const firstSentence = (text) => {
    const trimmed = text.trim();
    const match = trimmed.match(/.*?[.!?](\s|$)/);
    return match ? match[0].trim() : trimmed;
  };

  document.querySelectorAll(".carousel").forEach((carousel) => {
    const slides = Array.from(carousel.querySelectorAll(".carousel-slide"));
    if (!slides.length) return;

    slides.forEach((slide) => {
      slide.classList.remove("expanded");
      slide.querySelectorAll(".text-block").forEach((block) => {
        const full = block.querySelector(".full-text");
        const preview = block.querySelector(".preview-text");
        if (full && preview) {
          // Only set preview if it's empty (preserve manually set previews like "{ "seed": 1,")
          if (!preview.textContent.trim()) {
            preview.textContent = firstSentence(full.textContent || "");
          }
        }
      });
      const toggle = slide.querySelector(".see-more");
      if (toggle) {
        const originalText = toggle.textContent.trim();
        const isParameters = originalText.includes("parameters");
        toggle.addEventListener("click", () => {
          slide.classList.toggle("expanded");
          const expanded = slide.classList.contains("expanded");
          if (isParameters) {
            toggle.textContent = expanded ? "Hide full parameters" : "Show full parameters";
          } else {
            toggle.textContent = expanded ? "Hide full prompt" : "Show full prompt";
          }
        });
      }
    });

    let index = slides.findIndex((s) => s.classList.contains("is-active"));
    if (index < 0) {
      index = 0;
      slides[0].classList.add("is-active");
    }

    const intervalMs = parseInt(carousel.dataset.interval || "5000", 10);
    const show = (nextIndex) => {
      slides[index].classList.remove("is-active", "expanded");
      const priorToggle = slides[index].querySelector(".see-more");
      if (priorToggle) {
        const originalText = priorToggle.textContent.trim();
        const isParameters = originalText.includes("parameters");
        priorToggle.textContent = isParameters ? "Show full parameters" : "Show full prompt";
      }
      index = (nextIndex + slides.length) % slides.length;
      slides[index].classList.add("is-active");
    };

    const next = () => show(index + 1);
    const prev = () => show(index - 1);

    let timer = slides.length > 1 ? setInterval(next, intervalMs) : null;
    const resetTimer = () => {
      if (!timer) return;
      clearInterval(timer);
      timer = setInterval(next, intervalMs);
    };

    carousel.querySelectorAll(".carousel-btn.next").forEach((btn) => {
      btn.addEventListener("click", () => {
        next();
        resetTimer();
      });
    });
    carousel.querySelectorAll(".carousel-btn.prev").forEach((btn) => {
      btn.addEventListener("click", () => {
        prev();
        resetTimer();
      });
    });
  });
});
</script>

## Quality Enhancements: Transfer 2.5 vs Transfer 1

Compared to Cosmos Transfer 1, Cosmos Transfer 2.5 offers significant improvements in both **video quality** and **inference speed**. The examples below show side-by-side comparisons where each video transitions between Transfer 1 results and Transfer 2.5 results, illustrating the quality of improvements achieved in the latest version.

### Examples

<div class="carousel" data-interval="5000">
  <div class="carousel-track">
    <article class="carousel-slide is-active">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/robot1_t1_t2.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
    </article>

    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/robot2_t1_t2.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
    </article>

  </div>
</div>

---

## Document Information

**Publication Date:** November 12, 2025

### Citation

If you use this content or reference this work, please cite it as:

```bibtex
@misc{cosmos_cookbook_robotics_gallery_2025,
  title={Robotics Domain Adaptation Gallery},
  author={Wagwani, Raju and Sriram, Jathavan and Yarlett, Richard and Bapst, Joshua and Gu, Jinwei},
  year={2025},
  month={November},
  howpublished={\url{https://nvidia-cosmos.github.io/cosmos-cookbook/gallery/robotics_inference.html}},
  note={NVIDIA Cosmos Cookbook}
}
```

**Suggested text citation:**

> Raju Wagwani, Jathavan Sriram, Richard Yarlett, Joshua Bapst, & Jinwei Gu (2025). Robotics Domain Adaptation Gallery. In *NVIDIA Cosmos Cookbook*. Accessible at <https://nvidia-cosmos.github.io/cosmos-cookbook/gallery/robotics_inference.html>
