# Autonomous Vehicle Domain Adaptation Gallery

> **Authors:** [Raju Wagwani](https://www.linkedin.com/in/raju-wagwani-a4746027/) • [Nikolay Matveiev](https://www.linkedin.com/) • [Joshua Bapst](https://www.linkedin.com/in/joshbapst/) • [Jinwei Gu](https://www.linkedin.com/in/jinweigu/)

> **Organization:** NVIDIA

## Overview

This page showcases a collection of results generated using Cosmos Transfer 2.5 for autonomous vehicle (AV) applications. The examples demonstrate how to transform real-world or simulation-based driving videos across various environmental conditions such as different weather, lighting, and time of day. These results are intended to serve as inspiration for users exploring how to leverage the model for domain adaptation and synthetic data augmentation in autonomous driving use cases.

## Driving Scene 1

- **Multi control**: The model uses different controls, each with different control weights to produce an output. This gives the user more control over how the output would look like.
  - **depth**: Maintains 3D realism and spatial consistency
  - **edge**: Preserves original structure, shape, and layout
  - **seg**: Enables structural changes and semantic replacement
  - **vis**: Preserves background, lighting, and overall visual appearance

For detailed explanations of control modalities, see [Control Modalities Overview](../core_concepts/control_modalities/overview.md).

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
  font-family: var(--md-code-font, monospace);
  font-size: 0.9em;
}

.text-block .full-text {
  display: none;
  margin-top: 0.25rem;
  white-space: pre-wrap;
  font-family: var(--md-code-font, monospace);
  font-size: 0.9em;
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
</style>

### Input Video

This scene shows a driving video captured from a dashcam perspective. The examples demonstrate how different environmental conditions (weather, lighting, time of day) can be generated from the same input video while preserving the structure and motion of the driving scene.

<div class="carousel" data-interval="5000">
  <div class="carousel-track">
    <article class="carousel-slide is-active">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/av_car_input.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text">{
  "seed": 5000,
}</span>
          <span class="full-text">{
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
}</span>
        </div>
        <button class="see-more" type="button">Show full parameters</button>
      </div>
    </article>
  </div>
</div>

### Examples

<style>
.masonry-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.1rem;
}

@media (max-width: 1200px) {
  .masonry-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 720px) {
  .masonry-grid {
    grid-template-columns: 1fr;
  }
}

.masonry-card {
  position: relative;
  overflow: hidden;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
  background: #000;
  border-radius: 0px;
}

.masonry-card video {
  width: 100%;
  display: block;
}

.masonry-overlay {
  position: absolute;
  inset: 0;
  background: rgba(0, 0, 0, 0.88);
  color: #fff;
  padding: 1rem;
  display: flex;
  flex-direction: column;
  justify-content: flex-start;
  gap: 0.5rem;
  opacity: 0;
  transition: opacity 150ms ease;
  overflow-y: auto;
}

.masonry-card:hover .masonry-overlay {
  opacity: 1;
}

.masonry-overlay .label {
  font-weight: 700;
  text-transform: uppercase;
  font-size: 0.75rem;
  letter-spacing: 0.04em;
  color: #e2e8f0;
}

.masonry-overlay .prompt,
.masonry-overlay .params {
  font-size: 0.9rem;
  line-height: 1.4;
  white-space: normal;
}

.masonry-card:hover .masonry-overlay .prompt,
.masonry-card:hover .masonry-overlay .params {
  font-size: 0.75rem;
}

</style>

<div class="masonry-grid">
  <div class="masonry-card">
    <video autoplay loop muted playsinline>
      <source src="assets/av_car_output_1.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="masonry-overlay">
      <div class="label">Input Prompt</div>
      <div class="prompt">The video is a driving scene through a modern urban environment, likely captured from a dashcam or a similar fixed camera setup inside a vehicle. The scene unfolds on a wide, multi-lane road flanked by tall, modern buildings with glass facades. The road is relatively empty, with only a few cars visible, including a black car directly ahead of the camera, maintaining a steady pace. The camera remains static, providing a consistent view of the road and surroundings as the vehicle moves forward.On the left side of the road, there are several trees lining the sidewalk, providing a touch of greenery amidst the urban setting. Pedestrians are visible on the sidewalks, some walking leisurely, while others stand near the buildings. The buildings are a mix of architectural styles, with some featuring large glass windows and others having more traditional concrete exteriors. A few commercial signs and logos are visible on the buildings, indicating the presence of businesses and offices.Traffic cones are placed on the road ahead, suggesting some form of roadwork or lane closure, guiding the vehicles to merge or change lanes. The road markings are clear, with white arrows indicating the direction of travel. Throughout the video, the vehicle maintains a steady speed, and the camera captures the gradual approach towards the intersection, where the road splits into different directions. The overall atmosphere is calm and orderly, typical of a city during non-peak hours.  heavy rain, wet road with puddles</div>
      <div class="label">Parameters</div>
      <div class="params">seed: 4000, guidance: 3, depth: 0.5, edge: 0.1, seg: 0.35, vis: 0.0</div>
    </div>
  </div>

  <div class="masonry-card">
    <video autoplay loop muted playsinline>
      <source src="assets/av_car_output_2.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="masonry-overlay">
      <div class="label">Input Prompt</div>
      <div class="prompt">The video is a driving scene through a modern urban environment, likely captured from a dashcam or a similar fixed camera setup inside a vehicle. The scene unfolds on a wide, multi-lane road flanked by tall, modern buildings with glass facades. The road is relatively empty, with only a few cars visible, including a black car directly ahead of the camera, maintaining a steady pace. The camera remains static, providing a consistent view of the road and surroundings as the vehicle moves forward.On the left side of the road, there are several trees lining the sidewalk, providing a touch of greenery amidst the urban setting. Pedestrians are visible on the sidewalks, some walking leisurely, while others stand near the buildings. The buildings are a mix of architectural styles, with some featuring large glass windows and others having more traditional concrete exteriors. A few commercial signs and logos are visible on the buildings, indicating the presence of businesses and offices.Traffic cones are placed on the road ahead, suggesting some form of roadwork or lane closure, guiding the vehicles to merge or change lanes. The road markings are clear, with white arrows indicating the direction of travel. Throughout the video, the vehicle maintains a steady speed, and the camera captures the gradual approach towards the intersection, where the road splits into different directions. The overall atmosphere is calm and orderly, typical of a city during non-peak hours.  night time, bright street lamps and colorful neon lights on buildings</div>
      <div class="label">Parameters</div>
      <div class="params">seed: 4000, guidance: 7, depth: 0.5, edge: 0.1, seg: 0.35, vis: 0.0</div>
    </div>
  </div>

  <div class="masonry-card">
    <video autoplay loop muted playsinline>
      <source src="assets/av_car_output_3.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="masonry-overlay">
      <div class="label">Input Prompt</div>
      <div class="prompt">Dashcam video, driving through a modern urban environment, sunset with beautiful clouds</div>
      <div class="label">Parameters</div>
      <div class="params">seed: 5000, guidance: 3, depth: 0.35, edge: 0.1, seg: 0.35, vis: 0.0</div>
    </div>
  </div>

  <div class="masonry-card">
    <video autoplay loop muted playsinline>
      <source src="assets/av_car_output_4.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="masonry-overlay">
      <div class="label">Input Prompt</div>
      <div class="prompt">Dashcam video, driving through a modern urban environment, heavy rain, wet road with puddles</div>
      <div class="label">Parameters</div>
      <div class="params">seed: 5000, guidance: 3, depth: 0.35, edge: 0.1, seg: 0.35, vis: 0.0</div>
    </div>
  </div>

  <div class="masonry-card">
    <video autoplay loop muted playsinline>
      <source src="assets/av_car_output_5.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="masonry-overlay">
      <div class="label">Input Prompt</div>
      <div class="prompt">Dashcam video, driving through a modern urban environment, winter with heavy snow storm, trees and sidewalks covered in snow</div>
      <div class="label">Parameters</div>
      <div class="params">seed: 8001, guidance: 3, depth: 0.35, edge: 0.15, seg: 0.35, vis: 0.15</div>
    </div>
  </div>

  <div class="masonry-card">
    <video autoplay loop muted playsinline>
      <source src="assets/av_car_output_6.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="masonry-overlay">
      <div class="label">Input Prompt</div>
      <div class="prompt">Dashcam video, driving through a modern urban environment, heavy rain, wet road with puddles</div>
      <div class="label">Parameters</div>
      <div class="params">seed: 5000, guidance: 6, depth: 0.6, edge: 0.1, seg: 0.4, vis: 0.1</div>
    </div>
  </div>

  <div class="masonry-card">
    <video autoplay loop muted playsinline>
      <source src="assets/av_car_output_7.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="masonry-overlay">
      <div class="label">Input Prompt</div>
      <div class="prompt">Dashcam video, driving through a modern urban environment, sunset with beautiful clouds</div>
      <div class="label">Parameters</div>
      <div class="params">seed: 7000, guidance: 3, depth: 0.35, edge: 0.1, seg: 0.35, vis: 0.05</div>
    </div>
  </div>

  <div class="masonry-card">
    <video autoplay loop muted playsinline>
      <source src="assets/av_car_output_8.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="masonry-overlay">
      <div class="label">Input Prompt</div>
      <div class="prompt">Dashcam video, driving through a modern urban environment, winter with heavy snow storm, trees and sidewalks covered in snow</div>
      <div class="label">Parameters</div>
      <div class="params">seed: 7001, guidance: 3, depth: 0.35, edge: 0.1, seg: 0.35, vis: 0.05</div>
    </div>
  </div>

  <div class="masonry-card">
    <video autoplay loop muted playsinline>
      <source src="assets/av_car_output_9.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="masonry-overlay">
      <div class="label">Input Prompt</div>
      <div class="prompt">Dashcam video, driving through a modern urban environment, twilight or early morning, partly cloudy</div>
      <div class="label">Parameters</div>
      <div class="params">seed: 5000, guidance: 3, depth: 0.4, edge: 0.1, seg: 0.5, vis: 0.1</div>
    </div>
  </div>

  <div class="masonry-card">
    <video autoplay loop muted playsinline>
      <source src="assets/av_car_output_10.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="masonry-overlay">
      <div class="label">Input Prompt</div>
      <div class="prompt">Dashcam video, driving through a modern urban environment, night time, bright street lamps lighting up the fog</div>
      <div class="label">Parameters</div>
      <div class="params">seed: 5000, guidance: 3, depth: 0.4, edge: 0.1, seg: 0.5, vis: 0.1</div>
    </div>
  </div>
</div>

## Quality Enhancements: Transfer 2.5 vs Transfer 1

Compared to Cosmos Transfer 1, Cosmos Transfer 2.5 offers significant improvements in both **video quality** and **inference speed**. The examples below show side-by-side comparisons where each video transitions between Transfer 1 results and Transfer 2.5 results, illustrating the quality improvements achieved in the latest version.

<div class="carousel" data-interval="5000">
  <div class="carousel-track">
    <article class="carousel-slide is-active">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/av1_t1_t2.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Comparison</div>
          <span class="preview-text">Example A: Side-by-side comparison showing quality improvements from Transfer 1 to Transfer 2.5.</span>
          <span class="full-text">Example A: This comparison video demonstrates the quality improvements achieved in Cosmos Transfer 2.5 compared to Transfer 1. The video transitions between Transfer 1 results and Transfer 2.5 results, showcasing enhanced video quality, better temporal consistency, and improved inference speed.</span>
        </div>
        <button class="see-more" type="button">Show full description</button>
      </div>
    </article>

    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/av2_t1_t2.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Comparison</div>
          <span class="preview-text">Example B: Side-by-side comparison showing quality improvements from Transfer 1 to Transfer 2.5.</span>
          <span class="full-text">Example B: This comparison video demonstrates the quality improvements achieved in Cosmos Transfer 2.5 compared to Transfer 1. The video transitions between Transfer 1 results and Transfer 2.5 results, showcasing enhanced video quality, better temporal consistency, and improved inference speed.</span>
        </div>
        <button class="see-more" type="button">Show full description</button>
      </div>
    </article>

    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/av3_t1_t2.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Comparison</div>
          <span class="preview-text">Example C: Side-by-side comparison showing quality improvements from Transfer 1 to Transfer 2.5.</span>
          <span class="full-text">Example C: This comparison video demonstrates the quality improvements achieved in Cosmos Transfer 2.5 compared to Transfer 1. The video transitions between Transfer 1 results and Transfer 2.5 results, showcasing enhanced video quality, better temporal consistency, and improved inference speed.</span>
        </div>
        <button class="see-more" type="button">Show full description</button>
      </div>
    </article>

    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="assets/av4_t1_t2.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Comparison</div>
          <span class="preview-text">Example D: Side-by-side comparison showing quality improvements from Transfer 1 to Transfer 2.5.</span>
          <span class="full-text">Example D: This comparison video demonstrates the quality improvements achieved in Cosmos Transfer 2.5 compared to Transfer 1. The video transitions between Transfer 1 results and Transfer 2.5 results, showcasing enhanced video quality, better temporal consistency, and improved inference speed.</span>
        </div>
        <button class="see-more" type="button">Show full description</button>
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
          // Only set preview if it's empty (preserve things like `{ "seed": 1,`)
          if (!preview.textContent.trim()) {
            preview.textContent = firstSentence(full.textContent || "");
          }
        }
      });
      const toggle = slide.querySelector(".see-more");
      if (toggle) {
        const originalText = toggle.textContent.trim();
        const isDescription = originalText.includes("description");
        const isParameters = originalText.includes("parameters");
        toggle.addEventListener("click", () => {
          slide.classList.toggle("expanded");
          const expanded = slide.classList.contains("expanded");
          if (isParameters) {
            toggle.textContent = expanded ? "Hide full parameters" : "Show full parameters";
          } else {
            toggle.textContent = expanded ? "Hide full description" : "Show full description";
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
        if (originalText.includes("parameters")) {
          priorToggle.textContent = "Show full parameters";
        } else {
          priorToggle.textContent = "Show full description";
        }
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
