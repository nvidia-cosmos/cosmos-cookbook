# Vision AI Gallery

> **Authors:** [Aiden Chang](https://www.linkedin.com/in/aiden-chang/) • [Akul Santhosh](https://www.linkedin.com/in/akulsanthosh/)

> **Organization:** NVIDIA

We provide a dedicated Brev instance to help you follow along with these examples. The default configuration uses 8× H100 GPUs, but you can switch to 1× H100 to reduce costs (with slower inference performance).

[![Brev Instance](./vs_assets/nv-lb-dark.svg)](https://brev.nvidia.com/launchable/deploy?launchableID=env-36KJKKHoOQkFr3FJPpDtinC7QuX)

## Overview

This page showcases results generated with Cosmos Transfer 2.5 for Vision AI applications. The examples demonstrate sim-to-real transfer across a variety of urban and roadway scenarios, illustrating how source videos can be transformed to reflect different times of day, lighting conditions, weather, environmental effects, and scene elements.

To understand what each control modality does, please refer to our [control modality concepts page](../core_concepts/control_modalities/overview.md). This page will be focused on showing some different results that we can make.

For a detailed explanation of each control modality, please refer to the [control modality concepts page](../core_concepts/control_modalities/overview.md). This gallery focuses on visual results, highlighting the range of transformations achievable with Cosmos Transfer 2.5.

**Use Case**: Vision based applications can leverage these techniques to train, test, and validate perception systems under diverse and challenging conditions without additional data collection.

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
}

.carousel-slide.is-active {
  display: flex;
}

.media-wrap {
  position: relative;
  overflow: visible;
  background: #000;
  margin-bottom: 0.5rem;
}

.media-wrap video {
  width: 100%;
  display: block;
}

/*Override Material theme default border-radius*/
.carousel-slide,
.media-wrap,
.media-wrap video,
.see-more,
.masonry-card {
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
}
</style>

### Input Video

We showcase the different input control modalities used for this highway scene.

<div class="carousel" data-interval="5000">
  <div class="carousel-track">
    <article class="carousel-slide is-active">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="./vs_assets/clip_1_short.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Original RGB Video</div>
          <span class="preview-text"></span>
        </div>
      </div>
    </article>
    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="./vs_assets/clip_1_edge.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Edge Control</div>
        </div>
      </div>
    </article>
    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="./vs_assets/clip_1_seg.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Segmentation Control</div>
        </div>
      </div>
    </article>
    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="./vs_assets/clip_1_depth.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Depth Control</div>
        </div>
      </div>
    </article>
    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="./vs_assets/clip_1_vis.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Vis Control</div>
        </div>
      </div>
    </article>
    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="./vs_assets/clip_1_mask.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Mask used</div>
        </div>
      </div>
    </article>
  </div>
</div>

We now show example results generated using these control modalities.

### Examples

<style>
.masonry-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0.1rem;
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
      <source src="./vs_assets/vs_fog.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="masonry-overlay">
      <div class="label">Parameters - Fog</div>
      <div class="params">guidance: 3, edge: 0.5, depth: 1.0</div>
      <div class="label">Input Prompt</div>
      <div class="prompt">A video of a highway with a dense, heavy fog hangs low over the highway, dramatically reducing visibility and softening the outlines of the surrounding hills and leafless trees. A white sedan travels away from the camera in the right lane, its taillights glowing dimly through the fog. The scene conveys slow-moving traffic under conditions with near-whiteout visibility.</div>
    </div>
  </div>
  <div class="masonry-card">
    <video autoplay loop muted playsinline>
      <source src="./vs_assets/vs_morning_sun.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="masonry-overlay">
      <div class="label">Parameters - Morning Sunlight</div>
      <div class="params">guidance: 3, edge: 1.0, depth: 0.9</div>
      <div class="label">Input Prompt</div>
      <div class="prompt">A video of a winding four-lane divided highway cutting through a rural landscape of rolling hills under clear morning sunlight. The low sun casts long, soft shadows across the gently curving roadway and illuminates dry brown grass and leafless trees along the roadside with a warm, golden glow. A white sedan travels away from the camera in the right lane. The sky is pale blue with thin, high clouds, and the scene captures the calm flow of light traffic in crisp, early-day conditions.</div>
    </div>
  </div>
  <div class="masonry-card">
    <video autoplay loop muted playsinline>
      <source src="./vs_assets/vs_night.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="masonry-overlay">
      <div class="label">Parameters - Night</div>
      <div class="params">guidance: 3, edge: 0.5, depth: 1.0</div>
      <div class="label">Input Prompt</div>
      <div class="prompt">A video of a winding four-lane divided highway cutting through a rural landscape of rolling hills at night. The scene is illuminated primarily by vehicle headlights and sparse roadside lighting, with reflective lane markings and road signs glowing against the dark asphalt. The surrounding hills and leafless trees fade into deep shadows beyond the roadway. A white sedan travels away from the camera in the right lane, its red taillights tracing the gentle S-curve. The sky is black and clouded, and the scene conveys light traffic moving steadily through a quiet, nighttime rural environment.</div>
    </div>
  </div>
  <div class="masonry-card">
    <video autoplay loop muted playsinline>
      <source src="./vs_assets/vs_rain.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="masonry-overlay">
      <div class="label">Parameters - Rain</div>
      <div class="params">guidance: 3, edge: 0.9, depth: 1.0</div>
      <div class="label">Input Prompt</div>
      <div class="prompt">A video of a winding four-lane divided highway cutting through a rural landscape of rolling hills, now soaked by a severe rainstorm. The roadway is partially flooded, with standing water pooling across multiple lanes and flowing toward the shoulders, where drainage ditches have overflowed. Dark, rain-slick asphalt reflects headlights and the gray sky above. A white sedan travels away from the camera in the right lane, sending up wide sprays of water. Sheets of rain reduce visibility, and low clouds hang heavy over the scene, conveying hazardous driving conditions during a flood event.
  </div>
    </div>
  </div>
  <div class="masonry-card">
    <video autoplay loop muted playsinline>
      <source src="./vs_assets/vs_snow.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="masonry-overlay">
      <div class="label">Parameters - Snow</div>
      <div class="params">guidance: 3, edge: 0.9, depth: 1.0</div>
      <div class="label">Input Prompt</div>
      <div class="prompt">A video of a winding four-lane divided highway cutting through a rural landscape of rolling hills blanketed in snow, with patches of icy pavement and snowbanks lining the shoulders. Leafless trees are dusted with fresh snow. A white sedan travels away from the camera in the right lane. The scene captures the flow of light traffic under a cold, gray, overcast winter sky.</div>
    </div>
  </div>
  <div class="masonry-card">
    <video autoplay loop muted playsinline>
      <source src="./vs_assets/wooden_road_1.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="masonry-overlay">
      <div class="label">Parameters - Wooden Road</div>
      <div class="params">guidance: 7, edge: 0.6, seg: 0.4</div>
      <div class="label">Input Prompt</div>
      <div class="prompt">A video of a winding four-lane divided highway cutting through a rural landscape of rolling hills, dry brown grass, and leafless trees. The roadway is constructed from long, weathered wooden planks laid lengthwise, with visible seams, grain patterns, and slight warping between boards. The wooden surface follows the gentle curves of the highway and shows subtle wear from traffic. A white sedan travels away from the camera in the right lane. The scene captures the flow of light traffic on a gray, overcast day.</div>
    </div>
  </div>
  <div class="masonry-card">
    <video autoplay loop muted playsinline>
      <source src="./vs_assets/object.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="masonry-overlay">
      <div class="label">Parameters - Debris</div>
      <div class="params">guidance: 7, edge: 0.5, seg: 0.8, depth: 0.4</div>
      <div class="label">Input Prompt</div>
      <div class="prompt">A video of a winding four-lane divided highway cutting through a rural landscape of rolling hills, dry brown grass, and leafless trees under a gray, overcast sky. A large brown bear stands in the middle of the roadway near the center divide, facing slightly toward the oncoming lanes.
      </div>
    </div>
  </div>
  <div class="masonry-card">
    <video autoplay loop muted playsinline>
      <source src="./vs_assets/small_car.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="masonry-overlay">
      <div class="label">Parameters - Small Car</div>
      <div class="params">guidance: 3, edge: 0.5, seg: 0.4, seg_mask: True, depth: 1.0</div>
      <div class="label">Input Prompt</div>
      <div class="prompt">A video of a winding four-lane divided highway cutting through a rural landscape of rolling hills, dry brown grass, and leafless trees. A blue Smart Fortwo microcar travels away from the camera in the right lane, appearing notably small against the wide roadway. The scene captures the flow of light traffic on a gray, overcast day.</div>
    </div>
  </div>
  <div class="masonry-card">
    <video autoplay loop muted playsinline>
      <source src="./vs_assets/van.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="masonry-overlay">
      <div class="label">Parameters - Van</div>
      <div class="params">guidance: 7, edge: 0.5, seg: 0.8, seg_mask: True, depth: 0.5</div>
      <div class="label">Input Prompt</div>
      <div class="prompt">A video of a winding four-lane divided highway cutting through a rural landscape of rolling hills, dry brown grass, and leafless trees. A black Ford Transit cargo van travels away from the camera in the right lane, its tall, boxy profile clearly visible against the wide roadway. The scene captures the flow of light traffic on a gray, overcast day.</div>
    </div>
  </div>
  <div class="masonry-card">
    <video autoplay loop muted playsinline>
      <source src="./vs_assets/people_generation.mp4" type="video/mp4">
      Your browser does not support the video tag.
    </video>
    <div class="masonry-overlay">
      <div class="label">Parameters - People Generation</div>
      <div class="params">guidance: 3, edge: 0.5, seg: 0.7, depth: 0.5</div>
      <div class="label">Input Prompt</div>
      <div class="prompt">A video of a winding four-lane divided highway cutting through a rural landscape of rolling hills, dry brown grass, and leafless trees. A white sedan travels away from the camera in the right lane. Both sides of the road are lined with wide sidewalks densely populated with pedestrians—dozens of clearly visible people walking in clusters and alone. Individuals wear jackets, hats, and backpacks, some talking to each other, others looking at their phones or walking dogs. The constant movement of people along the sidewalks is a dominant visual element, contrasting with the light vehicle traffic on the road. The scene unfolds under a gray, overcast sky, emphasizing a cool, busy daytime atmosphere.</div>
    </div>
  </div>

</div>

## Edge & Depth Control only for Environmental Variations

This example demonstrates how to transform videos into scenes with different environmental conditions and surface materials using edge and depth control. Edge control preserves the original scene structure and motion, while depth control maintains the spatial relationships between objects. All the prompts are the same as the above examples.

### Fog Changes

This scene shows different fog augmentations generated by varying the control modalities.

<div class="carousel" data-interval="5000">
  <div class="carousel-track">
    <article class="carousel-slide is-active">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="./vs_assets/vs_fog.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">guidance: 3, edge: 0.5, depth: 1.0</span>
        </div>
      </div>
    </article>
    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="./vs_assets/vs_fog_1.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">guidance: 7, edge: 0.5, depth: 1.0</span>
        </div>
      </div>
    </article>
  </div>
</div>

### Lighting Changes

This scene shows different lighting conditions generated by varying the control modalities.

<div class="carousel" data-interval="5000">
  <div class="carousel-track">
    <article class="carousel-slide is-active">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="./vs_assets/vs_morning_sun_1.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">guidance: 3, edge: 0.9, depth: 1.0</span>
        </div>
      </div>
    </article>
    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="./vs_assets/vs_morning_sun_2.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">guidance: 3, edge: 0.5, depth: 1.0</span>
        </div>
      </div>
    </article>
    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="./vs_assets/vs_morning_sun.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">guidance: 3, edge: 1.0, depth: 0.9</span>
        </div>
      </div>
    </article>
    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="./vs_assets/vs_morning_sun_3.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">guidance: 7, edge: 0.9, depth: 1.0</span>
        </div>
      </div>
    </article>
  </div>
</div>

### Night Augmentations

This scene shows different night conditions generated by varying the control modalities.

<div class="carousel" data-interval="5000">
  <div class="carousel-track">
    <article class="carousel-slide is-active">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="./vs_assets/vs_night.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">guidance: 3, edge: 0.5, depth: 1.0</span>
        </div>
      </div>
    </article>
    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="./vs_assets/vs_night_1.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">guidance: 3, edge: 0.9, depth: 1.0</span>
        </div>
      </div>
    </article>
    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="./vs_assets/vs_night_2.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">guidance: 7, edge: 0.5, depth: 1.0</span>
        </div>
      </div>
    </article>
  </div>
</div>

### Rain Augmentations

This scene shows different rainy conditions generated by varying the control modalities.

<div class="carousel" data-interval="5000">
  <div class="carousel-track">
    <article class="carousel-slide is-active">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="./vs_assets/vs_rain.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">guidance: 3, edge: 0.9, depth: 1.0</span>
        </div>
      </div>
    </article>
    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="./vs_assets/vs_rain_1.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">guidance: 3, edge: 1.0, depth: 0.9</span>
        </div>
      </div>
    </article>
  </div>
</div>

### Snow Augmentations

This scene shows different snowy conditions generated by varying the control modalities.

<div class="carousel" data-interval="5000">
  <div class="carousel-track">
    <article class="carousel-slide is-active">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="./vs_assets/vs_snow_1.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">guidance: 3, edge: 1.0</span>
        </div>
      </div>
    </article>
    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="./vs_assets/vs_snow.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Parameters</div>
          <span class="preview-text"></span>
          <span class="full-text">guidance: 3, edge: 1.0, depth: 0.9</span>
        </div>
      </div>
    </article>
  </div>
</div>

## Other Video Examples

Here are some results from similar other videos.

### Video Example 1

<div class="carousel" data-interval="5000">
  <div class="carousel-track">
    <article class="carousel-slide is-active">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="./vs_assets/clip_2_short.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Original RGB Video</div>
        </div>
      </div>
    </article>
    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="./vs_assets/clip_2_lighting_augment.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Parameters - Lighting Augmentation</div>
          <span class="preview-text"></span>
          <span class="full-text">guidance: 3, edge: 1.0, depth: 0.9</span>
        </div>
        <div class="text-block">
          <div class="label">Input Prompt</div>
          <span class="preview-text"></span>
          <span class="full-text">A video overlooking a roadway intersection in bright morning sunlight. Soft golden light casts long, gentle shadows across the pavement, replacing the earlier overcast atmosphere. In the foreground, a black SUV navigates a sweeping curved lane moving from right to left. Beyond a grassy median, a silver sedan travels along a multi-lane main road that runs past a large concrete building and leafless trees. The scene captures a quiet suburban traffic flow, with crisp visibility and the highway stretching into the distance under a clear early-day sky.</span>
        </div>
        <button class="see-more" type="button">Show full prompt</button>
      </div>
    </article>
  </div>
</div>

### Video Example 2

<div class="carousel" data-interval="5000">
  <div class="carousel-track">
    <article class="carousel-slide is-active">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="./vs_assets/clip_3_short.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Original RGB Video</div>
        </div>
      </div>
    </article>
    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="./vs_assets/clip_3_night_augment.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Parameters - Night Augmentation</div>
          <span class="preview-text"></span>
          <span class="full-text">guidance: 3, edge: 0.5, depth: 1.0</span>
        </div>
        <div class="text-block">
          <div class="label">Input Prompt</div>
          <span class="preview-text"></span>
          <span class="full-text">A video looking down at a busy multi-lane intersection at night. Streetlights and traffic signals illuminate the scene, casting pools of warm light and reflections on the dark asphalt. Traffic accelerates forward from the stop line, led by a dark gray sedan and a silver sedan, followed closely by a black muscle car with distinctive white racing stripes. To the right, a black SUV turns onto the cross street, passing a red pickup truck parked on the shoulder. In the distance, a large white FedEx truck travels beneath a metal overhead gantry, its headlights and taillights glowing against embankments of dry grass and leafless trees silhouetted in the darkness.</span>
        </div>
        <button class="see-more" type="button">Show full prompt</button>
      </div>
    </article>
  </div>
</div>

### Video Example 3

<div class="carousel" data-interval="5000">
  <div class="carousel-track">
    <article class="carousel-slide is-active">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="./vs_assets/clip_4_short.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Original RGB Video</div>
        </div>
      </div>
    </article>
    <article class="carousel-slide">
      <div class="media-wrap">
        <video autoplay loop muted playsinline>
          <source src="./vs_assets/clip_4_rain_augment.mp4" type="video/mp4">
          Your browser does not support the video tag.
        </video>
        <button class="carousel-btn prev" type="button" aria-label="Previous">‹</button>
        <button class="carousel-btn next" type="button" aria-label="Next">›</button>
      </div>
      <div class="text-stack">
        <div class="text-block">
          <div class="label">Parameters - Rain Augmentation</div>
          <span class="preview-text"></span>
          <span class="full-text">guidance: 3, edge: 0.5, depth: 1.0</span>
        </div>
        <div class="text-block">
          <div class="label">Input Prompt</div>
          <span class="preview-text"></span>
          <span class="full-text">A static, high-angle shot overlooking a wide bridge during steady rain. The roadway is darkened and slick with water, reflecting headlights and taillights across multiple lanes of traffic. Heavy congestion fills the lanes moving away from the camera, where vehicles—including a white sedan in the foreground—are stopped or inching forward slowly. In contrast, the oncoming lanes to the right remain relatively clear with sparse traffic. Raindrops and light mist soften the view of large overhead metal gantries spanning the road in the distance, set against an industrial backdrop beneath a low, overcast sky.</span>
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

    // let timer = slides.length > 1 ? setInterval(next, intervalMs) : null;
    // const resetTimer = () => {
    //   if (!timer) return;
    //   clearInterval(timer);
    //   timer = setInterval(next, intervalMs);
    // };

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
