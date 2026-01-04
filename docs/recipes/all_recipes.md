# All recipes

<style>
  .recipe-page {
    --recipe-right-bleed: clamp(2rem, 12vw, 18rem);
    width: calc(100% + var(--recipe-right-bleed));
    margin-right: calc(-1 * var(--recipe-right-bleed));
    padding-right: 0.5rem;
  }

  .recipe-board {
    display: flex;
    flex-direction: column;
    gap: 2rem;
    margin-top: 1.5rem;
  }

  .recipe-intro {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    margin-bottom: 1.5rem;
  }

  .recipe-intro p {
    margin: 0;
  }

  .recipe-category {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }

  .recipe-category-body {
    padding: 1.1rem;
    border: 1px solid rgba(255, 255, 255, 0.35);
    background: var(--md-default-bg-color, #111111);
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.25);
    border-radius: 0;
  }

  .category-header {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    /*Stretch header text to full available width so it wraps later (matches carousel width).*/
    align-items: stretch;
  }

  .category-header h2 {
    margin: 0;
  }

  .category-header p {
    margin: 0;
    color: var(--md-default-fg-color--light, #b7bec8);
    /*Don't artificially constrain the description width; let it use the full content width.*/
    max-width: none;
  }

  .recipe-track {
    display: flex;
    gap: 1rem;
    overflow-x: auto;
    padding: 0.25rem 0.25rem 0.75rem;
    scroll-snap-type: x mandatory;
    align-items: stretch;
    position: relative;
    z-index: 0;
  }

  .recipe-track::-webkit-scrollbar {
    height: 8px;
  }

  .recipe-track::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.08);
  }

  .recipe-track::-webkit-scrollbar-thumb {
    background: var(--md-accent-fg-color, #76b900);
    border-radius: 0;
  }

  .recipe-card {
    flex: 0 0 250px;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    padding: 0.8rem;
    border: 1px solid rgba(255, 255, 255, 0.35);
    text-decoration: none;
    color: var(--md-default-fg-color, #f2f2f2);
    background: var(--md-default-bg-color, #111111);
    scroll-snap-align: start;
    transition: border-color 150ms ease, transform 150ms ease, box-shadow 150ms ease;
    border-radius: 0;
  }

  .recipe-card:hover {
    border-color: var(--md-accent-fg-color, #76b900);
    transform: translateY(-2px);
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.3);
  }

  .recipe-media {
    width: 100%;
    height: 150px;
    border-radius: 0;
    border: 1px solid rgba(255, 255, 255, 0.18);
    background: repeating-linear-gradient(
      135deg,
      rgba(118, 185, 0, 0.1),
      rgba(118, 185, 0, 0.1) 14px,
      rgba(118, 185, 0, 0.2) 14px,
      rgba(118, 185, 0, 0.2) 28px
    );
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--md-default-fg-color--light, #b7bec8);
    font-weight: 700;
    letter-spacing: 0.02em;
    position: relative;
    overflow: hidden;
  }

  /*For cards where we show a real image, use a clean background instead of the placeholder pattern.*/
  .recipe-media--image {
    background: #ffffff;
  }

  .recipe-media img {
    width: 100%;
    height: 100%;
    /*Avoid cropping hero images in carousels.*/
    object-fit: contain;
    /*When `object-fit: contain` letterboxes, keep the empty area white in dark mode too.*/
    background: #ffffff;
    display: block;
  }

  .recipe-media--video {
    /*Match image cards: white background so letterboxing isn't black.*/
    background: #ffffff;
  }

  .recipe-media video {
    width: 100%;
    height: 100%;
    object-fit: contain;
    /*Match image cards: white background so letterboxing isn't black.*/
    background: #ffffff;
    display: block;
    /*Ensure the whole card remains clickable; video shouldn't capture pointer events.*/
    pointer-events: none;
  }

  .recipe-title {
    font-weight: 700;
    line-height: 1.3;
    color: var(--md-accent-fg-color, #76b900);
  }

  .recipe-tag {
    align-self: flex-start;
    margin-top: auto;
    display: inline-flex;
    align-items: center;
    padding: 0.15rem 0.4rem;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    line-height: 1.1;
    border-radius: 0;
    color: #ffffff;
  }

  .recipe-tag--inference {
    background-color: #76b900;
  }

  .recipe-tag--post-training {
    background-color: #4aa3ff;
  }

  .recipe-tag--curation {
    background-color: #f5a623;
  }

  .recipe-tag--workflow {
    background-color: #36c2b2;
  }

  .recipe-board .is-hidden {
    display: none;
  }

  .recipe-show-more {
    flex: 0 0 auto;
    align-self: flex-end;
    margin-left: 0.25rem;
    margin-bottom: 0.1rem;
    padding: 0.3rem 0.6rem;
    font-size: 0.75rem;
    line-height: 1;
    border-radius: 0;
    position: relative;
    z-index: 2;
    pointer-events: auto;
    cursor: pointer;
  }

  .recipe-show-more.md-button {
    border-radius: 0;
  }

  .md-sidebar--secondary {
    display: none;
  }

  .md-content__inner {
    overflow: visible;
  }

  @media (max-width: 640px) {
    .recipe-card {
      flex-basis: 210px;
    }
  }
</style>

<div class="recipe-page">
  <div class="recipe-intro">
    <p>Discover Cosmos recipes across every domain in one place.</p>
  </div>

  <div class="recipe-board">
  <section class="recipe-category" id="robotics">
    <div class="category-header">
      <h2>Robotics</h2>
      <p>Manipulation, navigation, and embodied reasoning workflows for robot training.</p>
    </div>
    <div class="recipe-category-body">
      <div class="recipe-track" data-page-size="6" aria-label="Robotics recipes">
        <a class="recipe-card" href="./inference/reason2/intbot_showcase/inference.html">
          <div class="recipe-media recipe-media--image" aria-hidden="true">
            <img src="./inference/reason2/intbot_showcase/assets/IntBot-GTC.jpg" alt="" loading="lazy" />
          </div>
          <div class="recipe-title">Egocentric Social and Physical Reasoning with Cosmos-Reason2-8B</div>
          <div class="recipe-tag recipe-tag--inference">Inference</div>
        </a>
        <a class="recipe-card" href="./inference/transfer1/inference-warehouse-mv/inference.html">
          <div class="recipe-media recipe-media--image" aria-hidden="true">
            <img src="./inference/transfer1/inference-warehouse-mv/assets/multi_world_simulation.png" alt="" loading="lazy" />
          </div>
          <div class="recipe-title">Cosmos Transfer 1 Sim2Real for Multi-View Warehouse Detection and Tracking</div>
          <div class="recipe-tag recipe-tag--inference">Inference</div>
        </a>
        <a class="recipe-card" href="./inference/transfer1/gr00t-mimic/inference.html">
          <div class="recipe-media recipe-media--image" aria-hidden="true">
            <img src="./inference/transfer1/gr00t-mimic/assets/hero.png" alt="" loading="lazy" />
          </div>
          <div class="recipe-title">Isaac GR00T-Mimic for Synthetic Manipulation Motion Generation</div>
          <div class="recipe-tag recipe-tag--inference">Inference</div>
        </a>
        <a class="recipe-card" href="./inference/transfer1/inference-x-mobility/inference.html">
          <div class="recipe-media recipe-media--image" aria-hidden="true">
            <img src="./inference/transfer1/inference-x-mobility/assets/training.png" alt="" loading="lazy" />
          </div>
          <div class="recipe-title">Cosmos Transfer Sim2Real for Robotics Navigation Tasks</div>
          <div class="recipe-tag recipe-tag--inference">Inference</div>
        </a>
        <a class="recipe-card" href="./post_training/predict2/gr00t-dreams/post_training.html">
          <div class="recipe-media recipe-media--image" aria-hidden="true">
            <img src="./post_training/predict2/gr00t-dreams/assets/hero.png" alt="" loading="lazy" />
          </div>
          <div class="recipe-title">Isaac GR00T-Dreams for Synthetic Trajectory Data Generation</div>
          <div class="recipe-tag recipe-tag--post-training">Post-Training</div>
        </a>
        <a class="recipe-card" href="./post_training/reason1/spatial-ai-warehouse/post_training.html">
          <div class="recipe-media recipe-media--image" aria-hidden="true">
            <img src="./post_training/reason1/spatial-ai-warehouse/assets/e2e_workflow.png" alt="" loading="lazy" />
          </div>
          <div class="recipe-title">Spatial AI for Warehouse Post-Training with Cosmos Reason 1</div>
          <div class="recipe-tag recipe-tag--post-training">Post-Training</div>
        </a>
        <a class="recipe-card" href="./post_training/reason1/temporal_localization/post_training.html">
          <div class="recipe-media recipe-media--image" aria-hidden="true">
            <img src="./post_training/reason1/temporal_localization/assets/events_timeline.png" alt="" loading="lazy" />
          </div>
          <div class="recipe-title">Cosmos Reason for Mimic Gen temporal localization</div>
          <div class="recipe-tag recipe-tag--post-training">Post-Training</div>
        </a>
        <button class="md-button recipe-show-more" type="button">+ Show more</button>
      </div>
    </div>
  </section>

  <section class="recipe-category" id="autonomous-vehicles">
    <div class="category-header">
      <h2>Autonomous Vehicles</h2>
      <p>Simulation, traffic scenarios, and autonomous-vehicle-scale data generation and evaluation.</p>
    </div>
    <div class="recipe-category-body">
      <div class="recipe-track" data-page-size="6" aria-label="Autonomous Vehicles recipes">
        <a class="recipe-card" href="./post_training/transfer2_5/av_world_scenario_maps/post_training.html">
          <div class="recipe-media recipe-media--video" aria-hidden="true">
            <video autoplay loop muted playsinline preload="none" tabindex="-1">
              <source src="./post_training/transfer2_5/av_world_scenario_maps/assets/av_rgb_front_wide.mp4" type="video/mp4">
              Your browser does not support the video tag.
            </video>
          </div>
          <div class="recipe-title">Cosmos Transfer 2.5 Multiview Generation with World Scenario Map Control</div>
          <div class="recipe-tag recipe-tag--post-training">Post-Training</div>
        </a>
        <a class="recipe-card" href="./inference/predict2/inference-its/inference.html">
          <div class="recipe-media recipe-media--image" aria-hidden="true">
            <img src="./inference/predict2/inference-its/assets/output.jpg" alt="" loading="lazy" />
          </div>
          <div class="recipe-title">Cosmos Predict 2 Text2Image for Intelligent Transportation System (ITS) Images</div>
          <div class="recipe-tag recipe-tag--inference">Inference</div>
        </a>
        <a class="recipe-card" href="./inference/transfer1/inference-its-weather-augmentation/inference.html">
          <div class="recipe-media recipe-media--image" aria-hidden="true">
            <img src="./inference/transfer1/inference-its-weather-augmentation/assets/rainy_night_all_09.jpg" alt="" loading="lazy" />
          </div>
          <div class="recipe-title">Cosmos Transfer 1 Weather Augmentation for Intelligent Transportation System (ITS) Images</div>
          <div class="recipe-tag recipe-tag--inference">Inference</div>
        </a>
        <a class="recipe-card" href="./inference/transfer2_5/inference-carla-sdg-augmentation/inference.html">
          <div class="recipe-media recipe-media--image" aria-hidden="true">
            <img src="./inference/transfer2_5/inference-carla-sdg-augmentation/assets/augmentation_matrix_grid.gif" alt="" loading="lazy" />
          </div>
          <div class="recipe-title">Cosmos Transfer 2.5 Sim2Real for Simulator Videos</div>
          <div class="recipe-tag recipe-tag--inference">Inference</div>
        </a>
        <a class="recipe-card" href="./post_training/predict2/its-accident/post_training.html">
          <div class="recipe-media recipe-media--image" aria-hidden="true">
            <img src="./inference/transfer2_5/inference-carla-sdg-augmentation/assets/augmented_anomaly_trajectory.gif" alt="" loading="lazy" />
          </div>
          <div class="recipe-title">Traffic Anomaly Generation with Cosmos Predict2</div>
          <div class="recipe-tag recipe-tag--post-training">Post-Training</div>
        </a>
        <a class="recipe-card" href="./post_training/reason1/intelligent-transportation/post_training.html">
          <div class="recipe-media recipe-media--image" aria-hidden="true">
            <img src="./post_training/reason1/intelligent-transportation/assets/e2e_workflow.png" alt="" loading="lazy" />
          </div>
          <div class="recipe-title">Intelligent Transportation Post-Training with Cosmos Reason 1</div>
          <div class="recipe-tag recipe-tag--post-training">Post-Training</div>
        </a>
        <a class="recipe-card" href="./post_training/reason1/av_video_caption_vqa/post_training.html">
          <div class="recipe-media recipe-media--image" aria-hidden="true">
            <img src="./post_training/reason1/av_video_caption_vqa/assets/sft_results.png" alt="" loading="lazy" />
          </div>
          <div class="recipe-title">SFT for AV video captioning and VQA</div>
          <div class="recipe-tag recipe-tag--post-training">Post-Training</div>
        </a>
        <a class="recipe-card" href="./end2end/smart_city_sdg/workflow_e2e.html">
          <div class="recipe-media recipe-media--image" aria-hidden="true">
            <img src="./end2end/smart_city_sdg/assets/main_workflow.png" alt="" loading="lazy" />
          </div>
          <div class="recipe-title">Synthetic Data Generation (SDG) for Traffic Scenarios</div>
          <div class="recipe-tag recipe-tag--workflow">Workflow</div>
        </a>
        <button class="md-button recipe-show-more" type="button">+ Show more</button>
      </div>
    </div>
  </section>

  <section class="recipe-category" id="vision-ai">
    <div class="category-header">
      <h2>Vision AI</h2>
      <p>Visual generation, curation, and domain transfer across image and video modalities.</p>
    </div>
    <div class="recipe-category-body">
      <div class="recipe-track" data-page-size="6" aria-label="Vision AI recipes">
        <a class="recipe-card" href="./inference/transfer2_5/biotrove_augmentation/inference.html">
          <div class="recipe-media recipe-media--image" aria-hidden="true">
            <img src="./inference/transfer2_5/biotrove_augmentation/assets/moth_biotrove.webp" alt="" loading="lazy" />
          </div>
          <div class="recipe-title">Domain Transfer for BioTrove Moths with Cosmos Transfer 2.5</div>
          <div class="recipe-tag recipe-tag--inference">Inference</div>
        </a>
        <a class="recipe-card" href="./inference/transfer2_5/inference-real-augmentation/inference.html">
          <div class="recipe-media recipe-media--image" aria-hidden="true">
            <img src="./inference/transfer2_5/inference-real-augmentation/assets/omniverse_background_change_recipe.png" alt="" loading="lazy" />
          </div>
          <div class="recipe-title">Multi-Control Recipes with Cosmos Transfer 2.5</div>
          <div class="recipe-tag recipe-tag--inference">Inference</div>
        </a>
        <a class="recipe-card" href="./inference/transfer2_5/inference-image-prompt/inference.html">
          <div class="recipe-media recipe-media--video" aria-hidden="true">
            <video autoplay loop muted playsinline preload="none" tabindex="-1">
              <source src="./inference/transfer2_5/inference-image-prompt/assets/example1_generation-from-edge-sunset.mp4" type="video/mp4">
              Your browser does not support the video tag.
            </video>
          </div>
          <div class="recipe-title">Style-Guided Video Generation with Cosmos Transfer 2.5</div>
          <div class="recipe-tag recipe-tag--inference">Inference</div>
        </a>
        <a class="recipe-card" href="./post_training/predict2_5/sports/post_training.html">
          <div class="recipe-media recipe-media--video" aria-hidden="true">
            <video autoplay loop muted playsinline preload="none" tabindex="-1">
              <source src="./post_training/predict2_5/sports/assets/post_trained/12.mp4" type="video/mp4">
              Your browser does not support the video tag.
            </video>
          </div>
          <div class="recipe-title">LoRA Post-training for Sports Video Generation</div>
          <div class="recipe-tag recipe-tag--post-training">Post-Training</div>
        </a>
        <a class="recipe-card" href="./post_training/reason1/physical-plausibility-check/post_training.html">
          <div class="recipe-media recipe-media--image" aria-hidden="true">
            <img src="./post_training/reason1/physical-plausibility-check/assets/correlation_bar_graph.png" alt="" loading="lazy" />
          </div>
          <div class="recipe-title">Physical Plausibility Prediction with Cosmos Reason 1</div>
          <div class="recipe-tag recipe-tag--post-training">Post-Training</div>
        </a>
        <a class="recipe-card" href="./post_training/reason1/wafermap_classification/post_training.html">
          <div class="recipe-media recipe-media--image" aria-hidden="true">
            <img src="./post_training/reason1/wafermap_classification/assets/Picture6.png" alt="" loading="lazy" />
          </div>
          <div class="recipe-title">Wafer Map Anomaly Classification with Cosmos Reason 1</div>
          <div class="recipe-tag recipe-tag--post-training">Post-Training</div>
        </a>
        <a class="recipe-card" href="./data_curation/predict2_data/data_curation.html">
          <div class="recipe-media recipe-media--image" aria-hidden="true">
            <img src="../core_concepts/data_curation/images/grid_preview.png" alt="" loading="lazy" />
          </div>
          <div class="recipe-title">Curate data for Cosmos Predict Fine-Tuning using Cosmos Curator</div>
          <div class="recipe-tag recipe-tag--curation">Curation</div>
        </a>
        <button class="md-button recipe-show-more" type="button">+ Show more</button>
      </div>
    </div>
  </section>
</div>
</div>

<script>
  const initRecipeCarousels = () => {
    const defaultPageSize = 6;

    document.querySelectorAll(".recipe-category").forEach((section) => {
      const track = section.querySelector(".recipe-track");
      if (!track) {
        return;
      }

      const cards = Array.from(track.querySelectorAll(".recipe-card"));
      const pageSize = parseInt(track.dataset.pageSize || defaultPageSize, 10);
      const button = track.querySelector(".recipe-show-more");

      if (cards.length <= pageSize) {
        if (button) {
          button.classList.add("is-hidden");
        }
        return;
      }

      cards.forEach((card, index) => {
        if (index >= pageSize) {
          card.classList.add("is-hidden");
        }
      });

      if (!button) {
        return;
      }

      let visibleCount = pageSize;

      button.addEventListener("click", () => {
        visibleCount = Math.min(visibleCount + pageSize, cards.length);
        cards.slice(0, visibleCount).forEach((card) => card.classList.remove("is-hidden"));

        if (visibleCount >= cards.length) {
          button.classList.add("is-hidden");
        }
      });
    });
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initRecipeCarousels);
  } else {
    initRecipeCarousels();
  }
</script>
