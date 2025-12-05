# All Recipes

Explore the full Cosmos Cookbook catalog in one place. Each category below is a horizontal carousel you can scroll independently. For this test, we cap the initial view to four recipes per category—use **+ Show more** to reveal additional entries.

<style>
.recipe-board {
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

.recipe-category {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  padding: 1rem;
  border: 1px solid var(--md-default-fg-color--lightest, #e2e8f0);
  background: var(--md-default-bg-color, #fff);
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.04);
  border-radius: 0px;
}

.category-header {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 0.75rem;
  align-items: baseline;
  margin-bottom: 0.75rem;
}

.category-header h2 {
  margin: 0;
}

.category-header p {
  margin: 0;
  color: var(--md-default-fg-color--light, #5b6472);
}

.recipe-track {
  display: flex;
  gap: 1rem;
  overflow-x: auto;
  padding: 0.25rem 0.25rem 0.75rem;
  scroll-snap-type: x mandatory;
}

.recipe-track::-webkit-scrollbar {
  height: 10px;
}

.recipe-track::-webkit-scrollbar-thumb {
  background: var(--md-accent-fg-color, #76b900);
  border-radius: 0px;
}

.recipe-card {
  flex: 0 0 240px;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  padding: 0.75rem;
  border: 1px solid var(--md-default-fg-color--lightest, #e2e8f0);
  text-decoration: none;
  color: inherit;
  background: var(--md-default-bg-color, #fff);
  scroll-snap-align: start;
  transition: border-color 150ms ease, transform 150ms ease, box-shadow 150ms ease;
  border-radius: 0px;
}

.recipe-card:hover {
  border-color: var(--md-accent-fg-color, #76b900);
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
}

.recipe-card.is-hidden {
  display: none;
}

.recipe-media {
  width: 100%;
  height: 150px;
  border-radius: 0px;
  background: repeating-linear-gradient(
    135deg,
    rgba(118, 185, 0, 0.08),
    rgba(118, 185, 0, 0.08) 12px,
    rgba(118, 185, 0, 0.16) 12px,
    rgba(118, 185, 0, 0.16) 24px
  );
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--md-default-fg-color--light, #5b6472);
  font-weight: 700;
  letter-spacing: 0.02em;
}

.recipe-title {
  font-weight: 700;
  line-height: 1.3;
}

.recipe-show-more {
  display: inline-flex;
  margin-top: 0.5rem;
  margin-left: auto;
}

@media (max-width: 640px) {
  .recipe-card {
    flex-basis: 200px;
  }
}
</style>

<div class="recipe-board">

<section class="recipe-category" id="autonomous-driving">
  <div class="category-header">
    <h2>Autonomous Driving</h2>
    <p>Sim2Real, captioning, and AV-scale workflows for data generation and evaluation.</p>
  </div>
  <div class="recipe-track" data-page-size="4">
    <a class="recipe-card" href="./end2end/smart_city_sdg/workflow_e2e.html">
      <div class="recipe-media" aria-hidden="true">Placeholder</div>
      <div class="recipe-title">Synthetic Data Generation for Smart Cities</div>
    </a>
    <a class="recipe-card" href="./inference/transfer2_5/inference-carla-sdg-augmentation/inference.html">
      <div class="recipe-media" aria-hidden="true">Placeholder</div>
      <div class="recipe-title">Transfer for Drive Simulator</div>
    </a>
    <a class="recipe-card" href="./inference/transfer2_5/inference-real-augmentation/inference.html">
      <div class="recipe-media" aria-hidden="true">Placeholder</div>
      <div class="recipe-title">Transfer Multi-Control Video Editing</div>
    </a>
    <a class="recipe-card" href="./post_training/reason1/av_video_caption_vqa/post_training.html">
      <div class="recipe-media" aria-hidden="true">Placeholder</div>
      <div class="recipe-title">Reason for AV video Captioning and VQA</div>
    </a>
  </div>
  <button class="md-button recipe-show-more" type="button">+ Show more</button>
</section>

<section class="recipe-category" id="intelligent-transportation">
  <div class="category-header">
    <h2>Intelligent Transportation System</h2>
    <p>Weather augmentation, content creation, and VLM post-training for connected roadways.</p>
  </div>
  <div class="recipe-track" data-page-size="4">
    <a class="recipe-card" href="./inference/predict2/inference-its/inference.html">
      <div class="recipe-media" aria-hidden="true">Placeholder</div>
      <div class="recipe-title">Predict for Data Augmentation in ITS</div>
    </a>
    <a class="recipe-card" href="./inference/transfer1/inference-its-weather-augmentation/inference.html">
      <div class="recipe-media" aria-hidden="true">Placeholder</div>
      <div class="recipe-title">Transfer for Data Augmentation in ITS</div>
    </a>
    <a class="recipe-card" href="./post_training/reason1/intelligent-transportation/post_training.html">
      <div class="recipe-media" aria-hidden="true">Placeholder</div>
      <div class="recipe-title">Reason for Visual Q&amp;A in ITS</div>
    </a>
    <a class="recipe-card" href="./post_training/predict2/its-accident/post_training.html">
      <div class="recipe-media" aria-hidden="true">Placeholder</div>
      <div class="recipe-title">Predict for Traffic Anomaly Generation</div>
    </a>
  </div>
  <button class="md-button recipe-show-more" type="button">+ Show more</button>
</section>

<section class="recipe-category" id="robotics-embodied-ai">
  <div class="category-header">
    <h2>Robotics &amp; Embodied AI</h2>
    <p>GR00T and robotics-first pipelines for navigation, manipulation, and temporal reasoning.</p>
  </div>
  <div class="recipe-track" data-page-size="4">
    <a class="recipe-card" href="./inference/transfer1/gr00t-mimic/inference.html">
      <div class="recipe-media" aria-hidden="true">Placeholder</div>
      <div class="recipe-title">Transfer for Gr00t-Mimic</div>
    </a>
    <a class="recipe-card" href="./inference/transfer1/inference-x-mobility/inference.html">
      <div class="recipe-media" aria-hidden="true">Placeholder</div>
      <div class="recipe-title">Transfer for Robotics Navigation Tasks</div>
    </a>
    <a class="recipe-card" href="./post_training/predict2/gr00t-dreams/post-training.html">
      <div class="recipe-media" aria-hidden="true">Placeholder</div>
      <div class="recipe-title">Predict for Gr00t-Dreams</div>
    </a>
    <a class="recipe-card" href="./post_training/reason1/physical-plausibility-check/post_training.html">
      <div class="recipe-media" aria-hidden="true">Placeholder</div>
      <div class="recipe-title">Reason for Physical Plausibility Check</div>
    </a>
    <a class="recipe-card" href="./post_training/reason1/temporal_localization/post_training.html">
      <div class="recipe-media" aria-hidden="true">Placeholder</div>
      <div class="recipe-title">Reason for MimicGen temporal localization</div>
    </a>
  </div>
  <button class="md-button recipe-show-more" type="button">+ Show more</button>
</section>

<section class="recipe-category" id="smart-spaces">
  <div class="category-header">
    <h2>Smart Spaces (Indoor)</h2>
    <p>Warehouse and facility-focused flows for safety, tracking, and domain-transfer enrichment.</p>
  </div>
  <div class="recipe-track" data-page-size="4">
    <a class="recipe-card" href="./inference/transfer1/inference-warehouse-mv/inference.html">
      <div class="recipe-media" aria-hidden="true">Placeholder</div>
      <div class="recipe-title">Transfer for Warehouse Safety</div>
    </a>
    <a class="recipe-card" href="./post_training/reason1/spatial-ai-warehouse/post_training.html">
      <div class="recipe-media" aria-hidden="true">Placeholder</div>
      <div class="recipe-title">Reason for Warehouse Safety</div>
    </a>
    <a class="recipe-card" href="./inference/transfer2_5/biotrove_augmentation/inference.html">
      <div class="recipe-media" aria-hidden="true">Placeholder</div>
      <div class="recipe-title">Transfer for Data Augmentation in Biodiversity</div>
    </a>
    <a class="recipe-card" href="./data_curation/predict2_data/data_curation.html">
      <div class="recipe-media" aria-hidden="true">Placeholder</div>
      <div class="recipe-title">Curate data for Cosmos-Predict Fine-Tuning</div>
    </a>
  </div>
  <button class="md-button recipe-show-more" type="button">+ Show more</button>
</section>

</div>

<script>
document.addEventListener("DOMContentLoaded", () => {
  const defaultPageSize = 4;

  document.querySelectorAll(".recipe-category").forEach((section) => {
    const track = section.querySelector(".recipe-track");
    if (!track) return;

    const cards = Array.from(track.querySelectorAll(".recipe-card"));
    const pageSize = parseInt(track.dataset.pageSize || defaultPageSize, 10);
    const button = section.querySelector(".recipe-show-more");

    if (cards.length <= pageSize) {
      if (button) button.style.display = "none";
      return;
    }

    cards.forEach((card, index) => {
      if (index >= pageSize) {
        card.classList.add("is-hidden");
      }
    });

    if (!button) return;

    let visibleCount = pageSize;

    button.addEventListener("click", () => {
      visibleCount = Math.min(visibleCount + pageSize, cards.length);
      cards.slice(0, visibleCount).forEach((card) => card.classList.remove("is-hidden"));

      if (visibleCount >= cards.length) {
        button.style.display = "none";
      }
    });
  });
});
</script>
