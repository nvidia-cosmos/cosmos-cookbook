# Cloud Platform

Pick your cloud to get started. Each section below is a horizontal carousel you can scroll through; only Brev is wired up today, while the rest are placeholders for upcoming guides.

<style>
.platform-board {
  display: flex;
  flex-direction: column;
  gap: 2rem;
}

.platform-section {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  padding: 1rem;
  border: 1px solid var(--md-default-fg-color--lightest, #e2e8f0);
  background: var(--md-default-bg-color, #fff);
  box-shadow: 0 8px 30px rgba(0, 0, 0, 0.04);
  border-radius: 0px;
}

.platform-header {
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  gap: 0.75rem;
  align-items: baseline;
  margin-bottom: 0.75rem;
}

.platform-header h2 {
  margin: 0;
}

.platform-header p {
  margin: 0;
  color: var(--md-default-fg-color--light, #5b6472);
}

.platform-logo {
  height: 1.3em;
  width: auto;
  max-width: 2.8em;
  margin-right: 0.4em;
  vertical-align: middle;
  object-fit: contain;
}

.platform-logo-coreweave,
.platform-logo-bytedance {
  max-height: 0.9em;
}

.platform-logo-crusoe {
  max-height: 1.1em;
}

.platform-logo-nvidia {
  max-height: 1.3em;
}

.platform-logo-yotta {
  max-height: 1.7em;
}

.platform-track {
  display: flex;
  gap: 1rem;
  overflow-x: auto;
  padding: 0.25rem 0.25rem 0.75rem;
  scroll-snap-type: x mandatory;
}

.platform-track::-webkit-scrollbar {
  height: 10px;
}

.platform-track::-webkit-scrollbar-thumb {
  background: var(--md-accent-fg-color, #76b900);
  border-radius: 0px;
}

.platform-card {
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

.platform-card:hover {
  border-color: var(--md-accent-fg-color, #76b900);
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.08);
}

.platform-card.is-placeholder {
  pointer-events: none;
}

.platform-card.is-hidden {
  display: none;
}

.platform-media {
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

.platform-title {
  font-weight: 700;
  line-height: 1.3;
}

.platform-show-more {
  display: inline-flex;
  margin-top: 0.5rem;
  margin-left: auto;
}

@media (max-width: 640px) {
  .platform-card {
    flex-basis: 200px;
  }
}
</style>

<div class="platform-board">

<section class="platform-section" id="brev">
  <div class="platform-header">
    <h2><img class="platform-logo platform-logo-nvidia" src="../assets/images/clouds/brev.png" alt="Brev logo">Brev</h2>
    <p>Ready-to-launch notebooks on Brev workspaces.</p>
  </div>
  <div class="platform-track" data-page-size="4">
    <a class="platform-card" href="./brev/reason1/reason1_on_brev.html">
      <div class="platform-media" aria-hidden="true">Placeholder</div>
      <div class="platform-title">Brev on Reason1</div>
    </a>
    <a class="platform-card" href="./brev/transfer2_5/transfer_and_predict_on_brev.html">
      <div class="platform-media" aria-hidden="true">Placeholder</div>
      <div class="platform-title">Brev on Predict &amp; Transfer2.5</div>
    </a>
  </div>
  <button class="md-button platform-show-more" type="button">+ Show more</button>
</section>

<section class="platform-section" id="aws">
  <div class="platform-header">
    <h2><img class="platform-logo" src="../assets/images/clouds/aws.svg" alt="AWS logo">AWS</h2>
    <p>Placeholder for AWS GPU stacks and quickstarts.</p>
  </div>
  <div class="platform-track" data-page-size="4">
    <div class="platform-card is-placeholder">
      <div class="platform-media" aria-hidden="true">Placeholder</div>
      <div class="platform-title">Coming soon</div>
    </div>
    <div class="platform-card is-placeholder">
      <div class="platform-media" aria-hidden="true">Placeholder</div>
      <div class="platform-title">Coming soon</div>
    </div>
  </div>
  <button class="md-button platform-show-more" type="button">+ Show more</button>
</section>

<section class="platform-section" id="azure">
  <div class="platform-header">
    <h2><img class="platform-logo" src="../assets/images/clouds/azure.svg" alt="Azure logo">Azure</h2>
    <p>Placeholder for Azure deployment guides.</p>
  </div>
  <div class="platform-track" data-page-size="4">
    <div class="platform-card is-placeholder">
      <div class="platform-media" aria-hidden="true">Placeholder</div>
      <div class="platform-title">Coming soon</div>
    </div>
    <div class="platform-card is-placeholder">
      <div class="platform-media" aria-hidden="true">Placeholder</div>
      <div class="platform-title">Coming soon</div>
    </div>
  </div>
  <button class="md-button platform-show-more" type="button">+ Show more</button>
</section>

<section class="platform-section" id="gcp">
  <div class="platform-header">
    <h2><img class="platform-logo" src="../assets/images/clouds/gcp.svg" alt="GCP logo">GCP</h2>
    <p>Placeholder for GCP integrations.</p>
  </div>
  <div class="platform-track" data-page-size="4">
    <div class="platform-card is-placeholder">
      <div class="platform-media" aria-hidden="true">Placeholder</div>
      <div class="platform-title">Coming soon</div>
    </div>
    <div class="platform-card is-placeholder">
      <div class="platform-media" aria-hidden="true">Placeholder</div>
      <div class="platform-title">Coming soon</div>
    </div>
  </div>
  <button class="md-button platform-show-more" type="button">+ Show more</button>
</section>

<section class="platform-section" id="oci">
  <div class="platform-header">
    <h2><img class="platform-logo" src="../assets/images/clouds/oci.svg" alt="OCI logo">OCI</h2>
    <p>Placeholder for Oracle Cloud deployments.</p>
  </div>
  <div class="platform-track" data-page-size="4">
    <div class="platform-card is-placeholder">
      <div class="platform-media" aria-hidden="true">Placeholder</div>
      <div class="platform-title">Coming soon</div>
    </div>
    <div class="platform-card is-placeholder">
      <div class="platform-media" aria-hidden="true">Placeholder</div>
      <div class="platform-title">Coming soon</div>
    </div>
  </div>
  <button class="md-button platform-show-more" type="button">+ Show more</button>
</section>

<section class="platform-section" id="alibaba">
  <div class="platform-header">
    <h2><img class="platform-logo" src="../assets/images/clouds/alibaba.svg" alt="Alibaba Cloud logo">Alibaba</h2>
    <p>Placeholder for Alibaba Cloud playbooks.</p>
  </div>
  <div class="platform-track" data-page-size="4">
    <div class="platform-card is-placeholder">
      <div class="platform-media" aria-hidden="true">Placeholder</div>
      <div class="platform-title">Coming soon</div>
    </div>
    <div class="platform-card is-placeholder">
      <div class="platform-media" aria-hidden="true">Placeholder</div>
      <div class="platform-title">Coming soon</div>
    </div>
  </div>
  <button class="md-button platform-show-more" type="button">+ Show more</button>
</section>

<section class="platform-section" id="bytedance">
  <div class="platform-header">
    <h2><img class="platform-logo platform-logo-bytedance" src="../assets/images/clouds/bytedance.png" alt="ByteDance logo">Bytedance</h2>
    <p>Placeholder for Bytedance/FDU resources.</p>
  </div>
  <div class="platform-track" data-page-size="4">
    <div class="platform-card is-placeholder">
      <div class="platform-media" aria-hidden="true">Placeholder</div>
      <div class="platform-title">Coming soon</div>
    </div>
    <div class="platform-card is-placeholder">
      <div class="platform-media" aria-hidden="true">Placeholder</div>
      <div class="platform-title">Coming soon</div>
    </div>
  </div>
  <button class="md-button platform-show-more" type="button">+ Show more</button>
</section>

<section class="platform-section" id="coreweave">
  <div class="platform-header">
    <h2><img class="platform-logo platform-logo-coreweave" src="../assets/images/clouds/coreweave.png" alt="CoreWeave logo">Coreweave</h2>
    <p>Placeholder for Coreweave GPU recipes.</p>
  </div>
  <div class="platform-track" data-page-size="4">
    <div class="platform-card is-placeholder">
      <div class="platform-media" aria-hidden="true">Placeholder</div>
      <div class="platform-title">Coming soon</div>
    </div>
    <div class="platform-card is-placeholder">
      <div class="platform-media" aria-hidden="true">Placeholder</div>
      <div class="platform-title">Coming soon</div>
    </div>
  </div>
  <button class="md-button platform-show-more" type="button">+ Show more</button>
</section>

<section class="platform-section" id="nebius">
  <div class="platform-header">
    <h2><img class="platform-logo" src="../assets/images/clouds/nebius.png" alt="Nebius logo">Nebius</h2>
    <p>Placeholder for Nebius cloud workflows.</p>
  </div>
  <div class="platform-track" data-page-size="4">
    <div class="platform-card is-placeholder">
      <div class="platform-media" aria-hidden="true">Placeholder</div>
      <div class="platform-title">Coming soon</div>
    </div>
    <div class="platform-card is-placeholder">
      <div class="platform-media" aria-hidden="true">Placeholder</div>
      <div class="platform-title">Coming soon</div>
    </div>
  </div>
  <button class="md-button platform-show-more" type="button">+ Show more</button>
</section>

<section class="platform-section" id="lambda">
  <div class="platform-header">
    <h2><img class="platform-logo" src="../assets/images/clouds/lambda.svg" alt="Lambda logo">Lambda</h2>
    <p>Placeholder for Lambda Cloud guides.</p>
  </div>
  <div class="platform-track" data-page-size="4">
    <div class="platform-card is-placeholder">
      <div class="platform-media" aria-hidden="true">Placeholder</div>
      <div class="platform-title">Coming soon</div>
    </div>
    <div class="platform-card is-placeholder">
      <div class="platform-media" aria-hidden="true">Placeholder</div>
      <div class="platform-title">Coming soon</div>
    </div>
  </div>
  <button class="md-button platform-show-more" type="button">+ Show more</button>
</section>

<section class="platform-section" id="yotta">
  <div class="platform-header">
    <h2><img class="platform-logo platform-logo-yotta" src="../assets/images/clouds/yotta.png" alt="Yotta logo">Yotta</h2>
    <p>Placeholder for Yotta deployments.</p>
  </div>
  <div class="platform-track" data-page-size="4">
    <div class="platform-card is-placeholder">
      <div class="platform-media" aria-hidden="true">Placeholder</div>
      <div class="platform-title">Coming soon</div>
    </div>
    <div class="platform-card is-placeholder">
      <div class="platform-media" aria-hidden="true">Placeholder</div>
      <div class="platform-title">Coming soon</div>
    </div>
  </div>
  <button class="md-button platform-show-more" type="button">+ Show more</button>
</section>

<section class="platform-section" id="crusoe">
  <div class="platform-header">
    <h2><img class="platform-logo platform-logo-crusoe" src="../assets/images/clouds/crusoe.png" alt="Crusoe logo">Crusoe</h2>
    <p>Placeholder for Crusoe GPU stacks.</p>
  </div>
  <div class="platform-track" data-page-size="4">
    <div class="platform-card is-placeholder">
      <div class="platform-media" aria-hidden="true">Placeholder</div>
      <div class="platform-title">Coming soon</div>
    </div>
    <div class="platform-card is-placeholder">
      <div class="platform-media" aria-hidden="true">Placeholder</div>
      <div class="platform-title">Coming soon</div>
    </div>
  </div>
  <button class="md-button platform-show-more" type="button">+ Show more</button>
</section>

</div>

<script>
document.addEventListener("DOMContentLoaded", () => {
  document.body.classList.add("cloud-nav-open");
  // Clean up the flag when leaving or if SPA navigation swaps pages
  const clearFlag = () => document.body.classList.remove("cloud-nav-open");
  window.addEventListener("beforeunload", clearFlag);
  window.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "hidden") clearFlag();
  });

  const defaultPageSize = 4;

  document.querySelectorAll(".platform-section").forEach((section) => {
    const track = section.querySelector(".platform-track");
    if (!track) return;

    const cards = Array.from(track.querySelectorAll(".platform-card"));
    const pageSize = parseInt(track.dataset.pageSize || defaultPageSize, 10);
    const button = section.querySelector(".platform-show-more");

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
