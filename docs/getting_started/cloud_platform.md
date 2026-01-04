# Cloud Platform

Pick your cloud and access the deployment guide to get started.

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

.platform-logo-nvidia {
  max-height: 1.3em;
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
  background: #fff;
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--md-default-fg-color--light, #5b6472);
  font-weight: 700;
  letter-spacing: 0.02em;
}

.platform-media img,
.platform-media video {
  width: 100%;
  height: 100%;
  object-fit: contain;
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
    <a class="platform-card" href="./brev/reason2/reason2_on_brev.html">
      <div class="platform-media" aria-hidden="true">
        <img src="https://private-user-images.githubusercontent.com/815124/497340553-28f2d612-bbd6-44a3-8795-833d05e9f05f.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njc1NDgyMzgsIm5iZiI6MTc2NzU0NzkzOCwicGF0aCI6Ii84MTUxMjQvNDk3MzQwNTUzLTI4ZjJkNjEyLWJiZDYtNDRhMy04Nzk1LTgzM2QwNWU5ZjA1Zi5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTA0JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDEwNFQxNzMyMThaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT0yOWQ2NzM2NDk0MTAxMDEwYjViZjFkN2FiZGNkZDJmNzhmMTUwYjQwMzQ1ODQzM2U1NjNiNjM5M2Y4MTg5OWUwJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.g0G2lIb4nFPIdv_ldSzSpc4p-Lm39H7UrZpWq7_oMZk" alt="Reason 2">
      </div>
      <div class="platform-title">Reason 2 on Brev</div>
    </a>
    <a class="platform-card" href="./brev/reason1/reason1_on_brev.html">
      <div class="platform-media" aria-hidden="true">
        <video autoplay loop muted playsinline>
          <source src="./brev/reason1/images/nvidia-cosmos-reason1.mp4" type="video/mp4">
        </video>
      </div>
      <div class="platform-title">Reason 1 on Brev</div>
    </a>
    <a class="platform-card" href="./brev/transfer2_5/transfer_and_predict_on_brev.html">
      <div class="platform-media" aria-hidden="true">
        <video autoplay loop muted playsinline>
          <source src="./brev/transfer2_5/images/nvidia-cosmos-transfer-new.mp4" type="video/mp4">
        </video>
      </div>
      <div class="platform-title">Transfer &amp; Predict 2.5 on Brev</div>
    </a>
  </div>
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
