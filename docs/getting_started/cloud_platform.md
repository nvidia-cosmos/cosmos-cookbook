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

.platform-media {
  width: 100%;
  height: 150px;
  background: var(--md-default-bg-color, #fff);
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

@media (max-width: 640px) {
  .platform-card {
    flex-basis: 200px;
  }
}

.platform-logo-small {
  max-height: 1.5em;
  margin-right: 0.5rem;
  vertical-align: middle;
}
</style>

<div class="platform-board">

<section class="platform-section" id="brev">
  <div class="platform-header">
    <h2><img class="platform-logo platform-logo-small" src="../assets/images/clouds/brev.png" alt="Brev logo"> Brev</h2>
    <p>Ready-to-launch notebooks on Brev workspaces.</p>
  </div>
  <div class="platform-track">
    <a class="platform-card" href="./brev/reason2/reason2_on_brev.html">
      <div class="platform-media" aria-hidden="true">
        <img src="./brev/reason2/reason2.png" alt="Reason 2 on Brev">
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
      <div class="platform-title">Transfer & Predict 2.5 on Brev</div>
    </a>
  </div>
</section>

</div>
