# Model Evaluation Reason

## Standard Benchmarks

Cosmos Reason models can be evaluated using standardized benchmarks that assess reasoning capabilities across diverse scenarios. The [Cosmos Reason 1 Benchmark Example](https://github.com/nvidia-cosmos/cosmos-reason1/blob/main/examples/benchmark/README.md) provides instructions for running evaluation subsets, including physical reasoning, spatial understanding, and temporal consistency assessments.

## Custom Evaluation on Your Data

Use your own video data (e.g., robotics, egocentric) to probe task‑specific reasoning.

### Prompt Templates

- "What is happening in this clip?"
- "Describe the motion"
- Domain-specific questions tailored to your use case

### What to Measure

- **Answer correctness**: Manual review or LLM-as-a-judge
- **Consistency across time**: Temporal coherence of responses
- **Groundedness**: References what is actually visible
- **Precision vs hallucination**: Especially important post-training

## Automatic Metrics (During Post-Training)

### Instruction Tuning (SFT)

Generate answers on a held‑out set and measure:

- **Per-token loss / perplexity**: Used on held‑out instruction–response pairs
- **Text similarity**: BLEU, ROUGE, METEOR vs ground‑truth captions
- **Embedding similarity**: CLIPScore, BERTScore vs reference answers

### Video–Caption Post-Training

When post‑training on `<video, caption>` pairs:

- Build an evaluation set in **MCQ/BCQ** format with ground truth
- Track whether the model improves at video understanding over time
- Monitor reasoning and comprehension improvements
