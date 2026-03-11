# Physical Plausibility Check — Cosmos Reason 1 Post-Training

## Model

`nvidia/Cosmos-Reason1-7B`

## Data Source

<!--
  Access: Gated — requires accepting the NVIDIA Open Model License
  at https://huggingface.co/nvidia/Cosmos-Reason1-7B
  Training dataset: VideoPhy-2 dataset (see recipe for access)
  Size: ~14GB (model)
  License: NVIDIA Open Model License
-->

**Access:** Gated — requires accepting the NVIDIA Open Model License at https://huggingface.co/nvidia/Cosmos-Reason1-7B
**Size:** ~14GB
**License:** NVIDIA Open Model License

```bash
huggingface-cli download nvidia/Cosmos-Reason1-7B --repo-type model --local-dir ./models/Cosmos-Reason1-7B
```

## Compute Requirements

1x A100 80GB (min)

## Cosmos Metadata

| Field     | Value                                                                                                                                                                                     |
|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Workload  | post-training                                                                                                                                                                             |
| Domain    | domain:robotics, domain:autonomous-vehicles                                                                                                                                               |
| Technique | technique:post-training, technique:data-curation                                                                                                                                          |
| Tags      | post-training, reason-1, physical-plausibility, data-quality                                                                                                                              |
| Summary   | Fine-tunes Cosmos Reason 1 on the VideoPhy-2 dataset to score video physical plausibility on a 1–5 scale, enabling quality filtering of synthetically generated video in Cosmos Predict/Transfer SDG pipelines. |
