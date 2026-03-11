# Intelligent Transportation Scene Understanding — Cosmos Reason 1 Post-Training

## Model

`nvidia/Cosmos-Reason1-7B`

## Data Source

<!--
  Access: Gated — requires accepting the NVIDIA Open Model License
  at https://huggingface.co/nvidia/Cosmos-Reason1-7B
  Training dataset: WovenTraffic Safety dataset (see recipe for access)
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

8x A100 80GB (min)

## Cosmos Metadata

| Field     | Value                                                                                                                                                                           |
|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Workload  | post-training                                                                                                                                                                   |
| Domain    | domain:autonomous-vehicles                                                                                                                                                      |
| Technique | technique:post-training                                                                                                                                                         |
| Tags      | post-training, reason-1, transportation                                                                                                                                         |
| Summary   | Fine-tunes Cosmos Reason 1-7B on the WovenTraffic Safety dataset for intelligent transportation scene understanding, road attribute recognition, and pedestrian situation analysis using SFT. |
