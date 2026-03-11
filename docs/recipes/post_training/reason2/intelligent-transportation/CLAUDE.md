# Intelligent Transportation Scene Understanding — Cosmos Reason 2 Post-Training

## Model

`nvidia/Cosmos-Reason2-8B`

## Data Source

<!--
  Access: Gated — requires accepting the NVIDIA Open Model License
  at https://huggingface.co/nvidia/Cosmos-Reason2-8B
  Training dataset: WovenTraffic Safety dataset (see recipe for access)
  Size: ~16GB (model)
  License: NVIDIA Open Model License
-->

**Access:** Gated — requires accepting the NVIDIA Open Model License at https://huggingface.co/nvidia/Cosmos-Reason2-8B
**Size:** ~16GB
**License:** NVIDIA Open Model License

```bash
huggingface-cli download nvidia/Cosmos-Reason2-8B --repo-type model --local-dir ./models/Cosmos-Reason2-8B
```

## Compute Requirements

8x A100 80GB (min)

## Cosmos Metadata

| Field     | Value                                                                                                                                                                           |
|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Workload  | post-training                                                                                                                                                                   |
| Domain    | domain:autonomous-vehicles                                                                                                                                                      |
| Technique | technique:post-training                                                                                                                                                         |
| Tags      | post-training, reason-2, transportation                                                                                                                                         |
| Summary   | Fine-tunes Cosmos Reason 2-8B on the WovenTraffic Safety dataset for intelligent transportation scene understanding, road attribute recognition, and pedestrian situation analysis using SFT. |
