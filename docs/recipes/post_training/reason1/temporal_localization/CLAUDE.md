# Temporal Localization for Robot Manipulation — Cosmos Reason 1 Post-Training

## Model

`nvidia/Cosmos-Reason1-7B`

## Data Source

<!--
  Access: Gated — requires accepting the NVIDIA Open Model License
  at https://huggingface.co/nvidia/Cosmos-Reason1-7B
  Training dataset: MimicGen simulation videos (see recipe for access)
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

1x A100 80GB (min, 24 GB VRAM)

## Cosmos Metadata

| Field     | Value                                                                                                                                                                                                             |
|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Workload  | post-training                                                                                                                                                                                                     |
| Domain    | domain:robotics                                                                                                                                                                                                   |
| Technique | technique:post-training                                                                                                                                                                                           |
| Tags      | post-training, reason-1, temporal-localization, manipulation                                                                                                                                                      |
| Summary   | Post-trains Cosmos Reason 1 to automatically generate timestamp annotations for robot manipulation subtask boundaries in simulation videos, enabling scalable MimicGen dataset creation from a small number of human demonstrations. |
