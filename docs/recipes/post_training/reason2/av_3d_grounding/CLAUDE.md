# AV 3D Bounding Box Grounding — Cosmos Reason Post-Training

## Model

- `nvidia/Cosmos-Reason1-7B`
- `nvidia/Cosmos-Reason2-8B`

## Data Source

<!--
  Access: Gated — requires accepting the NVIDIA Open Model License
  at https://huggingface.co/nvidia/Cosmos-Reason2-8B
  Training dataset: internal AV camera image dataset; not publicly available
  Size: ~16GB (Reason2-8B model) or ~14GB (Reason1-7B model)
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

| Field     | Value                                                                                                                                                                        |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Workload  | post-training                                                                                                                                                                |
| Domain    | domain:autonomous-vehicles                                                                                                                                                   |
| Technique | technique:post-training                                                                                                                                                      |
| Tags      | post-training, reason-1, reason-2, 3d-grounding                                                                                                                              |
| Summary   | Fine-tunes Cosmos Reason 1 and 2 via supervised fine-tuning to predict 3D bounding box coordinates (position, dimensions, orientation) for vehicles in autonomous driving camera images. |
