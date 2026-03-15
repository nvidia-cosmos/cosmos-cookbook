# Wafer Map Defect Classification — Cosmos Reason 1 Post-Training

## Model

`nvidia/Cosmos-Reason1-7B`

## Data Source

<!--
  Access: Gated — requires accepting the NVIDIA Open Model License
  at https://huggingface.co/nvidia/Cosmos-Reason1-7B
  Training dataset: WM-811K Wafer Map dataset from Mir Lab (http://mirlab.org/dataSet/public/)
  Size: ~14GB (model) + ~1GB (dataset)
  License: NVIDIA Open Model License (model); Mir Lab dataset terms (see recipe)
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

| Field     | Value                                                                                                                                                   |
|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| Workload  | post-training                                                                                                                                           |
| Domain    | domain:industrial                                                                                                                                       |
| Technique | technique:post-training                                                                                                                                 |
| Tags      | post-training, reason-1, wafermap, semiconductor                                                                                                        |
| Summary   | Fine-tunes Cosmos Reason 1-7B on wafer map defect images to classify 8 semiconductor manufacturing defect patterns using SFT, achieving 96.8% classification accuracy. |
