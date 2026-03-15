# Spatial AI Warehouse — Cosmos Reason 1 Post-Training

## Model

`nvidia/Cosmos-Reason1-7B`

## Data Source

<!--
  Access: Restricted — synthetic warehouse dataset generated internally; not publicly available
  Model requires accepting the NVIDIA Open Model License
  at https://huggingface.co/nvidia/Cosmos-Reason1-7B
  Size: ~14GB (model)
  License: NVIDIA Open Model License
-->

**Access:** Restricted — training dataset is a synthetic internal warehouse dataset; see recipe for details
**Size:** ~14GB (model weights)
**License:** NVIDIA Open Model License

```bash
huggingface-cli download nvidia/Cosmos-Reason1-7B --repo-type model --local-dir ./models/Cosmos-Reason1-7B
```

## Compute Requirements

8x A100 80GB (min)

## Cosmos Metadata

| Field     | Value                                                                                                                                                              |
|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Workload  | post-training                                                                                                                                                      |
| Domain    | domain:industrial                                                                                                                                                  |
| Technique | technique:post-training                                                                                                                                            |
| Tags      | post-training, reason-1, spatial-ai, warehouse                                                                                                                     |
| Summary   | Fine-tunes Cosmos Reason 1-7B on a synthetic warehouse dataset for spatial intelligence tasks including distance estimation, object counting, and spatial relationship reasoning among pallets and transporters. |
