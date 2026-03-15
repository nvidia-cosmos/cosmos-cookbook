# Video Search and Summarization — Cosmos Reason 2 Inference

## Model

`nvidia/Cosmos-Reason2-8B`

## Data Source

<!--
  Access: Gated — requires accepting the NVIDIA Open Model License
  at https://huggingface.co/nvidia/Cosmos-Reason2-8B
  Size: ~16GB
  License: NVIDIA Open Model License
-->

**Access:** Gated — requires accepting the NVIDIA Open Model License at https://huggingface.co/nvidia/Cosmos-Reason2-8B
**Size:** ~16GB
**License:** NVIDIA Open Model License

```bash
huggingface-cli download nvidia/Cosmos-Reason2-8B --repo-type model --local-dir ./models/Cosmos-Reason2-8B
```

## Compute Requirements

1x H100 80GB (single-GPU profile) or 8x H100 80GB (multi-GPU profile)

## Cosmos Metadata

| Field     | Value                                                                                                                                                                                                                        |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Workload  | inference                                                                                                                                                                                                                    |
| Domain    | domain:smart-city, domain:industrial                                                                                                                                                                                         |
| Technique | technique:reasoning                                                                                                                                                                                                          |
| Tags      | inference, reason-2, video-search                                                                                                                                                                                            |
| Summary   | Deploys Cosmos Reason 2 as the vision-language backbone in NVIDIA's Video Search and Summarization Blueprint for large-scale video captioning, semantic Q&A, and streaming event alerts across warehouses, factories, and smart city infrastructure. |
