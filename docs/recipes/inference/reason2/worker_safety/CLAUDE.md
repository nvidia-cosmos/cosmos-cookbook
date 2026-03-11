# Worker Safety — Cosmos Reason 2 Inference

## Model

`nvidia/Cosmos-Reason2-2B`

## Data Source

<!--
  Access: Gated — requires accepting the NVIDIA Open Model License
  at https://huggingface.co/nvidia/Cosmos-Reason2-2B
  Size: ~4GB
  License: NVIDIA Open Model License
-->

**Access:** Gated — requires accepting the NVIDIA Open Model License at https://huggingface.co/nvidia/Cosmos-Reason2-2B
**Size:** ~4GB
**License:** NVIDIA Open Model License

```bash
huggingface-cli download nvidia/Cosmos-Reason2-2B --repo-type model --local-dir ./models/Cosmos-Reason2-2B
```

## Compute Requirements

1x A100 80GB (~30 min)

## Cosmos Metadata

| Field     | Value                                                                                                      |
|-----------|------------------------------------------------------------------------------------------------------------|
| Workload  | inference                                                                                                  |
| Domain    | domain:industrial                                                                                          |
| Technique | technique:reasoning                                                                                        |
| Tags      | inference, reason-2, safety                                                                                |
| Summary   | Zero-shot warehouse safety inspection using Cosmos Reason 2 to classify worker behaviors from video without custom model training. |
