# IntBot Showcase — Cosmos Reason 2 Inference

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

1x A100 80GB (inference only)

## Cosmos Metadata

| Field     | Value                                                                                                                                                                                                            |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Workload  | inference                                                                                                                                                                                                        |
| Domain    | domain:robotics                                                                                                                                                                                                  |
| Technique | technique:reasoning                                                                                                                                                                                              |
| Tags      | inference, reason-2, inspection                                                                                                                                                                                  |
| Summary   | Evaluates Cosmos-Reason2-8B on egocentric video tests for humanoid robot social and physical reasoning, benchmarked against Qwen3-VL-8B-Instruct on greetings, object motion, shared attention, and social context tasks. |
