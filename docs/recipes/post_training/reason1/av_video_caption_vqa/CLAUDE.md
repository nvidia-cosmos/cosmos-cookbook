# AV Video Captioning and VQA — Cosmos Reason 1 Post-Training

## Model

`nvidia/Cosmos-Reason1-7B`

## Data Source

<!--
  Access: Restricted — internal NVIDIA AV dataset; not publicly downloadable
  Model requires accepting the NVIDIA Open Model License
  at https://huggingface.co/nvidia/Cosmos-Reason1-7B
  Size: ~14GB (model)
  License: NVIDIA Open Model License
-->

**Access:** Restricted — training dataset is an internal NVIDIA AV dataset; see recipe for details
**Size:** ~14GB (model weights)
**License:** NVIDIA Open Model License

```bash
huggingface-cli download nvidia/Cosmos-Reason1-7B --repo-type model --local-dir ./models/Cosmos-Reason1-7B
```

## Compute Requirements

8x A100 80GB (min)

## Cosmos Metadata

| Field     | Value                                                                                                                                                                               |
|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Workload  | post-training                                                                                                                                                                       |
| Domain    | domain:autonomous-vehicles                                                                                                                                                          |
| Technique | technique:post-training                                                                                                                                                             |
| Tags      | post-training, reason-1, captioning, vqa                                                                                                                                            |
| Summary   | Fine-tunes Cosmos Reason 1 on an internal NVIDIA AV dataset to produce high-quality domain-specific video captions and labels for autonomous vehicle scenario retrieval and training data curation. |
