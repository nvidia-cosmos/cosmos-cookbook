# AV Video Captioning and VQA — Cosmos Reason 2 Post-Training

## Model

`nvidia/Cosmos-Reason2-8B`

## Data Source

<!--
  Access: Restricted — training dataset from Uber collaboration; not publicly available
  Model requires accepting the NVIDIA Open Model License
  at https://huggingface.co/nvidia/Cosmos-Reason2-8B
  Size: ~16GB (model)
  License: NVIDIA Open Model License
-->

**Access:** Restricted — training dataset from an Uber–NVIDIA collaboration; not publicly downloadable
**Size:** ~16GB (model weights)
**License:** NVIDIA Open Model License

```bash
huggingface-cli download nvidia/Cosmos-Reason2-8B --repo-type model --local-dir ./models/Cosmos-Reason2-8B
```

## Compute Requirements

8x A100 80GB (min)

## Cosmos Metadata

| Field     | Value                                                                                                                                                                              |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Workload  | post-training                                                                                                                                                                      |
| Domain    | domain:autonomous-vehicles                                                                                                                                                         |
| Technique | technique:post-training                                                                                                                                                            |
| Tags      | post-training, reason-2, captioning, vqa                                                                                                                                           |
| Summary   | Fine-tunes Cosmos Reason 2 in collaboration with Uber to produce AV-specific video captions and improve VQA accuracy for autonomous vehicle scenario retrieval, safety validation, and model training data curation. |
