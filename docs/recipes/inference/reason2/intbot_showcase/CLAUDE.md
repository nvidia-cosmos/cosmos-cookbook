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

1x H100-80GB (8B model requires ~75 GB VRAM)

## Execution

Headless execution for Claude or any agent on a Linux GPU machine.
No runnable notebook exists for this recipe — demo.sh runs representative
tests directly against the recipe's asset images.

### Pre-flight

```bash
nvidia-smi
```

Confirm H100-80GB is visible with >= 75 GB VRAM free.

### Run

```bash
export HF_TOKEN=hf_...
bash deploy/reason2/intbot_showcase/demo.sh
```

### What it runs

Three representative tests from the recipe using local asset images:
1. **Fist-bump gesture** — does the person intend to fist-bump the robot?
2. **Hat trajectory** — is the hat moving toward or away from the robot?
3. **Shared attention** — are people talking to each other or engaging the robot?

### Success criteria

- `CUDA available: True` during setup
- `intbot_results.json` contains entries with `"status": "success"`
- Responses reference robot-relative spatial reasoning (not generic descriptions)

## Cosmos Metadata

| Field     | Value                                                                                                                                                                                                            |
|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Workload  | inference                                                                                                                                                                                                        |
| Domain    | domain:robotics                                                                                                                                                                                                  |
| Technique | technique:reasoning                                                                                                                                                                                              |
| Tags      | inference, reason-2, inspection                                                                                                                                                                                  |
| Summary   | Evaluates Cosmos-Reason2-8B on egocentric video tests for humanoid robot social and physical reasoning, benchmarked against Qwen3-VL-8B-Instruct on greetings, object motion, shared attention, and social context tasks. |
