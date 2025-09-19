# nuScenes

Post-training example using [nuScenes](https://www.nuscenes.org/nuscenes).

```shell
# Install the package
cd cosmos-playbook/experimental/joallen/cosmos-reason1
just install
source .venv/bin/activate

# Download the dataset
s5cmd cp --show-progress "s3://lha-datasets/debug/joallen/nuscenes/v0/shard/v0/*" ~/datasets/nuscenes/v0/shard/

# Convert webdataset to huggingface dataset
./cosmos_curate/download.py ~/datasets/nuscenes/v0/shard ~/datasets/nuscenes/v0/hf

# [Optional] Generate caption with VLM
./cosmos_curate/generate_caption.py ~/datasets/nuscenes/v0/hf ~/datasets/nuscenes/v0/cosmos-rl --prompt nuscenes/v0/vlm_prompt_0.yaml --generation-config nuscenes/v0/generation_config.json --vision-config nuscenes/v0/vision_config.json --model nvidia/Cosmos-Reason1-7B [--max-samples 1 -v]

# [Optional] Extract VLM QA
./cosmos_curate/extract_qa.py ~/datasets/nuscenes/v0/hf ~/datasets/nuscenes/v0/cosmos-rl

# [Optional] Generate QA with LLM
# Authenticate: https://gitlab-master.nvidia.com/dir/nvidia-cosmos/cosmos-infra/-/blob/main/scripts/enterprise-api/README.md?ref_type=heads#authentication
./cosmos_curate/generate_qa.py  ~/datasets/nuscenes/v0/shard ~/datasets/nuscenes/v0/hf --prompt <prompt> [--max-samples 1 -v]

# Run SFT
cosmos-rl --config nuscenes/sft.toml hf_cosmos_sft.py
```
