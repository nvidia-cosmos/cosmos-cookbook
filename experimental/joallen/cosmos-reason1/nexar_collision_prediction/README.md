# Nexar Collision Prediction

Post-training example using [Nexar Collision Prediction dataset](https://huggingface.co/datasets/nexar-ai/nexar_collision_prediction).

```sh
# Install the repository
cd cosmos-playbook/experimental/joallen/cosmos-reason1
just install
source .venv/bin/activate

# Download the dataset
./nexar_collision_prediction/download.py ~/datasets/nexar_collision_prediction/cosmos-rl

# Run SFT
cosmos-rl --config nexar_collision_prediction/sft.toml hf_cosmos_sft.py
```
