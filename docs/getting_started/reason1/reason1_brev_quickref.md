# Cosmos Reason1 on Brev - Quick Reference

## Instance Requirements

- **GPU**: H100 (80GB)
- **OS**: Ubuntu 22.04 or 24.04
- **Storage**: 200GB minimum
- **Memory**: 80GB GPU, 64GB+ RAM recommended

## Quick Setup Commands

### 1. System Setup (5 minutes)

```bash
# Update system
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git wget curl

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 2. Hugging Face Setup (2 minutes)

```bash
# Install HF CLI
uv tool install -U "huggingface_hub[cli]"

# Login (requires token)
~/.local/bin/hf auth login
```

### 3. Clone Repository (1 minute)

```bash
git clone https://github.com/nvidia-cosmos/cosmos-reason1.git
cd cosmos-reason1
```

### 4. Run Inference (5 minutes first run, includes model download)

```bash
# Minimal example
uv run scripts/inference_sample.py

# Video captioning
./scripts/inference.py --prompt prompts/caption.yaml --videos assets/sample.mp4 -v

# Question answering
./scripts/inference.py --prompt prompts/question.yaml \
  --question 'What are the safety hazards?' \
  --reasoning --videos assets/sample.mp4 -v
```

### 5. Setup Post-Training (5 minutes)

```bash
cd examples/post_training_hf
just install
source .venv/bin/activate

# Download sample dataset (10 samples)
./scripts/download_nexar_collision_prediction.py data/sft --split "train[:10]"
```

### 6. Run Training (varies by dataset size)

```bash
cosmos-rl --config configs/sft.toml scripts/custom_sft.py
```

## Common Issues & Quick Fixes

### CUDA Out of Memory
```bash
# Edit configs/sft.toml
# Reduce: per_device_train_batch_size
# Enable: gradient_checkpointing = true
```

### Model Download Fails
```bash
# Verify authentication
~/.local/bin/hf whoami

# Manual download
huggingface-cli download nvidia/Cosmos-Reason1-7B
```

### SSH Connection Lost
- Check Brev dashboard for instance status
- Restart instance if needed
- Reconnect using SSH command from dashboard

## File Paths

| Purpose | Location |
|---------|----------|
| Inference scripts | `scripts/inference.py`, `scripts/inference_sample.py` |
| Prompts | `prompts/*.yaml` |
| Model configs | `configs/sampling_params.yaml`, `configs/vision_config.yaml` |
| Post-training | `examples/post_training_hf/` |
| Training configs | `examples/post_training_hf/configs/sft.toml` |
| Output checkpoints | `examples/post_training_hf/outputs/sft/checkpoints/` |

## Resource Usage

| Task | GPU Memory | Time (Approx) |
|------|------------|---------------|
| Model Download | N/A | 5-10 min (first time) |
| Inference (single video) | ~15GB | 10-30 sec |
| SFT (10 samples) | ~40GB | 10-15 min |
| SFT (100 samples) | ~40-60GB | 1-2 hours |

## Important Links

- Guide: `docs/getting_started/reason1_on_brev.md`
- Model: https://huggingface.co/nvidia/Cosmos-Reason1-7B
- Repository: https://github.com/nvidia-cosmos/cosmos-reason1
- Brev: https://brev.dev

## Cost Saving Tips

1. **Stop instance when not in use** - Brev charges by the hour
2. **Download model once** - Cache HF models to avoid re-downloading
3. **Test with small datasets** - Use `train[:10]` for testing before full training
4. **Monitor GPU usage** - Use `nvidia-smi` to check utilization
5. **Save checkpoints** - Copy to cloud storage before deleting instance

## Verification Commands

```bash
# Check GPU
nvidia-smi

# Check CUDA
nvcc --version

# Check Python
python3 --version

# Check uv
uv --version

# Check HF auth
~/.local/bin/hf whoami

# Check disk space
df -h

# Check GPU memory usage
watch -n 1 nvidia-smi
```

## Next Steps After Setup

1. ✅ Run inference examples
2. ✅ Try custom videos
3. ✅ Experiment with different prompts
4. ✅ Test post-training with sample data
5. ✅ Create custom dataset
6. ✅ Full training run
7. ✅ Evaluate fine-tuned model
8. ✅ Deploy to production

## Support

- **Cosmos Issues**: https://github.com/nvidia-cosmos/cosmos-reason1/issues
- **Brev Support**: https://brev.dev/support
- **License Questions**: cosmos-license@nvidia.com

