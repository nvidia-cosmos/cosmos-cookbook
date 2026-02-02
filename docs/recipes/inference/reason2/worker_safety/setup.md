# Setup and System Requirements

This guide summarizes the setup process for installing **Cosmos Reason2** on **Ubuntu Linux** with an **NVIDIA GPU**, using **uv** for dependency management, and running notebooks via **JupyterLab** with a dedicated kernel.

---

## 1) System prerequisites (Ubuntu)

Install required system packages:

```bash
sudo apt-get update
sudo apt-get install -y curl ffmpeg git git-lfs
git lfs install
```

Verify your GPU + NVIDIA driver + CUDA support:

```bash
nvidia-smi
```

> In this setup, the machine is a **laptop with an NVIDIA RTX PRO 5000 Blackwell GPU** running **Ubuntu 24.04** and supporting **CUDA 13.0**, so we use the **cu130** environment extra.

---

## 2) Install `uv` (one-time, per user)

Install **uv** (Astral):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv --version
```

---

## 3) Clone the Cosmos Reason2 repository

```bash
REPO_ROOT=/path/to/your/preferred/projects/directory  # e.g., $HOME/Documents/GitHub or $HOME/projects
mkdir -p "$REPO_ROOT"
cd "$REPO_ROOT"
git clone https://github.com/nvidia-cosmos/cosmos-reason2.git
cd cosmos-reason2
git lfs pull
```

---

## 4) Authenticate with Hugging Face (required for model downloads)

```bash
uvx hf auth login
```

---

## 5) Create the Cosmos Reason2 environment (CUDA 13.0)

Create the environment and install dependencies (this creates `./.venv` inside the repo):

```bash
uv sync --extra cu130
```

Activate the environment:

```bash
source .venv/bin/activate
```

Verify PyTorch + CUDA availability:

```bash
python -c "import torch; print(torch.__version__); print('cuda', torch.cuda.is_available())"
```

Expected:

- `cuda True`

---

## 6) Install additional packages in the same environment

### Install FiftyOne

```bash
pip install -U fiftyone
```

### Install JupyterLab + kernel support

```bash
pip install -U jupyterlab ipykernel
```

---

## 7) Register the environment as a Jupyter kernel (findable in JupyterLab)

Register this environment so it appears as a selectable kernel:

```bash
python -m ipykernel install --user --name cosmos-reason2 --display-name "Python (cosmos-reason2)"
```

Confirm it exists:

```bash
jupyter kernelspec list
```

You should see something like:

```
cosmos-reason2    /home/<user>/.local/share/jupyter/kernels/cosmos-reason2
```

---

## 8) Run JupyterLab and select the correct kernel

Start JupyterLab (recommended from the repo root):

```bash
cd ~/Documents/GitHub/cosmos-reason2
source .venv/bin/activate
jupyter lab
```

In the JupyterLab UI:

- **Kernel → Change Kernel → `Python (cosmos-reason2)`**

---

## 9) Running scripts from inside a notebook (important)

If your notebook lives in:

```
cosmos-reason2/notebooks/
```

and you run:

```python
!python ../scripts/inference_sample.py
```

the script will run relative to the notebook directory, which can affect paths.

### Recommended way (run from repo root)

Run the script from the repo root in one line:

```python
!cd .. && python scripts/inference_sample.py
```

Or in two steps:

```python
%cd ..
!python scripts/inference_sample.py
```

This ensures the script uses the correct repo-relative paths.

---

## Quick troubleshooting notes

- If port conflicts happen (e.g. `Address already in use`), try a different port for local services.
- If video decoding fails, ensure `ffmpeg` is installed system-wide and available via `ffmpeg -version`.

---

**Done!** You now have:

- a dedicated `cosmos-reason2` Python environment (`.venv`)
- Cosmos Reason2 and all granted models from Hugging Face (per the Cosmos Reason2 repo)
- FiftyOne installed inside that env
- JupyterLab installed inside that env
- a selectable kernel in JupyterLab: **Python (cosmos-reason2)**
