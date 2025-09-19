# Cosmos Post-Training Playbook

- [Documentation](https://cosmos-playbook-7663d3.gitlab-master-pages.nvidia.com/)
- [Contributing](CONTRIBUTING.md)

## Setup

### Install system dependencies

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv tool install -U rust-just

# Optional useful tools
uv tool install -U s5cmd
uv tool install -U streamlit
uv tool install -U yt-dlp
```

### Install repository

```shell
git clone https://github.com/nvidia-cosmos/cosmos-playbook.git
cd cosmos-playbook
just install
source .venv/bin/activate
```

## Jupyter

Notebooks should be written in [Jupytext Markdown](https://jupytext.readthedocs.io/en/latest/formats-markdown.html#jupytext-markdown)

Copy the header from the [example markdown](docs/post_training/post_training_predict.md).

Sync notebooks:

```shell
pre-commit run --all-files jupytext
```

Open the [example notebook](docs/post_training/post_training_predict.ipynb) in VS Code:

```shell
code . docs/post_training/post_training_predict.ipynb
```

Run the notebook. When prompted to select a kernel, choose "Python Environments" and select `.venv`.
