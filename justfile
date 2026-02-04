default:
  just --list

# Setup the repository
setup:
  uv tool install -U pre-commit
  pre-commit install -c .pre-commit-config-base.yaml

# Install the repository
install: setup
  uv sync

# Template the documentation
_template *args:
  rm -rf build/docs
  uvx makejinja@2.7.2 -q -i docs -o build/docs --jinja-suffix=.md --keep-jinja-suffix --undefined strict {{ args }}

# Serve the internal documentation locally
serve-internal:
  just _template -d internal.yaml
  uv run mkdocs serve -f mkdocs-internal.yml

# Serve the external documentation locally
serve-external:
  just _template -d external.yaml
  uv run mkdocs serve

# Run linting and formatting
lint: setup
  pre-commit run --all-files || pre-commit run --all-files

# Run tests
test: lint
  # Test the internal documentation
  just _template -d internal.yaml
  uv run mkdocs build -f mkdocs-internal.yml --strict

  # Test the external documentation
  just _template -d external.yaml
  uv run mkdocs build --strict

# CI: Run linting
ci-lint:
  uvx pre-commit run -c .pre-commit-config-base.yaml --all-files --show-diff-on-failure
  uvx pre-commit run --all-files --show-diff-on-failure

# CI: Deploy the internal documentation
ci-deploy-internal:
  rm -rf public
  just _template -d internal.yaml
  uv run mkdocs build -f mkdocs-internal.yml --site-dir public --strict

# CI: Deploy the external documentation
# Set COOKIE_TEST_MODE=true to use test cookie script (for forks)
ci-deploy-external:
  rm -rf external
  just _template -d external.yaml
  uv run mkdocs build {{ if env("COOKIE_TEST_MODE", "") == "true" { "-f mkdocs-internal.yml" } else { "" } }} --site-dir external/site --strict
  mkdir -p external/cosmos-cookbook
  cp -r scripts external/cosmos-cookbook/scripts

# Sync jupytext notebooks
notebooks-sync:
  uvx --with "ruff==0.14.8" jupytext --sync scripts/examples/*.ipynb docs/recipes/**/*.ipynb --pipe 'ruff format -'
