# VSS — Cosmos Reason 2 Deployment

The Video Search and Summarization (VSS) Blueprint is a full multi-service stack
(Docker Compose + vector DB + graph DB + LLM + Cosmos Reason). It has its own
official Brev Launchable maintained by the NVIDIA VSS team.

## Launch on Brev

Use the official VSS Brev Launchable:
https://docs.nvidia.com/vss/latest/content/cloud_brev.html

## Full deployment docs

- Supported platforms: https://docs.nvidia.com/vss/latest/content/supported_platforms.html
- Prerequisites: https://docs.nvidia.com/vss/latest/content/prereqs_x86.html
- Deployment guide: https://docs.nvidia.com/vss/latest/content/vss_dep_docker_compose_x86.html

## Hardware

- Single GPU: 1x H100-80GB (single GPU deployment profile)
- Full stack: 8x H100-80GB (recommended for production-scale video processing)

## Recipe

See `docs/recipes/inference/reason2/vss/inference.md` for the full recipe description.
