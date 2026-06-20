# Cosmos Cookbook Skills

`.claude/` contains agent skills usable by cosmos-cookbook contributors. The
`/byo-video` skill in `commands/byo-video.md` walks a contributor through
deploying a Cosmos VLM to a Gradio web UI on their own GPU — it handles
environment validation, model selection, container launch, and the web UI
wiring so a contributor can go from a fresh GPU box to a working
"bring-your-own-video" demo without leaving the chat.

## Quick Reference

| Command | What It Does | When to Use |
|---------|-------------|-------------|
| `/byo-video` | Deploy a Cosmos VLM to a local Gradio web UI on your own GPU | Standing up a self-hosted video understanding demo |

---

## `/byo-video`

**Purpose:** Walk a contributor through deploying a Cosmos VLM to a Gradio web
UI on a GPU they control. The skill validates the environment, selects a
compatible model, launches the inference container, and wires up the web UI so
the contributor can upload a video and see model output in a browser.

**Human usage:**
```
/byo-video
```
Run this on a host with an NVIDIA GPU. The skill will check prerequisites,
guide model selection, start the inference backend, and open the Gradio UI.

**Agent usage:** Invoke when the user wants a self-hosted Cosmos VLM demo with
a web frontend. The skill is interactive — it expects to ask the user about
model choice and confirm long-running steps.

**Flow:**
1. Validate GPU, drivers, container runtime, and disk space
2. Offer a list of compatible Cosmos VLMs and let the user pick one
3. Pull and launch the inference container
4. Start the Gradio web UI and surface the local URL
5. Hand off to the user for interactive video uploads and prompts

See `commands/byo-video.md` for the full skill definition and step-by-step
agent guidance.

---

## How Agents Discover and Use These Skills

Claude Code automatically reads `CLAUDE.md` files in the working directory and
its subdirectories. When a session starts in the cosmos-cookbook repo root,
the `.claude/commands/` directory makes the slash commands above available to
the agent, and any per-recipe `CLAUDE.md` is loaded when the user navigates
into a recipe directory.
