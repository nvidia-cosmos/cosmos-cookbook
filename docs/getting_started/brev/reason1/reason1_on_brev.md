# Get Started with Cosmos Reason1 on Brev: Inference and Post-Training
>
> **Author:** [Saurav Nanda](https://www.linkedin.com/in/sauravnanda/)
> **Organization:** NVIDIA

This guide walks you through setting up NVIDIA Cosmos Reason1 on a [Brev](https://brev.dev) H100 GPU instance for both inference and post-training workflows. Brev provides on-demand cloud GPUs with pre-configured environments, making it easy to get started with Cosmos models.

## Overview

[Brev.dev](https://brev.dev) is a cloud GPU platform that provides instant access to high-performance GPUs like the H100. This guide will help you:

1. Set up a Brev instance with H100 GPU
2. Configure the environment for Cosmos Reason1
3. Run inference on the Reason1 model
4. Perform post-training (SFT) on custom datasets

## Prerequisites

- A Brev account ([sign up here](https://brev.dev))
- Install the CLI as shown in [https://docs.nvidia.com/brev/latest/brev-cli.html](https://docs.nvidia.com/brev/latest/brev-cli.html)
- See the quickstart to get a feel for the platform: [https://docs.nvidia.com/brev/latest/quick-start.html](https://docs.nvidia.com/brev/latest/quick-start.html). The handy Brev docs are linked from the Brev page too.
- A Hugging Face account with access to [Cosmos-Reason1-7B](https://huggingface.co/nvidia/Cosmos-Reason1-7B)

## The cheat code: Launchables

[Launchables](https://docs.nvidia.com/brev/latest/launchables.html) are an easy way to bundle a hardware and software environment into an easily shareable link. Once you've dialed in your Cosmos setup, a Launchable is the most convenient way to save time and share your configuration with others.

> **Note:** Cosmos and Brev are evolving. You may encounter minor UI and other differences in the steps below over time.

## Step 1: Create a Brev Launchable

1. Log in to your [Brev account](https://brev.dev)
2. Find the Launchable section of the Brev website.
![Launchables Menu](./images/brev-01-launchable-menu.png)
3. Click the **Create Launchable** button.
![Create Launchable Button](./images/brev-02-create-launchable-button.png)
4. Enter the Cosmos Reason [GitHub URL](https://github.com/nvidia-cosmos/cosmos-reason1) ![https://github.com/nvidia-cosmos/cosmos-reason1](./images/brev-03-chose-repo.png)

5. Add a setup script for Cosmos Reason. See [sample setup script](./setup_script.sh) for an example.
![Setup Script](./images/brev-06-startup-script.png)

6. If you don't need Jupyter, remove it. Tell Brev which ports to open if you plan to setup some other custom server(if any). ![Jupyter](./images/brev-05-chosejupyter.png)

7. Choose H100 GPU instance with 80GB VRAM.
![H100 Instance](./images/brev04-choose-compute.png)

8. Name your Launchable and configure access.(usually takes 2-3 minutes). ![create](./images/brev-07-create-launchable.png)

## Step 2: Deploy and Connect to Your Instance

1. From the list of launchables, click "Deploy Now" button.
![Deploy Now](./images/brev-08-deploy-0.png)

2. Now click on "Deploy Launchable" button from the details page.
![Deploy Launchable](./images/brev-08-deploy-1.png)

3. Click "Go to Instance Page" button.
![Go to Instance Page](./images/brev-08-deploy-2.png)

4. Once your instance is ready, Brev will provide SSH connection details.
![instance](./images/brev-09-access-or-stop.png)

### Option 1: Open Jupyter Notebook

![Notebook](./images/brev-10-notebook.png)

### Option 2: Copy the SSH command from your Brev dashboard

```bash
brev login --token <YOUR_TOKEN>
```

Open a terminal locally

```bash
brev shell sample-reason1-fa3124
```

OR Open in Code Editor

```bash
brev open sample-reason1-fa3124 cursor
```

## Step 3: Authenticate Hugging Face CLI

The Hugging Face Token is required to download the Cosmos Reason1 model:

```bash
# Authenticate with Hugging Face
~/.local/bin/hf auth login
```

When prompted, enter your Hugging Face token. You can create a token at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

**Important**: Make sure you have access to the [Cosmos-Reason1-7B](https://huggingface.co/nvidia/Cosmos-Reason1-7B) model. Request access if needed.

## Step 4: Run Inference and Post-Training

Now you're ready to run inference with Cosmos Reason1!

Follow the steps provided in the [Cosmos Reason GitHub repo](https://github.com/nvidia-cosmos/cosmos-reason1) to run the inference and post-training examples.

## Troubleshooting

### Model Download Issues

If the model fails to download:

1. Verify your Hugging Face authentication: `~/.local/bin/hf whoami`
2. Ensure you have access to the Cosmos-Reason1-7B model
3. Check your internet connection
4. Try downloading manually: `huggingface-cli download nvidia/Cosmos-Reason1-7B`

### SSH Connection Issues

If you lose SSH connection:

1. Brev instances may pause after inactivity
2. Check your Brev dashboard for instance status
3. Restart the instance if needed
4. Reconnect using the SSH command

## Resource Management

### Stopping Your Instance

To avoid unnecessary charges:

1. Go to your Brev dashboard
2. Select your instance
3. Click **"Stop"** or **"Delete"** when done

### Saving Your Work

Before stopping your instance:

```bash
# Save model checkpoints to cloud storage (e.g., S3, GCS)
# Or download them to your local machine
scp -r ubuntu@<your-instance-ip>:~/cosmos-reason1/examples/post_training_hf/outputs ./local-outputs
```

## Additional Resources

- [Cosmos Reason1 GitHub Repository](https://github.com/nvidia-cosmos/cosmos-reason1)
- [Cosmos Reason1 Model on Hugging Face](https://huggingface.co/nvidia/Cosmos-Reason1-7B)
- [Cosmos Reason1 Paper](https://arxiv.org/abs/2503.15558)
- [Brev Documentation](https://docs.brev.dev)
- [Cosmos Cookbook](https://github.com/nvidia-cosmos/cosmos-cookbook)

## Support

For issues related to:

- **Cosmos Reason1**: Open an issue on the [GitHub repository](https://github.com/nvidia-cosmos/cosmos-reason1/issues)
- **Brev Platform**: Contact [Brev support](https://brev.dev/support)
