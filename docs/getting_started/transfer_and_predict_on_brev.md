# Get Started with Transfer2.5 and Predict2.5 on Brev

> **Authors:** Carlos Casanova
> **Organization:** NVIDIA

## Explore Brev
NVIDIA's Brev is an excellent platform for experimenting with Cosmos. To get started,

1. Create an account at [https://brev.nvidia.com](https://brev.nvidia.com)
2. Install the CLI as shown in [https://docs.nvidia.com/brev/latest/brev-cli.html](https://docs.nvidia.com/brev/latest/brev-cli.html)
3. See the quickstart to get a feel for the platform: [https://docs.nvidia.com/brev/latest/quick-start.html](https://docs.nvidia.com/brev/latest/quick-start.html). The handy Brev docs are linked from the Brev page too.

While lower spec'd GPUs can work for some workflows, prefer those with 80GB of VRAM. Note also that the Transfer2.5 AV Multiview model requires instances with 8 or more GPUs.

## The cheat code: Launchables!
[Launchables](https://docs.nvidia.com/brev/latest/launchables.html) are an easy way to bundle a hardware and software environment into an easily shareable link. Once you've dialed in your Cosmos setup, a Launchable is the most convenient way to save time and share your configuration with others.

In this section, we'll walk through building a Launchable for Transfer2.5. The steps are nearly identical for Predict2.5.

> **Note:** Cosmos and Brev are evolving. You may encounter minor UI and other differences in the steps below over time. The spirit will be correct.

1. Find the Launchable section of the Brev website.

    ![Launchables Menu](../assets/images/brev01-launchable-menu.png)

2. Click the **Create Launchable** button.

    ![Create Launchable Button](../assets/images/brev02-create-launchable-button.png)

3. Enter the Cosmos Transfer URL [https://github.com/nvidia-cosmos/cosmos-transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5)

    ![Cosmos Transfer URL](../assets/images/brev03-create-launchable-step1.png)

4. Add a setup script. This script should follow the setup instructions from the [Cosmos Transfer2.5 repo](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/setup.md). See [Sample setup script](#sample-setup-script) for an example.

    ![Add setup script](../assets/images/brev04-create-launchable-step2.png)

5. If you don't need Jupyter, remove it. Tell Brev which ports to open if you plan to setup the sample Gradio server from the Transfer repo (or some other custom server). 

    ![Add ports](../assets/images/brev05-create-launchable-step3.png)

6. Now for the fun part: choosing the compute! Below, we're filtering on 8+ GPUs so that we could run Transfer2.5 AV Multiview.

    ![Choose compute](../assets/images/brev06-create-launchable-step4.png)

7. Name your Launchable and configure access.

    ![Name and configure access](../assets/images/brev07-create-launchable-step5.png)

8. You're ready to deploy! Notice the **View All Options** link. That's how to change the compute.

    ![Ready to deploy](../assets/images/brev08-launchable-ready-to-deploy.png)

### Sample setup script
The sample setup script below builds a Transfer2.5 Docker image and creates a script in the home folder of your Brev environment to run it. Once inside the container, do not forget to run `hf auth login` to enable checkpoint downloads. See also [Transfer2.5 downloading checkpoints](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/setup.md#downloading-checkpoints) for more info.

```bash
#!/bin/bash

# Detect the ultimate Brev user. We will run the script as them.
if id "nvidia" &>/dev/null; then
  RUN_USER="nvidia"
elif id "shadeform" &>/dev/null; then
  RUN_USER="shadeform"
elif id "ubuntu" &>/dev/null; then
  RUN_USER="ubuntu"
else
  RUN_USER="root"
fi

sudo -u $RUN_USER bash -lc '
# Move into $HOME/cosmos-transfer2.5
cd $HOME/cosmos-transfer2.5

# Build the Cosmos Transfer 2.5 docker image
docker build --ulimit nofile=131071:131071 -f Dockerfile . -t transfer2.5

# Create folders to share HuggingFace files and .venv with the container
HF_HOME=$HOME/.cache/huggingface
VENV_DIR=$HOME/.venv_transfer2.5
mkdir -p $HF_HOME
mkdir -p $VENV_DIR

# Find out number of GPUs and CPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NUM_CPUS=$(nproc)

# Set OMP_NUM_THREADS to FLOOR(NUM_CPUS/NUM_GPUS)
sudo apt-get update && sudo apt-get install -y bc # ensure bc is installed
OMP_NUM_THREADS=$(echo "scale=0; $NUM_CPUS/$NUM_GPUS" | bc)

# Create script to run the container
VOL="-v $HF_HOME:/root/.cache/huggingface -v $HOME/cosmos-transfer2.5:/workspace -v $VENV_DIR:/workspace/.venv"
ENV="-e HF_HOME=/root/.cache/huggingface -e OMP_NUM_THREADS=$OMP_NUM_THREADS -e NUM_GPUS=$NUM_GPUS"
RUN_CMD="docker run -it --rm --ipc=host --name transfer2.5 $VOL $ENV transfer2.5"
echo "$RUN_CMD" > $HOME/run_transfer2.5_docker.sh
chmod +x $HOME/run_transfer2.5_docker.sh
'
```

## Notes, tips and gotchas
- Prefer GPUs with 80GB+ of VRAM.
- Prefer instances with a few TB+ of storage. With less than 2TB you might sometimes run out of space!
- Don't forget to shutdown (delete) your instances when you're done.
- As of November 2025, most instances suitable for Transfer and Predict do not support the pause and resume (start/stop) feature.
- Note Brev's deployment time estimate when evaluating instance types, ex: "Ready in 7minutes". 
- Deployment can fail on occasion and the driver version might not be what you expect when trying a new provider. For these reasons, set aside 3 times the estimated ready time in your mind and you will be happy 😀
- Your favorite cloud provider might not always be available. 
- You can change a Launchable's compute! Reasons you might want to include
    - ☁️ The preferred cloud provider is not available
    - 💰 You want to save money with a different configuration
    - 🏎️ You want to try higher specs
