# Get Started with Transfer2.5 and Predict2.5 on Brev

> **Authors:** [Carlos Casanova](https://www.linkedin.com/in/carloscasanova/)
> **Organization:** NVIDIA

## Explore Brev

NVIDIA Brev is an excellent platform for experimenting with Cosmos. Follow these steps to get started:

1. Create an account at [https://brev.nvidia.com](https://brev.nvidia.com).
2. Install the CLI as shown in [https://docs.nvidia.com/brev/latest/brev-cli.html](https://docs.nvidia.com/brev/latest/brev-cli.html).
3. Refer to the [Brev Quickstart](https://docs.nvidia.com/brev/latest/quick-start.html) to get a feel for the platform. The Brev documentation is also linked from the Brev page.

While lower spec GPUs can work for some workflows, GPUs with 80GB of VRAM are recommended for Cosmos. Also note that the Transfer 2.5 AV Multiview model requires instances with 8 or more GPUs.

## The cheat code: Launchables

[Launchables](https://docs.nvidia.com/brev/latest/launchables.html) are an easy way to bundle a hardware and software environment into an easily shareable link. Once you've dialed in your Cosmos setup, a Launchable is the most convenient way to save time and share your configuration with others.

In this section, we'll walk through building a Launchable for Transfer2.5. Setting up Predict2.5 is nearly identical to the below steps. Refer to the [Predict2.5 setup guide](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/setup.md) and adjust the setup script accordingly. You can also set up both models at once.

> **Note**: Cosmos and Brev are evolving. You may encounter minor UI and other differences in the steps below as Brev changes over time.

1. Find the **Launchable** section of the Brev website.

   ![Launchables Menu](images/brev01-launchable-menu.png)

2. Click the **Create Launchable** button.

   ![Create Launchable Button](images/brev02-create-launchable-button.png)

3. Enter the Cosmos Transfer URL: [https://github.com/nvidia-cosmos/cosmos-transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5)

   ![Cosmos Transfer URL](images/brev03-create-launchable-step1.png)

4. Add a setup script. Brev will run it after cloning the repo. This script should follow the setup instructions from the [Cosmos Transfer2.5 repo](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/setup.md). In this example, we use the [sample script](#sample-setup-script) from later in this guide, which builds the Transfer2.5 Docker image and creates another script in the home folder of your Brev environment to launch the container.

   ![Add setup script](images/brev04-create-launchable-step2.png)

5. If you don't need Jupyter, remove it. You can open other ports on Brev if you plan to set up a custom server.

   ![Add ports](images/brev05-create-launchable-step3.png)

   > Setting up Predict2.5 is nearly identical to the above steps. Refer to the [Predict2.5 setup guide](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/setup.md) and adjust the setup script accordingly. Want to setup both at once? Nothing's stopping you. The world is your oyster.

6. Choose the desired level of compute. The screenshot below shows filtering on 8+ GPUs to run the Transfer 2.5 AV Multiview model.

   ![Choose compute](images/brev06-create-launchable-step4.png)

7. Name your Launchable and configure access.

   ![Name and configure access](images/brev07-create-launchable-step5.png)

   You're ready to deploy! Notice the **View All Options** link, which allows you to change the compute.

   ![Ready to deploy](images/brev08-launchable-ready-to-deploy.png)

8. After deploying, visit the instance page to find helpful examples of how to connect to the instance. Note the **Delete** button, which allows you to delete your instance when you're done. This can also be done with the `brev delete` CLI command. Instances that support pause and resume can be stopped from this page.

   ![Instance page](images/brev09-instance-page.png)

9. Connect to the instance. This example runs the generated `run_transfer2.5_docker.sh` script to start the container. Once the prompt appears, run `hf auth login` to enable checkpoint downloads. Transfer2.5 won't work without the checkpoints.

   ![Docker prompt](images/brev10-docker-prompt.png)

   > The Docker entrypoint pulls dependencies, and since share the Python virtual environment (venv) folder is shared with the container, subsequent runs will already have the deps installed.

### Sample setup script

The sample setup script below builds a Transfer2.5 Docker image and creates another script in the home folder of your Brev environment to launch the container. Once inside the container, run the `hf auth login` command to enable checkpoint downloads. Refer to the [Transfer2.5 Downloading Checkpoints](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/setup.md#downloading-checkpoints) section for more info.

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
# Install required packages
sudo apt-get update && sudo apt-get install -y git-lfs bc

# Move into $HOME/cosmos-transfer2.5
cd $HOME/cosmos-transfer2.5

# Initialize git-lfs and pull LFS files to ensure complete clone
git lfs install
git lfs pull

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
OMP_NUM_THREADS=$(echo "scale=0; $NUM_CPUS/$NUM_GPUS" | bc)

# Create script to run the container
VOL="-v $HF_HOME:/root/.cache/huggingface -v $HOME/cosmos-transfer2.5:/workspace -v $VENV_DIR:/workspace/.venv"
ENV="-e HF_HOME=/root/.cache/huggingface -e OMP_NUM_THREADS=$OMP_NUM_THREADS -e NUM_GPUS=$NUM_GPUS"
RUN_CMD="docker run -it --rm --ipc=host --name transfer2.5 $VOL $ENV transfer2.5"
echo "$RUN_CMD" > $HOME/run_transfer2.5_docker.sh
chmod +x $HOME/run_transfer2.5_docker.sh
'
```

## Notes and Tips

- We recommend using GPUs with 80GB+ of VRAM.
- We recommend using instances with a 2 or more terabytes of storage. With less than 2 terabytes, you might run out of space.
- Don't forget to shutdown (i.e. delete) your instances when you're done.
- As of November 2025, most instances suitable for Transfer 2.5 and Predict 2.5 do not support the pause and resume (start/stop) feature.
- Note the Brev deployment time estimate when evaluating instance types (e.g. "Ready in 7minutes").
- Deployment can fail on occasion, and the driver version might not be what you expect when trying a new provider. For these reasons, set aside 3x your estimated ready time and you will be happy 😀
- Your favorite cloud provider might not always be available.
- You can change the compute for a Launchable. Here are some reasons you might want to do this:
  <ul>
    <li>☁️ The preferred cloud provider is not available.</li>
    <li>💰 You want to save money with a different configuration.</li>
    <li>🏎️ You want to try higher specs.</li>
  </ul>