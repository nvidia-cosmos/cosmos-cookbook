# Slurm setup (optional)

Use this only if you want to run finetuning via **Slurm** with `scripts/run_finetuning.sh` (multi-node, scheduled jobs). If you have a single server without Slurm, use [run_finetuning_standalone.sh](scripts/run_finetuning_standalone.sh) instead; see [Post-training §4](README.md#4-finetuning).

## When you need Slurm

- You want to use `sbatch scripts/run_finetuning.sh` as in the main recipe.
- You have one or more machines and want a job scheduler (queues, multi-user, multi-node).

## Quick reference

1. **Official docs:** [Slurm Quick Start](https://slurm.schedmd.com/quickstart.html) and [Configuration](https://slurm.schedmd.com/slurm.conf.html).
2. **Single-node (one machine):** Install controller + compute on the same node, then configure and start services.
3. **Multi-node:** Install `slurmctld` on one controller node and `slurmd` on each compute node; shared config and Munge for auth.

## Single-node Ubuntu (controller + compute on one machine)

Install and create dirs:

```bash
sudo apt update
sudo apt install -y slurm-wlm slurm-wlm-doc munge
sudo mkdir -p /etc/slurm-llnl /var/lib/slurm-llnl /var/log/slurm-llnl
sudo chown slurm:slurm /var/lib/slurm-llnl /var/log/slurm-llnl
```

Create `/etc/slurm-llnl/slurm.conf` (minimal; replace `YOURHOSTNAME` with `hostname -s`):

```ini
ClusterName=cosmos
SlurmctldHost=YOURHOSTNAME
MpiDefault=none
ProctrackType=proctrack/linuxproc
ReturnToService=2
SlurmctldPidFile=/var/run/slurmctld.pid
SlurmctldPort=6817
SlurmdPidFile=/var/run/slurmd.pid
SlurmdPort=6818
SlurmdSpoolDir=/var/lib/slurm-llnl/slurmd
StateSaveLocation=/var/lib/slurm-llnl/slurmctld
SwitchType=switch/none
TaskPlugin=task/none
NodeName=YOURHOSTNAME CPUs=8 RealMemory=32000 State=UNKNOWN
PartitionName=batch Nodes=YOURHOSTNAME Default=YES MaxTime=INFINITE State=UP
```

If you have GPUs, add a GRES line (example for 8 GPUs) and create `/etc/slurm-llnl/gres.conf`; see [Slurm GRES](https://slurm.schedmd.com/gres.html). Then:

```bash
sudo systemctl enable munge slurmctld slurmd
sudo systemctl start munge slurmctld slurmd
sinfo
```

## Using the recipe’s Slurm script

After Slurm is running, copy or adapt `scripts/run_finetuning.sh` to your cluster:

- Set **partition**, **account**, and **container image/mounts** (or remove container and run in a venv).
- Ensure the partition has enough nodes/GPUs (script requests 4 nodes × 8 GPUs by default).
- Submit: `mkdir -p logs && sbatch scripts/run_finetuning.sh`.

For a single server without Slurm, use **`scripts/run_finetuning_standalone.sh`** instead (no `sbatch`).
