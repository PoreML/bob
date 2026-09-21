#!/bin/bash
#SBATCH --job-name=bob_mgpu_parity
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=demo/multi_gpu_parity/output/slurm-%j.log

# Multi-GPU parity demonstration: every scenario at 1/2/8 devices, sharded
# along its adversarial axis. Compute-only (--run); render the GIFs afterwards:
#   uv run python demo/multi_gpu_parity/multi_gpu_parity_demo.py --render --gpus 1 2 8
# Submit from the repository root:  sbatch demo/multi_gpu_parity/run_parity.slurm
# The #SBATCH partition/QOS settings above are site-specific; adapt them to your cluster.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p demo/multi_gpu_parity/output

nvidia-smi --query-gpu=index,name --format=csv,noheader
uv run --extra cuda python demo/multi_gpu_parity/multi_gpu_parity_demo.py --run --gpus 1 2 8
