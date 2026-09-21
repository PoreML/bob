#!/bin/bash
#SBATCH --job-name=bob_mgpu_halo4
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=00:40:00
#SBATCH --output=demo/multi_gpu_halo/output/slurm-%j.log

# Reduced sweep for a 4-GPU allocation: 256^3, gspmd and halo modes, 1/2/4 GPUs,
# fp32 + fp64, into its own CSV (the full 8-GPU sweep is run_halo.slurm).
# Submit from the repository root (partition is site-specific):
#   sbatch demo/multi_gpu_halo/run_halo_4g.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
mkdir -p demo/multi_gpu_halo/output

nvidia-smi --query-gpu=index,name --format=csv,noheader
DEMO=demo/multi_gpu_halo/multi_gpu_halo_demo.py
CSV=halo_scaling_4g.csv
rm -f "demo/multi_gpu_halo/output/$CSV"

for mode in gspmd halo; do
  for n in 1 2 4; do
    uv run --extra cuda python "$DEMO" --gpus "$n" --size 256 --mode "$mode" --steps 200 --block 50 --csv "$CSV"
    uv run --extra cuda python "$DEMO" --gpus "$n" --size 256 --mode "$mode" --fp64 --steps 200 --block 50 --csv "$CSV"
  done
done

uv run --extra cuda python "$DEMO" --report --csv "$CSV"
