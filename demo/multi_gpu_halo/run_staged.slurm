#!/bin/bash
#SBATCH --job-name=bob_mgpu_staged
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=demo/multi_gpu_halo/output/slurm-%j.log

# Staged thin-halo backend (phi depth-4 + populations depth-1 exchanges), run after
# run_halo.slurm: appends to halo_scaling.csv so the report shows all three modes
# side by side. Submit from the repository root (partition is site-specific):
#   sbatch demo/multi_gpu_halo/run_staged.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
mkdir -p demo/multi_gpu_halo/output

nvidia-smi --query-gpu=index,name --format=csv,noheader
DEMO=demo/multi_gpu_halo/multi_gpu_halo_demo.py

# ---- phase 1: fp32 (solver default) — all sizes/counts ----------------------
for size in 64 128 256; do
  for n in 1 2 4 8; do
    uv run --extra cuda python "$DEMO" --gpus "$n" --size "$size" --mode staged --steps 200 --block 50
  done
done

# ---- phase 2: fp64 256^3 — comparability with demo/multi_gpu_scaling ---------
for n in 1 2 4 8; do
  uv run --extra cuda python "$DEMO" --gpus "$n" --size 256 --mode staged --fp64 --steps 200 --block 50
done

uv run --extra cuda python "$DEMO" --report
