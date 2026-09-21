#!/bin/bash
#SBATCH --job-name=bob_mgpu_halo
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=demo/multi_gpu_halo/output/slurm-%j.log

# Halo-step (packed shard_map ghost-plane exchange) vs the GSPMD baseline, same
# drainage benchmark protocol as demo/multi_gpu_scaling.
# Submit from the repository root (partition is site-specific — adapt it):
#   sbatch demo/multi_gpu_halo/run_halo.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repo root}"
mkdir -p demo/multi_gpu_halo/output

nvidia-smi --query-gpu=index,name --format=csv,noheader
DEMO=demo/multi_gpu_halo/multi_gpu_halo_demo.py

rm -f demo/multi_gpu_halo/output/halo_scaling.csv   # fresh sweep

# ---- phase 1: fp32 (solver default) — both modes, all sizes/counts ----------
for size in 64 128 256; do
  for mode in gspmd halo; do
    for n in 1 2 4 8; do
      uv run --extra cuda python "$DEMO" --gpus "$n" --size "$size" --mode "$mode" --steps 200 --block 50
    done
  done
done

# ---- phase 2: fp64 256^3 — comparability with demo/multi_gpu_scaling ---------
for mode in gspmd halo; do
  for n in 1 2 4 8; do
    uv run --extra cuda python "$DEMO" --gpus "$n" --size 256 --mode "$mode" --fp64 --steps 200 --block 50
  done
done

uv run --extra cuda python "$DEMO" --report
