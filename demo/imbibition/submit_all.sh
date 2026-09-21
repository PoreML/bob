#!/bin/bash
# Submit all 9 capillary-imbibition runs (3 geometries x {96, 128, 256}).
# 96/128 run on 1 GPU; 256 shards across 4 (bob.multigpu z-sharding).
# Usage:  bash demo/imbibition/submit_all.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repository root

for size in 96 128 256; do
  gpus=1
  [[ "$size" == "256" ]] && gpus=4
  for i in 0 1 2; do
    stem=$(printf "sphere_%04d" "$i")
    run_dir="demo/imbibition/sphere/${size}/${stem}"
    SIZE=$size STEM=$stem GPUS=$gpus \
      sbatch --job-name="imb_${size}_${i}" --gres="gpu:${gpus}" \
      --output="${run_dir}/slurm-%j.log" --error="${run_dir}/slurm-%j.log" \
      demo/imbibition/run_imbibition.slurm
  done
done
squeue -u "$USER" -o "%i %j %T %b %R"
