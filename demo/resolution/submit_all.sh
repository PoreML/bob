#!/bin/bash
# Submit the three resolution-study drainage runs (capillary regime, Ca=1e-5,
# mirror of capillary_s005_nu004) — one per resolution of the same geometry.
# GPU counts: 256^3 -> 8 cards (z-sharded), 128^3 -> 2, 64^3 -> 1.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repository root
mkdir -p demo/resolution/output

SIZE=256 GPUS=8 sbatch --job-name=res256_ca1e-5 --gres=gpu:8 demo/resolution/run_resolution.slurm
SIZE=128 GPUS=2 sbatch --job-name=res128_ca1e-5 --gres=gpu:2 demo/resolution/run_resolution.slurm
SIZE=64  GPUS=1 sbatch --job-name=res64_ca1e-5  --gres=gpu:1 demo/resolution/run_resolution.slurm

squeue -u "$USER" -o '%.10i %.16j %.8T %.10M %.6D %b'
