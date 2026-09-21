#!/bin/bash
# Submit tuned-sigma/nu capillary runs on the full 256^3 pack, one GPU job each,
# 7-day wall clock, automatic resume on resubmit. At fixed Ca=1e-5,
# u_in = Ca*sigma/nu0, so lower nu0 = fewer steps to breakthrough at the same
# capillary number. Output folders are dedicated (output/capillary_<tag>),
# separate from the preset output/capillary run.
#
# Usage: bash demo/lenormand/submit_tuned.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repository root
mkdir -p demo/lenormand/output

submit() {  # tag sigma nu0
    REGIME=capillary SIGMA="$2" NU0="$3" TAG="$1" FP32=1 \
        sbatch --job-name="lenormand_cap_$1" demo/lenormand/run_lenormand_tuned.slurm
}

# Stability limits (64^3 crop sweep): sigma stays 0.05 — sigma=0.10 destabilizes
# late (u_max 0.5-0.6) and raises entry pressures. The usable axis is nu0 down:
# on 64^3 the nu_blue floor lies between 0.0061 (stable) and 0.0046 (immediate NaN).
# The full 256^3 pack is less tolerant by about one rung: nu0=0.0333 (3x, stable
# on 64^3) goes NaN at step 2000 on the full pack — it samples ~64x more throats,
# so a single extreme site is enough. Keep >=1 rung of 64^3 margin.
submit s005_nu005 0.05 0.05     # 2x:   u_in=1e-5, 64^3 u_max flat 0.15
submit s005_nu004 0.05 0.04     # 2.5x: u_in=1.25e-5, between the rungs

squeue -u "$USER" -o "%.10i %.24j %.8T %.12M %R"
