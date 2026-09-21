#!/bin/bash
# Submit the three Lenormand regimes as parallel slurm jobs (one GPU each).
# Each regime runs run_lenormand.slurm, which auto-resumes from the newest
# checkpoint in demo/lenormand/output/<regime> — resubmitting this script after
# a wall-clock timeout continues the runs (relevant for capillary, ~1e7 steps).
#
#   bash demo/lenormand/submit_all.sh            # all three regimes
#   bash demo/lenormand/submit_all.sh capillary  # a subset
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # repository root
mkdir -p demo/lenormand/output

regimes=("$@")
[[ ${#regimes[@]} -eq 0 ]] && regimes=(stable viscous capillary)
for r in "${regimes[@]}"; do
    sbatch --job-name="lenormand_$r" --export=ALL,REGIME="$r" \
        demo/lenormand/run_lenormand.slurm
done
