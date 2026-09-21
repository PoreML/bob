#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --job-name=fig4_drain
#SBATCH --output=demo/capillary_fill/slurm-%j.log

# Dynamic drainage demo (demo/capillary_fill/drainage_fill_demo.py) on one GPU.
# ~2.4M cells, ~6.8M steps at Ca 1e-4: about 17 h at ~110 steps/s (H200). The demo
# checkpoints every 10 blocks; resubmit with the same command to resume (--resume is
# passed automatically when a checkpoint exists; FRESH=1 starts over).
#
# Submit from the repository root (partition/QOS above are site-specific — adapt them):
#   sbatch demo/capillary_fill/run_fig4.slurm            # extra demo flags via ARGS="..."
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
OUT="${OUT:-demo/capillary_fill/output_fig4_ca1e-4}"
mkdir -p "$OUT"

# Prepend the venv's CUDA libraries so they take priority over a system CUDA install
# (otherwise cuSPARSE fails to load and JAX falls back to CPU).
PY="${PY:-.venv/bin/python}"
NV=$("$PY" -c "import nvidia, os; print(os.path.dirname(nvidia.__file__))")
LIBS=$(find "$NV" -name '*.so*' -path '*lib*' -exec dirname {} \; | sort -u | tr '\n' ':')
export LD_LIBRARY_PATH="${LIBS}${LD_LIBRARY_PATH:-}"

RESUME_ARGS=()
if [[ "${FRESH:-0}" != "1" && -f "$OUT/checkpoint.npz" ]]; then
    RESUME_ARGS=(--resume)
    echo "=== resuming from $OUT/checkpoint.npz ==="
fi

echo "=== node $(hostname)  job ${SLURM_JOB_ID:-?}  $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
"$PY" -c "import jax; print('JAX devices:', jax.devices())"

"$PY" -u demo/capillary_fill/drainage_fill_demo.py --out "$OUT" "${RESUME_ARGS[@]}" ${ARGS:-} \
    2>&1 | tee -a "$OUT/run.log"

echo "=== done $(date) ==="
