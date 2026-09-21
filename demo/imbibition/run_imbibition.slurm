#!/bin/bash
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
# Partition, QOS and resource limits above are site-specific: adapt them to your cluster.

# Capillary-dominated imbibition on one sphere-pack geometry (demo/imbibition).
# The capillary-regime drainage configuration with one change: theta 50 deg (red
# wetting) instead of the drainage 135 deg — piston inlet, blue-pinned outlet,
# sigma 0.05, nu0 0.04, Ca 1e-5, M=1, MRT chi 0.8, fp32, 2x-breakthrough run length.
# GPUs come from --gres at submit time (bob.multigpu z-sharding; the GPU count must
# divide the geometry z-extent).
#
# Submit from the repository root, via submit_all.sh or singly:
#   SIZE=256 STEM=sphere_0000 GPUS=4 \
#     sbatch --job-name=imb_256_0000 --gres=gpu:4 \
#     --output=demo/imbibition/sphere/256/sphere_0000/slurm-%j.log \
#     demo/imbibition/run_imbibition.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

SIZE="${SIZE:?set SIZE=96|128|256}"
STEM="${STEM:?set STEM (e.g. sphere_0000)}"
GPUS="${GPUS:-1}"
RUN_DIR="demo/imbibition/sphere/${SIZE}/${STEM}"
STRUCTURE="${RUN_DIR}/${STEM}.npy"
[[ -f "$STRUCTURE" ]] || { echo "missing geometry $STRUCTURE"; exit 1; }

# Prepend the venv's nvidia libs so cuSPARSE loads (system CUDA otherwise shadows
# it and JAX silently falls back to CPU).
PY="${PY:-.venv/bin/python}"
NV=$("$PY" -c "import nvidia, os; print(os.path.dirname(nvidia.__file__))")
LIBS=$(find "$NV" -name '*.so*' -path '*lib*' -exec dirname {} \; | sort -u | tr '\n' ':')
export LD_LIBRARY_PATH="${LIBS}${LD_LIBRARY_PATH:-}"

RESUME_ARGS=()
if [[ "${FRESH:-0}" != "1" ]] && compgen -G "$RUN_DIR/ckpt_step*.npz" > /dev/null; then
    RESUME_ARGS=(--resume "$RUN_DIR")
    echo "=== resuming from newest checkpoint in $RUN_DIR ==="
fi

echo "=== node $(hostname)  job ${SLURM_JOB_ID:-?}  ${SIZE}^3 ${STEM}  gpus $GPUS  $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
"$PY" -c "import jax; print('JAX devices:', jax.devices())"

# Single-process job: run python directly (no srun — its cpu-bind step fails when
# two jobs share a node).
"$PY" -u demo/lenormand/lenormand_demo.py \
    --regime capillary --m 1 --sigma 0.05 --nu0 0.04 \
    --theta 50 \
    --structure "$STRUCTURE" --name "imb_${SIZE}_${STEM}" \
    --after-breakthrough-x 1.0 \
    --gpus "$GPUS" --field-every 20 \
    --out "$RUN_DIR" "${RESUME_ARGS[@]}"

echo "=== done $(date) ==="
