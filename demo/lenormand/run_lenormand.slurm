#!/bin/bash
#SBATCH --job-name=lenormand
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output=demo/lenormand/output/%x-%j.out
#SBATCH --error=demo/lenormand/output/%x-%j.out
# Partition, QOS and resource limits above are site-specific: adapt them to your cluster.

# One Lenormand regime on the full 256^3 sphere pack: leak-free piston inlet +
# blue-pinned pressure outlet + mrt_chi=0.8 (see demo/lenormand/README.md).
#
# Submit from the repository root (log paths above are relative to it; the
# demo/lenormand/output directory must exist — submit_all.sh creates it).
#
# The demo stops at breakthrough on its own (invading red reaches the outlet);
# --steps is only a safety cap (default effectively uncapped). The capillary
# regime (Ca=1e-5, ~1e7 steps) outlives the 3-day wall clock: this script
# resumes automatically from the newest checkpoint in the output dir, so resubmit
# the same job when it times out. Set FRESH=1 to ignore checkpoints and start
# over (the demo then clears stale frames itself; delete the output dir first
# if you also want clean metrics).
#
# Submit one job per regime (or use submit_all.sh):
#   REGIME=stable    sbatch --job-name=lenormand_stable    demo/lenormand/run_lenormand.slurm
#   REGIME=viscous   sbatch --job-name=lenormand_viscous   demo/lenormand/run_lenormand.slurm
#   REGIME=capillary sbatch --job-name=lenormand_capillary demo/lenormand/run_lenormand.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

REGIME="${REGIME:?set REGIME=stable|viscous|capillary}"
STEPS="${STEPS:-100000000}"  # effectively uncapped; the run stops at breakthrough first
FIELD_EVERY="${FIELD_EVERY:-20}"  # blocks between HDF5 field dumps (fields.h5 + .xdmf for ParaView)
OUT="demo/lenormand/output/$REGIME"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Prepend the venv's nvidia libs so cuSPARSE loads (system CUDA otherwise shadows it
# and JAX silently falls back to CPU).
PY="${PY:-.venv/bin/python}"
NV=$("$PY" -c "import nvidia, os; print(os.path.dirname(nvidia.__file__))")
LIBS=$(find "$NV" -name '*.so*' -path '*lib*' -exec dirname {} \; | sort -u | tr '\n' ':')
export LD_LIBRARY_PATH="${LIBS}${LD_LIBRARY_PATH:-}"

RESUME_ARGS=()
if [[ "${FRESH:-0}" != "1" ]] && compgen -G "$OUT/ckpt_step*.npz" > /dev/null; then
    RESUME_ARGS=(--resume "$OUT")
    echo "=== resuming from newest checkpoint in $OUT ==="
fi

echo "=== node $(hostname)  job ${SLURM_JOB_ID:-?}  regime $REGIME  steps $STEPS  $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
"$PY" -c "import jax; print('JAX devices:', jax.devices())"

# Single-process job: run python directly (no srun — its cpu-bind step fails when
# two jobs share a node).
"$PY" -u demo/lenormand/lenormand_demo.py --regime "$REGIME" --steps "$STEPS" \
    --field-every "$FIELD_EVERY" "${RESUME_ARGS[@]}"

echo "=== done $(date) ==="
