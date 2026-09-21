#!/bin/bash
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00

# Submit from the repository root; partition/QOS are site-specific.
#
# One flip-chip solder-ball underfill run (case/underfill/underfill_demo.py:
# capillary Zou-He dP=0 fill on a data/flipchip geometry sample, 1000-step
# log/2D+3D frame/h5 interval). Stage the geometry first with
# `campaign.py prepare` (or any run of underfill_demo.py), then:
#   NAME=fc0007_M10_th40 SAMPLE=7 M=10 THETA=40 \
#     sbatch --job-name=uf_fc0007_M10_th40 \
#     --output=case/underfill/runs/fc0007_M10_th40/slurm-%j.log \
#     case/underfill/run_underfill.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

NAME="${NAME:?set NAME=<run folder name>}"
SAMPLE="${SAMPLE:?set SAMPLE=<dataset sample index>}"
M="${M:?set M=<viscosity ratio>}"
THETA="${THETA:?set THETA=<contact angle deg>}"
# Runs end at the driver's fill-target criterion (array fill >= 0.99 + settle
# blocks); this step cap is only a safety backstop. The driver's own default is
# 3,000,000; raise STEPS for slow (high-M) samples.
STEPS="${STEPS:-1200000}"
RUN_DIR="case/underfill/runs/${NAME}"
[[ -f "${RUN_DIR}/${NAME}.npy" ]] || { echo "missing geometry ${RUN_DIR}/${NAME}.npy (run campaign.py prepare)"; exit 1; }

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

echo "=== node $(hostname)  job ${SLURM_JOB_ID:-?}  ${NAME}  $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
"$PY" -c "import jax; print('JAX devices:', jax.devices())"

# Single-process job: run python directly (no srun — its cpu-bind step fails when
# two jobs share a node).
"$PY" -u case/underfill/underfill_demo.py \
    --sample "$SAMPLE" \
    --m-ratio "$M" \
    --theta "$THETA" \
    --out "$RUN_DIR" \
    --name "$NAME" \
    --steps "$STEPS" \
    "${RESUME_ARGS[@]}"

echo "=== done $(date) ==="
