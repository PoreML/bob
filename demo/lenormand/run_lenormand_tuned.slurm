#!/bin/bash
#SBATCH --job-name=lenormand_tuned
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --output=demo/lenormand/output/%x-%j.out
#SBATCH --error=demo/lenormand/output/%x-%j.out
# Partition, QOS and resource limits above are site-specific: adapt them to your cluster.

# Tuned-sigma/nu Lenormand run on the full 256^3 pack.
# Same driver as run_lenormand.slurm, but sigma/nu0 (and optionally Ca) come from
# the environment and the output goes to a dedicated folder output/<regime>_<TAG>,
# separate from the preset output/<regime> runs. At fixed Ca, u_in = Ca*sigma/nu0,
# so larger sigma or lower nu0 = proportionally fewer steps to breakthrough (see
# the stability limits in demo/lenormand/README.md).
#
# Submit from the repository root (log paths above are relative to it; the
# demo/lenormand/output directory must exist). 7-day wall clock; resumes
# automatically from the newest checkpoint in OUT, so resubmit the same command
# when it times out.
#
#   REGIME=capillary SIGMA=0.10 NU0=0.05 TAG=s010_nu005 \
#     sbatch --job-name=lenormand_cap_s010_nu005 demo/lenormand/run_lenormand_tuned.slurm
#
# Multi-GPU (z-sharded via bob.multigpu; 256^3 = 4.46x on 8 H100s compute-side):
#   REGIME=capillary SIGMA=0.05 NU0=0.05 TAG=s005_nu005_8g GPUS=8 \
#     sbatch --job-name=lenormand_cap_s005_nu005_8g --gres=gpu:8 demo/lenormand/run_lenormand_tuned.slurm
# The solver runs float32 by default (FP64=1 opts out); the 8-GPU fp32 run measured
# 55 steps/s wall (~975 MLUPS effective) vs ~20 steps/s for 1-GPU fp32.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

REGIME="${REGIME:?set REGIME=stable|viscous|capillary}"
SIGMA="${SIGMA:?set SIGMA (e.g. 0.10)}"
NU0="${NU0:?set NU0 (e.g. 0.05)}"
TAG="${TAG:?set TAG for the output folder (e.g. s010_nu005)}"
CA_ARGS=()
[[ -n "${CA:-}" ]] && CA_ARGS=(--ca "$CA")
STEPS="${STEPS:-100000000}"       # effectively uncapped; the run stops at breakthrough first
FIELD_EVERY="${FIELD_EVERY:-20}"  # blocks between HDF5 field dumps
GPUS="${GPUS:-1}"                 # z-shard across N GPUs (bob.multigpu); submit with --gres=gpu:N
OUT="demo/lenormand/output/${REGIME}_${TAG}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

# float32 is the solver default (~2x steps/s, half the memory; validated at
# parity with float64 on a 64^3 crop). FP64=1 opts back into float64; FP32=1 is
# also accepted and forces fp32 explicitly.
if [[ "${FP64:-0}" == "1" ]]; then
    export BOB_FP64=1
    echo "=== float64 solver requested ==="
elif [[ "${FP32:-0}" == "1" ]]; then
    export BOB_FP32=1
fi

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

echo "=== node $(hostname)  job ${SLURM_JOB_ID:-?}  regime $REGIME  sigma $SIGMA  nu0 $NU0  ca ${CA:-preset}  out $OUT  $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
"$PY" -c "import jax; print('JAX devices:', jax.devices())"

# Single-process job: run python directly (no srun — its cpu-bind step fails when
# two jobs share a node).
"$PY" -u demo/lenormand/lenormand_demo.py --regime "$REGIME" --steps "$STEPS" \
    --sigma "$SIGMA" --nu0 "$NU0" "${CA_ARGS[@]}" --gpus "$GPUS" \
    --field-every "$FIELD_EVERY" --out "$OUT" "${RESUME_ARGS[@]}"

echo "=== done $(date) ==="
