#!/bin/bash
#SBATCH --job-name=resolution
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --output=demo/resolution/output/%x-%j.out
#SBATCH --error=demo/resolution/output/%x-%j.out
# Partition, QOS and resource limits above are site-specific: adapt them to your cluster.

# Resolution study: the capillary_s005_nu004 Lenormand setup (capillary regime,
# Ca=1e-5, M=30, sigma=0.05, nu0=0.04, piston inlet, chi=0.8, fp32) run on the
# resolution-study sphere pack at three resolutions of the same
# geometry (data/resolution: 256^3 original, naive 128^3 and 64^3 downsamples).
#
# Submit from the repository root (log paths above are relative to it; the
# demo/resolution/output directory must exist — submit_all.sh creates it) with
# SIZE and a matching --gres:
#   SIZE=256 GPUS=8 sbatch --job-name=res256_ca1e-5 --gres=gpu:8 demo/resolution/run_resolution.slurm
#   SIZE=128 GPUS=2 sbatch --job-name=res128_ca1e-5 --gres=gpu:2 demo/resolution/run_resolution.slurm
#   SIZE=64          sbatch --job-name=res64_ca1e-5              demo/resolution/run_resolution.slurm
#
# 7-day wall clock; auto-resumes from the newest checkpoint in OUT on resubmit.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

SIZE="${SIZE:?set SIZE=256|128|64}"
GPUS="${GPUS:-1}"                 # z-shard across N GPUs (bob.multigpu); submit with --gres=gpu:N
STEPS="${STEPS:-100000000}"       # effectively uncapped; the run stops at breakthrough first
FIELD_EVERY="${FIELD_EVERY:-20}"  # blocks between HDF5 field dumps

# Mirrors the demo/lenormand capillary_s005_nu004 configuration:
REGIME=capillary                  # Ca=1e-5 preset, M=30, chi=0.8, theta=135
SIGMA=0.05
NU0=0.04

# resolution-study dataset; default assumes a geometry checkout next to this repository.
STRUCT_DIR="${STRUCT_DIR:-../geometry/data/resolution}"
case "$SIZE" in
    256) STRUCTURE="$STRUCT_DIR/resolution_0000.npy" ;;
    128) STRUCTURE="$STRUCT_DIR/resolution_0000_ds128.npy" ;;
    64)  STRUCTURE="$STRUCT_DIR/resolution_0000_ds64.npy" ;;
    *)   echo "unknown SIZE=$SIZE" >&2; exit 1 ;;
esac
OUT="demo/resolution/output/res${SIZE}"
NAME="resolution_${SIZE}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false

# float32 is the solver default; FP64=1 opts back into float64.
if [[ "${FP64:-0}" == "1" ]]; then
    export BOB_FP64=1
    echo "=== float64 solver requested ==="
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

echo "=== node $(hostname)  job ${SLURM_JOB_ID:-?}  size $SIZE  gpus $GPUS  structure $STRUCTURE  out $OUT  $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
"$PY" -c "import jax; print('JAX devices:', jax.devices())"

# Single-process job: run python directly (no srun — its cpu-bind step fails when
# two jobs share a node).
"$PY" -u demo/lenormand/lenormand_demo.py --regime "$REGIME" --steps "$STEPS" \
    --sigma "$SIGMA" --nu0 "$NU0" --gpus "$GPUS" \
    --structure "$STRUCTURE" --name "$NAME" \
    --field-every "$FIELD_EVERY" --out "$OUT" "${RESUME_ARGS[@]}"

echo "=== done $(date) ==="
