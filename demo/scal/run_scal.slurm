#!/bin/bash
#SBATCH --qos=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=5-00:00:00

# SCAL pressure-ladder drainage + imbibition on Bentheimer 128 (demo/scal).
# Quasi-static Zou-He pressure inlet ladder + porous-plate/scrubber membrane
# outlet; auto-resumes mid-ladder from the newest checkpoint (FRESH=1 restarts),
# so a run cut by the wall-time limit continues on resubmission. Single GPU by
# default (128^3, no sharding).
# The #SBATCH partition/QOS settings above are site-specific; adapt them to your cluster.
#
# Submit from the repository root (RUN_DIR/EXTRA/STRUCTURE/LADDER env-overridable
# for variants, e.g. the fine ladder: RUN_DIR=demo/scal/bentheimer128_fine
# EXTRA="--dpc 0.00035 ..."):
#   sbatch --job-name=scal_bent128 --gres=gpu:1 \
#     --output=demo/scal/bentheimer128/slurm-%j.log \
#     --error=demo/scal/bentheimer128/slurm-%j.log demo/scal/run_scal.slurm
# 256^3 fine ladder (own ladder calibration -- the 128 numbers do not transfer):
#   RUN_DIR=demo/scal/bentheimer256_fine STRUCTURE=test/assets/bentheimer256.npy \
#   LADDER=demo/scal/plate_check/output/ladder256.json \
#   EXTRA="--structure test/assets/bentheimer256.npy \
#     --ladder-json demo/scal/plate_check/output/ladder256.json \
#     --name scal_bent256_fine --dpc 0.000434 --stall-n 8 --pre-blocks 5 \
#     --tol-ca 1e-5 --tol-eq 2e-4 --max-blocks 40 --frame-blocks 10 --gpus 4" \
#   sbatch --job-name=scal_bent256_fine --gres=gpu:4 \
#     --output=demo/scal/bentheimer256_fine/slurm-%j.log \
#     --error=demo/scal/bentheimer256_fine/slurm-%j.log demo/scal/run_scal.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

RUN_DIR="${RUN_DIR:-demo/scal/bentheimer128}"
EXTRA="${EXTRA:-}"
STRUCTURE="${STRUCTURE:-test/assets/bentheimer128.npy}"
LADDER="${LADDER:-demo/scal/plate_check/output/ladder.json}"
mkdir -p "$RUN_DIR"

PY="${PY:-.venv/bin/python}"
# put the CUDA libraries bundled with the venv ahead of any system CUDA
NV=$("$PY" -c "import nvidia, os; print(os.path.dirname(nvidia.__file__))")
LIBS=$(find "$NV" -name '*.so*' -path '*lib*' -exec dirname {} \; | sort -u | tr '\n' ':')
export LD_LIBRARY_PATH="${LIBS}${LD_LIBRARY_PATH:-}"

# the ladder json is a cheap regenerable artifact (CPU ~1-5 min)
[[ -f "$LADDER" ]] || "$PY" demo/scal/plate_check/throat_ladder.py --structure "$STRUCTURE" --out "$LADDER"

RESUME_ARGS=()
if [[ "${FRESH:-0}" != "1" ]] && compgen -G "$RUN_DIR/ckpt_step*.npz" > /dev/null; then
    RESUME_ARGS=(--resume "$RUN_DIR")
    echo "=== resuming from newest checkpoint in $RUN_DIR ==="
fi

echo "=== node $(hostname)  job ${SLURM_JOB_ID:-?}  scal $RUN_DIR  $(date) ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
"$PY" -c "import jax; print('JAX devices:', jax.devices())"

# shellcheck disable=SC2086  # EXTRA is intentionally word-split
"$PY" -u demo/scal/scal_demo.py --out "$RUN_DIR" $EXTRA "${RESUME_ARGS[@]}"

"$PY" demo/scal/publication_plot.py "$RUN_DIR" || true
echo "=== done $(date) ==="
