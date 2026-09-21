#!/bin/bash
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00        # default; submit_phase.py overrides per GPU class (1-GPU 12h, 4-GPU 48h)
#SBATCH --requeue

# Submit from the repository root; partition/QOS are site-specific.
#
# One array task = one campaign run. The array index selects a line from the TSV
# manifest that submit_phase.py wrote; this script dispatches it to the right
# case driver. There are no job dependencies: SLURM's --array=...%N throttle
# starts the next task the moment a slot frees, so the campaign holds a fixed GPU
# footprint and never waits on a predecessor to finish.
#
# --qos=normal (lowest priority) on purpose: a dataset-scale campaign at a
# high-priority QOS would displace every other user of a shared cluster.
#
# --cpus-per-task=4 / --mem=48G are deliberately small. A low-priority job only
# starts early if backfill can fit it in a gap without delaying higher-priority
# work, and on a GPU cluster the free CPUs on a node are often the binding
# resource, so a small host footprint lets more tasks fit per node. The solver
# runs on the GPU; the host side is per-block metrics, one PyVista render and a
# gzip h5 append, all effectively serial, so 4 cores is ample.
#
# Walltime is set per GPU class by submit_phase.py:
#   1-GPU  12h  — one segment finishes these runs, and a short request backfills
#                 into gaps a long one cannot reach
#   4-GPU  48h  — large-domain runs can take longer than a day; at 12 h each would need
#                 several resubmissions, re-paying JAX compile and a multi-GB checkpoint read
# The drainage and GDL drivers checkpoint, so a task killed at the walltime loses one
# checkpoint interval, not the run; re-running submit_phase.py picks up whatever did
# not finish.
#
# The %M in --array=0-N%M is SLURM's concurrency throttle: every task is queued,
# but at most M run at once, which is what fixes the campaign's GPU footprint
# (20 x 1 GPU + 3 x 4 GPUs = 32).
#
# Launched by case/submit_phase.py; not usually run by hand. To do so:
#   MANIFEST=case/_manifests/<manifest>.tsv sbatch --array=0-15%20 --gres=gpu:1 \
#     case/run_array.slurm
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

MANIFEST="${MANIFEST:?set MANIFEST=<tsv written by submit_phase.py>}"
IDX="${SLURM_ARRAY_TASK_ID:?this script must run as a job array}"

# TSV columns: case, name, gpus, out, geometry path, M, theta
LINE=$(sed -n "$((IDX + 1))p" "$MANIFEST")
[[ -n "$LINE" ]] || { echo "no line $((IDX + 1)) in $MANIFEST"; exit 1; }
IFS=$'\t' read -r CASE NAME GPUS OUT GEOM M THETA <<< "$LINE"
mkdir -p "$OUT"

# Prepend the venv's nvidia libs so cuSPARSE loads (the system CUDA otherwise
# shadows it and JAX silently falls back to CPU).
PY="${PY:-.venv/bin/python}"
NV=$("$PY" -c "import nvidia, os; print(os.path.dirname(nvidia.__file__))")
LIBS=$(find "$NV" -name '*.so*' -path '*lib*' -exec dirname {} \; | sort -u | tr '\n' ':')
export LD_LIBRARY_PATH="${LIBS}${LD_LIBRARY_PATH:-}"

echo "=== node $(hostname) | job ${SLURM_JOB_ID} (array ${SLURM_ARRAY_JOB_ID}[${IDX}]) | $(date)"
echo "=== case $CASE | run $NAME | gpus $GPUS | M $M | theta $THETA"
echo "=== geometry $GEOM"
echo "=== out $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$PY" -c "import jax; print('JAX devices:', jax.devices())"

# Connectivity gate. The draw already rejects non-percolating geometries, but a
# manifest can outlive the draw that produced it (arrays are re-submittable, and
# a manifest is read at task start), so the check is repeated here, where a bad
# geometry would cost a GPU slot. A cube with no inlet->outlet path cannot drain:
# the run pushes dp into its cap at a few percent of the step budget. ~0.5 s.
case "$CASE" in
  drainage|GDL|trapping)
    if ! "$PY" case/geometry_qc.py --check "$GEOM"; then
      echo "=== ABORT $NAME: geometry has no inlet->outlet path, it cannot drain."
      echo "=== The frozen draw excludes such geometries: write a fresh manifest with case/submit_phase.py."
      exit 1
    fi
    ;;
esac

# Resume if this run already has state on disk (the array is re-submittable: a
# drainage/GDL task that hit the walltime continues from its newest checkpoint).
case "$CASE" in
  drainage|GDL)
    RESUME=()
    compgen -G "$OUT/ckpt_step*.npz" > /dev/null && RESUME=(--resume "$OUT")
    DEMO=$([[ "$CASE" == "drainage" ]] && echo case/drainage/drainage_demo.py || echo case/GDL/drainage_gdl_demo.py)
    "$PY" -u "$DEMO" \
        --structure "$GEOM" --out "$OUT" --name "$NAME" \
        --m "$M" --theta "$THETA" --gpus "$GPUS" --beta 0.99 "${RESUME[@]}"
    ;;
  trapping)
    # two-stage driver: skip stage 1 once its equilibrated state exists; a flood
    # without stage-prefixed checkpoints starts again from that state
    # --pmi-workers matches --cpus-per-task: PMI forks that many processes for the
    # stage-1 morphological init, and 8 on 4 cores just oversubscribes
    ARGS=(--structure "$GEOM" --out "$OUT" --name "$NAME" --m "$M" --theta "$THETA" --gpus "$GPUS"
          --pmi-workers "${SLURM_CPUS_PER_TASK:-4}"
          --beta 0.99 --mob-tol 0.003)
    if [[ -f "$OUT/equilibrated_state.npz" ]]; then
        ARGS+=(--stage flood)
        compgen -G "$OUT/ckpt_flood_step*.npz" > /dev/null && ARGS+=(--resume)
    fi
    "$PY" -u case/trapping/trapping_demo.py "${ARGS[@]}"
    ;;
  *)
    echo "unknown case '$CASE'"; exit 1 ;;
esac

echo "=== done $(date)"
