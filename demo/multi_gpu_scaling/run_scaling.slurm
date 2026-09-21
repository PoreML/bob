#!/bin/bash
#SBATCH --job-name=bob_mgpu_scaling
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=demo/multi_gpu_scaling/output/slurm-%j.log

# Strong-scaling benchmark: same drainage run on 1/2/4/8 of the allocated GPUs,
# then XLA communication-overlap flag variants at 256^3 (overlapping the per-roll
# halo CollectivePermutes with interior compute).
# Submit from the repository root:  sbatch demo/multi_gpu_scaling/run_scaling.slurm
# The #SBATCH partition/QOS settings above are site-specific; adapt them to your cluster.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p demo/multi_gpu_scaling/output

nvidia-smi --query-gpu=index,name --format=csv,noheader
DEMO=demo/multi_gpu_scaling/multi_gpu_scaling_demo.py

# ---- phase 1: main sweep, default flags -------------------------------------
rm -f demo/multi_gpu_scaling/output/scaling.csv   # fresh sweep
for size in 64 128 256; do
  for n in 1 2 4 8; do
    uv run --extra cuda python "$DEMO" --gpus "$n" --size "$size" --steps 200 --block 50
  done
done
uv run --extra cuda python "$DEMO" --report

# ---- phase 2: XLA flag variants at 256^3 ------------------------------------
# lhs      : latency-hiding scheduler (overlap collectives with compute)
# lhs_cpd  : + decompose CollectivePermute into async send/recv pairs
# lhs_p2p  : + pipelined point-to-point communication
declare -A FLAGSETS=(
  [lhs]="--xla_gpu_enable_latency_hiding_scheduler=true"
  [lhs_cpd]="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_collective_permute_decomposer_threshold=0"
  [lhs_p2p]="--xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_collective_permute_decomposer_threshold=0 --xla_gpu_enable_pipelined_p2p=true"
)
for name in lhs lhs_cpd lhs_p2p; do
  csv="flags_${name}.csv"
  rm -f "demo/multi_gpu_scaling/output/$csv"
  for n in 1 4 8; do
    # a flag set that XLA rejects (pipelined p2p aborts the compiler on this program) must not end the sweep
    XLA_FLAGS="${FLAGSETS[$name]}" \
      uv run --extra cuda python "$DEMO" --gpus "$n" --size 256 --steps 200 --block 50 --csv "$csv" \
      || echo "=== flag set '$name' failed on $n GPU(s); continuing ==="
  done
  uv run --extra cuda python "$DEMO" --report --csv "$csv"
done
