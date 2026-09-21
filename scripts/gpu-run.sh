#!/usr/bin/env bash
# Run a command under `uv run` with JAX's pip-bundled CUDA libraries taking
# priority on LD_LIBRARY_PATH. Without this, a system CUDA install earlier on
# LD_LIBRARY_PATH shadows the wheels and JAX silently falls back to CPU
# ("a CUDA-enabled jaxlib is not installed. Falling back to cpu").
#
# JAX still auto-detects the hardware: with the CUDA plugin reachable it uses
# an available GPU, and on a machine with no GPU it falls back to CPU. So this
# wrapper is safe to use everywhere.
#
#   ./scripts/gpu-run.sh python demo/laplace/laplace_demo.py
#   ./scripts/gpu-run.sh pytest -q
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NV="$(find "$ROOT/.venv" -type d -path '*nvidia*/lib' 2>/dev/null | paste -sd: -)"
if [[ -n "$NV" ]]; then
  export LD_LIBRARY_PATH="${NV}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec uv run "$@"
