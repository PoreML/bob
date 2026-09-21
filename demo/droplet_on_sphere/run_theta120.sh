#!/usr/bin/env bash
# Run the theta=120 contact-angle validation on its own (GIFs/frames are
# tagged t120_* so the two angles' artifacts coexist in output/).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYVISTA_OFF_SCREEN=true
exec "$ROOT/scripts/gpu-run.sh" python "$ROOT/demo/droplet_on_sphere/droplet_on_sphere_demo.py" --thetas 120 "$@"
