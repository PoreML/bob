# Multi-GPU parity — sharded runs reproduce the single-device physics

Verifies that domain decomposition does not change the physics. Four scenarios
re-run the signature physics of existing demos at 1/2/N devices through
`bob.multigpu`, each sharded along an adversarial axis so that the slabs cut
through walls, inlets or pressure boundary planes. The companion benchmark
`demo/multi_gpu_scaling` covers the performance of the standard configuration
(porous x-flow sharded along z).

## Setup

The scenarios use plain GSPMD sharding (`multigpu.mesh(n)` +
`multigpu.shard(..., axis=...)`): the state and every 3D field the step closes
over are placed on a 1D device mesh and XLA's partitioner compiles each
streaming `jnp.roll` across a shard boundary into a one-plane halo exchange.
This is the path that supports all three spatial axes. The z-only `shard_map`
halo backend (`multigpu.staged_halo_step`) is benchmarked in
`demo/multi_gpu_halo`.

| scenario | pattern of | shard axis | what lies on the sharded axis |
|---|---|---|---|
| wetting | `demo/wetting` (sessile drop, sealed z-walls) | z | both wall planes live inside the first/last slab |
| washburn | `demo/washburn` (Zou-He capillary rise) | x | both pressure-BC planes |
| drainage | rate-controlled velocity inlet + pressure outlet on the 64^3 sphere-pack crop | y | non-default axis on a porous geometry |
| pc | `demo/capillary_pressure` (fixed-pressure layer slabs) | x | both pressure slabs |

All runs are float64. Each scenario records its phase field and its physics
observable once per block (20 blocks per run).

## How to run

```bash
sbatch demo/multi_gpu_parity/run_parity.slurm     # compute phase on an 8-GPU node
uv run python demo/multi_gpu_parity/multi_gpu_parity_demo.py --render --gpus 1 2 8   # GIFs + report
# CPU smoke test (4 emulated devices, no GPU needed):
XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
  uv run python demo/multi_gpu_parity/multi_gpu_parity_demo.py --run --gpus 1 4 --smoke
```

`--run` needs no display stack; `--render` post-processes the saved snapshots
and can run on a workstation.

## Validation criterion

The physics observable (contact angle / imbibed length / red saturation / S_w)
must match the single-device run within a per-scenario tolerance (0.2 deg,
0.5 lu, 1e-3, 1e-3). The raw phase-field difference max |Δφ| is reported
alongside but is not the criterion: over a few dozen steps the sharded state
matches the single-device state to float64 roundoff (pinned at 1e-12 on every
axis by `test/test_multigpu.py`), but a partitioned compilation may differ by
~1 ULP per step (GSPMD changes XLA's fusion order), and a moving contact line
amplifies such a seed exponentially. Under CPU emulation this drift is
localised at the interface, not at slab boundaries (max |Δφ| 5e-16 -> 2.5e-6
over 100 steps while the contact angle agrees to 6 digits) — the same
decorrelation any MPI decomposition exhibits. `<name>_drift.png` plots it.

## Results (8x H100, float64)

All scenarios pass, and on the GPUs the runs are bit-identical (max |Δφ| = 0.0
everywhere) across 1/2/8 devices for the full runs:

| scenario | shard axis | steps | metric, 1 GPU = 2 = 8 | max \|Δφ\| |
|---|---|---:|---|---:|
| wetting (60° imposed) | z (walls on axis) | 4000 | contact angle 61.6032° | 0.0 |
| washburn | x (Zou-He planes on axis) | 12000 | imbibed length 45.1 lu | 0.0 |
| drainage (sphere64) | y | 3000 | red saturation 0.220866 | 0.0 |
| pc (square tube) | x (pressure slabs on axis) | 6000 | S_w 0.789 | 0.0 |

## Implementation note: plane boundary conditions under GSPMD

In jax 0.6.2 the GSPMD partitioner miscompiles a slab update
`.at[:, :, :, :n].set(...)` that is followed by a second slice-update at the far
end of the same sharded axis; no error is raised, and an inlet + outlet pair
written that way under `axis="x"` deviates by 5.6e-2 after one step, although
each boundary condition alone is exact. Plane and slab boundary conditions in
bob (`bc.inlet_reservoir`, `bc.outlet_zero_gradient`, the capillary-pressure
layer slabs) are therefore written as elementwise `jnp.where` on an index mask (plus
`jnp.roll` for neighbour-plane copies), which partitions exactly. The `clamp`
case of `test/test_multigpu.py` pins this on every shard axis. User-written boundary
conditions should follow the same pattern.

## Outputs

In `output/` (gitignored), per scenario: `<name>_parity.gif` (1 device |
N devices | |Δφ| midplane heat map, one frame per block), `<name>_final.png`,
`frames/<name>/`; plus `parity.csv`, `report.md` and the raw snapshots in
`runs/`. The analysis curves `<name>_metric.png` (metric curves per device
count, which must coincide) and `<name>_drift.png` (log-scale |Δφ| growth) are
written only with `--plots`.
