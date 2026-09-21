# Multi-GPU strong scaling — 3D drainage under plain GSPMD sharding

Strong-scaling benchmark of the production drainage pipeline
(`porous3d.drain`, default MRT + CSF + akai wetting stack, rate-controlled
velocity inlet) on 1/2/4/8 GPUs, with a cross-device check that sharding leaves
the physics unchanged.

## Setup

- **Decomposition.** `bob.multigpu` places the state on a 1D device mesh
  sharded along z (`multigpu.mesh(n)` + `multigpu.shard(...)`). The solver is
  untouched: it is pure `jnp` code under `jit`/`scan`, so XLA's GSPMD
  partitioner compiles each streaming `jnp.roll` that crosses a shard boundary
  into a one-plane halo exchange (CollectivePermute), the equivalent of a
  hand-written MPI slab decomposition. This benchmark measures that plain GSPMD
  path. The `shard_map` backend with per-stage thin ghost-plane halos (phi at
  depth 4 + populations at depth 1, `multigpu.staged_halo_step`) is benchmarked
  separately in `demo/multi_gpu_halo`.
- **Axis.** z is the natural axis for this pipeline: flow and its inlet/outlet
  BCs run along x (each device owns its local piece of every x-plane), and the
  z walls are ordinary solid cells. `multigpu.shard(..., axis="y"/"x")` slices
  the other axes when a domain is laid out differently; `demo/multi_gpu_parity`
  demonstrates that walls and BC planes on the sharded axis stay correct.
- **Domain.** The sphere pack `test/assets/sphere256.npy`: the canonical 64^3
  crop, a centered 128^3 crop, or the full 256^3, each with 10-cell x-buffers;
  omega 1.5, sigma 0.05, theta 135°, Ca 1e-4; float64; one untimed warm-up
  block, then 200 timed steps in 50-step jitted blocks.
- **Metric.** MLUPS = million lattice-site updates per second over the whole
  domain (solid cells included, since they are computed too).

## How to run

```bash
sbatch demo/multi_gpu_scaling/run_scaling.slurm      # from the repository root (8-GPU node; sweep + XLA flag variants)
# or manually, one row per invocation:
uv run --extra cuda python demo/multi_gpu_scaling/multi_gpu_scaling_demo.py --gpus 2 --size 256
uv run python demo/multi_gpu_scaling/multi_gpu_scaling_demo.py --report
# CPU smoke test (no GPU needed):
XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
  uv run python demo/multi_gpu_scaling/multi_gpu_scaling_demo.py --gpus 4 --size 64 --steps 20 --block 10
```

## Validation criterion

Sharded results equal single-device results to 1e-12 (float64 roundoff). This
is pinned hardware-independently by `test/test_multigpu.py` (4 emulated CPU
devices) and re-checked here across GPU counts through the end-state saturation
column of `scaling.csv`: `--report` flags a drift above 1e-9 as FAIL.

## Results (8x H100 80GB, float64, 200 timed steps)

| size  | GPUs | MLUPS | speedup | efficiency |
|------:|-----:|------:|--------:|-----------:|
| 64^3  | 1    | 192.0 | 1.00x   | 100%       |
| 64^3  | 2    | 64.5  | 0.34x   | 17%        |
| 64^3  | 4    | 74.8  | 0.39x   | 10%        |
| 64^3  | 8    | 84.8  | 0.44x   | 6%         |
| 128^3 | 1    | 166.2 | 1.00x   | 100%       |
| 128^3 | 2    | 175.6 | 1.06x   | 53%        |
| 128^3 | 4    | 262.7 | 1.58x   | 40%        |
| 128^3 | 8    | 358.8 | 2.16x   | 27%        |
| 256^3 | 1    | 161.0 | 1.00x   | 100%       |
| 256^3 | 2    | 241.2 | 1.50x   | 75%        |
| 256^3 | 4    | 426.7 | 2.65x   | 66%        |
| 256^3 | 8    | 718.4 | 4.46x   | 56%        |

End-state saturation drift across device counts is 0.00e+00 at every size: the
sharded runs are bit-identical to the single-GPU run.

This is the strong-scaling shape expected of a slab decomposition. 64^3 is
halo- and launch-bound (each GPU owns 8-32 z-planes), so sharding a small
domain slows it down. 256^3 is compute-heavy enough for 1.50x / 2.65x / 4.46x
on 2/4/8 GPUs, with per-doubling ratios of 1.77x (2→4) and 1.68x (4→8).
Efficiency tracks slab thickness (planes per GPU), so larger volumes scale
better at a fixed device count.

### XLA communication-overlap flags

Three XLA communication-overlap configurations were benchmarked at 256^3
(`flags_*.csv`, phase 2 of the SLURM script) against the default:

| config (256^3), MLUPS | 1 GPU | 4 GPU | 8 GPU |
|---|---:|---:|---:|
| default | 161.0 | 426.7 | 718.4 |
| `--xla_gpu_enable_latency_hiding_scheduler=true` | 160.9 | 425.3 | 707.4 |
| + `--xla_gpu_collective_permute_decomposer_threshold=0` | 160.8 | 425.9 | 707.5 |
| + `--xla_gpu_enable_pipelined_p2p=true` | 160.9 | XLA SIGABRT | — |

No configuration improves on the default (all differences are within
run-to-run noise), and pipelined p2p aborts XLA's compiler on this program in
jax 0.6.2. The default schedule already hides the collective latency; the
residual efficiency loss is the partitioner's per-roll slice/concat traffic and
the surface-to-volume ratio of thin slabs. The effective throughput levers are
thicker slabs (larger domains or fewer GPUs per volume), float32 (about 2x at
parity on this stack, and it halves the halo bytes), and the thin-halo
`shard_map` backend of `demo/multi_gpu_halo`.

### Dispatch-loop requirement

In a dispatch loop (`state = run_block(state)`), pin `out_shardings` to the
input shardings and pass `donate_argnums=0`, as this demo does. Otherwise XLA
can return a differently laid-out output at large sizes; every call is then a
jit cache miss, and jax 0.6.2 fails while re-lowering (`TypeError: ... Got
UnspecifiedValue`). Donation also halves device memory. See the `bob.multigpu`
module docstring.

## Outputs

In `output/` (gitignored): `scaling.csv` (size, ngpu, steps, seconds, mlups,
saturation), `report.md`, the `flags_*` CSV/report variants and the SLURM log.
The `speedup.svg` curve (and its `flags_*` variant) is written only when
`--report` is passed `--plots`.
