# multi_gpu_halo — halo-exchange domain decomposition vs GSPMD

Benchmark of bob's three multi-GPU backends on the 3D drainage stack (default MRT + CSF +
Akai wetting, rate-controlled inlet), all sharding the domain into slabs along z:

| mode | backend | communication per step |
|---|---|---|
| `gspmd` | automatic partitioning (`multigpu.shard`) | one collective per z-crossing `jnp.roll` — about 100 small messages |
| `halo` | `multigpu.halo_step` | one packed ghost-plane exchange per color (depth 5) |
| `staged` | `multigpu.staged_halo_step` (**bob's default**) | φ at depth 4 before collision + populations at depth 1 before streaming |

## Mechanism

**Why plain GSPMD scales poorly.** Under automatic partitioning, XLA emits a separate
slice/concatenate + `CollectivePermute` for every `jnp.roll` that crosses the sharded axis.
A default-stack step contains 2 × 19 streaming rolls plus the four-stage gradient/curvature
chain of the CSF force, so each step sends on the order of a hundred one-plane messages.
The cost is the number of messages, not their size: parallel efficiency is 49 % on 8 GPUs
(fp32, 256³).

**Packed halo (`halo`).** `multigpu.halo_step` wraps the step in `shard_map`. Each device
pads its z-slab with `required_halo(params)` ghost planes received from its two ring
neighbours — one packed `lax.ppermute` pair per color per step — runs the unmodified
`color3d.step` on the padded slab, and discards the ghost planes. The ghost depth equals
the reach of the step's neighbour-operation chain: 5 for the default CSF stack (φ
extrapolation 1 + gradient 1 + normal extrapolation 1 + curvature 1 + streaming 1).
`test/test_multigpu.py::test_halo_step_depth_is_minimal` pins this depth as minimal (depth
4 breaks parity). Boundary conditions stay outside `shard_map`: they act on x planes and
need no communication under z-sharding. The trade-off: about 100 small messages collapse
into 4 packed ones, but each slab recomputes 2 × 5 ghost planes (31 % redundant compute at
256³ on 8 GPUs, 8 % on 2 GPUs).

**Staged thin halo (`staged`).** The depth-5 requirement comes almost entirely from the
scalar φ chain; collision is pointwise and streaming needs depth 1 only.
`multigpu.staged_halo_step` therefore exchanges ρ_N at depth `required_phi_halo` (4)
before collision and the post-collision populations at depth 1 before streaming, using the
stage split of `color3d.step` (`surface_force` / `collide_recolor` / `wall_restore` /
`stream_bounce` / `wall_seal`). MRT collision and recoloring never run on ghost planes, and
per-side traffic drops from 5 × 38 plane-fields to 4 × 1 + 1 × 38 (about 4.5× less). The φ
depth of 4 is pinned as minimal by `test_staged_phi_halo_depth_is_minimal` (depth 3 breaks
parity).

**Parity.** Halo-step drainage equals single-device drainage to float64 round-off
(≤ 5 × 10⁻¹⁶ over 30 steps; piston inlet bit-exact) on 2- and 4-device CPU emulation —
`test/test_multigpu.py` cases `halo_*` and `staged_*`. On GPU, `--report` cross-checks the
end-state saturation across device counts and across modes (tolerance 10⁻⁹ in fp64, 10⁻⁴
in fp32, where round-off-level differences between distinct XLA programs are expected at
the contact line).

## How to run

```bash
sbatch demo/multi_gpu_halo/run_halo.slurm      # from the repository root: 8-GPU node, gspmd + halo, fp32 sweep + fp64 256^3
sbatch demo/multi_gpu_halo/run_staged.slurm    # staged mode, appends to the same CSV
sbatch demo/multi_gpu_halo/run_halo_4g.slurm   # reduced sweep for a 4-GPU allocation
# or manually, one row per invocation:
uv run --extra cuda python demo/multi_gpu_halo/multi_gpu_halo_demo.py --gpus 8 --size 256 --mode staged
uv run --extra cuda python demo/multi_gpu_halo/multi_gpu_halo_demo.py --gpus 8 --size 256 --mode halo
uv run --extra cuda python demo/multi_gpu_halo/multi_gpu_halo_demo.py --gpus 8 --size 256 --mode gspmd
uv run python demo/multi_gpu_halo/multi_gpu_halo_demo.py --report
# CPU smoke test (no GPU needed):
XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
  uv run python demo/multi_gpu_halo/multi_gpu_halo_demo.py --gpus 4 --size 64 --steps 20 --block 10 --mode halo
```

`--halo N` overrides the ghost depth for message-size experiments (parity requires the
full depth).

## Results

256³ drainage on one 8 × H100 node, 200 timed steps, throughput in MLUPS (million lattice
updates per second). Efficiency is relative to the same mode on 1 GPU.

| precision | mode | 1 GPU | 2 GPU | 4 GPU | 8 GPU | efficiency (2/4/8) | vs gspmd (2/4/8) |
|:----:|:-----:|------:|------:|------:|------:|:------------------:|:----------------:|
| fp32 | gspmd | 316.5 | 430.1 | 725.7 | 1232.1 | 68% / 57% / 49% | — |
| fp32 | halo | 306.2 | 547.6 | 922.9 | 1535.8 | 89% / 75% / 63% | 1.27x / 1.27x / 1.25x |
| fp32 | staged | 300.2 | 691.9 | 1323.4 | **2168.9** | **115% / 110% / 90%** | 1.61x / 1.82x / 1.76x |
| fp64 | gspmd | 161.0 | 240.6 | 425.7 | 718.5 | 75% / 66% / 56% | — |
| fp64 | halo | 159.1 | 296.1 | 541.4 | 884.8 | 93% / 85% / 70% | 1.23x / 1.27x / 1.23x |
| fp64 | staged | 158.0 | 303.4 | 584.2 | **1003.3** | **96% / 92% / 79%** | 1.26x / 1.37x / 1.40x |

- The fp64 GSPMD rows reproduce the `demo/multi_gpu_scaling` table (same protocol), so the
  backends are compared on equal terms.
- The packed halo exchange gains 14–25 efficiency points over GSPMD at every device count
  (1.23–1.27× throughput at equal count) despite the ghost-plane recompute; its single-GPU
  overhead is 1–4 %.
- The staged backend is the fastest at every count ≥ 2: 2169 MLUPS at 90 % efficiency on
  8 GPUs in fp32, 1.41× the packed halo at equal count. Its super-linear 2- and 4-GPU fp32
  efficiencies are relative to its own 1-GPU baseline, which is slightly slower than the
  other modes (extra concatenations with the whole domain resident); the equal-count ratio
  against GSPMD is the mode-independent measure.
- Physics parity: the end-state saturation agrees across device counts and across modes in
  both precisions — the difference is exactly 0 between `gspmd` and `halo`, and at most
  round-off level with `staged`.
- fp32 delivers about 2× the fp64 throughput; its efficiency is a few points lower at equal
  count because the halved compute makes communication relatively larger.
- Overlapping the depth-1 population exchange with the interior streaming (splitting
  streaming into interior + edge strips) does not pay off: it is 15 % slower on 8 GPUs in
  fp32 (1851 vs 2169 MLUPS), because the extra full-array concatenation and edge kernels
  cost more than the 5–8 % of communication they can hide.

## Outputs

`output/` (gitignored): `halo_scaling.csv` (one row per invocation: size, mode, precision,
halo depth, GPU count, steps, seconds, MLUPS, saturation), `report.md` and the SLURM log.
The `speedup.svg` curve is written only when `--report` is passed `--plots`.
