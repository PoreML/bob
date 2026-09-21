# Capillary underfill of the MT6225A flip-chip package

Reproduces case 3 of Wang, Hao, Zhou, Zhang & Li (Microelectron. Eng. 2016,
PII S0167931715300575) with bob's color-gradient solver: capillary underfill of
a 12 x 12 mm TFBGA package. The geometry is reconstructed from the MediaTek
datasheet ball map in `mt6225a_package_geometry.json`: 264 balls on a 17x17 /
0.65 mm-pitch grid (25 depopulated sites form the bump-free center channels),
ball diameter 0.30 mm, standoff (= underfill gap) 0.21 mm. The observable
compared with the paper's Fig. 8 is the melt-front shape at 42 % fill.

## Setup

- **Geometry** (`geometry.py`): substrate and die planes at z = 0 / nz−1, balls
  as spheres truncated by both plates (`--bump-shape cylinder` gives full-gap
  cylinders instead), sealed chip side edges at both y ends, x as the flow axis
  with open inlet/outlet buffers outside the chip footprint.
- **Dispensing and drive**: I-type instantaneous dispensation (paper Fig. 5) —
  the encapsulant (red, wetting, the more viscous phase) is seeded along the
  low-x edge and then imbibes under capillary action alone: Zou-He pressure
  inlet and outlet with ΔP = 0 (`porous3d.drain(inlet="zouhe")`, the boundary
  pair validated against the Washburn law in `demo/washburn`), default MRT +
  akai + CSF stack.
- **Parameters**: θ = 30° on all walls, σ = 0.05, β = 0.95, ν_red = 0.05,
  ν_blue = 0.033. The viscosity ratio M = ν_red/ν_blue = 1.5 is a
  lattice-stability compromise (real epoxy/air is ~1e4). Fill fraction and
  front profiles are always measured over the chip footprint only.

### Correspondence to the paper

- Case 3 names the MT6225A MCU (encapsulant TC-2002W), so the datasheet
  reconstruction is the paper's chip. The paper publishes no lattice
  dimensions, gap height or margin width; the datasheet values are used.
- Case 3 assigns the substrate, die and bump walls the same wall-interaction
  strength (g^w = −1.8 / −1.8 / −1.8, paper Table 3), so a single global
  contact angle is faithful. Cases 1 (−1.8/−1.8/−1.6) and 2 (−1.5/−1.5/−2.0)
  differ per wall and are not reproduced here. The paper's Shan-Chen wall
  strength has no closed-form mapping to a contact angle; θ = 30° stands in for
  a strongly wetting encapsulant.
- The paper's flow region is the main filling part plus a flow margin (paper
  Fig. 4). The bump-free edge channel inside the chip footprint follows from
  the datasheet (the 17x17 grid spans 10.4 mm inside the 12 mm body, a ~0.8 mm
  rim). The x margins are the inlet/outlet buffers. The side (y) flow margin,
  which carries the detour flow around the chip side edges, is the optional
  `--margin <mm>`: bump-free strips at gap height along both y edges (the die
  side edge as a vertical wall is not modelled). The default is 0 (sealed
  walls at the chip edge); the paper gives no width, so the margin is a free
  parameter (0.5–1.0 mm is a reasonable range).
- bob's color-gradient MRT + akai + CSF stack replaces the paper's Shan-Chen
  LBGK model.

### Resolution presets

| | default | `--hires` |
|---|---|---|
| dx | 0.03 mm | 0.015 mm |
| domain | 9x402x420 (1.5 M cells) | 16x802x840 (10.8 M cells) |
| gap fluid layers | 7 | 14 |
| ball diameter | 10 cells (~2 interface widths) | 20 cells (~4-5 interface widths) |
| cost to 45 % fill (H200, M = 1.5) | 98k steps, ~9 min (184 steps/s) | 177k steps, ~2.5 h (20 steps/s) |
| output | `output/run` | `output/hires` |

`--hires` only fills the flags left at their defaults; explicit
`--dx/--buffer/--steps/--out` take precedence. The diffuse interface is about
4-5 lattice cells wide regardless of dx, so refining the lattice shrinks it
relative to the geometry. This matters for trapped air: at dx = 0.03 mm the air
pockets left in the wake of the balls dissolve through the interface before
they can disconnect, and no pockets remain in the final state; at
dx = 0.015 mm a population of pockets persists, located mid-gap on the
downstream faces of the balls (wall-attached ones pinned, free ones advected
with the melt). Statements about voids therefore require the `--hires` preset,
with the default run as the resolution control.

## How to run

```bash
uv run python demo/underfill_mt/plot_package.py                 # geometry views (ball map, voxel slices, 3D)
uv run python demo/underfill_mt/underfill_demo.py --theta 30 --omega 1.538 --fill-target 0.45 --steps 250000
uv run python demo/underfill_mt/underfill_demo.py --hires --fill-target 0.45   # ~2.5 h on an H200
uv run python demo/underfill_mt/underfill_demo.py --steps 2000 --block 200     # smoke test
uv run python demo/underfill_mt/compare_viscosity_ratio.py      # viscosity-ratio figure from finished runs
uv run python demo/underfill_mt/publication_plot.py             # paper figures from output/hires
```

`finish_run.py <run dir>` regenerates the end-of-run artifacts (curves, GIF,
`report.md`) of a run from its checkpoint, `state42.npz`, `metrics.csv` and
`run_meta.json` without re-simulating.

## Results (H200, float32)

**Reference run** (`output/run`, M = 1.5): the fill rises from 0.6 % to 45 % in
98k steps (532 s, 184 steps/s) with an inlet-connected red fraction of 0.9999.
At the paper's Fig. 8 comparison point (42 % fill, step 91k) the melt front
shows the expected morphology: the ball-free edge channels lead, the
depopulated center region lags slightly, and the ball rows locally pull the
meniscus forward through their extra wetted surface — the paper's observation
that the flow in the bump region is a little faster.

**Viscosity-ratio ladder** (dx = 0.03 mm, sphere bumps; for M ≥ 10, ν_blue is
held at its stability floor of 0.025 and M is raised through ν_red only):

| M    | ν_red / ν_blue | omega / omega2  | steps to 42 % / 45 % |
|------|----------------|-----------------|----------------------|
| 1.5  | 0.05 / 0.033   | 1.538 / 1.667   | 91k / 98k            |
| 10   | 0.25 / 0.025   | 0.8 / 1.739     | 204k / 227k          |
| 20   | 0.5  / 0.025   | 0.5 / 1.739     | 350k / 395k          |
| 30   | 0.75 / 0.025   | 0.3636 / 1.739  | 495k / 561k          |

All rungs are stable end-to-end, and the fill time scales approximately
linearly with M (the drag of the red phase dominates; the early air-limited
stage barely changes). `compare_viscosity_ratio.py` plots the ladder from the
run directories `output/run`, `output/m10_sphere`, `output/m20_sphere` and
`output/m30_sphere` (set with `--omega/--omega2/--out`). The
bump shape has no significant effect on the fill time (cylinder bumps at
M = 10: 231k steps to 45 %, against 227k for spheres).

## Numerical limits

- **Stability floor on ν_blue.** At σ = 0.05 the run diverges within 1000 steps
  for ν_blue ≤ 0.02 (omega2 ≥ 1.786), independent of ν_red (tested 0.05–0.33)
  and of the bump shape; ν_blue = 0.025 is stable. The limit is set by the thin
  layer of defending air in the 7-cell gap, not by the ball geometry.
- **High M.** For M ≥ 20, tau_red reaches 2.0–2.75, where the BGK/MRT
  truncation error grows. This is acceptable for these viscous capillary
  fronts, but absolute permeabilities should not be derived from those runs.
- **Divergence detection.** A NaN state reads as fill = 0.0 through
  red-majority counts (NaN compares false), so the demo monitors
  `porous3d.saturation`, which propagates NaN, and stops with a DIVERGED status.

## Outputs

In the run directory (`output/run` by default, gitignored): `frames/` +
`underfill.gif` (top-view melt-front maps), `state42.npz` (the frozen 42 %
state), `metrics.csv`, `run_meta.json`, `report.md`, checkpoints. The analysis
curves `fill_curve.png`, `front_profile.png` (x_front(y) at 42 % and at the
end) and `fill.svg` are written only with `--plots`.
`plot_package.py` writes `output/package_*.png`; `compare_viscosity_ratio.py`
writes `output/viscosity_ratio_ladder.(png|pdf)`; `publication_plot.py` writes
`publication_melt_front` and `publication_fill_front` (PNG + PDF) into the run
directory.

## References

- Wang, Hao, Zhou, Zhang & Li, *Microelectronic Engineering* (2016),
  PII S0167931715300575 — case 3 (MT6225A, I-type dispensing).
- MediaTek MT6225 datasheet Rev 1.02, Table 1 — package dimensions and ball map
  (`mt6225a_package_geometry.json`).
