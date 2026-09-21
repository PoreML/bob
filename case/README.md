# case/ — dataset campaigns

Production campaigns that generate the published pore-scale two-phase flow
datasets with the `bob` color-gradient lattice-Boltzmann solver (D3Q19, JAX).
Every run is a time-resolved 3D simulation of one displacement process in one
porous geometry, logged at a fixed cadence (fields, metrics, renders, metadata),
for training and evaluating data-driven models of multiphase flow in porous
media.

| Case | Process | Geometries | Driver |
|---|---|---|---|
| `drainage/` | constant-rate primary drainage of a rock against a porous plate | synthetic and micro-CT sandstone cubes | `drainage/drainage_demo.py` |
| `GDL/` | constant-rate through-plane drainage (non-wetting liquid intrusion) of a fuel-cell gas-diffusion layer | synthetic fiber mats and micro-CT carbon paper | `GDL/drainage_gdl_demo.py` |
| `trapping/` | waterflood to residual (trapped) oil, two-stage piston protocol | the drainage rock families | `trapping/trapping_demo.py` |
| `underfill/` | capillary underfill of flip-chip solder-ball gaps | designed solder-ball arrays | `underfill/underfill_demo.py` |

`drainage`, `GDL` and `trapping` share one frozen, seeded schedule (this file);
`underfill` is documented in `underfill/README.md`.

Shared machinery:

| File | Role |
|---|---|
| `campaign_draw.py` | the frozen, seeded draw of every run (geometry, M, theta, name); generates the manifest below the marker line of this file |
| `submit_phase.py` | submits unfinished runs as throttled SLURM job arrays from write-once manifests |
| `run_array.slurm` | array task: reads one manifest line and dispatches it to the case driver |
| `geometry_qc.py`, `geometry_qc.json` | inlet-to-outlet percolation check and its cached verdicts |
| `geometry_throats.csv` | critical throat radius `r_c` of every rock cube (frozen table used by the draw) |
| `trapping/PMI.py`, `trapping/pmi_search.py` | pore-morphology initialiser (vendored; Tavakkoli et al., Phys. Fluids 37(9), 2025, doi:10.1063/5.0285656) and its full-domain kernel search |

## Physics

All cases use the solver's default stack: MRT collision, the Akai geometric
wetting boundary condition (closed-form rotation of the interface normal at the
wall; `Params.theta` is the contact angle of the red phase, in degrees), CSF
surface tension, `recolor_emag` recoloring and `wall_grad="fluid"`, in single
precision (fp32; `BOB_FP64=1` switches a driver to fp64). Both fluids have equal
density. Red is the non-wetting phase and blue the wetting phase in drainage,
GDL and trapping; theta is always the contact angle of the red phase.

| Parameter | Value |
|---|---|
| surface tension sigma | 0.05 |
| recoloring sharpness beta | 0.99 (passed by `run_array.slurm`; the drivers' own default is 0.7) |
| MRT spectrum factor chi (`mrt_chi`, Leclaire) | 0.8: non-shear relaxation rates = chi x omega (drainage, GDL, trapping) |
| baseline kinematic viscosity nu0 | 0.04, split geometrically: nu_red = nu0 sqrt(M), nu_blue = nu0 / sqrt(M) |
| viscosity ratio M | nu_red / nu_blue = mu_non-wetting / mu_wetting |
| capillary number Ca | 1e-5, referenced to nu0: u_in = Ca sigma / nu0 = 1.25e-5, independent of M |
| inlet | piston: exactly u_in A of the injected phase per step, inlet pressure floats |
| precision | fp32 |

At the extreme ratios the thin phase reaches nu = 0.0089 (M = 0.05: red;
M = 20: blue). chi = 0.8 ties the non-shear MRT rates to the local viscous
rate omega, which keeps these dual-viscosity runs stable.

## Cases and protocols

All porous domains are assembled along the flow axis x (arrays are `(nz, ny, nx)`,
`True` = solid) with solid no-slip walls on the four z/y faces (core-holder
convention). The slab next to the piston face carries its own contact angle so
the piston face is wetted by the injected phase only.

**drainage** — `[piston face + inlet reservoir (7) | rock | porous plate (20) | outlet buffer (6)]` (slabs along x).
Red is injected at constant rate into a blue-saturated rock. The water-wet
porous plate (5 x 5-voxel pores on a pitch-8 lattice, theta 150) is in direct
contact with the rock outlet face and retains the red phase like a laboratory
semi-permeable membrane; a per-block scrubber converts red that passes the
plate to blue (mass- and momentum-conserving, logged as `scrubbed`). The run
ends at 0.75 rock pore volumes injected (`pv_cap`) or when the pressure drop
reaches half the plate entry pressure, dp >= 0.5 x 2 sigma |cos 150°| / 2.5 =
0.0173 (`dp_cap`), beyond which the plate itself could drain. Breakthrough at
the plate face is logged and checkpointed but does not end the run.

**GDL** — `[piston face + inlet reservoir (7) | slab | open outlet buffer (6)]`,
no plate. Flow is through-plane. Breakthrough (red reaching the last slab
layer) is the timing anchor, recorded as `breakthrough_step` /
`breakthrough_sat`; the run ends at 1.2x the breakthrough step (`bt_cap`) or at
0.5 pore volumes injected (`pv_cap`), whichever comes first. Red leaving the
slab is scrubbed to blue in the outlet buffer, so it is produced and never
returns.

**trapping** — `[piston face + water buffer (10) | rock | water buffer (6)]`,
open zero-gradient outlet, two stages in one run folder:

1. *Stage 1, `stab`.* Oil (red) is placed by the pore-morphology initialiser at
   a target oil saturation of 0.6 (morphology contact angle 35°; integer kernels
   quantize the reachable saturation, the achieved value is recorded as
   `so_init`). The piston is off and the system relaxes until |dS_w| per
   1000-step block stays below 5e-6 for 3 blocks (`equilibrated`). The state is
   saved as `equilibrated_state.npz`. A geometry in which the initialiser places
   no oil is refused.
2. *Stage 2, `flood`.* The piston injects blue at Ca = 1e-5 until the oil body
   is immobile or 1.2 pore volumes are injected (`pv_cap`). Immobility
   (`red_immobile`) is a set comparison on the oil body itself: the red-majority
   voxel mask in the rock is compared with the mask 20,000 steps earlier, and
   the run ends when fewer than 0.3% of its voxels changed in each of 3
   consecutive, non-overlapping windows (`--mob-tol 0.003`, passed by
   `run_array.slurm`; the driver default is 0.01). Saturation and mean |u| are
   not used: oil can be produced at constant saturation, and spurious interface
   currents dominate mean |u|.

**underfill** — see `underfill/README.md`.

## Schedule

The base phase holds 432 runs; each (case, family, domain) set draws evenly
across its sources:

| Case | Family | Sources (even split) | Domain | GPUs | Runs | M | theta |
|---|---|---|---|---:|---:|---|---|
| drainage | generated | blob, poly, sphere | 128 | 1 | 64 | 0.05, 0.1, 0.2, 1 | 120, 130, 140, 150 |
| drainage | generated | blob, poly, sphere | 256 | 4 | 8 | 0.05, 0.1, 0.2, 1 | 120, 150 |
| drainage | ct | bentheimer, buffberea, castlegate | 128 | 1 | 64 | 0.05, 0.1, 0.2, 1 | 120, 130, 140, 150 |
| drainage | ct | bentheimer, buffberea, castlegate | 256 | 4 | 8 | 0.05, 0.1, 0.2, 1 | 120, 150 |
| trapping | generated | blob, poly, sphere | 128 | 1 | 64 | 0.05, 0.1, 0.2, 1 | 120, 130, 140, 150 |
| trapping | generated | blob, poly, sphere | 256 | 4 | 8 | 0.05, 0.1, 0.2, 1 | 120, 150 |
| trapping | ct | bentheimer, buffberea, castlegate | 128 | 1 | 64 | 0.05, 0.1, 0.2, 1 | 120, 130, 140, 150 |
| trapping | ct | bentheimer, buffberea, castlegate | 256 | 4 | 8 | 0.05, 0.1, 0.2, 1 | 120, 150 |
| GDL | generated | fiber | 128x128x64 | 1 | 64 | 1, 5, 10, 20 | 120, 130, 140, 150 |
| GDL | generated | fiber | 256x256x128 | 4 | 8 | 1, 5, 10, 20 | 120, 150 |
| GDL | ct | gdl_ct, gdl_ct_20, gdl_ct_40 | 128x128x64 | 1 | 64 | 1, 5, 10, 20 | 120, 130, 140, 150 |
| GDL | ct | gdl_ct, gdl_ct_20, gdl_ct_40 | 256x256x128 | 4 | 8 | 1, 5, 10, 20 | 120, 150 |
| **total** | | | | | **432** | | |

- **Even source split.** Three sources into 64 runs give 22/21/21, into 8 runs
  3/3/2 (sorted source order takes the remainder).
- **Balanced parameters.** Each 64-run set covers all 16 (M, theta) cells
  exactly 4 times; each 8-run large-domain set takes the ends of the
  contact-angle range and covers the 4 M x 2 theta grid exactly once.
- **Domains.** `128` / `256` are cubic rocks of that edge length; GDL slabs are
  128 x 128 x 64 and 256 x 256 x 128 with the short axis through-plane (the flow
  axis). Large domains run on 4 GPUs, sharded along z with
  `multigpu.staged_halo_step`; small domains run on 1 GPU.

The `extra_beta_1g` expansion phase adds 96 runs under the identical protocol —
16 more 1-GPU runs for each of the six small-domain (case, family) sets:

| Case | Family | Sources (even split) | Domain | GPUs | Runs | M | theta |
|---|---|---|---|---:|---:|---|---|
| drainage | generated | blob, poly, sphere | 128 | 1 | 16 | 0.05, 0.1, 0.2, 1 | 120, 130, 140, 150 |
| drainage | ct | bentheimer, buffberea, castlegate | 128 | 1 | 16 | 0.05, 0.1, 0.2, 1 | 120, 130, 140, 150 |
| trapping | generated | blob, poly, sphere | 128 | 1 | 16 | 0.05, 0.1, 0.2, 1 | 120, 130, 140, 150 |
| trapping | ct | bentheimer, buffberea, castlegate | 128 | 1 | 16 | 0.05, 0.1, 0.2, 1 | 120, 130, 140, 150 |
| GDL | generated | fiber | 128x128x64 | 1 | 16 | 1, 5, 10, 20 | 120, 130, 140, 150 |
| GDL | ct | gdl_ct, gdl_ct_20, gdl_ct_40 | 128x128x64 | 1 | 16 | 1, 5, 10, 20 | 120, 130, 140, 150 |
| **total** | | | | | **96** | | |

- It is its own frozen draw (seed 20260830), taken strictly after replaying the
  base draw, so the base schedule cannot move; a sha256 pin over the base rows
  in `test/test_campaign.py` enforces this.
- Every (M, theta) cell appears exactly once per set, so joint base + expansion
  coverage is 5x per cell, per set.
- Each set inherits the base draw's used/unused ledger: an unused screened
  geometry is always preferred, and where a pool is exhausted a geometry is
  repeated at a different (M, theta) — never at a triple already drawn — and
  flagged `reused`. The GDL generated set is all reuse, because the base draw
  uses all 64 fiber samples.

## Geometry sources and screening

Geometries come from the geometry dataset tree, laid out as
`<dataset>/<domain>/<stem>.npy` boolean volumes (`True` = solid), 64 samples per
dataset and domain. `campaign_draw.py` reads it from `$GEOMETRY_DATA`, and by
default from the sibling checkout `<repo>/../geometry/data`.

| Family | Sources |
|---|---|
| generated rock | `blob` (thresholded Gaussian random fields), `poly` (polydisperse overlapping spheres), `sphere` (monodisperse overlapping spheres) |
| CT rock | `bentheimer`, `buffberea`, `castlegate` — crops of sandstone micro-CT volumes |
| generated GDL | `fiber` — synthetic in-plane fiber mats |
| CT GDL | `gdl_ct`, `gdl_ct_20`, `gdl_ct_40` — crops of carbon-paper GDL micro-CT scans at 5 %, 20 % and 40 % PTFE loading |

Two screens are applied to the candidate pools before anything is drawn:

1. **Percolation QC** (`geometry_qc.py`, all cases). The four z/y faces are
   sealed as the drivers seal them, the pore space is labelled with
   18-connectivity (D3Q19 streams along faces and edges), and a geometry is kept
   only if one pore component touches both the inlet and the outlet face.
   Verdicts are cached in `geometry_qc.json`; a missing entry is computed, never
   skipped. `run_array.slurm` repeats the check when a task starts.
2. **Throat screen** (drainage and trapping). `geometry_throats.csv` holds the
   critical throat radius `r_c` of every rock cube, in lattice units. The pool
   is cut to `r_c > 5`, so the narrowest controlling throat is resolved by the
   diffuse interface; this floor is never waived. A rock is then eligible at
   contact angle theta only if it clears the drainage pressure cap with margin
   `r_c / r_cap(theta) >= 1.3`, where `r_cap(theta) = 2 sigma |cos theta| / dp_stop`
   is the widest throat the cap forbids:

   | theta | 120 | 130 | 140 | 150 |
   |---|---:|---:|---:|---:|
   | `r_cap` (lu) | 2.89 | 3.71 | 4.42 | 5.00 |
   | eligible `r_c` (lu): above the floor and >= 1.3 `r_cap` | > 5 | > 5 | >= 5.75 | >= 6.50 |

   Geometries are assigned hardest-theta-first, so the theta-150 runs pick
   before the others. Two fallbacks exist, both flagged in the manifest: where a
   source has no unused eligible rock left, an already-used one is reused at a
   different (M, theta) (`reused`; castlegate/128 keeps 17 rocks above the floor
   for a 21-run quota); where not even a reuse clears the margin, the largest
   remaining margin is taken (`margin x.xx`). GDL slabs are not in the throat
   table and pass percolation QC only.

`geometry_qc.json` is keyed by the geometry path below the dataset root
(`<dataset>/<domain>/<stem>.npy`), so the cached verdicts hold wherever the dataset
lives; a geometry missing from the cache is computed on first use and appended.

## Layout and run naming

```
case/<case>/runs/<dataset>/<domain>/<run name>/
```

Run names embed the full assignment, `<prefix>_<geometry>_<domain>_M<M>_th<theta>_b99`,
for example `drain_blob128_0007_128_M0.05_th140_b99`. Prefixes are `drain`,
`trap` and `gdl`; `_b99` records the recoloring sharpness beta = 0.99. The three
GDL CT datasets share `gdl_XXXX` stems, so their run names embed the dataset
(`gdl_gdl_ct_20_0027_128x128x64_M20_th120_b99`). `runs/`, fields and frames are not tracked in git.

## Per-run outputs

Drainage and GDL runs log every 5000 steps; trapping logs every 1000 steps in
stage 1 and every 5000 in stage 2. A campaign run writes `metrics.csv`,
`run_meta.json`, the fields file, checkpoints, `report.md`, frames and the GIF;
the analysis curves are opt-in and stay unwritten unless the driver is given
`--plots`.

| Artifact | Content |
|---|---|
| `run_meta.json` | reproducibility sidecar, rewritten atomically during the run: command line, all solver `Params`, precision, code commit, geometry shape / porosity / sha256 / source, devices, steps/s, elapsed and device hours, the case protocol constants, and on completion `finish_type` + `finish_detail` |
| `metrics.csv` | one row per block: saturation, front position, red mass vs the ideal injected volume (colour ledger), pressures at inlet / mid-rock / outlet and their difference, spurious-velocity diagnostics, and the case-specific columns (`scrubbed`, `plate_leak`, `outbuf_red`; trapping: trapped-blob inventory, produced water / oil, `red_moved`) |
| `<run name>.h5` + `.xdmf` | fields appended every block, solid cells masked: `phi` (colour field), `rho`, `p`, `umag`, `u`; the run parameters and the static `run_meta` sections are mirrored in the file attributes |
| `frames/`, `<run name>.gif` | PyVista 3D phase-field renders (one per block) and their animation |
| `saturation.svg`, `pressure_curve.svg`, `ledger_curve.svg`, `spurious_curve.svg` | monitoring curves, refreshed every block — opt-in, written only when the run is given `--plots` (trapping: `<stage>_curves.svg`, `sw_<stage>.svg`; underfill: `fill.svg`) |
| `report.md` | human-readable summary of setup, termination and result |
| `wall_normals.npz` | the wall-normal field of the wetting boundary condition (preprocessing output) |
| `ckpt_step<step>_sat<S>.npz` | populations `fR`, `fB` + step, every 10 blocks; the 2 newest are kept |

Trapping writes one set per stage with `stab` / `flood` prefixes or suffixes
(`run_meta_stab.json`, `run_meta_flood.json`, `metrics_flood.csv`,
`flood_<run name>.h5`, `frames_flood/`, `ckpt_flood_*.npz`, ...) plus
`equilibrated_state.npz`. Stage 1 writes no fields file by default (`--stab-h5`
enables it); the flood is the deliverable.

## Running

All commands run from the repository root with the project environment
(`uv sync --extra cuda`). `$GEOMETRY_DATA` selects the geometry tree.

```bash
.venv/bin/python case/campaign_draw.py --check    # verify this README matches the frozen draw
.venv/bin/python case/campaign_draw.py            # regenerate the manifest below + refresh the Status column
.venv/bin/python case/geometry_qc.py --report     # list geometries rejected by the percolation QC

.venv/bin/python case/submit_phase.py --dry-run   # show the job arrays; writes nothing, submits nothing
.venv/bin/python case/submit_phase.py             # submit every unfinished run of both phases
.venv/bin/python case/submit_phase.py --phase extra_beta_1g --tag _x1 --small-slots 8
.venv/bin/python case/submit_phase.py --case drainage --only drain_blob128_0007_128_M0.05_th140_b99
```

The draw is deterministic (`random.Random(seed)` over sorted pools), so the
schedule is fully specified before anything is submitted. The Status column of
the manifest is read from each run's `run_meta.json` on disk and is `—` where
no run data is present; `--check` ignores it.

`submit_phase.py` submits one job array per GPU class with a concurrency
throttle (`--array=0-N%slots`); an array task starts the moment a slot frees, so
the campaign holds a fixed GPU footprint without job dependencies:

| Array | Tasks (base phase) | `--gres` | Throttle | Walltime | GPUs held |
|---|---:|---|---|---|---:|
| small (128-class) | 384 | `gpu:1` | `%20` | 12 h | 20 |
| large (256-class) | 48 | `gpu:4` | `%3` | 48 h | 12 |

Each submission writes its own manifest, `_manifests/beta<tag>_<stamp>_<n>gpu.tsv`.
A queued job array reads its manifest when each task starts, so manifests are
immutable once written: a dry run writes nothing and no submission reuses a
file name. `--phase` restricts a submission to one phase; use it while another
phase's arrays are still queued, otherwise its unfinished runs are submitted
twice. The QOS in `run_array.slurm` is site-specific, and the SLURM scripts in this
repository name no partition: set `SBATCH_PARTITION` (and `SBATCH_ACCOUNT` where
required) in the environment. `PY` overrides the interpreter (default
`.venv/bin/python`).

Re-running the submit is the resume path. Runs with a recorded `finish_type`
are skipped. Drainage and GDL tasks continue from their newest checkpoint.
A trapping task skips stage 1 once `equilibrated_state.npz` exists; an
interrupted flood starts again from that state.

A single run outside SLURM:

```bash
.venv/bin/python case/drainage/drainage_demo.py --structure $GEOMETRY_DATA/blob/128/blob128_0007.npy \
    --out case/drainage/runs/blob/128/drain_blob128_0007_128_M0.05_th140_b99 \
    --name drain_blob128_0007_128_M0.05_th140_b99 --m 0.05 --theta 140 --gpus 1 --beta 0.99
```

Trapping additionally takes `--mob-tol 0.003` to match the campaign
(`run_array.slurm` shows the exact command line of every case).

## Hardware footprint

Each task requests 4 CPU cores and 48 GB of host memory; the solver runs on the
GPU and the host side (metrics, one render and one compressed h5 append per
block) is serial. The published runs were produced on NVIDIA H200 GPUs with
JAX 0.6.2. Wall time per run, from the `run_meta.json` files of the scheduled
runs (median / maximum, hours):

| Case | 1 GPU (128-class) | 4 GPUs (256-class) |
|---|---|---|
| drainage | 3.9 / 7.1 | 19.3 / 25.9 |
| GDL | 1.5 / 2.1 | 6.7 / 10.8 |
| trapping (both stages) | 1.8 / 10.1 | 6.4 / 9.0 |

In total about 3,200 GPU-hours. Run data on disk: drainage 1.6 TB, GDL 1.5 TB,
trapping 1.0 TB; a 128-class drainage run is about 5 GB and a 256-class run
about 55 GB, dominated by the fields file.

## Deviation of the published data from the manifest

The manifest entry `gdl_fiber_0019_128x128x64_M10_th140_b99` (GDL, generated,
128x128x64) diverged and has no published data. The dataset contains
`gdl_fiber_0018_128x128x64_M10_th140_b99` in its place — the same M and theta on
the neighbouring fiber sample. This run is not part of the seeded draw.

<!-- campaign_draw.py manifest — everything below is generated; do not edit by hand -->

## Manifest — the frozen 432-run draw

Seed 20260825. Regenerate/refresh with `.venv/bin/python case/campaign_draw.py`; `--check` verifies the draw is unchanged. `r_c` is the critical throat radius from `geometry_throats.csv` (lattice units, pool floor r_c > 5), `margin` is r_c / r_cap(theta) against the drainage dp cap (eligibility >= 1.3). Flags: `reused` = geometry repeated at a different (M, theta) because its source ran out of unused wide rocks; `margin x.xx` = best remaining rock fell short of 1.3 (the floor is never waived).

| Case | Family | Sources (even split) | Domain | GPUs | Runs | flagged |
|---|---|---|---|---:|---:|---:|
| drainage | generated | blob, poly, sphere | 128 | 1 | 64 | — |
| drainage | generated | blob, poly, sphere | 256 | 4 | 8 | — |
| drainage | ct | bentheimer, buffberea, castlegate | 128 | 1 | 64 | 4 |
| drainage | ct | bentheimer, buffberea, castlegate | 256 | 4 | 8 | — |
| trapping | generated | blob, poly, sphere | 128 | 1 | 64 | — |
| trapping | generated | blob, poly, sphere | 256 | 4 | 8 | — |
| trapping | ct | bentheimer, buffberea, castlegate | 128 | 1 | 64 | 4 |
| trapping | ct | bentheimer, buffberea, castlegate | 256 | 4 | 8 | 1 |
| GDL | generated | fiber | 128x128x64 | 1 | 64 | — |
| GDL | generated | fiber | 256x256x128 | 4 | 8 | — |
| GDL | ct | gdl_ct, gdl_ct_20, gdl_ct_40 | 128x128x64 | 1 | 64 | — |
| GDL | ct | gdl_ct, gdl_ct_20, gdl_ct_40 | 256x256x128 | 4 | 8 | — |
| **total** | | | | | **432** | **9** |


## Termination

What ends a run, per case — the `finish_type` recorded in its `run_meta.json`:

| Case | Stops on |
|---|---|
| `drainage` | `pv_cap` 0.75 PV, or `dp_cap` when dp reaches 0.5x the porous-plate entry pressure — beyond that the plate itself risks draining and it stops being a valid porous-plate experiment |
| `GDL` | `bt_cap` 1.2x the breakthrough step, or `pv_cap` 0.5 PV, whichever comes first; breakthrough is recorded as `breakthrough_step` / `breakthrough_sat` |
| `trapping` | two stages. Stage 1 `equilibrated` when |dS_w| per 1000-step block < 5e-6 held 3 blocks. Stage 2 `red_immobile` at `--mob-tol 0.003` — the oil body moved less than 0.3% of its voxels in each of 3 consecutive, non-overlapping 20,000-step windows — or `pv_cap` at 1.2 PV |

Every case also records `diverged` on a non-finite saturation, and `step_cap` (trapping: `cap`) if it reaches its safety step cap first.


## drainage — 144 runs

**144 runs** — 144 finished, 0 running.

| # | Family | Domain | GPUs | Run | Geometry | r_c | margin | M | theta | Flags | Status |
|---:|---|---|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | generated | 128 | 1 | `drain_blob128_0007_128_M0.05_th140_b99` | `blob/128/blob128_0007.npy` | 6.16 | 1.39 | 0.05 | 140 |  | dp_cap |
| 2 | generated | 128 | 1 | `drain_poly128_0061_128_M0.2_th130_b99` | `poly/128/poly128_0061.npy` | 5.47 | 1.47 | 0.2 | 130 |  | dp_cap |
| 3 | generated | 128 | 1 | `drain_blob128_0006_128_M0.1_th150_b99` | `blob/128/blob128_0006.npy` | 8.05 | 1.61 | 0.1 | 150 |  | dp_cap |
| 4 | generated | 128 | 1 | `drain_poly128_0001_128_M1_th120_b99` | `poly/128/poly128_0001.npy` | 5.09 | 1.76 | 1 | 120 |  | pv_cap |
| 5 | generated | 128 | 1 | `drain_blob128_0042_128_M0.2_th120_b99` | `blob/128/blob128_0042.npy` | 9.79 | 3.39 | 0.2 | 120 |  | dp_cap |
| 6 | generated | 128 | 1 | `drain_sphere128_0054_128_M1_th150_b99` | `sphere/128/sphere128_0054.npy` | 7.00 | 1.40 | 1 | 150 |  | dp_cap |
| 7 | generated | 128 | 1 | `drain_blob128_0063_128_M0.05_th130_b99` | `blob/128/blob128_0063.npy` | 6.40 | 1.72 | 0.05 | 130 |  | dp_cap |
| 8 | generated | 128 | 1 | `drain_poly128_0005_128_M0.1_th140_b99` | `poly/128/poly128_0005.npy` | 5.99 | 1.35 | 0.1 | 140 |  | dp_cap |
| 9 | generated | 128 | 1 | `drain_sphere128_0006_128_M0.1_th130_b99` | `sphere/128/sphere128_0006.npy` | 7.99 | 2.15 | 0.1 | 130 |  | pv_cap |
| 10 | generated | 128 | 1 | `drain_poly128_0012_128_M0.2_th140_b99` | `poly/128/poly128_0012.npy` | 6.16 | 1.39 | 0.2 | 140 |  | dp_cap |
| 11 | generated | 128 | 1 | `drain_sphere128_0000_128_M0.05_th150_b99` | `sphere/128/sphere128_0000.npy` | 7.34 | 1.47 | 0.05 | 150 |  | dp_cap |
| 12 | generated | 128 | 1 | `drain_poly128_0051_128_M0.05_th120_b99` | `poly/128/poly128_0051.npy` | 6.00 | 2.08 | 0.05 | 120 |  | pv_cap |
| 13 | generated | 128 | 1 | `drain_sphere128_0056_128_M1_th130_b99` | `sphere/128/sphere128_0056.npy` | 5.66 | 1.52 | 1 | 130 |  | dp_cap |
| 14 | generated | 128 | 1 | `drain_blob128_0010_128_M0.2_th150_b99` | `blob/128/blob128_0010.npy` | 7.13 | 1.43 | 0.2 | 150 |  | dp_cap |
| 15 | generated | 128 | 1 | `drain_sphere128_0051_128_M0.1_th120_b99` | `sphere/128/sphere128_0051.npy` | 10.99 | 3.81 | 0.1 | 120 |  | pv_cap |
| 16 | generated | 128 | 1 | `drain_blob128_0049_128_M1_th140_b99` | `blob/128/blob128_0049.npy` | 9.00 | 2.03 | 1 | 140 |  | dp_cap |
| 17 | generated | 128 | 1 | `drain_blob128_0003_128_M1_th140_b99` | `blob/128/blob128_0003.npy` | 5.99 | 1.35 | 1 | 140 |  | dp_cap |
| 18 | generated | 128 | 1 | `drain_sphere128_0031_128_M0.1_th130_b99` | `sphere/128/sphere128_0031.npy` | 6.99 | 1.88 | 0.1 | 130 |  | pv_cap |
| 19 | generated | 128 | 1 | `drain_blob128_0001_128_M0.05_th150_b99` | `blob/128/blob128_0001.npy` | 8.48 | 1.70 | 0.05 | 150 |  | dp_cap |
| 20 | generated | 128 | 1 | `drain_poly128_0009_128_M0.2_th120_b99` | `poly/128/poly128_0009.npy` | 6.16 | 2.13 | 0.2 | 120 |  | pv_cap |
| 21 | generated | 128 | 1 | `drain_blob128_0060_128_M0.2_th140_b99` | `blob/128/blob128_0060.npy` | 7.98 | 1.80 | 0.2 | 140 |  | dp_cap |
| 22 | generated | 128 | 1 | `drain_poly128_0058_128_M0.05_th120_b99` | `poly/128/poly128_0058.npy` | 5.64 | 1.96 | 0.05 | 120 |  | pv_cap |
| 23 | generated | 128 | 1 | `drain_sphere128_0026_128_M1_th150_b99` | `sphere/128/sphere128_0026.npy` | 9.00 | 1.80 | 1 | 150 |  | pv_cap |
| 24 | generated | 128 | 1 | `drain_poly128_0033_128_M0.2_th130_b99` | `poly/128/poly128_0033.npy` | 5.82 | 1.57 | 0.2 | 130 |  | dp_cap |
| 25 | generated | 128 | 1 | `drain_blob128_0062_128_M0.1_th140_b99` | `blob/128/blob128_0062.npy` | 5.91 | 1.34 | 0.1 | 140 |  | dp_cap |
| 26 | generated | 128 | 1 | `drain_blob128_0023_128_M0.1_th150_b99` | `blob/128/blob128_0023.npy` | 9.89 | 1.98 | 0.1 | 150 |  | dp_cap |
| 27 | generated | 128 | 1 | `drain_sphere128_0040_128_M0.05_th130_b99` | `sphere/128/sphere128_0040.npy` | 6.32 | 1.70 | 0.05 | 130 |  | pv_cap |
| 28 | generated | 128 | 1 | `drain_sphere128_0027_128_M1_th120_b99` | `sphere/128/sphere128_0027.npy` | 6.31 | 2.19 | 1 | 120 |  | pv_cap |
| 29 | generated | 128 | 1 | `drain_blob128_0038_128_M1_th130_b99` | `blob/128/blob128_0038.npy` | 9.63 | 2.60 | 1 | 130 |  | dp_cap |
| 30 | generated | 128 | 1 | `drain_blob128_0002_128_M0.1_th120_b99` | `blob/128/blob128_0002.npy` | 7.87 | 2.73 | 0.1 | 120 |  | dp_cap |
| 31 | generated | 128 | 1 | `drain_blob128_0029_128_M0.05_th140_b99` | `blob/128/blob128_0029.npy` | 9.69 | 2.19 | 0.05 | 140 |  | pv_cap |
| 32 | generated | 128 | 1 | `drain_poly128_0007_128_M0.2_th150_b99` | `poly/128/poly128_0007.npy` | 7.27 | 1.45 | 0.2 | 150 |  | dp_cap |
| 33 | generated | 128 | 1 | `drain_blob128_0061_128_M0.1_th120_b99` | `blob/128/blob128_0061.npy` | 6.99 | 2.42 | 0.1 | 120 |  | pv_cap |
| 34 | generated | 128 | 1 | `drain_sphere128_0005_128_M1_th150_b99` | `sphere/128/sphere128_0005.npy` | 7.35 | 1.47 | 1 | 150 |  | dp_cap |
| 35 | generated | 128 | 1 | `drain_blob128_0044_128_M0.2_th130_b99` | `blob/128/blob128_0044.npy` | 9.10 | 2.45 | 0.2 | 130 |  | pv_cap |
| 36 | generated | 128 | 1 | `drain_blob128_0058_128_M0.05_th140_b99` | `blob/128/blob128_0058.npy` | 6.99 | 1.58 | 0.05 | 140 |  | dp_cap |
| 37 | generated | 128 | 1 | `drain_blob128_0016_128_M0.2_th140_b99` | `blob/128/blob128_0016.npy` | 6.40 | 1.45 | 0.2 | 140 |  | dp_cap |
| 38 | generated | 128 | 1 | `drain_sphere128_0003_128_M1_th130_b99` | `sphere/128/sphere128_0003.npy` | 5.74 | 1.55 | 1 | 130 |  | pv_cap |
| 39 | generated | 128 | 1 | `drain_poly128_0022_128_M0.1_th150_b99` | `poly/128/poly128_0022.npy` | 6.55 | 1.31 | 0.1 | 150 |  | dp_cap |
| 40 | generated | 128 | 1 | `drain_poly128_0000_128_M0.05_th120_b99` | `poly/128/poly128_0000.npy` | 5.65 | 1.96 | 0.05 | 120 |  | dp_cap |
| 41 | generated | 128 | 1 | `drain_poly128_0047_128_M0.2_th150_b99` | `poly/128/poly128_0047.npy` | 7.60 | 1.52 | 0.2 | 150 |  | dp_cap |
| 42 | generated | 128 | 1 | `drain_blob128_0019_128_M0.1_th140_b99` | `blob/128/blob128_0019.npy` | 6.70 | 1.52 | 0.1 | 140 |  | dp_cap |
| 43 | generated | 128 | 1 | `drain_poly128_0002_128_M1_th120_b99` | `poly/128/poly128_0002.npy` | 5.19 | 1.80 | 1 | 120 |  | pv_cap |
| 44 | generated | 128 | 1 | `drain_poly128_0050_128_M0.05_th130_b99` | `poly/128/poly128_0050.npy` | 5.10 | 1.37 | 0.05 | 130 |  | dp_cap |
| 45 | generated | 128 | 1 | `drain_sphere128_0021_128_M0.2_th120_b99` | `sphere/128/sphere128_0021.npy` | 7.54 | 2.61 | 0.2 | 120 |  | dp_cap |
| 46 | generated | 128 | 1 | `drain_poly128_0056_128_M0.1_th130_b99` | `poly/128/poly128_0056.npy` | 5.91 | 1.59 | 0.1 | 130 |  | dp_cap |
| 47 | generated | 128 | 1 | `drain_blob128_0000_128_M1_th140_b99` | `blob/128/blob128_0000.npy` | 6.32 | 1.43 | 1 | 140 |  | dp_cap |
| 48 | generated | 128 | 1 | `drain_sphere128_0039_128_M0.05_th150_b99` | `sphere/128/sphere128_0039.npy` | 6.99 | 1.40 | 0.05 | 150 |  | dp_cap |
| 49 | generated | 128 | 1 | `drain_sphere128_0022_128_M0.05_th120_b99` | `sphere/128/sphere128_0022.npy` | 8.99 | 3.11 | 0.05 | 120 |  | pv_cap |
| 50 | generated | 128 | 1 | `drain_blob128_0014_128_M1_th150_b99` | `blob/128/blob128_0014.npy` | 6.62 | 1.32 | 1 | 150 |  | dp_cap |
| 51 | generated | 128 | 1 | `drain_blob128_0046_128_M0.2_th140_b99` | `blob/128/blob128_0046.npy` | 7.48 | 1.69 | 0.2 | 140 |  | dp_cap |
| 52 | generated | 128 | 1 | `drain_sphere128_0048_128_M0.1_th130_b99` | `sphere/128/sphere128_0048.npy` | 5.73 | 1.54 | 0.1 | 130 |  | dp_cap |
| 53 | generated | 128 | 1 | `drain_sphere128_0015_128_M1_th140_b99` | `sphere/128/sphere128_0015.npy` | 12.44 | 2.81 | 1 | 140 |  | pv_cap |
| 54 | generated | 128 | 1 | `drain_poly128_0042_128_M0.2_th150_b99` | `poly/128/poly128_0042.npy` | 9.70 | 1.94 | 0.2 | 150 |  | pv_cap |
| 55 | generated | 128 | 1 | `drain_poly128_0060_128_M0.05_th130_b99` | `poly/128/poly128_0060.npy` | 7.67 | 2.07 | 0.05 | 130 |  | pv_cap |
| 56 | generated | 128 | 1 | `drain_poly128_0052_128_M0.1_th120_b99` | `poly/128/poly128_0052.npy` | 5.99 | 2.07 | 0.1 | 120 |  | pv_cap |
| 57 | generated | 128 | 1 | `drain_sphere128_0002_128_M0.05_th150_b99` | `sphere/128/sphere128_0002.npy` | 9.20 | 1.84 | 0.05 | 150 |  | pv_cap |
| 58 | generated | 128 | 1 | `drain_sphere128_0043_128_M1_th130_b99` | `sphere/128/sphere128_0043.npy` | 8.99 | 2.42 | 1 | 130 |  | pv_cap |
| 59 | generated | 128 | 1 | `drain_poly128_0043_128_M0.2_th120_b99` | `poly/128/poly128_0043.npy` | 5.48 | 1.90 | 0.2 | 120 |  | dp_cap |
| 60 | generated | 128 | 1 | `drain_poly128_0034_128_M0.1_th140_b99` | `poly/128/poly128_0034.npy` | 5.83 | 1.32 | 0.1 | 140 |  | dp_cap |
| 61 | generated | 128 | 1 | `drain_sphere128_0012_128_M0.2_th130_b99` | `sphere/128/sphere128_0012.npy` | 5.38 | 1.45 | 0.2 | 130 |  | pv_cap |
| 62 | generated | 128 | 1 | `drain_sphere128_0036_128_M1_th120_b99` | `sphere/128/sphere128_0036.npy` | 6.31 | 2.19 | 1 | 120 |  | pv_cap |
| 63 | generated | 128 | 1 | `drain_poly128_0026_128_M0.1_th150_b99` | `poly/128/poly128_0026.npy` | 8.00 | 1.60 | 0.1 | 150 |  | pv_cap |
| 64 | generated | 128 | 1 | `drain_sphere128_0001_128_M0.05_th140_b99` | `sphere/128/sphere128_0001.npy` | 6.39 | 1.44 | 0.05 | 140 |  | pv_cap |
| 65 | generated | 256 | 4 | `drain_blob256_0045_256_M0.1_th150_b99` | `blob/256/blob256_0045.npy` | 8.77 | 1.75 | 0.1 | 150 |  | dp_cap |
| 66 | generated | 256 | 4 | `drain_poly256_0011_256_M1_th120_b99` | `poly/256/poly256_0011.npy` | 8.19 | 2.84 | 1 | 120 |  | dp_cap |
| 67 | generated | 256 | 4 | `drain_poly256_0005_256_M0.2_th150_b99` | `poly/256/poly256_0005.npy` | 8.24 | 1.65 | 0.2 | 150 |  | dp_cap |
| 68 | generated | 256 | 4 | `drain_sphere256_0036_256_M0.05_th120_b99` | `sphere/256/sphere256_0036.npy` | 6.16 | 2.13 | 0.05 | 120 |  | pv_cap |
| 69 | generated | 256 | 4 | `drain_blob256_0001_256_M0.05_th150_b99` | `blob/256/blob256_0001.npy` | 10.99 | 2.20 | 0.05 | 150 |  | dp_cap |
| 70 | generated | 256 | 4 | `drain_blob256_0062_256_M0.2_th120_b99` | `blob/256/blob256_0062.npy` | 6.39 | 2.21 | 0.2 | 120 |  | dp_cap |
| 71 | generated | 256 | 4 | `drain_poly256_0034_256_M0.1_th120_b99` | `poly/256/poly256_0034.npy` | 5.83 | 2.02 | 0.1 | 120 |  | dp_cap |
| 72 | generated | 256 | 4 | `drain_sphere256_0015_256_M1_th150_b99` | `sphere/256/sphere256_0015.npy` | 9.90 | 1.98 | 1 | 150 |  | pv_cap |
| 73 | ct | 128 | 1 | `drain_buff128_0048_128_M0.05_th120_b99` | `buffberea/128/buff128_0048.npy` | 5.98 | 2.07 | 0.05 | 120 |  | pv_cap |
| 74 | ct | 128 | 1 | `drain_buff128_0036_128_M0.1_th130_b99` | `buffberea/128/buff128_0036.npy` | 5.38 | 1.45 | 0.1 | 130 |  | dp_cap |
| 75 | ct | 128 | 1 | `drain_bent128_0014_128_M1_th140_b99` | `bentheimer/128/bent128_0014.npy` | 5.99 | 1.35 | 1 | 140 |  | dp_cap |
| 76 | ct | 128 | 1 | `drain_castle128_0023_128_M0.2_th150_b99` | `castlegate/128/castle128_0023.npy` | 6.71 | 1.34 | 0.2 | 150 |  | dp_cap |
| 77 | ct | 128 | 1 | `drain_bent128_0022_128_M1_th150_b99` | `bentheimer/128/bent128_0022.npy` | 6.70 | 1.34 | 1 | 150 |  | dp_cap |
| 78 | ct | 128 | 1 | `drain_buff128_0022_128_M0.05_th130_b99` | `buffberea/128/buff128_0022.npy` | 7.55 | 2.03 | 0.05 | 130 |  | pv_cap |
| 79 | ct | 128 | 1 | `drain_castle128_0056_128_M0.2_th140_b99` | `castlegate/128/castle128_0056.npy` | 6.00 | 1.36 | 0.2 | 140 |  | pv_cap |
| 80 | ct | 128 | 1 | `drain_castle128_0000_128_M0.1_th120_b99` | `castlegate/128/castle128_0000.npy` | 5.38 | 1.86 | 0.1 | 120 |  | dp_cap |
| 81 | ct | 128 | 1 | `drain_castle128_0006_128_M0.05_th140_b99` | `castlegate/128/castle128_0006.npy` | 6.70 | 1.52 | 0.05 | 140 |  | dp_cap |
| 82 | ct | 128 | 1 | `drain_bent128_0007_128_M1_th130_b99` | `bentheimer/128/bent128_0007.npy` | 5.10 | 1.37 | 1 | 130 |  | dp_cap |
| 83 | ct | 128 | 1 | `drain_bent128_0046_128_M0.1_th150_b99` | `bentheimer/128/bent128_0046.npy` | 6.78 | 1.36 | 0.1 | 150 |  | pv_cap |
| 84 | ct | 128 | 1 | `drain_buff128_0028_128_M0.2_th120_b99` | `buffberea/128/buff128_0028.npy` | 12.08 | 4.18 | 0.2 | 120 |  | pv_cap |
| 85 | ct | 128 | 1 | `drain_bent128_0035_128_M0.2_th130_b99` | `bentheimer/128/bent128_0035.npy` | 5.64 | 1.52 | 0.2 | 130 |  | dp_cap |
| 86 | ct | 128 | 1 | `drain_castle128_0021_128_M0.1_th140_b99` | `castlegate/128/castle128_0021.npy` | 5.83 | 1.32 | 0.1 | 140 |  | dp_cap |
| 87 | ct | 128 | 1 | `drain_castle128_0030_128_M1_th120_b99` | `castlegate/128/castle128_0030.npy` | 5.10 | 1.77 | 1 | 120 |  | pv_cap |
| 88 | ct | 128 | 1 | `drain_castle128_0036_128_M0.05_th150_b99` | `castlegate/128/castle128_0036.npy` | 6.92 | 1.38 | 0.05 | 150 |  | dp_cap |
| 89 | ct | 128 | 1 | `drain_buff128_0007_128_M1_th130_b99` | `buffberea/128/buff128_0007.npy` | 5.46 | 1.47 | 1 | 130 |  | pv_cap |
| 90 | ct | 128 | 1 | `drain_castle128_0007_128_M0.2_th120_b99` | `castlegate/128/castle128_0007.npy` | 5.38 | 1.86 | 0.2 | 120 |  | dp_cap |
| 91 | ct | 128 | 1 | `drain_castle128_0049_128_M0.1_th150_b99` | `castlegate/128/castle128_0049.npy` | 6.62 | 1.32 | 0.1 | 150 |  | dp_cap |
| 92 | ct | 128 | 1 | `drain_castle128_0008_128_M0.05_th140_b99` | `castlegate/128/castle128_0008.npy` | 6.77 | 1.53 | 0.05 | 140 |  | dp_cap |
| 93 | ct | 128 | 1 | `drain_buff128_0029_128_M0.05_th150_b99` | `buffberea/128/buff128_0029.npy` | 7.99 | 1.60 | 0.05 | 150 |  | dp_cap |
| 94 | ct | 128 | 1 | `drain_bent128_0020_128_M0.2_th130_b99` | `bentheimer/128/bent128_0020.npy` | 9.89 | 2.66 | 0.2 | 130 |  | dp_cap |
| 95 | ct | 128 | 1 | `drain_castle128_0045_128_M0.1_th140_b99` | `castlegate/128/castle128_0045.npy` | 6.16 | 1.39 | 0.1 | 140 |  | dp_cap |
| 96 | ct | 128 | 1 | `drain_buff128_0011_128_M1_th120_b99` | `buffberea/128/buff128_0011.npy` | 9.85 | 3.41 | 1 | 120 |  | pv_cap |
| 97 | ct | 128 | 1 | `drain_bent128_0040_128_M1_th150_b99` | `bentheimer/128/bent128_0040.npy` | 8.99 | 1.80 | 1 | 150 |  | dp_cap |
| 98 | ct | 128 | 1 | `drain_buff128_0033_128_M0.05_th120_b99` | `buffberea/128/buff128_0033.npy` | 6.99 | 2.42 | 0.05 | 120 |  | dp_cap |
| 99 | ct | 128 | 1 | `drain_buff128_0059_128_M0.2_th140_b99` | `buffberea/128/buff128_0059.npy` | 12.76 | 2.88 | 0.2 | 140 |  | pv_cap |
| 100 | ct | 128 | 1 | `drain_buff128_0004_128_M0.1_th130_b99` | `buffberea/128/buff128_0004.npy` | 5.38 | 1.45 | 0.1 | 130 |  | dp_cap |
| 101 | ct | 128 | 1 | `drain_bent128_0052_128_M0.2_th150_b99` | `bentheimer/128/bent128_0052.npy` | 6.54 | 1.31 | 0.2 | 150 |  | pv_cap |
| 102 | ct | 128 | 1 | `drain_bent128_0032_128_M1_th140_b99` | `bentheimer/128/bent128_0032.npy` | 5.99 | 1.35 | 1 | 140 |  | dp_cap |
| 103 | ct | 128 | 1 | `drain_bent128_0063_128_M0.05_th130_b99` | `bentheimer/128/bent128_0063.npy` | 9.69 | 2.61 | 0.05 | 130 |  | pv_cap |
| 104 | ct | 128 | 1 | `drain_buff128_0038_128_M0.1_th120_b99` | `buffberea/128/buff128_0038.npy` | 6.32 | 2.19 | 0.1 | 120 |  | pv_cap |
| 105 | ct | 128 | 1 | `drain_buff128_0014_128_M1_th140_b99` | `buffberea/128/buff128_0014.npy` | 6.16 | 1.39 | 1 | 140 |  | dp_cap |
| 106 | ct | 128 | 1 | `drain_buff128_0032_128_M0.05_th150_b99` | `buffberea/128/buff128_0032.npy` | 7.34 | 1.47 | 0.05 | 150 |  | pv_cap |
| 107 | ct | 128 | 1 | `drain_castle128_0058_128_M0.2_th120_b99` | `castlegate/128/castle128_0058.npy` | 5.74 | 1.99 | 0.2 | 120 |  | dp_cap |
| 108 | ct | 128 | 1 | `drain_bent128_0055_128_M0.1_th130_b99` | `bentheimer/128/bent128_0055.npy` | 5.09 | 1.37 | 0.1 | 130 |  | dp_cap |
| 109 | ct | 128 | 1 | `drain_buff128_0021_128_M0.1_th150_b99` | `buffberea/128/buff128_0021.npy` | 7.00 | 1.40 | 0.1 | 150 |  | dp_cap |
| 110 | ct | 128 | 1 | `drain_bent128_0003_128_M0.05_th140_b99` | `bentheimer/128/bent128_0003.npy` | 6.08 | 1.37 | 0.05 | 140 |  | dp_cap |
| 111 | ct | 128 | 1 | `drain_castle128_0005_128_M0.2_th130_b99` | `castlegate/128/castle128_0005.npy` | 5.46 | 1.47 | 0.2 | 130 |  | dp_cap |
| 112 | ct | 128 | 1 | `drain_buff128_0044_128_M1_th120_b99` | `buffberea/128/buff128_0044.npy` | 6.99 | 2.42 | 1 | 120 |  | pv_cap |
| 113 | ct | 128 | 1 | `drain_castle128_0009_128_M1_th150_b99` | `castlegate/128/castle128_0009.npy` | 13.92 | 2.78 | 1 | 150 |  | dp_cap |
| 114 | ct | 128 | 1 | `drain_castle128_0051_128_M0.2_th140_b99` | `castlegate/128/castle128_0051.npy` | 7.81 | 1.77 | 0.2 | 140 |  | dp_cap |
| 115 | ct | 128 | 1 | `drain_bent128_0043_128_M0.05_th120_b99` | `bentheimer/128/bent128_0043.npy` | 9.04 | 3.13 | 0.05 | 120 |  | pv_cap |
| 116 | ct | 128 | 1 | `drain_castle128_0058_128_M0.1_th120_b99` | `castlegate/128/castle128_0058.npy` | 5.74 | 1.99 | 0.1 | 120 | reused | dp_cap |
| 117 | ct | 128 | 1 | `drain_bent128_0049_128_M0.05_th130_b99` | `bentheimer/128/bent128_0049.npy` | 5.99 | 1.61 | 0.05 | 130 |  | pv_cap |
| 118 | ct | 128 | 1 | `drain_castle128_0008_128_M0.1_th140_b99` | `castlegate/128/castle128_0008.npy` | 6.77 | 1.53 | 0.1 | 140 | reused | dp_cap |
| 119 | ct | 128 | 1 | `drain_bent128_0050_128_M0.2_th150_b99` | `bentheimer/128/bent128_0050.npy` | 6.93 | 1.39 | 0.2 | 150 |  | dp_cap |
| 120 | ct | 128 | 1 | `drain_bent128_0016_128_M1_th130_b99` | `bentheimer/128/bent128_0016.npy` | 7.99 | 2.15 | 1 | 130 |  | pv_cap |
| 121 | ct | 128 | 1 | `drain_bent128_0047_128_M0.1_th150_b99` | `bentheimer/128/bent128_0047.npy` | 7.20 | 1.44 | 0.1 | 150 |  | dp_cap |
| 122 | ct | 128 | 1 | `drain_castle128_0001_128_M1_th130_b99` | `castlegate/128/castle128_0001.npy` | 5.38 | 1.45 | 1 | 130 |  | dp_cap |
| 123 | ct | 128 | 1 | `drain_buff128_0055_128_M0.05_th120_b99` | `buffberea/128/buff128_0055.npy` | 5.38 | 1.86 | 0.05 | 120 |  | dp_cap |
| 124 | ct | 128 | 1 | `drain_bent128_0038_128_M0.2_th140_b99` | `bentheimer/128/bent128_0038.npy` | 6.08 | 1.37 | 0.2 | 140 |  | pv_cap |
| 125 | ct | 128 | 1 | `drain_buff128_0031_128_M0.05_th130_b99` | `buffberea/128/buff128_0031.npy` | 5.10 | 1.37 | 0.05 | 130 |  | dp_cap |
| 126 | ct | 128 | 1 | `drain_buff128_0024_128_M1_th150_b99` | `buffberea/128/buff128_0024.npy` | 8.05 | 1.61 | 1 | 150 |  | pv_cap |
| 127 | ct | 128 | 1 | `drain_buff128_0041_128_M0.1_th140_b99` | `buffberea/128/buff128_0041.npy` | 7.87 | 1.78 | 0.1 | 140 |  | dp_cap |
| 128 | ct | 128 | 1 | `drain_castle128_0056_128_M0.2_th120_b99` | `castlegate/128/castle128_0056.npy` | 6.00 | 2.08 | 0.2 | 120 | reused | pv_cap |
| 129 | ct | 128 | 1 | `drain_buff128_0026_128_M0.05_th150_b99` | `buffberea/128/buff128_0026.npy` | 7.21 | 1.44 | 0.05 | 150 |  | dp_cap |
| 130 | ct | 128 | 1 | `drain_bent128_0033_128_M0.1_th130_b99` | `bentheimer/128/bent128_0033.npy` | 7.54 | 2.03 | 0.1 | 130 |  | dp_cap |
| 131 | ct | 128 | 1 | `drain_bent128_0025_128_M1_th140_b99` | `bentheimer/128/bent128_0025.npy` | 8.30 | 1.88 | 1 | 140 |  | dp_cap |
| 132 | ct | 128 | 1 | `drain_castle128_0007_128_M1_th120_b99` | `castlegate/128/castle128_0007.npy` | 5.38 | 1.86 | 1 | 120 | reused | dp_cap |
| 133 | ct | 128 | 1 | `drain_bent128_0036_128_M0.2_th150_b99` | `bentheimer/128/bent128_0036.npy` | 9.20 | 1.84 | 0.2 | 150 |  | pv_cap |
| 134 | ct | 128 | 1 | `drain_castle128_0012_128_M0.2_th130_b99` | `castlegate/128/castle128_0012.npy` | 5.64 | 1.52 | 0.2 | 130 |  | dp_cap |
| 135 | ct | 128 | 1 | `drain_buff128_0046_128_M0.05_th140_b99` | `buffberea/128/buff128_0046.npy` | 6.70 | 1.52 | 0.05 | 140 |  | dp_cap |
| 136 | ct | 128 | 1 | `drain_bent128_0024_128_M0.1_th120_b99` | `bentheimer/128/bent128_0024.npy` | 6.48 | 2.24 | 0.1 | 120 |  | pv_cap |
| 137 | ct | 256 | 4 | `drain_castle256_0024_256_M0.05_th120_b99` | `castlegate/256/castle256_0024.npy` | 5.10 | 1.77 | 0.05 | 120 |  | dp_cap |
| 138 | ct | 256 | 4 | `drain_bent256_0053_256_M0.1_th150_b99` | `bentheimer/256/bent256_0053.npy` | 6.99 | 1.40 | 0.1 | 150 |  | dp_cap |
| 139 | ct | 256 | 4 | `drain_buff256_0034_256_M0.2_th150_b99` | `buffberea/256/buff256_0034.npy` | 7.20 | 1.44 | 0.2 | 150 |  | dp_cap |
| 140 | ct | 256 | 4 | `drain_buff256_0010_256_M1_th120_b99` | `buffberea/256/buff256_0010.npy` | 5.38 | 1.86 | 1 | 120 |  | dp_cap |
| 141 | ct | 256 | 4 | `drain_castle256_0041_256_M0.2_th120_b99` | `castlegate/256/castle256_0041.npy` | 5.66 | 1.96 | 0.2 | 120 |  | dp_cap |
| 142 | ct | 256 | 4 | `drain_buff256_0032_256_M1_th150_b99` | `buffberea/256/buff256_0032.npy` | 7.06 | 1.41 | 1 | 150 |  | dp_cap |
| 143 | ct | 256 | 4 | `drain_bent256_0000_256_M0.1_th120_b99` | `bentheimer/256/bent256_0000.npy` | 5.38 | 1.86 | 0.1 | 120 |  | dp_cap |
| 144 | ct | 256 | 4 | `drain_bent256_0036_256_M0.05_th150_b99` | `bentheimer/256/bent256_0036.npy` | 6.99 | 1.40 | 0.05 | 150 |  | dp_cap |

## trapping — 144 runs

**144 runs** — 144 finished, 0 running.

| # | Family | Domain | GPUs | Run | Geometry | r_c | margin | M | theta | Flags | Status |
|---:|---|---|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | generated | 128 | 1 | `trap_blob128_0009_128_M0.2_th120_b99` | `blob/128/blob128_0009.npy` | 7.21 | 2.50 | 0.2 | 120 |  | red_immobile |
| 2 | generated | 128 | 1 | `trap_poly128_0012_128_M0.05_th140_b99` | `poly/128/poly128_0012.npy` | 6.16 | 1.39 | 0.05 | 140 |  | red_immobile |
| 3 | generated | 128 | 1 | `trap_poly128_0057_128_M1_th150_b99` | `poly/128/poly128_0057.npy` | 8.83 | 1.77 | 1 | 150 |  | red_immobile |
| 4 | generated | 128 | 1 | `trap_sphere128_0016_128_M0.1_th130_b99` | `sphere/128/sphere128_0016.npy` | 5.37 | 1.45 | 0.1 | 130 |  | red_immobile |
| 5 | generated | 128 | 1 | `trap_blob128_0012_128_M0.05_th150_b99` | `blob/128/blob128_0012.npy` | 9.22 | 1.84 | 0.05 | 150 |  | red_immobile |
| 6 | generated | 128 | 1 | `trap_sphere128_0031_128_M1_th140_b99` | `sphere/128/sphere128_0031.npy` | 6.99 | 1.58 | 1 | 140 |  | red_immobile |
| 7 | generated | 128 | 1 | `trap_sphere128_0015_128_M0.1_th120_b99` | `sphere/128/sphere128_0015.npy` | 12.44 | 4.31 | 0.1 | 120 |  | red_immobile |
| 8 | generated | 128 | 1 | `trap_poly128_0061_128_M0.2_th130_b99` | `poly/128/poly128_0061.npy` | 5.47 | 1.47 | 0.2 | 130 |  | red_immobile |
| 9 | generated | 128 | 1 | `trap_blob128_0018_128_M1_th120_b99` | `blob/128/blob128_0018.npy` | 8.11 | 2.81 | 1 | 120 |  | red_immobile |
| 10 | generated | 128 | 1 | `trap_sphere128_0006_128_M0.1_th140_b99` | `sphere/128/sphere128_0006.npy` | 7.99 | 1.81 | 0.1 | 140 |  | red_immobile |
| 11 | generated | 128 | 1 | `trap_blob128_0026_128_M0.05_th130_b99` | `blob/128/blob128_0026.npy` | 5.99 | 1.61 | 0.05 | 130 |  | red_immobile |
| 12 | generated | 128 | 1 | `trap_sphere128_0005_128_M0.2_th150_b99` | `sphere/128/sphere128_0005.npy` | 7.35 | 1.47 | 0.2 | 150 |  | red_immobile |
| 13 | generated | 128 | 1 | `trap_blob128_0049_128_M0.1_th150_b99` | `blob/128/blob128_0049.npy` | 9.00 | 1.80 | 0.1 | 150 |  | red_immobile |
| 14 | generated | 128 | 1 | `trap_poly128_0003_128_M0.2_th140_b99` | `poly/128/poly128_0003.npy` | 5.90 | 1.33 | 0.2 | 140 |  | red_immobile |
| 15 | generated | 128 | 1 | `trap_poly128_0002_128_M1_th130_b99` | `poly/128/poly128_0002.npy` | 5.19 | 1.40 | 1 | 130 |  | red_immobile |
| 16 | generated | 128 | 1 | `trap_sphere128_0032_128_M0.05_th120_b99` | `sphere/128/sphere128_0032.npy` | 7.13 | 2.47 | 0.05 | 120 |  | red_immobile |
| 17 | generated | 128 | 1 | `trap_blob128_0007_128_M1_th120_b99` | `blob/128/blob128_0007.npy` | 6.16 | 2.13 | 1 | 120 |  | red_immobile |
| 18 | generated | 128 | 1 | `trap_poly128_0026_128_M0.2_th140_b99` | `poly/128/poly128_0026.npy` | 8.00 | 1.81 | 0.2 | 140 |  | red_immobile |
| 19 | generated | 128 | 1 | `trap_blob128_0041_128_M0.1_th130_b99` | `blob/128/blob128_0041.npy` | 5.38 | 1.45 | 0.1 | 130 |  | red_immobile |
| 20 | generated | 128 | 1 | `trap_sphere128_0026_128_M0.05_th150_b99` | `sphere/128/sphere128_0026.npy` | 9.00 | 1.80 | 0.05 | 150 |  | red_immobile |
| 21 | generated | 128 | 1 | `trap_sphere128_0012_128_M0.2_th130_b99` | `sphere/128/sphere128_0012.npy` | 5.38 | 1.45 | 0.2 | 130 |  | red_immobile |
| 22 | generated | 128 | 1 | `trap_poly128_0023_128_M0.05_th120_b99` | `poly/128/poly128_0023.npy` | 5.19 | 1.80 | 0.05 | 120 |  | red_immobile |
| 23 | generated | 128 | 1 | `trap_poly128_0060_128_M1_th140_b99` | `poly/128/poly128_0060.npy` | 7.67 | 1.73 | 1 | 140 |  | red_immobile |
| 24 | generated | 128 | 1 | `trap_sphere128_0014_128_M0.1_th150_b99` | `sphere/128/sphere128_0014.npy` | 7.07 | 1.41 | 0.1 | 150 |  | red_immobile |
| 25 | generated | 128 | 1 | `trap_poly128_0042_128_M0.05_th140_b99` | `poly/128/poly128_0042.npy` | 9.70 | 2.19 | 0.05 | 140 |  | red_immobile |
| 26 | generated | 128 | 1 | `trap_blob128_0060_128_M0.2_th150_b99` | `blob/128/blob128_0060.npy` | 7.98 | 1.60 | 0.2 | 150 |  | red_immobile |
| 27 | generated | 128 | 1 | `trap_poly128_0007_128_M1_th130_b99` | `poly/128/poly128_0007.npy` | 7.27 | 1.96 | 1 | 130 |  | red_immobile |
| 28 | generated | 128 | 1 | `trap_sphere128_0036_128_M0.1_th120_b99` | `sphere/128/sphere128_0036.npy` | 6.31 | 2.19 | 0.1 | 120 |  | red_immobile |
| 29 | generated | 128 | 1 | `trap_blob128_0038_128_M1_th150_b99` | `blob/128/blob128_0038.npy` | 9.63 | 1.93 | 1 | 150 |  | red_immobile |
| 30 | generated | 128 | 1 | `trap_sphere128_0019_128_M0.05_th130_b99` | `sphere/128/sphere128_0019.npy` | 6.99 | 1.88 | 0.05 | 130 |  | red_immobile |
| 31 | generated | 128 | 1 | `trap_blob128_0044_128_M0.1_th140_b99` | `blob/128/blob128_0044.npy` | 9.10 | 2.06 | 0.1 | 140 |  | red_immobile |
| 32 | generated | 128 | 1 | `trap_blob128_0014_128_M0.2_th120_b99` | `blob/128/blob128_0014.npy` | 6.62 | 2.29 | 0.2 | 120 |  | red_immobile |
| 33 | generated | 128 | 1 | `trap_poly128_0031_128_M0.2_th150_b99` | `poly/128/poly128_0031.npy` | 6.62 | 1.32 | 0.2 | 150 |  | red_immobile |
| 34 | generated | 128 | 1 | `trap_blob128_0031_128_M0.1_th130_b99` | `blob/128/blob128_0031.npy` | 5.64 | 1.52 | 0.1 | 130 |  | red_immobile |
| 35 | generated | 128 | 1 | `trap_sphere128_0043_128_M0.05_th140_b99` | `sphere/128/sphere128_0043.npy` | 8.99 | 2.03 | 0.05 | 140 |  | red_immobile |
| 36 | generated | 128 | 1 | `trap_sphere128_0028_128_M1_th120_b99` | `sphere/128/sphere128_0028.npy` | 7.27 | 2.52 | 1 | 120 |  | red_immobile |
| 37 | generated | 128 | 1 | `trap_sphere128_0047_128_M0.1_th140_b99` | `sphere/128/sphere128_0047.npy` | 6.00 | 1.36 | 0.1 | 140 |  | red_immobile |
| 38 | generated | 128 | 1 | `trap_sphere128_0022_128_M1_th150_b99` | `sphere/128/sphere128_0022.npy` | 8.99 | 1.80 | 1 | 150 |  | red_immobile |
| 39 | generated | 128 | 1 | `trap_poly128_0028_128_M0.2_th130_b99` | `poly/128/poly128_0028.npy` | 5.74 | 1.55 | 0.2 | 130 |  | red_immobile |
| 40 | generated | 128 | 1 | `trap_poly128_0056_128_M0.05_th120_b99` | `poly/128/poly128_0056.npy` | 5.91 | 2.05 | 0.05 | 120 |  | red_immobile |
| 41 | generated | 128 | 1 | `trap_sphere128_0058_128_M0.05_th150_b99` | `sphere/128/sphere128_0058.npy` | 6.63 | 1.33 | 0.05 | 150 |  | red_immobile |
| 42 | generated | 128 | 1 | `trap_poly128_0055_128_M1_th130_b99` | `poly/128/poly128_0055.npy` | 5.64 | 1.52 | 1 | 130 |  | red_immobile |
| 43 | generated | 128 | 1 | `trap_blob128_0032_128_M0.2_th140_b99` | `blob/128/blob128_0032.npy` | 7.06 | 1.60 | 0.2 | 140 |  | red_immobile |
| 44 | generated | 128 | 1 | `trap_sphere128_0039_128_M0.1_th120_b99` | `sphere/128/sphere128_0039.npy` | 6.99 | 2.42 | 0.1 | 120 |  | red_immobile |
| 45 | generated | 128 | 1 | `trap_poly128_0062_128_M0.2_th120_b99` | `poly/128/poly128_0062.npy` | 5.83 | 2.02 | 0.2 | 120 |  | red_immobile |
| 46 | generated | 128 | 1 | `trap_poly128_0058_128_M0.05_th130_b99` | `poly/128/poly128_0058.npy` | 5.64 | 1.52 | 0.05 | 130 |  | red_immobile |
| 47 | generated | 128 | 1 | `trap_poly128_0005_128_M1_th140_b99` | `poly/128/poly128_0005.npy` | 5.99 | 1.35 | 1 | 140 |  | red_immobile |
| 48 | generated | 128 | 1 | `trap_blob128_0006_128_M0.1_th150_b99` | `blob/128/blob128_0006.npy` | 8.05 | 1.61 | 0.1 | 150 |  | red_immobile |
| 49 | generated | 128 | 1 | `trap_poly128_0051_128_M0.05_th140_b99` | `poly/128/poly128_0051.npy` | 6.00 | 1.36 | 0.05 | 140 |  | red_immobile |
| 50 | generated | 128 | 1 | `trap_blob128_0001_128_M0.1_th150_b99` | `blob/128/blob128_0001.npy` | 8.48 | 1.70 | 0.1 | 150 |  | red_immobile |
| 51 | generated | 128 | 1 | `trap_blob128_0046_128_M1_th120_b99` | `blob/128/blob128_0046.npy` | 7.48 | 2.59 | 1 | 120 |  | red_immobile |
| 52 | generated | 128 | 1 | `trap_sphere128_0035_128_M0.2_th130_b99` | `sphere/128/sphere128_0035.npy` | 5.19 | 1.40 | 0.2 | 130 |  | red_immobile |
| 53 | generated | 128 | 1 | `trap_poly128_0034_128_M0.1_th120_b99` | `poly/128/poly128_0034.npy` | 5.83 | 2.02 | 0.1 | 120 |  | red_immobile |
| 54 | generated | 128 | 1 | `trap_poly128_0046_128_M1_th140_b99` | `poly/128/poly128_0046.npy` | 6.39 | 1.45 | 1 | 140 |  | red_immobile |
| 55 | generated | 128 | 1 | `trap_blob128_0023_128_M0.05_th150_b99` | `blob/128/blob128_0023.npy` | 9.89 | 1.98 | 0.05 | 150 |  | red_immobile |
| 56 | generated | 128 | 1 | `trap_sphere128_0002_128_M0.2_th150_b99` | `sphere/128/sphere128_0002.npy` | 9.20 | 1.84 | 0.2 | 150 |  | red_immobile |
| 57 | generated | 128 | 1 | `trap_sphere128_0027_128_M1_th130_b99` | `sphere/128/sphere128_0027.npy` | 6.31 | 1.70 | 1 | 130 |  | pv_cap |
| 58 | generated | 128 | 1 | `trap_blob128_0027_128_M0.1_th130_b99` | `blob/128/blob128_0027.npy` | 5.18 | 1.40 | 0.1 | 130 |  | red_immobile |
| 59 | generated | 128 | 1 | `trap_sphere128_0040_128_M0.05_th120_b99` | `sphere/128/sphere128_0040.npy` | 6.32 | 2.19 | 0.05 | 120 |  | red_immobile |
| 60 | generated | 128 | 1 | `trap_blob128_0062_128_M0.2_th140_b99` | `blob/128/blob128_0062.npy` | 5.91 | 1.34 | 0.2 | 140 |  | red_immobile |
| 61 | generated | 128 | 1 | `trap_poly128_0047_128_M1_th150_b99` | `poly/128/poly128_0047.npy` | 7.60 | 1.52 | 1 | 150 |  | red_immobile |
| 62 | generated | 128 | 1 | `trap_blob128_0010_128_M0.2_th120_b99` | `blob/128/blob128_0010.npy` | 7.13 | 2.47 | 0.2 | 120 |  | red_immobile |
| 63 | generated | 128 | 1 | `trap_blob128_0005_128_M0.05_th130_b99` | `blob/128/blob128_0005.npy` | 6.00 | 1.62 | 0.05 | 130 |  | red_immobile |
| 64 | generated | 128 | 1 | `trap_blob128_0029_128_M0.1_th140_b99` | `blob/128/blob128_0029.npy` | 9.69 | 2.19 | 0.1 | 140 |  | red_immobile |
| 65 | generated | 256 | 4 | `trap_blob256_0034_256_M0.1_th150_b99` | `blob/256/blob256_0034.npy` | 6.70 | 1.34 | 0.1 | 150 |  | red_immobile |
| 66 | generated | 256 | 4 | `trap_blob256_0046_256_M0.2_th120_b99` | `blob/256/blob256_0046.npy` | 7.00 | 2.42 | 0.2 | 120 |  | red_immobile |
| 67 | generated | 256 | 4 | `trap_poly256_0029_256_M0.05_th120_b99` | `poly/256/poly256_0029.npy` | 5.38 | 1.86 | 0.05 | 120 |  | red_immobile |
| 68 | generated | 256 | 4 | `trap_sphere256_0052_256_M1_th150_b99` | `sphere/256/sphere256_0052.npy` | 9.11 | 1.82 | 1 | 150 |  | red_immobile |
| 69 | generated | 256 | 4 | `trap_poly256_0021_256_M1_th120_b99` | `poly/256/poly256_0021.npy` | 5.09 | 1.76 | 1 | 120 |  | red_immobile |
| 70 | generated | 256 | 4 | `trap_blob256_0061_256_M0.05_th150_b99` | `blob/256/blob256_0061.npy` | 8.30 | 1.66 | 0.05 | 150 |  | red_immobile |
| 71 | generated | 256 | 4 | `trap_sphere256_0037_256_M0.1_th120_b99` | `sphere/256/sphere256_0037.npy` | 5.09 | 1.76 | 0.1 | 120 |  | red_immobile |
| 72 | generated | 256 | 4 | `trap_poly256_0031_256_M0.2_th150_b99` | `poly/256/poly256_0031.npy` | 7.86 | 1.57 | 0.2 | 150 |  | red_immobile |
| 73 | ct | 128 | 1 | `trap_castle128_0009_128_M0.05_th150_b99` | `castlegate/128/castle128_0009.npy` | 13.92 | 2.78 | 0.05 | 150 |  | red_immobile |
| 74 | ct | 128 | 1 | `trap_bent128_0028_128_M0.2_th120_b99` | `bentheimer/128/bent128_0028.npy` | 5.38 | 1.86 | 0.2 | 120 |  | red_immobile |
| 75 | ct | 128 | 1 | `trap_castle128_0056_128_M1_th140_b99` | `castlegate/128/castle128_0056.npy` | 6.00 | 1.36 | 1 | 140 |  | red_immobile |
| 76 | ct | 128 | 1 | `trap_castle128_0001_128_M0.1_th130_b99` | `castlegate/128/castle128_0001.npy` | 5.38 | 1.45 | 0.1 | 130 |  | red_immobile |
| 77 | ct | 128 | 1 | `trap_castle128_0007_128_M1_th120_b99` | `castlegate/128/castle128_0007.npy` | 5.38 | 1.86 | 1 | 120 |  | red_immobile |
| 78 | ct | 128 | 1 | `trap_castle128_0030_128_M0.2_th130_b99` | `castlegate/128/castle128_0030.npy` | 5.10 | 1.37 | 0.2 | 130 |  | red_immobile |
| 79 | ct | 128 | 1 | `trap_buff128_0033_128_M0.05_th140_b99` | `buffberea/128/buff128_0033.npy` | 6.99 | 1.58 | 0.05 | 140 |  | red_immobile |
| 80 | ct | 128 | 1 | `trap_castle128_0023_128_M0.1_th150_b99` | `castlegate/128/castle128_0023.npy` | 6.71 | 1.34 | 0.1 | 150 |  | red_immobile |
| 81 | ct | 128 | 1 | `trap_castle128_0058_128_M1_th130_b99` | `castlegate/128/castle128_0058.npy` | 5.74 | 1.55 | 1 | 130 |  | red_immobile |
| 82 | ct | 128 | 1 | `trap_castle128_0008_128_M0.2_th140_b99` | `castlegate/128/castle128_0008.npy` | 6.77 | 1.53 | 0.2 | 140 |  | red_immobile |
| 83 | ct | 128 | 1 | `trap_castle128_0012_128_M0.1_th120_b99` | `castlegate/128/castle128_0012.npy` | 5.64 | 1.96 | 0.1 | 120 |  | red_immobile |
| 84 | ct | 128 | 1 | `trap_buff128_0011_128_M1_th150_b99` | `buffberea/128/buff128_0011.npy` | 9.85 | 1.97 | 1 | 150 |  | red_immobile |
| 85 | ct | 128 | 1 | `trap_buff128_0014_128_M0.05_th120_b99` | `buffberea/128/buff128_0014.npy` | 6.16 | 2.13 | 0.05 | 120 |  | red_immobile |
| 86 | ct | 128 | 1 | `trap_bent128_0009_128_M0.05_th130_b99` | `bentheimer/128/bent128_0009.npy` | 5.65 | 1.52 | 0.05 | 130 |  | red_immobile |
| 87 | ct | 128 | 1 | `trap_castle128_0006_128_M0.2_th150_b99` | `castlegate/128/castle128_0006.npy` | 6.70 | 1.34 | 0.2 | 150 |  | red_immobile |
| 88 | ct | 128 | 1 | `trap_castle128_0045_128_M0.1_th140_b99` | `castlegate/128/castle128_0045.npy` | 6.16 | 1.39 | 0.1 | 140 |  | red_immobile |
| 89 | ct | 128 | 1 | `trap_castle128_0008_128_M0.05_th120_b99` | `castlegate/128/castle128_0008.npy` | 6.77 | 2.35 | 0.05 | 120 | reused | red_immobile |
| 90 | ct | 128 | 1 | `trap_bent128_0040_128_M0.1_th140_b99` | `bentheimer/128/bent128_0040.npy` | 8.99 | 2.03 | 0.1 | 140 |  | red_immobile |
| 91 | ct | 128 | 1 | `trap_buff128_0022_128_M1_th130_b99` | `buffberea/128/buff128_0022.npy` | 7.55 | 2.03 | 1 | 130 |  | red_immobile |
| 92 | ct | 128 | 1 | `trap_castle128_0036_128_M0.2_th150_b99` | `castlegate/128/castle128_0036.npy` | 6.92 | 1.38 | 0.2 | 150 |  | red_immobile |
| 93 | ct | 128 | 1 | `trap_bent128_0024_128_M1_th140_b99` | `bentheimer/128/bent128_0024.npy` | 6.48 | 1.47 | 1 | 140 |  | red_immobile |
| 94 | ct | 128 | 1 | `trap_buff128_0032_128_M0.2_th120_b99` | `buffberea/128/buff128_0032.npy` | 7.34 | 2.54 | 0.2 | 120 |  | red_immobile |
| 95 | ct | 128 | 1 | `trap_bent128_0005_128_M0.1_th150_b99` | `bentheimer/128/bent128_0005.npy` | 7.00 | 1.40 | 0.1 | 150 |  | red_immobile |
| 96 | ct | 128 | 1 | `trap_buff128_0048_128_M0.05_th130_b99` | `buffberea/128/buff128_0048.npy` | 5.98 | 1.61 | 0.05 | 130 |  | red_immobile |
| 97 | ct | 128 | 1 | `trap_castle128_0005_128_M0.2_th130_b99` | `castlegate/128/castle128_0005.npy` | 5.46 | 1.47 | 0.2 | 130 |  | red_immobile |
| 98 | ct | 128 | 1 | `trap_buff128_0055_128_M1_th120_b99` | `buffberea/128/buff128_0055.npy` | 5.38 | 1.86 | 1 | 120 |  | red_immobile |
| 99 | ct | 128 | 1 | `trap_buff128_0059_128_M0.05_th150_b99` | `buffberea/128/buff128_0059.npy` | 12.76 | 2.55 | 0.05 | 150 |  | red_immobile |
| 100 | ct | 128 | 1 | `trap_castle128_0051_128_M0.2_th140_b99` | `castlegate/128/castle128_0051.npy` | 7.81 | 1.77 | 0.2 | 140 |  | red_immobile |
| 101 | ct | 128 | 1 | `trap_buff128_0028_128_M0.1_th130_b99` | `buffberea/128/buff128_0028.npy` | 12.08 | 3.25 | 0.1 | 130 |  | red_immobile |
| 102 | ct | 128 | 1 | `trap_castle128_0021_128_M0.05_th140_b99` | `castlegate/128/castle128_0021.npy` | 5.83 | 1.32 | 0.05 | 140 |  | red_immobile |
| 103 | ct | 128 | 1 | `trap_bent128_0025_128_M1_th150_b99` | `bentheimer/128/bent128_0025.npy` | 8.30 | 1.66 | 1 | 150 |  | red_immobile |
| 104 | ct | 128 | 1 | `trap_buff128_0038_128_M0.1_th120_b99` | `buffberea/128/buff128_0038.npy` | 6.32 | 2.19 | 0.1 | 120 |  | red_immobile |
| 105 | ct | 128 | 1 | `trap_bent128_0045_128_M0.1_th130_b99` | `bentheimer/128/bent128_0045.npy` | 5.38 | 1.45 | 0.1 | 130 |  | red_immobile |
| 106 | ct | 128 | 1 | `trap_castle128_0005_128_M0.2_th120_b99` | `castlegate/128/castle128_0005.npy` | 5.46 | 1.89 | 0.2 | 120 | reused | red_immobile |
| 107 | ct | 128 | 1 | `trap_buff128_0024_128_M1_th140_b99` | `buffberea/128/buff128_0024.npy` | 8.05 | 1.82 | 1 | 140 |  | red_immobile |
| 108 | ct | 128 | 1 | `trap_bent128_0050_128_M0.05_th150_b99` | `bentheimer/128/bent128_0050.npy` | 6.93 | 1.39 | 0.05 | 150 |  | red_immobile |
| 109 | ct | 128 | 1 | `trap_buff128_0007_128_M0.2_th130_b99` | `buffberea/128/buff128_0007.npy` | 5.46 | 1.47 | 0.2 | 130 |  | red_immobile |
| 110 | ct | 128 | 1 | `trap_bent128_0041_128_M1_th120_b99` | `bentheimer/128/bent128_0041.npy` | 6.39 | 2.21 | 1 | 120 |  | red_immobile |
| 111 | ct | 128 | 1 | `trap_castle128_0049_128_M0.1_th150_b99` | `castlegate/128/castle128_0049.npy` | 6.62 | 1.32 | 0.1 | 150 |  | red_immobile |
| 112 | ct | 128 | 1 | `trap_bent128_0030_128_M0.05_th140_b99` | `bentheimer/128/bent128_0030.npy` | 5.83 | 1.32 | 0.05 | 140 |  | red_immobile |
| 113 | ct | 128 | 1 | `trap_buff128_0026_128_M0.1_th140_b99` | `buffberea/128/buff128_0026.npy` | 7.21 | 1.63 | 0.1 | 140 |  | red_immobile |
| 114 | ct | 128 | 1 | `trap_castle128_0007_128_M0.05_th120_b99` | `castlegate/128/castle128_0007.npy` | 5.38 | 1.86 | 0.05 | 120 | reused | red_immobile |
| 115 | ct | 128 | 1 | `trap_bent128_0020_128_M0.2_th150_b99` | `bentheimer/128/bent128_0020.npy` | 9.89 | 1.98 | 0.2 | 150 |  | red_immobile |
| 116 | ct | 128 | 1 | `trap_bent128_0038_128_M1_th130_b99` | `bentheimer/128/bent128_0038.npy` | 6.08 | 1.64 | 1 | 130 |  | red_immobile |
| 117 | ct | 128 | 1 | `trap_bent128_0016_128_M1_th150_b99` | `bentheimer/128/bent128_0016.npy` | 7.99 | 1.60 | 1 | 150 |  | red_immobile |
| 118 | ct | 128 | 1 | `trap_buff128_0049_128_M0.05_th130_b99` | `buffberea/128/buff128_0049.npy` | 5.99 | 1.61 | 0.05 | 130 |  | red_immobile |
| 119 | ct | 128 | 1 | `trap_bent128_0003_128_M0.2_th140_b99` | `bentheimer/128/bent128_0003.npy` | 6.08 | 1.37 | 0.2 | 140 |  | red_immobile |
| 120 | ct | 128 | 1 | `trap_buff128_0004_128_M0.1_th120_b99` | `buffberea/128/buff128_0004.npy` | 5.38 | 1.87 | 0.1 | 120 |  | red_immobile |
| 121 | ct | 128 | 1 | `trap_buff128_0041_128_M0.1_th150_b99` | `buffberea/128/buff128_0041.npy` | 7.87 | 1.57 | 0.1 | 150 |  | red_immobile |
| 122 | ct | 128 | 1 | `trap_buff128_0029_128_M0.2_th140_b99` | `buffberea/128/buff128_0029.npy` | 7.99 | 1.81 | 0.2 | 140 |  | red_immobile |
| 123 | ct | 128 | 1 | `trap_buff128_0021_128_M0.05_th130_b99` | `buffberea/128/buff128_0021.npy` | 7.00 | 1.89 | 0.05 | 130 |  | red_immobile |
| 124 | ct | 128 | 1 | `trap_bent128_0055_128_M1_th120_b99` | `bentheimer/128/bent128_0055.npy` | 5.09 | 1.76 | 1 | 120 |  | red_immobile |
| 125 | ct | 128 | 1 | `trap_bent128_0008_128_M0.1_th120_b99` | `bentheimer/128/bent128_0008.npy` | 5.99 | 2.07 | 0.1 | 120 |  | red_immobile |
| 126 | ct | 128 | 1 | `trap_bent128_0036_128_M0.2_th150_b99` | `bentheimer/128/bent128_0036.npy` | 9.20 | 1.84 | 0.2 | 150 |  | red_immobile |
| 127 | ct | 128 | 1 | `trap_buff128_0046_128_M1_th140_b99` | `buffberea/128/buff128_0046.npy` | 6.70 | 1.52 | 1 | 140 |  | red_immobile |
| 128 | ct | 128 | 1 | `trap_castle128_0030_128_M0.05_th120_b99` | `castlegate/128/castle128_0030.npy` | 5.10 | 1.77 | 0.05 | 120 | reused | red_immobile |
| 129 | ct | 128 | 1 | `trap_bent128_0052_128_M0.2_th130_b99` | `bentheimer/128/bent128_0052.npy` | 6.54 | 1.76 | 0.2 | 130 |  | red_immobile |
| 130 | ct | 128 | 1 | `trap_buff128_0044_128_M0.05_th140_b99` | `buffberea/128/buff128_0044.npy` | 6.99 | 1.58 | 0.05 | 140 |  | red_immobile |
| 131 | ct | 128 | 1 | `trap_bent128_0047_128_M1_th150_b99` | `bentheimer/128/bent128_0047.npy` | 7.20 | 1.44 | 1 | 150 |  | red_immobile |
| 132 | ct | 128 | 1 | `trap_castle128_0000_128_M0.1_th130_b99` | `castlegate/128/castle128_0000.npy` | 5.38 | 1.45 | 0.1 | 130 |  | red_immobile |
| 133 | ct | 128 | 1 | `trap_buff128_0039_128_M0.2_th120_b99` | `buffberea/128/buff128_0039.npy` | 5.82 | 2.02 | 0.2 | 120 |  | red_immobile |
| 134 | ct | 128 | 1 | `trap_bent128_0046_128_M0.05_th150_b99` | `bentheimer/128/bent128_0046.npy` | 6.78 | 1.36 | 0.05 | 150 |  | red_immobile |
| 135 | ct | 128 | 1 | `trap_bent128_0033_128_M1_th130_b99` | `bentheimer/128/bent128_0033.npy` | 7.54 | 2.03 | 1 | 130 |  | red_immobile |
| 136 | ct | 128 | 1 | `trap_bent128_0043_128_M0.1_th140_b99` | `bentheimer/128/bent128_0043.npy` | 9.04 | 2.05 | 0.1 | 140 |  | red_immobile |
| 137 | ct | 256 | 4 | `trap_castle256_0023_256_M0.05_th150_b99` | `castlegate/256/castle256_0023.npy` | 8.30 | 1.66 | 0.05 | 150 |  | red_immobile |
| 138 | ct | 256 | 4 | `trap_bent256_0012_256_M0.1_th120_b99` | `bentheimer/256/bent256_0012.npy` | 5.19 | 1.80 | 0.1 | 120 |  | red_immobile |
| 139 | ct | 256 | 4 | `trap_buff256_0056_256_M1_th120_b99` | `buffberea/256/buff256_0056.npy` | 5.91 | 2.05 | 1 | 120 |  | red_immobile |
| 140 | ct | 256 | 4 | `trap_castle256_0023_256_M0.2_th150_b99` | `castlegate/256/castle256_0023.npy` | 8.30 | 1.66 | 0.2 | 150 | reused | red_immobile |
| 141 | ct | 256 | 4 | `trap_bent256_0034_256_M0.05_th120_b99` | `bentheimer/256/bent256_0034.npy` | 5.38 | 1.86 | 0.05 | 120 |  | red_immobile |
| 142 | ct | 256 | 4 | `trap_bent256_0040_256_M1_th150_b99` | `bentheimer/256/bent256_0040.npy` | 6.92 | 1.38 | 1 | 150 |  | red_immobile |
| 143 | ct | 256 | 4 | `trap_buff256_0023_256_M0.2_th120_b99` | `buffberea/256/buff256_0023.npy` | 5.74 | 1.99 | 0.2 | 120 |  | red_immobile |
| 144 | ct | 256 | 4 | `trap_buff256_0059_256_M0.1_th150_b99` | `buffberea/256/buff256_0059.npy` | 9.43 | 1.89 | 0.1 | 150 |  | red_immobile |

## GDL — 144 runs

**144 runs** — 143 finished, 0 running.

| # | Family | Domain | GPUs | Run | Geometry | r_c | margin | M | theta | Flags | Status |
|---:|---|---|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | generated | 128x128x64 | 1 | `gdl_fiber_0008_128x128x64_M20_th130_b99` | `fiber/128x128x64/fiber_0008.npy` | — | — | 20 | 130 |  | pv_cap |
| 2 | generated | 128x128x64 | 1 | `gdl_fiber_0015_128x128x64_M10_th120_b99` | `fiber/128x128x64/fiber_0015.npy` | — | — | 10 | 120 |  | pv_cap |
| 3 | generated | 128x128x64 | 1 | `gdl_fiber_0044_128x128x64_M1_th140_b99` | `fiber/128x128x64/fiber_0044.npy` | — | — | 1 | 140 |  | bt_cap |
| 4 | generated | 128x128x64 | 1 | `gdl_fiber_0061_128x128x64_M5_th150_b99` | `fiber/128x128x64/fiber_0061.npy` | — | — | 5 | 150 |  | bt_cap |
| 5 | generated | 128x128x64 | 1 | `gdl_fiber_0054_128x128x64_M20_th120_b99` | `fiber/128x128x64/fiber_0054.npy` | — | — | 20 | 120 |  | pv_cap |
| 6 | generated | 128x128x64 | 1 | `gdl_fiber_0026_128x128x64_M5_th140_b99` | `fiber/128x128x64/fiber_0026.npy` | — | — | 5 | 140 |  | pv_cap |
| 7 | generated | 128x128x64 | 1 | `gdl_fiber_0051_128x128x64_M10_th130_b99` | `fiber/128x128x64/fiber_0051.npy` | — | — | 10 | 130 |  | pv_cap |
| 8 | generated | 128x128x64 | 1 | `gdl_fiber_0050_128x128x64_M1_th150_b99` | `fiber/128x128x64/fiber_0050.npy` | — | — | 1 | 150 |  | bt_cap |
| 9 | generated | 128x128x64 | 1 | `gdl_fiber_0057_128x128x64_M10_th140_b99` | `fiber/128x128x64/fiber_0057.npy` | — | — | 10 | 140 |  | pv_cap |
| 10 | generated | 128x128x64 | 1 | `gdl_fiber_0058_128x128x64_M5_th120_b99` | `fiber/128x128x64/fiber_0058.npy` | — | — | 5 | 120 |  | pv_cap |
| 11 | generated | 128x128x64 | 1 | `gdl_fiber_0031_128x128x64_M20_th150_b99` | `fiber/128x128x64/fiber_0031.npy` | — | — | 20 | 150 |  | pv_cap |
| 12 | generated | 128x128x64 | 1 | `gdl_fiber_0034_128x128x64_M1_th130_b99` | `fiber/128x128x64/fiber_0034.npy` | — | — | 1 | 130 |  | bt_cap |
| 13 | generated | 128x128x64 | 1 | `gdl_fiber_0043_128x128x64_M20_th140_b99` | `fiber/128x128x64/fiber_0043.npy` | — | — | 20 | 140 |  | pv_cap |
| 14 | generated | 128x128x64 | 1 | `gdl_fiber_0059_128x128x64_M10_th150_b99` | `fiber/128x128x64/fiber_0059.npy` | — | — | 10 | 150 |  | pv_cap |
| 15 | generated | 128x128x64 | 1 | `gdl_fiber_0039_128x128x64_M5_th130_b99` | `fiber/128x128x64/fiber_0039.npy` | — | — | 5 | 130 |  | pv_cap |
| 16 | generated | 128x128x64 | 1 | `gdl_fiber_0055_128x128x64_M1_th120_b99` | `fiber/128x128x64/fiber_0055.npy` | — | — | 1 | 120 |  | bt_cap |
| 17 | generated | 128x128x64 | 1 | `gdl_fiber_0006_128x128x64_M5_th150_b99` | `fiber/128x128x64/fiber_0006.npy` | — | — | 5 | 150 |  | bt_cap |
| 18 | generated | 128x128x64 | 1 | `gdl_fiber_0030_128x128x64_M10_th130_b99` | `fiber/128x128x64/fiber_0030.npy` | — | — | 10 | 130 |  | pv_cap |
| 19 | generated | 128x128x64 | 1 | `gdl_fiber_0018_128x128x64_M1_th140_b99` | `fiber/128x128x64/fiber_0018.npy` | — | — | 1 | 140 |  | pv_cap |
| 20 | generated | 128x128x64 | 1 | `gdl_fiber_0020_128x128x64_M20_th120_b99` | `fiber/128x128x64/fiber_0020.npy` | — | — | 20 | 120 |  | bt_cap |
| 21 | generated | 128x128x64 | 1 | `gdl_fiber_0022_128x128x64_M5_th130_b99` | `fiber/128x128x64/fiber_0022.npy` | — | — | 5 | 130 |  | bt_cap |
| 22 | generated | 128x128x64 | 1 | `gdl_fiber_0019_128x128x64_M10_th140_b99` | `fiber/128x128x64/fiber_0019.npy` | — | — | 10 | 140 |  | — |
| 23 | generated | 128x128x64 | 1 | `gdl_fiber_0038_128x128x64_M20_th150_b99` | `fiber/128x128x64/fiber_0038.npy` | — | — | 20 | 150 |  | pv_cap |
| 24 | generated | 128x128x64 | 1 | `gdl_fiber_0029_128x128x64_M1_th120_b99` | `fiber/128x128x64/fiber_0029.npy` | — | — | 1 | 120 |  | pv_cap |
| 25 | generated | 128x128x64 | 1 | `gdl_fiber_0007_128x128x64_M5_th140_b99` | `fiber/128x128x64/fiber_0007.npy` | — | — | 5 | 140 |  | bt_cap |
| 26 | generated | 128x128x64 | 1 | `gdl_fiber_0028_128x128x64_M1_th150_b99` | `fiber/128x128x64/fiber_0028.npy` | — | — | 1 | 150 |  | bt_cap |
| 27 | generated | 128x128x64 | 1 | `gdl_fiber_0060_128x128x64_M20_th130_b99` | `fiber/128x128x64/fiber_0060.npy` | — | — | 20 | 130 |  | pv_cap |
| 28 | generated | 128x128x64 | 1 | `gdl_fiber_0017_128x128x64_M10_th120_b99` | `fiber/128x128x64/fiber_0017.npy` | — | — | 10 | 120 |  | bt_cap |
| 29 | generated | 128x128x64 | 1 | `gdl_fiber_0037_128x128x64_M1_th130_b99` | `fiber/128x128x64/fiber_0037.npy` | — | — | 1 | 130 |  | pv_cap |
| 30 | generated | 128x128x64 | 1 | `gdl_fiber_0004_128x128x64_M5_th120_b99` | `fiber/128x128x64/fiber_0004.npy` | — | — | 5 | 120 |  | pv_cap |
| 31 | generated | 128x128x64 | 1 | `gdl_fiber_0047_128x128x64_M20_th140_b99` | `fiber/128x128x64/fiber_0047.npy` | — | — | 20 | 140 |  | pv_cap |
| 32 | generated | 128x128x64 | 1 | `gdl_fiber_0042_128x128x64_M10_th150_b99` | `fiber/128x128x64/fiber_0042.npy` | — | — | 10 | 150 |  | bt_cap |
| 33 | generated | 128x128x64 | 1 | `gdl_fiber_0016_128x128x64_M1_th150_b99` | `fiber/128x128x64/fiber_0016.npy` | — | — | 1 | 150 |  | bt_cap |
| 34 | generated | 128x128x64 | 1 | `gdl_fiber_0052_128x128x64_M5_th130_b99` | `fiber/128x128x64/fiber_0052.npy` | — | — | 5 | 130 |  | pv_cap |
| 35 | generated | 128x128x64 | 1 | `gdl_fiber_0005_128x128x64_M10_th120_b99` | `fiber/128x128x64/fiber_0005.npy` | — | — | 10 | 120 |  | pv_cap |
| 36 | generated | 128x128x64 | 1 | `gdl_fiber_0013_128x128x64_M20_th140_b99` | `fiber/128x128x64/fiber_0013.npy` | — | — | 20 | 140 |  | bt_cap |
| 37 | generated | 128x128x64 | 1 | `gdl_fiber_0024_128x128x64_M5_th150_b99` | `fiber/128x128x64/fiber_0024.npy` | — | — | 5 | 150 |  | bt_cap |
| 38 | generated | 128x128x64 | 1 | `gdl_fiber_0032_128x128x64_M10_th140_b99` | `fiber/128x128x64/fiber_0032.npy` | — | — | 10 | 140 |  | pv_cap |
| 39 | generated | 128x128x64 | 1 | `gdl_fiber_0011_128x128x64_M1_th130_b99` | `fiber/128x128x64/fiber_0011.npy` | — | — | 1 | 130 |  | bt_cap |
| 40 | generated | 128x128x64 | 1 | `gdl_fiber_0045_128x128x64_M20_th120_b99` | `fiber/128x128x64/fiber_0045.npy` | — | — | 20 | 120 |  | pv_cap |
| 41 | generated | 128x128x64 | 1 | `gdl_fiber_0012_128x128x64_M1_th140_b99` | `fiber/128x128x64/fiber_0012.npy` | — | — | 1 | 140 |  | bt_cap |
| 42 | generated | 128x128x64 | 1 | `gdl_fiber_0053_128x128x64_M20_th130_b99` | `fiber/128x128x64/fiber_0053.npy` | — | — | 20 | 130 |  | bt_cap |
| 43 | generated | 128x128x64 | 1 | `gdl_fiber_0023_128x128x64_M5_th120_b99` | `fiber/128x128x64/fiber_0023.npy` | — | — | 5 | 120 |  | pv_cap |
| 44 | generated | 128x128x64 | 1 | `gdl_fiber_0021_128x128x64_M10_th150_b99` | `fiber/128x128x64/fiber_0021.npy` | — | — | 10 | 150 |  | bt_cap |
| 45 | generated | 128x128x64 | 1 | `gdl_fiber_0014_128x128x64_M5_th140_b99` | `fiber/128x128x64/fiber_0014.npy` | — | — | 5 | 140 |  | pv_cap |
| 46 | generated | 128x128x64 | 1 | `gdl_fiber_0035_128x128x64_M10_th130_b99` | `fiber/128x128x64/fiber_0035.npy` | — | — | 10 | 130 |  | pv_cap |
| 47 | generated | 128x128x64 | 1 | `gdl_fiber_0003_128x128x64_M1_th120_b99` | `fiber/128x128x64/fiber_0003.npy` | — | — | 1 | 120 |  | pv_cap |
| 48 | generated | 128x128x64 | 1 | `gdl_fiber_0046_128x128x64_M20_th150_b99` | `fiber/128x128x64/fiber_0046.npy` | — | — | 20 | 150 |  | pv_cap |
| 49 | generated | 128x128x64 | 1 | `gdl_fiber_0009_128x128x64_M10_th130_b99` | `fiber/128x128x64/fiber_0009.npy` | — | — | 10 | 130 |  | pv_cap |
| 50 | generated | 128x128x64 | 1 | `gdl_fiber_0033_128x128x64_M1_th150_b99` | `fiber/128x128x64/fiber_0033.npy` | — | — | 1 | 150 |  | bt_cap |
| 51 | generated | 128x128x64 | 1 | `gdl_fiber_0027_128x128x64_M5_th140_b99` | `fiber/128x128x64/fiber_0027.npy` | — | — | 5 | 140 |  | pv_cap |
| 52 | generated | 128x128x64 | 1 | `gdl_fiber_0049_128x128x64_M20_th120_b99` | `fiber/128x128x64/fiber_0049.npy` | — | — | 20 | 120 |  | pv_cap |
| 53 | generated | 128x128x64 | 1 | `gdl_fiber_0036_128x128x64_M10_th120_b99` | `fiber/128x128x64/fiber_0036.npy` | — | — | 10 | 120 |  | pv_cap |
| 54 | generated | 128x128x64 | 1 | `gdl_fiber_0001_128x128x64_M20_th130_b99` | `fiber/128x128x64/fiber_0001.npy` | — | — | 20 | 130 |  | bt_cap |
| 55 | generated | 128x128x64 | 1 | `gdl_fiber_0048_128x128x64_M1_th140_b99` | `fiber/128x128x64/fiber_0048.npy` | — | — | 1 | 140 |  | bt_cap |
| 56 | generated | 128x128x64 | 1 | `gdl_fiber_0010_128x128x64_M5_th150_b99` | `fiber/128x128x64/fiber_0010.npy` | — | — | 5 | 150 |  | pv_cap |
| 57 | generated | 128x128x64 | 1 | `gdl_fiber_0040_128x128x64_M10_th140_b99` | `fiber/128x128x64/fiber_0040.npy` | — | — | 10 | 140 |  | bt_cap |
| 58 | generated | 128x128x64 | 1 | `gdl_fiber_0000_128x128x64_M5_th130_b99` | `fiber/128x128x64/fiber_0000.npy` | — | — | 5 | 130 |  | pv_cap |
| 59 | generated | 128x128x64 | 1 | `gdl_fiber_0062_128x128x64_M1_th120_b99` | `fiber/128x128x64/fiber_0062.npy` | — | — | 1 | 120 |  | pv_cap |
| 60 | generated | 128x128x64 | 1 | `gdl_fiber_0025_128x128x64_M20_th150_b99` | `fiber/128x128x64/fiber_0025.npy` | — | — | 20 | 150 |  | bt_cap |
| 61 | generated | 128x128x64 | 1 | `gdl_fiber_0041_128x128x64_M10_th150_b99` | `fiber/128x128x64/fiber_0041.npy` | — | — | 10 | 150 |  | bt_cap |
| 62 | generated | 128x128x64 | 1 | `gdl_fiber_0056_128x128x64_M20_th140_b99` | `fiber/128x128x64/fiber_0056.npy` | — | — | 20 | 140 |  | pv_cap |
| 63 | generated | 128x128x64 | 1 | `gdl_fiber_0063_128x128x64_M1_th130_b99` | `fiber/128x128x64/fiber_0063.npy` | — | — | 1 | 130 |  | pv_cap |
| 64 | generated | 128x128x64 | 1 | `gdl_fiber_0002_128x128x64_M5_th120_b99` | `fiber/128x128x64/fiber_0002.npy` | — | — | 5 | 120 |  | bt_cap |
| 65 | generated | 256x256x128 | 4 | `gdl_fiber_0018_256x256x128_M20_th120_b99` | `fiber/256x256x128/fiber_0018.npy` | — | — | 20 | 120 |  | pv_cap |
| 66 | generated | 256x256x128 | 4 | `gdl_fiber_0005_256x256x128_M10_th150_b99` | `fiber/256x256x128/fiber_0005.npy` | — | — | 10 | 150 |  | bt_cap |
| 67 | generated | 256x256x128 | 4 | `gdl_fiber_0035_256x256x128_M1_th120_b99` | `fiber/256x256x128/fiber_0035.npy` | — | — | 1 | 120 |  | bt_cap |
| 68 | generated | 256x256x128 | 4 | `gdl_fiber_0030_256x256x128_M5_th150_b99` | `fiber/256x256x128/fiber_0030.npy` | — | — | 5 | 150 |  | bt_cap |
| 69 | generated | 256x256x128 | 4 | `gdl_fiber_0055_256x256x128_M1_th150_b99` | `fiber/256x256x128/fiber_0055.npy` | — | — | 1 | 150 |  | bt_cap |
| 70 | generated | 256x256x128 | 4 | `gdl_fiber_0029_256x256x128_M10_th120_b99` | `fiber/256x256x128/fiber_0029.npy` | — | — | 10 | 120 |  | pv_cap |
| 71 | generated | 256x256x128 | 4 | `gdl_fiber_0063_256x256x128_M5_th120_b99` | `fiber/256x256x128/fiber_0063.npy` | — | — | 5 | 120 |  | bt_cap |
| 72 | generated | 256x256x128 | 4 | `gdl_fiber_0042_256x256x128_M20_th150_b99` | `fiber/256x256x128/fiber_0042.npy` | — | — | 20 | 150 |  | bt_cap |
| 73 | ct | 128x128x64 | 1 | `gdl_gdl_ct_0062_128x128x64_M10_th140_b99` | `gdl_ct/128x128x64/gdl_0062.npy` | — | — | 10 | 140 |  | pv_cap |
| 74 | ct | 128x128x64 | 1 | `gdl_gdl_ct_20_0027_128x128x64_M20_th120_b99` | `gdl_ct_20/128x128x64/gdl_0027.npy` | — | — | 20 | 120 |  | bt_cap |
| 75 | ct | 128x128x64 | 1 | `gdl_gdl_ct_0003_128x128x64_M5_th150_b99` | `gdl_ct/128x128x64/gdl_0003.npy` | — | — | 5 | 150 |  | pv_cap |
| 76 | ct | 128x128x64 | 1 | `gdl_gdl_ct_0024_128x128x64_M1_th130_b99` | `gdl_ct/128x128x64/gdl_0024.npy` | — | — | 1 | 130 |  | pv_cap |
| 77 | ct | 128x128x64 | 1 | `gdl_gdl_ct_40_0048_128x128x64_M5_th130_b99` | `gdl_ct_40/128x128x64/gdl_0048.npy` | — | — | 5 | 130 |  | bt_cap |
| 78 | ct | 128x128x64 | 1 | `gdl_gdl_ct_40_0012_128x128x64_M10_th150_b99` | `gdl_ct_40/128x128x64/gdl_0012.npy` | — | — | 10 | 150 |  | bt_cap |
| 79 | ct | 128x128x64 | 1 | `gdl_gdl_ct_40_0023_128x128x64_M1_th120_b99` | `gdl_ct_40/128x128x64/gdl_0023.npy` | — | — | 1 | 120 |  | pv_cap |
| 80 | ct | 128x128x64 | 1 | `gdl_gdl_ct_40_0005_128x128x64_M20_th140_b99` | `gdl_ct_40/128x128x64/gdl_0005.npy` | — | — | 20 | 140 |  | bt_cap |
| 81 | ct | 128x128x64 | 1 | `gdl_gdl_ct_20_0018_128x128x64_M20_th150_b99` | `gdl_ct_20/128x128x64/gdl_0018.npy` | — | — | 20 | 150 |  | pv_cap |
| 82 | ct | 128x128x64 | 1 | `gdl_gdl_ct_40_0022_128x128x64_M5_th120_b99` | `gdl_ct_40/128x128x64/gdl_0022.npy` | — | — | 5 | 120 |  | bt_cap |
| 83 | ct | 128x128x64 | 1 | `gdl_gdl_ct_40_0009_128x128x64_M1_th140_b99` | `gdl_ct_40/128x128x64/gdl_0009.npy` | — | — | 1 | 140 |  | pv_cap |
| 84 | ct | 128x128x64 | 1 | `gdl_gdl_ct_0013_128x128x64_M10_th130_b99` | `gdl_ct/128x128x64/gdl_0013.npy` | — | — | 10 | 130 |  | pv_cap |
| 85 | ct | 128x128x64 | 1 | `gdl_gdl_ct_0015_128x128x64_M10_th120_b99` | `gdl_ct/128x128x64/gdl_0015.npy` | — | — | 10 | 120 |  | pv_cap |
| 86 | ct | 128x128x64 | 1 | `gdl_gdl_ct_20_0046_128x128x64_M5_th140_b99` | `gdl_ct_20/128x128x64/gdl_0046.npy` | — | — | 5 | 140 |  | pv_cap |
| 87 | ct | 128x128x64 | 1 | `gdl_gdl_ct_20_0053_128x128x64_M20_th130_b99` | `gdl_ct_20/128x128x64/gdl_0053.npy` | — | — | 20 | 130 |  | pv_cap |
| 88 | ct | 128x128x64 | 1 | `gdl_gdl_ct_40_0018_128x128x64_M1_th150_b99` | `gdl_ct_40/128x128x64/gdl_0018.npy` | — | — | 1 | 150 |  | bt_cap |
| 89 | ct | 128x128x64 | 1 | `gdl_gdl_ct_0054_128x128x64_M20_th150_b99` | `gdl_ct/128x128x64/gdl_0054.npy` | — | — | 20 | 150 |  | pv_cap |
| 90 | ct | 128x128x64 | 1 | `gdl_gdl_ct_40_0056_128x128x64_M1_th140_b99` | `gdl_ct_40/128x128x64/gdl_0056.npy` | — | — | 1 | 140 |  | bt_cap |
| 91 | ct | 128x128x64 | 1 | `gdl_gdl_ct_0025_128x128x64_M5_th120_b99` | `gdl_ct/128x128x64/gdl_0025.npy` | — | — | 5 | 120 |  | bt_cap |
| 92 | ct | 128x128x64 | 1 | `gdl_gdl_ct_40_0027_128x128x64_M10_th130_b99` | `gdl_ct_40/128x128x64/gdl_0027.npy` | — | — | 10 | 130 |  | bt_cap |
| 93 | ct | 128x128x64 | 1 | `gdl_gdl_ct_20_0020_128x128x64_M5_th150_b99` | `gdl_ct_20/128x128x64/gdl_0020.npy` | — | — | 5 | 150 |  | bt_cap |
| 94 | ct | 128x128x64 | 1 | `gdl_gdl_ct_0002_128x128x64_M20_th140_b99` | `gdl_ct/128x128x64/gdl_0002.npy` | — | — | 20 | 140 |  | pv_cap |
| 95 | ct | 128x128x64 | 1 | `gdl_gdl_ct_40_0014_128x128x64_M10_th120_b99` | `gdl_ct_40/128x128x64/gdl_0014.npy` | — | — | 10 | 120 |  | bt_cap |
| 96 | ct | 128x128x64 | 1 | `gdl_gdl_ct_40_0051_128x128x64_M1_th130_b99` | `gdl_ct_40/128x128x64/gdl_0051.npy` | — | — | 1 | 130 |  | bt_cap |
| 97 | ct | 128x128x64 | 1 | `gdl_gdl_ct_20_0006_128x128x64_M1_th120_b99` | `gdl_ct_20/128x128x64/gdl_0006.npy` | — | — | 1 | 120 |  | pv_cap |
| 98 | ct | 128x128x64 | 1 | `gdl_gdl_ct_0034_128x128x64_M5_th140_b99` | `gdl_ct/128x128x64/gdl_0034.npy` | — | — | 5 | 140 |  | bt_cap |
| 99 | ct | 128x128x64 | 1 | `gdl_gdl_ct_0017_128x128x64_M10_th150_b99` | `gdl_ct/128x128x64/gdl_0017.npy` | — | — | 10 | 150 |  | pv_cap |
| 100 | ct | 128x128x64 | 1 | `gdl_gdl_ct_20_0044_128x128x64_M20_th130_b99` | `gdl_ct_20/128x128x64/gdl_0044.npy` | — | — | 20 | 130 |  | bt_cap |
| 101 | ct | 128x128x64 | 1 | `gdl_gdl_ct_40_0020_128x128x64_M20_th120_b99` | `gdl_ct_40/128x128x64/gdl_0020.npy` | — | — | 20 | 120 |  | pv_cap |
| 102 | ct | 128x128x64 | 1 | `gdl_gdl_ct_20_0037_128x128x64_M10_th140_b99` | `gdl_ct_20/128x128x64/gdl_0037.npy` | — | — | 10 | 140 |  | bt_cap |
| 103 | ct | 128x128x64 | 1 | `gdl_gdl_ct_40_0057_128x128x64_M1_th150_b99` | `gdl_ct_40/128x128x64/gdl_0057.npy` | — | — | 1 | 150 |  | bt_cap |
| 104 | ct | 128x128x64 | 1 | `gdl_gdl_ct_20_0025_128x128x64_M5_th130_b99` | `gdl_ct_20/128x128x64/gdl_0025.npy` | — | — | 5 | 130 |  | pv_cap |
| 105 | ct | 128x128x64 | 1 | `gdl_gdl_ct_0047_128x128x64_M20_th140_b99` | `gdl_ct/128x128x64/gdl_0047.npy` | — | — | 20 | 140 |  | bt_cap |
| 106 | ct | 128x128x64 | 1 | `gdl_gdl_ct_40_0054_128x128x64_M1_th150_b99` | `gdl_ct_40/128x128x64/gdl_0054.npy` | — | — | 1 | 150 |  | bt_cap |
| 107 | ct | 128x128x64 | 1 | `gdl_gdl_ct_40_0053_128x128x64_M5_th120_b99` | `gdl_ct_40/128x128x64/gdl_0053.npy` | — | — | 5 | 120 |  | bt_cap |
| 108 | ct | 128x128x64 | 1 | `gdl_gdl_ct_0014_128x128x64_M10_th130_b99` | `gdl_ct/128x128x64/gdl_0014.npy` | — | — | 10 | 130 |  | bt_cap |
| 109 | ct | 128x128x64 | 1 | `gdl_gdl_ct_0061_128x128x64_M20_th120_b99` | `gdl_ct/128x128x64/gdl_0061.npy` | — | — | 20 | 120 |  | bt_cap |
| 110 | ct | 128x128x64 | 1 | `gdl_gdl_ct_0040_128x128x64_M5_th130_b99` | `gdl_ct/128x128x64/gdl_0040.npy` | — | — | 5 | 130 |  | bt_cap |
| 111 | ct | 128x128x64 | 1 | `gdl_gdl_ct_40_0004_128x128x64_M1_th140_b99` | `gdl_ct_40/128x128x64/gdl_0004.npy` | — | — | 1 | 140 |  | bt_cap |
| 112 | ct | 128x128x64 | 1 | `gdl_gdl_ct_0020_128x128x64_M10_th150_b99` | `gdl_ct/128x128x64/gdl_0020.npy` | — | — | 10 | 150 |  | pv_cap |
| 113 | ct | 128x128x64 | 1 | `gdl_gdl_ct_20_0012_128x128x64_M20_th130_b99` | `gdl_ct_20/128x128x64/gdl_0012.npy` | — | — | 20 | 130 |  | bt_cap |
| 114 | ct | 128x128x64 | 1 | `gdl_gdl_ct_40_0010_128x128x64_M10_th120_b99` | `gdl_ct_40/128x128x64/gdl_0010.npy` | — | — | 10 | 120 |  | pv_cap |
| 115 | ct | 128x128x64 | 1 | `gdl_gdl_ct_20_0051_128x128x64_M5_th150_b99` | `gdl_ct_20/128x128x64/gdl_0051.npy` | — | — | 5 | 150 |  | bt_cap |
| 116 | ct | 128x128x64 | 1 | `gdl_gdl_ct_20_0013_128x128x64_M1_th120_b99` | `gdl_ct_20/128x128x64/gdl_0013.npy` | — | — | 1 | 120 |  | bt_cap |
| 117 | ct | 128x128x64 | 1 | `gdl_gdl_ct_20_0039_128x128x64_M5_th140_b99` | `gdl_ct_20/128x128x64/gdl_0039.npy` | — | — | 5 | 140 |  | pv_cap |
| 118 | ct | 128x128x64 | 1 | `gdl_gdl_ct_0043_128x128x64_M1_th130_b99` | `gdl_ct/128x128x64/gdl_0043.npy` | — | — | 1 | 130 |  | bt_cap |
| 119 | ct | 128x128x64 | 1 | `gdl_gdl_ct_0057_128x128x64_M10_th140_b99` | `gdl_ct/128x128x64/gdl_0057.npy` | — | — | 10 | 140 |  | bt_cap |
| 120 | ct | 128x128x64 | 1 | `gdl_gdl_ct_0036_128x128x64_M20_th150_b99` | `gdl_ct/128x128x64/gdl_0036.npy` | — | — | 20 | 150 |  | pv_cap |
| 121 | ct | 128x128x64 | 1 | `gdl_gdl_ct_0045_128x128x64_M1_th120_b99` | `gdl_ct/128x128x64/gdl_0045.npy` | — | — | 1 | 120 |  | bt_cap |
| 122 | ct | 128x128x64 | 1 | `gdl_gdl_ct_0004_128x128x64_M10_th150_b99` | `gdl_ct/128x128x64/gdl_0004.npy` | — | — | 10 | 150 |  | pv_cap |
| 123 | ct | 128x128x64 | 1 | `gdl_gdl_ct_20_0043_128x128x64_M5_th140_b99` | `gdl_ct_20/128x128x64/gdl_0043.npy` | — | — | 5 | 140 |  | bt_cap |
| 124 | ct | 128x128x64 | 1 | `gdl_gdl_ct_0052_128x128x64_M20_th130_b99` | `gdl_ct/128x128x64/gdl_0052.npy` | — | — | 20 | 130 |  | pv_cap |
| 125 | ct | 128x128x64 | 1 | `gdl_gdl_ct_20_0058_128x128x64_M10_th140_b99` | `gdl_ct_20/128x128x64/gdl_0058.npy` | — | — | 10 | 140 |  | bt_cap |
| 126 | ct | 128x128x64 | 1 | `gdl_gdl_ct_40_0016_128x128x64_M5_th130_b99` | `gdl_ct_40/128x128x64/gdl_0016.npy` | — | — | 5 | 130 |  | bt_cap |
| 127 | ct | 128x128x64 | 1 | `gdl_gdl_ct_40_0061_128x128x64_M1_th150_b99` | `gdl_ct_40/128x128x64/gdl_0061.npy` | — | — | 1 | 150 |  | pv_cap |
| 128 | ct | 128x128x64 | 1 | `gdl_gdl_ct_40_0013_128x128x64_M20_th120_b99` | `gdl_ct_40/128x128x64/gdl_0013.npy` | — | — | 20 | 120 |  | bt_cap |
| 129 | ct | 128x128x64 | 1 | `gdl_gdl_ct_40_0033_128x128x64_M1_th140_b99` | `gdl_ct_40/128x128x64/gdl_0033.npy` | — | — | 1 | 140 |  | bt_cap |
| 130 | ct | 128x128x64 | 1 | `gdl_gdl_ct_0035_128x128x64_M10_th130_b99` | `gdl_ct/128x128x64/gdl_0035.npy` | — | — | 10 | 130 |  | bt_cap |
| 131 | ct | 128x128x64 | 1 | `gdl_gdl_ct_20_0003_128x128x64_M20_th150_b99` | `gdl_ct_20/128x128x64/gdl_0003.npy` | — | — | 20 | 150 |  | bt_cap |
| 132 | ct | 128x128x64 | 1 | `gdl_gdl_ct_20_0031_128x128x64_M5_th120_b99` | `gdl_ct_20/128x128x64/gdl_0031.npy` | — | — | 5 | 120 |  | bt_cap |
| 133 | ct | 128x128x64 | 1 | `gdl_gdl_ct_20_0061_128x128x64_M20_th140_b99` | `gdl_ct_20/128x128x64/gdl_0061.npy` | — | — | 20 | 140 |  | pv_cap |
| 134 | ct | 128x128x64 | 1 | `gdl_gdl_ct_20_0016_128x128x64_M5_th150_b99` | `gdl_ct_20/128x128x64/gdl_0016.npy` | — | — | 5 | 150 |  | bt_cap |
| 135 | ct | 128x128x64 | 1 | `gdl_gdl_ct_20_0001_128x128x64_M10_th120_b99` | `gdl_ct_20/128x128x64/gdl_0001.npy` | — | — | 10 | 120 |  | bt_cap |
| 136 | ct | 128x128x64 | 1 | `gdl_gdl_ct_20_0007_128x128x64_M1_th130_b99` | `gdl_ct_20/128x128x64/gdl_0007.npy` | — | — | 1 | 130 |  | bt_cap |
| 137 | ct | 256x256x128 | 4 | `gdl_gdl_ct_0032_256x256x128_M10_th150_b99` | `gdl_ct/256x256x128/gdl_0032.npy` | — | — | 10 | 150 |  | bt_cap |
| 138 | ct | 256x256x128 | 4 | `gdl_gdl_ct_20_0046_256x256x128_M20_th120_b99` | `gdl_ct_20/256x256x128/gdl_0046.npy` | — | — | 20 | 120 |  | bt_cap |
| 139 | ct | 256x256x128 | 4 | `gdl_gdl_ct_40_0003_256x256x128_M5_th150_b99` | `gdl_ct_40/256x256x128/gdl_0003.npy` | — | — | 5 | 150 |  | bt_cap |
| 140 | ct | 256x256x128 | 4 | `gdl_gdl_ct_40_0036_256x256x128_M1_th120_b99` | `gdl_ct_40/256x256x128/gdl_0036.npy` | — | — | 1 | 120 |  | bt_cap |
| 141 | ct | 256x256x128 | 4 | `gdl_gdl_ct_0053_256x256x128_M5_th120_b99` | `gdl_ct/256x256x128/gdl_0053.npy` | — | — | 5 | 120 |  | pv_cap |
| 142 | ct | 256x256x128 | 4 | `gdl_gdl_ct_20_0053_256x256x128_M20_th150_b99` | `gdl_ct_20/256x256x128/gdl_0053.npy` | — | — | 20 | 150 |  | bt_cap |
| 143 | ct | 256x256x128 | 4 | `gdl_gdl_ct_20_0038_256x256x128_M10_th120_b99` | `gdl_ct_20/256x256x128/gdl_0038.npy` | — | — | 10 | 120 |  | pv_cap |
| 144 | ct | 256x256x128 | 4 | `gdl_gdl_ct_0020_256x256x128_M1_th150_b99` | `gdl_ct/256x256x128/gdl_0020.npy` | — | — | 1 | 150 |  | bt_cap |

## Manifest — extra_beta_1g, the frozen 96-run expansion

Seed 20260830, drawn after (and never touching) the base draw above: 16 more 1-GPU runs per (case, family) under the same protocol — even source split, percolation QC, throat screen, and every (M, theta) cell of the 4x4 grid exactly once per set (joint base+expansion coverage: 5x per cell). A rock the base already ran is only repeated at a different (M, theta), flagged `reused` — the GDL generated set is all reuse, because the base draw uses all 64 fiber samples.

| Case | Family | Sources (even split) | Domain | GPUs | Runs | flagged |
|---|---|---|---|---:|---:|---:|
| drainage | generated | blob, poly, sphere | 128 | 1 | 16 | — |
| drainage | ct | bentheimer, buffberea, castlegate | 128 | 1 | 16 | 8 |
| trapping | generated | blob, poly, sphere | 128 | 1 | 16 | — |
| trapping | ct | bentheimer, buffberea, castlegate | 128 | 1 | 16 | 9 |
| GDL | generated | fiber | 128x128x64 | 1 | 16 | 16 |
| GDL | ct | gdl_ct, gdl_ct_20, gdl_ct_40 | 128x128x64 | 1 | 16 | — |
| **total** | | | | | **96** | **33** |


## extra_beta_1g — drainage — 32 runs

**32 runs** — 32 finished, 0 running.

| # | Family | Domain | GPUs | Run | Geometry | r_c | margin | M | theta | Flags | Status |
|---:|---|---|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | generated | 128 | 1 | `drain_poly128_0020_128_M0.05_th130_b99` | `poly/128/poly128_0020.npy` | 5.18 | 1.40 | 0.05 | 130 |  | dp_cap |
| 2 | generated | 128 | 1 | `drain_poly128_0055_128_M0.2_th120_b99` | `poly/128/poly128_0055.npy` | 5.64 | 1.96 | 0.2 | 120 |  | pv_cap |
| 3 | generated | 128 | 1 | `drain_blob128_0057_128_M1_th150_b99` | `blob/128/blob128_0057.npy` | 6.99 | 1.40 | 1 | 150 |  | dp_cap |
| 4 | generated | 128 | 1 | `drain_poly128_0006_128_M0.1_th140_b99` | `poly/128/poly128_0006.npy` | 5.83 | 1.32 | 0.1 | 140 |  | dp_cap |
| 5 | generated | 128 | 1 | `drain_sphere128_0035_128_M1_th130_b99` | `sphere/128/sphere128_0035.npy` | 5.19 | 1.40 | 1 | 130 |  | pv_cap |
| 6 | generated | 128 | 1 | `drain_poly128_0062_128_M0.2_th140_b99` | `poly/128/poly128_0062.npy` | 5.83 | 1.32 | 0.2 | 140 |  | dp_cap |
| 7 | generated | 128 | 1 | `drain_blob128_0009_128_M0.05_th120_b99` | `blob/128/blob128_0009.npy` | 7.21 | 2.50 | 0.05 | 120 |  | dp_cap |
| 8 | generated | 128 | 1 | `drain_blob128_0013_128_M0.1_th150_b99` | `blob/128/blob128_0013.npy` | 6.70 | 1.34 | 0.1 | 150 |  | dp_cap |
| 9 | generated | 128 | 1 | `drain_blob128_0032_128_M0.2_th150_b99` | `blob/128/blob128_0032.npy` | 7.06 | 1.41 | 0.2 | 150 |  | dp_cap |
| 10 | generated | 128 | 1 | `drain_sphere128_0024_128_M0.05_th140_b99` | `sphere/128/sphere128_0024.npy` | 5.90 | 1.33 | 0.05 | 140 |  | dp_cap |
| 11 | generated | 128 | 1 | `drain_sphere128_0019_128_M0.1_th120_b99` | `sphere/128/sphere128_0019.npy` | 6.99 | 2.42 | 0.1 | 120 |  | pv_cap |
| 12 | generated | 128 | 1 | `drain_sphere128_0050_128_M1_th140_b99` | `sphere/128/sphere128_0050.npy` | 7.79 | 1.76 | 1 | 140 |  | pv_cap |
| 13 | generated | 128 | 1 | `drain_blob128_0040_128_M0.2_th130_b99` | `blob/128/blob128_0040.npy` | 8.05 | 2.17 | 0.2 | 130 |  | pv_cap |
| 14 | generated | 128 | 1 | `drain_sphere128_0010_128_M1_th120_b99` | `sphere/128/sphere128_0010.npy` | 7.87 | 2.72 | 1 | 120 |  | pv_cap |
| 15 | generated | 128 | 1 | `drain_poly128_0057_128_M0.05_th150_b99` | `poly/128/poly128_0057.npy` | 8.83 | 1.77 | 0.05 | 150 |  | pv_cap |
| 16 | generated | 128 | 1 | `drain_blob128_0037_128_M0.1_th130_b99` | `blob/128/blob128_0037.npy` | 6.62 | 1.78 | 0.1 | 130 |  | dp_cap |
| 17 | ct | 128 | 1 | `drain_bent128_0041_128_M1_th140_b99` | `bentheimer/128/bent128_0041.npy` | 6.39 | 1.45 | 1 | 140 |  | dp_cap |
| 18 | ct | 128 | 1 | `drain_bent128_0005_128_M0.05_th150_b99` | `bentheimer/128/bent128_0005.npy` | 7.00 | 1.40 | 0.05 | 150 |  | dp_cap |
| 19 | ct | 128 | 1 | `drain_castle128_0021_128_M0.1_th120_b99` | `castlegate/128/castle128_0021.npy` | 5.83 | 2.02 | 0.1 | 120 | reused | dp_cap |
| 20 | ct | 128 | 1 | `drain_buff128_0049_128_M0.2_th130_b99` | `buffberea/128/buff128_0049.npy` | 5.99 | 1.61 | 0.2 | 130 |  | dp_cap |
| 21 | ct | 128 | 1 | `drain_buff128_0039_128_M0.05_th120_b99` | `buffberea/128/buff128_0039.npy` | 5.82 | 2.02 | 0.05 | 120 |  | dp_cap |
| 22 | ct | 128 | 1 | `drain_bent128_0018_128_M1_th130_b99` | `bentheimer/128/bent128_0018.npy` | 5.74 | 1.55 | 1 | 130 |  | pv_cap |
| 23 | ct | 128 | 1 | `drain_buff128_0022_128_M0.1_th150_b99` | `buffberea/128/buff128_0022.npy` | 7.55 | 1.51 | 0.1 | 150 | reused | pv_cap |
| 24 | ct | 128 | 1 | `drain_castle128_0006_128_M0.2_th140_b99` | `castlegate/128/castle128_0006.npy` | 6.70 | 1.52 | 0.2 | 140 | reused | dp_cap |
| 25 | ct | 128 | 1 | `drain_buff128_0029_128_M0.2_th120_b99` | `buffberea/128/buff128_0029.npy` | 7.99 | 2.77 | 0.2 | 120 | reused | pv_cap |
| 26 | ct | 128 | 1 | `drain_castle128_0049_128_M0.1_th130_b99` | `castlegate/128/castle128_0049.npy` | 6.62 | 1.78 | 0.1 | 130 | reused | dp_cap |
| 27 | ct | 128 | 1 | `drain_bent128_0030_128_M0.05_th140_b99` | `bentheimer/128/bent128_0030.npy` | 5.83 | 1.32 | 0.05 | 140 |  | pv_cap |
| 28 | ct | 128 | 1 | `drain_castle128_0036_128_M1_th150_b99` | `castlegate/128/castle128_0036.npy` | 6.92 | 1.38 | 1 | 150 | reused | dp_cap |
| 29 | ct | 128 | 1 | `drain_bent128_0028_128_M0.05_th130_b99` | `bentheimer/128/bent128_0028.npy` | 5.38 | 1.45 | 0.05 | 130 |  | dp_cap |
| 30 | ct | 128 | 1 | `drain_castle128_0012_128_M1_th120_b99` | `castlegate/128/castle128_0012.npy` | 5.64 | 1.96 | 1 | 120 | reused | dp_cap |
| 31 | ct | 128 | 1 | `drain_bent128_0010_128_M0.1_th140_b99` | `bentheimer/128/bent128_0010.npy` | 6.31 | 1.43 | 0.1 | 140 |  | dp_cap |
| 32 | ct | 128 | 1 | `drain_buff128_0044_128_M0.2_th150_b99` | `buffberea/128/buff128_0044.npy` | 6.99 | 1.40 | 0.2 | 150 | reused | dp_cap |

## extra_beta_1g — trapping — 32 runs

**32 runs** — 32 finished, 0 running.

| # | Family | Domain | GPUs | Run | Geometry | r_c | margin | M | theta | Flags | Status |
|---:|---|---|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | generated | 128 | 1 | `trap_sphere128_0011_128_M0.05_th130_b99` | `sphere/128/sphere128_0011.npy` | 5.99 | 1.62 | 0.05 | 130 |  | red_immobile |
| 2 | generated | 128 | 1 | `trap_blob128_0061_128_M1_th120_b99` | `blob/128/blob128_0061.npy` | 6.99 | 2.42 | 1 | 120 |  | red_immobile |
| 3 | generated | 128 | 1 | `trap_blob128_0003_128_M0.1_th140_b99` | `blob/128/blob128_0003.npy` | 5.99 | 1.35 | 0.1 | 140 |  | red_immobile |
| 4 | generated | 128 | 1 | `trap_sphere128_0060_128_M0.2_th150_b99` | `sphere/128/sphere128_0060.npy` | 6.99 | 1.40 | 0.2 | 150 |  | red_immobile |
| 5 | generated | 128 | 1 | `trap_poly128_0022_128_M0.05_th150_b99` | `poly/128/poly128_0022.npy` | 6.55 | 1.31 | 0.05 | 150 |  | red_immobile |
| 6 | generated | 128 | 1 | `trap_blob128_0040_128_M1_th140_b99` | `blob/128/blob128_0040.npy` | 8.05 | 1.82 | 1 | 140 |  | red_immobile |
| 7 | generated | 128 | 1 | `trap_sphere128_0021_128_M0.1_th130_b99` | `sphere/128/sphere128_0021.npy` | 7.54 | 2.03 | 0.1 | 130 |  | red_immobile |
| 8 | generated | 128 | 1 | `trap_sphere128_0056_128_M0.2_th120_b99` | `sphere/128/sphere128_0056.npy` | 5.66 | 1.96 | 0.2 | 120 |  | red_immobile |
| 9 | generated | 128 | 1 | `trap_sphere128_0050_128_M0.05_th120_b99` | `sphere/128/sphere128_0050.npy` | 7.79 | 2.70 | 0.05 | 120 |  | red_immobile |
| 10 | generated | 128 | 1 | `trap_blob128_0057_128_M0.1_th150_b99` | `blob/128/blob128_0057.npy` | 6.99 | 1.40 | 0.1 | 150 |  | red_immobile |
| 11 | generated | 128 | 1 | `trap_poly128_0033_128_M0.2_th140_b99` | `poly/128/poly128_0033.npy` | 5.82 | 1.32 | 0.2 | 140 |  | red_immobile |
| 12 | generated | 128 | 1 | `trap_poly128_0000_128_M1_th130_b99` | `poly/128/poly128_0000.npy` | 5.65 | 1.52 | 1 | 130 |  | red_immobile |
| 13 | generated | 128 | 1 | `trap_poly128_0045_128_M0.2_th130_b99` | `poly/128/poly128_0045.npy` | 6.47 | 1.74 | 0.2 | 130 |  | red_immobile |
| 14 | generated | 128 | 1 | `trap_blob128_0002_128_M0.1_th120_b99` | `blob/128/blob128_0002.npy` | 7.87 | 2.73 | 0.1 | 120 |  | red_immobile |
| 15 | generated | 128 | 1 | `trap_poly128_0015_128_M0.05_th140_b99` | `poly/128/poly128_0015.npy` | 5.82 | 1.32 | 0.05 | 140 |  | red_immobile |
| 16 | generated | 128 | 1 | `trap_blob128_0058_128_M1_th150_b99` | `blob/128/blob128_0058.npy` | 6.99 | 1.40 | 1 | 150 |  | red_immobile |
| 17 | ct | 128 | 1 | `trap_bent128_0063_128_M1_th150_b99` | `bentheimer/128/bent128_0063.npy` | 9.69 | 1.94 | 1 | 150 |  | red_immobile |
| 18 | ct | 128 | 1 | `trap_buff128_0028_128_M0.2_th140_b99` | `buffberea/128/buff128_0028.npy` | 12.08 | 2.73 | 0.2 | 140 | reused | red_immobile |
| 19 | ct | 128 | 1 | `trap_buff128_0031_128_M0.1_th130_b99` | `buffberea/128/buff128_0031.npy` | 5.10 | 1.37 | 0.1 | 130 |  | red_immobile |
| 20 | ct | 128 | 1 | `trap_bent128_0057_128_M0.05_th120_b99` | `bentheimer/128/bent128_0057.npy` | 5.64 | 1.96 | 0.05 | 120 |  | red_immobile |
| 21 | ct | 128 | 1 | `trap_castle128_0056_128_M0.05_th130_b99` | `castlegate/128/castle128_0056.npy` | 6.00 | 1.62 | 0.05 | 130 | reused | red_immobile |
| 22 | ct | 128 | 1 | `trap_buff128_0011_128_M0.1_th150_b99` | `buffberea/128/buff128_0011.npy` | 9.85 | 1.97 | 0.1 | 150 | reused | red_immobile |
| 23 | ct | 128 | 1 | `trap_buff128_0032_128_M1_th140_b99` | `buffberea/128/buff128_0032.npy` | 7.34 | 1.66 | 1 | 140 | reused | red_immobile |
| 24 | ct | 128 | 1 | `trap_castle128_0012_128_M0.2_th120_b99` | `castlegate/128/castle128_0012.npy` | 5.64 | 1.96 | 0.2 | 120 | reused | red_immobile |
| 25 | ct | 128 | 1 | `trap_bent128_0035_128_M0.2_th130_b99` | `bentheimer/128/bent128_0035.npy` | 5.64 | 1.52 | 0.2 | 130 |  | red_immobile |
| 26 | ct | 128 | 1 | `trap_castle128_0036_128_M0.05_th150_b99` | `castlegate/128/castle128_0036.npy` | 6.92 | 1.38 | 0.05 | 150 | reused | red_immobile |
| 27 | ct | 128 | 1 | `trap_castle128_0006_128_M0.1_th140_b99` | `castlegate/128/castle128_0006.npy` | 6.70 | 1.52 | 0.1 | 140 | reused | red_immobile |
| 28 | ct | 128 | 1 | `trap_bent128_0049_128_M1_th120_b99` | `bentheimer/128/bent128_0049.npy` | 5.99 | 2.07 | 1 | 120 |  | red_immobile |
| 29 | ct | 128 | 1 | `trap_castle128_0023_128_M0.2_th150_b99` | `castlegate/128/castle128_0023.npy` | 6.71 | 1.34 | 0.2 | 150 | reused | red_immobile |
| 30 | ct | 128 | 1 | `trap_bent128_0032_128_M0.1_th120_b99` | `bentheimer/128/bent128_0032.npy` | 5.99 | 2.08 | 0.1 | 120 |  | red_immobile |
| 31 | ct | 128 | 1 | `trap_buff128_0038_128_M0.05_th140_b99` | `buffberea/128/buff128_0038.npy` | 6.32 | 1.43 | 0.05 | 140 | reused | red_immobile |
| 32 | ct | 128 | 1 | `trap_bent128_0018_128_M1_th130_b99` | `bentheimer/128/bent128_0018.npy` | 5.74 | 1.55 | 1 | 130 |  | red_immobile |

## extra_beta_1g — GDL — 32 runs

**32 runs** — 32 finished, 0 running.

| # | Family | Domain | GPUs | Run | Geometry | r_c | margin | M | theta | Flags | Status |
|---:|---|---|---:|---|---|---:|---:|---:|---:|---|---|
| 1 | generated | 128x128x64 | 1 | `gdl_fiber_0015_128x128x64_M1_th130_b99` | `fiber/128x128x64/fiber_0015.npy` | — | — | 1 | 130 | reused | pv_cap |
| 2 | generated | 128x128x64 | 1 | `gdl_fiber_0020_128x128x64_M5_th120_b99` | `fiber/128x128x64/fiber_0020.npy` | — | — | 5 | 120 | reused | bt_cap |
| 3 | generated | 128x128x64 | 1 | `gdl_fiber_0021_128x128x64_M20_th150_b99` | `fiber/128x128x64/fiber_0021.npy` | — | — | 20 | 150 | reused | bt_cap |
| 4 | generated | 128x128x64 | 1 | `gdl_fiber_0056_128x128x64_M10_th140_b99` | `fiber/128x128x64/fiber_0056.npy` | — | — | 10 | 140 | reused | pv_cap |
| 5 | generated | 128x128x64 | 1 | `gdl_fiber_0044_128x128x64_M1_th120_b99` | `fiber/128x128x64/fiber_0044.npy` | — | — | 1 | 120 | reused | bt_cap |
| 6 | generated | 128x128x64 | 1 | `gdl_fiber_0011_128x128x64_M5_th150_b99` | `fiber/128x128x64/fiber_0011.npy` | — | — | 5 | 150 | reused | bt_cap |
| 7 | generated | 128x128x64 | 1 | `gdl_fiber_0003_128x128x64_M20_th130_b99` | `fiber/128x128x64/fiber_0003.npy` | — | — | 20 | 130 | reused | pv_cap |
| 8 | generated | 128x128x64 | 1 | `gdl_fiber_0017_128x128x64_M5_th140_b99` | `fiber/128x128x64/fiber_0017.npy` | — | — | 5 | 140 | reused | bt_cap |
| 9 | generated | 128x128x64 | 1 | `gdl_fiber_0050_128x128x64_M10_th130_b99` | `fiber/128x128x64/fiber_0050.npy` | — | — | 10 | 130 | reused | pv_cap |
| 10 | generated | 128x128x64 | 1 | `gdl_fiber_0036_128x128x64_M10_th150_b99` | `fiber/128x128x64/fiber_0036.npy` | — | — | 10 | 150 | reused | pv_cap |
| 11 | generated | 128x128x64 | 1 | `gdl_fiber_0027_128x128x64_M1_th140_b99` | `fiber/128x128x64/fiber_0027.npy` | — | — | 1 | 140 | reused | pv_cap |
| 12 | generated | 128x128x64 | 1 | `gdl_fiber_0026_128x128x64_M20_th120_b99` | `fiber/128x128x64/fiber_0026.npy` | — | — | 20 | 120 | reused | pv_cap |
| 13 | generated | 128x128x64 | 1 | `gdl_fiber_0047_128x128x64_M5_th130_b99` | `fiber/128x128x64/fiber_0047.npy` | — | — | 5 | 130 | reused | pv_cap |
| 14 | generated | 128x128x64 | 1 | `gdl_fiber_0053_128x128x64_M20_th140_b99` | `fiber/128x128x64/fiber_0053.npy` | — | — | 20 | 140 | reused | bt_cap |
| 15 | generated | 128x128x64 | 1 | `gdl_fiber_0032_128x128x64_M1_th150_b99` | `fiber/128x128x64/fiber_0032.npy` | — | — | 1 | 150 | reused | pv_cap |
| 16 | generated | 128x128x64 | 1 | `gdl_fiber_0051_128x128x64_M10_th120_b99` | `fiber/128x128x64/fiber_0051.npy` | — | — | 10 | 120 | reused | pv_cap |
| 17 | ct | 128x128x64 | 1 | `gdl_gdl_ct_0053_128x128x64_M5_th130_b99` | `gdl_ct/128x128x64/gdl_0053.npy` | — | — | 5 | 130 |  | bt_cap |
| 18 | ct | 128x128x64 | 1 | `gdl_gdl_ct_40_0036_128x128x64_M10_th140_b99` | `gdl_ct_40/128x128x64/gdl_0036.npy` | — | — | 10 | 140 |  | bt_cap |
| 19 | ct | 128x128x64 | 1 | `gdl_gdl_ct_40_0058_128x128x64_M20_th120_b99` | `gdl_ct_40/128x128x64/gdl_0058.npy` | — | — | 20 | 120 |  | bt_cap |
| 20 | ct | 128x128x64 | 1 | `gdl_gdl_ct_0022_128x128x64_M1_th150_b99` | `gdl_ct/128x128x64/gdl_0022.npy` | — | — | 1 | 150 |  | bt_cap |
| 21 | ct | 128x128x64 | 1 | `gdl_gdl_ct_0026_128x128x64_M1_th130_b99` | `gdl_ct/128x128x64/gdl_0026.npy` | — | — | 1 | 130 |  | pv_cap |
| 22 | ct | 128x128x64 | 1 | `gdl_gdl_ct_20_0055_128x128x64_M5_th150_b99` | `gdl_ct_20/128x128x64/gdl_0055.npy` | — | — | 5 | 150 |  | bt_cap |
| 23 | ct | 128x128x64 | 1 | `gdl_gdl_ct_40_0049_128x128x64_M10_th120_b99` | `gdl_ct_40/128x128x64/gdl_0049.npy` | — | — | 10 | 120 |  | pv_cap |
| 24 | ct | 128x128x64 | 1 | `gdl_gdl_ct_20_0033_128x128x64_M20_th140_b99` | `gdl_ct_20/128x128x64/gdl_0033.npy` | — | — | 20 | 140 |  | bt_cap |
| 25 | ct | 128x128x64 | 1 | `gdl_gdl_ct_20_0022_128x128x64_M20_th130_b99` | `gdl_ct_20/128x128x64/gdl_0022.npy` | — | — | 20 | 130 |  | pv_cap |
| 26 | ct | 128x128x64 | 1 | `gdl_gdl_ct_0051_128x128x64_M10_th150_b99` | `gdl_ct/128x128x64/gdl_0051.npy` | — | — | 10 | 150 |  | pv_cap |
| 27 | ct | 128x128x64 | 1 | `gdl_gdl_ct_40_0001_128x128x64_M1_th140_b99` | `gdl_ct_40/128x128x64/gdl_0001.npy` | — | — | 1 | 140 |  | bt_cap |
| 28 | ct | 128x128x64 | 1 | `gdl_gdl_ct_0029_128x128x64_M5_th120_b99` | `gdl_ct/128x128x64/gdl_0029.npy` | — | — | 5 | 120 |  | bt_cap |
| 29 | ct | 128x128x64 | 1 | `gdl_gdl_ct_0019_128x128x64_M20_th150_b99` | `gdl_ct/128x128x64/gdl_0019.npy` | — | — | 20 | 150 |  | pv_cap |
| 30 | ct | 128x128x64 | 1 | `gdl_gdl_ct_40_0034_128x128x64_M5_th140_b99` | `gdl_ct_40/128x128x64/gdl_0034.npy` | — | — | 5 | 140 |  | bt_cap |
| 31 | ct | 128x128x64 | 1 | `gdl_gdl_ct_20_0028_128x128x64_M10_th130_b99` | `gdl_ct_20/128x128x64/gdl_0028.npy` | — | — | 10 | 130 |  | bt_cap |
| 32 | ct | 128x128x64 | 1 | `gdl_gdl_ct_20_0021_128x128x64_M1_th120_b99` | `gdl_ct_20/128x128x64/gdl_0021.npy` | — | — | 1 | 120 |  | bt_cap |
