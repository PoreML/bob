# case/underfill — capillary underfill of flip-chip solder-ball gaps

Capillary underfill of the `data/flipchip` geometries: a thin
die/substrate gap crossed by a square staggered array of truncated-sphere solder
balls (uniform shared-stencil balls, 2 D side-wall clearance; per-sample ball
diameter D 32-54, gap 20-50, pitch 1.5-2.1 D, all in cells). The encapsulant
(red) wets the gap and displaces air (blue) by capillary suction alone. The
dataset sweeps the two physics knobs of that process:

- **viscosity ratio M = nu_red/nu_blue in {5, 10, 20, 30}** — nu_blue is pinned
  at 0.025 (the stability floor in these thin gaps; nu_blue <= 0.02 diverges)
  and nu_red = M x 0.025. Fill time in lattice steps is ~linear in M.
- **contact angle theta in {30, 40, 50} deg** of the encapsulant on all walls.

Physics: capillary suction only (Zou-He dP = 0 pressure inlet and outlet,
`porous3d.drain(inlet="zouhe")`), sigma 0.05, beta 0.99 (recoloring sharpness,
thin interface), solver defaults MRT + akai geometric wetting + CSF +
`recolor_emag` + `wall_grad="fluid"`, fp32. `physical_mapping.md` gives the
conversion of these lattice runs to physical units.

## Published runs

32 runs, one per geometry sample, under
`runs/flipchip/26x482x476/fill_flipchip_<sample>_26x482x476_M<M>_th<theta>/`.
`26x482x476` is a folder label, not the shape: the domains are
(22-48) x 482 x (432-476) cells. The 12 (M, theta) cells hold 2-3 runs each and
every M has 8 runs. All 32 runs ended `filled`; the longest took 1,265,000
steps, well under the driver's 3,000,000-step safety cap.

| Sample | M | theta | Domain (nz x ny x nx) | D | gap | pitch / D | balls | Steps |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 0000 | 5 | 30 | 26 x 482 x 476 | 32 | 24 | 2.00 | 33 | 170,000 |
| 0001 | 30 | 30 | 22 x 482 x 476 | 32 | 20 | 2.09 | 23 | 795,000 |
| 0002 | 20 | 40 | 44 x 482 x 444 | 48 | 42 | 1.58 | 14 | 383,000 |
| 0004 | 30 | 40 | 46 x 482 x 440 | 50 | 44 | 1.76 | 8 | 481,000 |
| 0006 | 30 | 50 | 44 x 482 x 438 | 51 | 42 | 1.96 | 8 | 518,000 |
| 0007 | 20 | 30 | 46 x 482 x 448 | 46 | 44 | 2.04 | 8 | 312,000 |
| 0011 | 5 | 40 | 42 x 482 x 436 | 52 | 40 | 1.62 | 8 | 117,000 |
| 0012 | 5 | 50 | 34 x 482 x 460 | 40 | 32 | 1.62 | 23 | 194,000 |
| 0013 | 10 | 30 | 35 x 482 x 452 | 44 | 33 | 2.00 | 8 | 197,000 |
| 0018 | 5 | 50 | 38 x 482 x 456 | 42 | 36 | 1.79 | 14 | 162,000 |
| 0019 | 10 | 30 | 29 x 482 x 456 | 42 | 27 | 1.57 | 23 | 251,000 |
| 0020 | 20 | 40 | 34 x 482 x 460 | 40 | 32 | 1.85 | 14 | 457,000 |
| 0023 | 10 | 40 | 28 x 482 x 464 | 38 | 26 | 1.74 | 23 | 299,000 |
| 0029 | 5 | 50 | 32 x 482 x 448 | 46 | 30 | 1.80 | 14 | 156,000 |
| 0030 | 30 | 30 | 47 x 482 x 436 | 52 | 45 | 1.77 | 8 | 416,000 |
| 0032 | 20 | 40 | 30 x 482 x 450 | 45 | 28 | 1.58 | 14 | 476,000 |
| 0033 | 10 | 30 | 36 x 482 x 464 | 38 | 34 | 1.92 | 16 | 229,000 |
| 0034 | 20 | 50 | 40 x 482 x 460 | 40 | 38 | 1.98 | 14 | 487,000 |
| 0036 | 5 | 30 | 30 x 482 x 476 | 32 | 28 | 1.78 | 33 | 172,000 |
| 0038 | 30 | 40 | 41 x 482 x 452 | 44 | 39 | 1.95 | 14 | 558,000 |
| 0039 | 30 | 40 | 34 x 482 x 456 | 42 | 32 | 1.88 | 14 | 641,000 |
| 0040 | 10 | 50 | 29 x 482 x 474 | 33 | 27 | 1.55 | 46 | 457,000 |
| 0041 | 20 | 30 | 32 x 482 x 444 | 48 | 30 | 1.79 | 8 | 345,000 |
| 0043 | 20 | 30 | 24 x 482 x 474 | 33 | 22 | 2.06 | 23 | 517,000 |
| 0044 | 20 | 50 | 41 x 482 x 432 | 54 | 39 | 1.85 | 8 | 350,000 |
| 0048 | 30 | 50 | 32 x 482 x 452 | 44 | 30 | 1.70 | 14 | 795,000 |
| 0050 | 5 | 40 | 38 x 482 x 462 | 39 | 36 | 1.59 | 23 | 167,000 |
| 0052 | 10 | 40 | 29 x 482 x 474 | 33 | 27 | 1.94 | 23 | 314,000 |
| 0055 | 5 | 40 | 41 x 482 x 452 | 44 | 39 | 1.98 | 9 | 125,000 |
| 0058 | 30 | 50 | 23 x 482 x 472 | 34 | 21 | 1.76 | 33 | 1,265,000 |
| 0060 | 10 | 50 | 48 x 482 x 438 | 51 | 46 | 1.80 | 8 | 205,000 |
| 0062 | 10 | 50 | 26 x 482 x 462 | 39 | 24 | 1.95 | 14 | 336,000 |

Each row is reproduced by
`underfill_demo.py --sample <Sample> --m-ratio <M> --theta <theta>`.

`campaign.py` holds a separate frozen draw (a two-round full factorial over
theta in {30, 40, 50, 60} on samples 0-31, folders `runs/fc<sample>_M<M>_th<theta>/`).
It is not the draw behind the table above, and its `submit` subcommand is
disabled; `prepare`, `status`, `cancel` and `verify-draw` work.

## Termination

A run stops when the ball-array fill fraction reaches `--fill-target` 0.99 plus
10 settle blocks (`finish_type` `filled`). It records `diverged` on a non-finite
saturation and `step_cap` at the `--steps` safety cap (default 3,000,000).

## Logging (every 1000 steps)

Per block: a `metrics.csv` row (fill, front, void inventory at blue fraction
fB > 0.5 / 0.25 / 0.1), a top-view frame (`frames/` -> `uf_<name>.gif`), a PyVista
3D frame (`frames3d/` -> `uf3d_<name>.gif`; red melt opaque, solid translucent)
and a fields `.h5/.xdmf` append (phi, rho, p, umag, u). Checkpoints and GIFs
refresh every 25 blocks. `run_meta.json` is the reproducibility sidecar (solver
parameters, precision, code commit, geometry sha256, devices, throughput,
`finish_type`), and `report.md` summarises the run. The `fill.svg` progress
curve is opt-in: it is written only with `--plots`.

Disk: each h5 append is ~100-200 MB (7-11 M-cell domains, 5 field groups), so
the full dataset is in the low-TB range. Trim `--field-vars` to reduce it.

## Files

- `underfill_demo.py` — single-run driver (`--sample --m-ratio --theta`)
- `campaign.py` — frozen standalone draw: `prepare | status | cancel | clean | verify-draw`
- `run_underfill.slurm` — 1 GPU per run, resumes from the newest checkpoint
- `physical_mapping.md` — lattice-to-physical unit conversion

## Usage

The geometry is read from `$GEOMETRY_DATA/flipchip/samples` (default: the
sibling checkout `<repo>/../geometry/data`). From the repository root:

```bash
# one run, directly
.venv/bin/python case/underfill/underfill_demo.py --sample 7 --m-ratio 10 --theta 40 \
    --out case/underfill/runs/fc0007_M10_th40 --name fc0007_M10_th40

# one run through SLURM (partition/QOS in the script are site-specific)
.venv/bin/python case/underfill/campaign.py prepare   # stage + sha256-verify the geometries
NAME=fc0007_M10_th40 SAMPLE=7 M=10 THETA=40 sbatch --job-name=uf_fc0007_M10_th40 \
    --output=case/underfill/runs/fc0007_M10_th40/slurm-%j.log case/underfill/run_underfill.slurm
```

`run_underfill.slurm` passes `--steps ${STEPS:-1200000}`; set `STEPS` higher for slow
(high-M) samples, since the longest published run needed 1,265,000 steps.
