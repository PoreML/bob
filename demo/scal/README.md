# SCAL — pressure-controlled drainage and imbibition on Bentheimer sandstone

A special-core-analysis (SCAL) style capillary-pressure experiment on a
micro-CT Bentheimer sample (`test/assets/bentheimer128.npy`, 2.25 um/voxel,
porosity 0.266): quasi-static drainage by raising the inlet pressure in small
steps, followed by imbibition by lowering it through Pc = 0 into negative
values (forced imbibition). The result is the full Pc–S_w hysteresis loop. The
pressure-ladder procedure follows `demo/capillary_pressure` (idealised tubes);
this demo applies it to a real rock with a porous-plate outlet.

## Setup

**Domain.** `[inlet buffer 10 | rock | porous plate 20 | outlet buffer 6]`
along the flow axis x, the plate in direct contact with the rock outlet face.
Solid no-slip walls seal the y/z faces over the whole domain (the core-holder
convention of `porous3d.with_buffers`).

**Pressure ladder.** Pc is imposed at the inlet (`bc.zou_he_inlet`, pure red,
`Pc = (rho_in − 1)·cs²`) against a fixed-pressure outlet (`rho_out = 1`). Each
rung is equilibrated block-wise (2000-step blocks; equilibrium = |ΔS_w| below
`--tol-eq` per block and Darcy-mean capillary number below `--tol-ca`; a rung
that does not reach the gate within `--max-blocks` is recorded with
`capped = 1`). Drainage ends when S_w stalls against rising Pc (|ΔS_w| < 0.005
over `--stall-n` consecutive rungs, counted only after the leg has activated —
flatness below the entry pressure is not a plateau) or at the ladder cap.
Imbibition is the reverse ladder, ending on the same stall rule (residual oil)
or at the −Pc_max floor. The initial state is pressure-consistent (red at
`rho_in` in the inlet buffer, blue at 1.0 beyond, u = 0).

**Ladder calibration** (`plate_check/throat_ladder.py` →
`plate_check/output/ladder.json`). The entry pressure follows from the critical
throat radius r_c, the largest r for which the pore cells with
distance-to-wall ≥ r still percolate along x: r_c = 6.3 voxels gives
entry Pc = 2σ cos45°/r_c = 0.0112. The ladder starts at half the entry
pressure (pc0 = 0.0056), steps by entry/8 (ΔPc = 0.0014) and is capped at
2.5 × entry = 0.028. Pore distance-to-wall p50/p90 = 3.3/8.1 voxels.

**Outlet membrane.** The laboratory method uses a semi-permeable porous plate.
On a voxel lattice the plate is limited by the diffuse interface, which is 4–5
cells wide: a pore narrower than the interface does not develop its
Young–Laplace entry pressure. A plate with 2-voxel holes breaches at Pc ≈ 0.015
(measured by `plate_check/plate_holdoff.py`; the sharp-interface estimate
2σcosθ/r is ≈ 0.1), and a plate with 1-voxel holes passes no water at all. The
demo therefore combines two elements:

- **Plate** — strongly water-wet (θ = 150°) slab pierced by 5 × 5-voxel square
  through-pores on a pitch-8 lattice, 20 voxels long. Its entry pressure,
  ≈ 4σ|cosθ|/w ≈ 0.035, lies above the ladder cap, and the pore length damps
  the dissolved-red concentration gradient ahead of the outlet.
- **Scrubber** (`--scrub`, on by default) — once per block any red mass beyond
  the plate is converted to blue, conserving mass and momentum per population,
  and accumulated in the `scrubbed` column. Non-wetting fluid that passes the
  plate is thus produced and never returns during imbibition; `plate_leak`
  records the per-block arrival before scrubbing.

**Fluids and solver.** σ = 0.05, ν = 0.04 in both phases (M = 1),
θ_rock = 135° (red = non-wetting oil), β = 0.7, MRT with χ = 0.8, float32, and
the default stack (MRT + akai + CSF + `recolor_emag` + `wall_grad="fluid"`).
The capillary number `Ca = μu/σ` is logged every block in three forms:
interface mean, bulk mean, and `Ca_flux = μ|<u_x>_rock|/σ`, the signed
Darcy-mean value. Spurious currents cancel in the signed mean, whereas the
|u|-based bulk value has a spurious floor of about 2e-4; `Ca_flux` is therefore
the quantity used in the equilibrium gate.

## How to run

```bash
uv run pytest test/test_scal.py -q                          # unit tests (geometry + ladder)
uv run python demo/scal/plate_check/throat_ladder.py        # ladder calibration (CPU, ~1 min)
uv run python demo/scal/plate_check/plate_holdoff.py        # narrow-pore plate breach/conductance (GPU, ~3 min)
sbatch --job-name=scal_bent128 --gres=gpu:1 \
  --output=demo/scal/bentheimer128/slurm-%j.log \
  --error=demo/scal/bentheimer128/slurm-%j.log demo/scal/run_scal.slurm

uv run python demo/scal/scal_demo.py --plots                # add the analysis curves (off by default)
```

The run resumes automatically from the newest checkpoint (the ladder state —
leg, Pc, stall counters — is stored in the `.npz`); `FRESH=1` restarts. A fresh
run rewrites the CSVs of its run directory, so point `RUN_DIR` (or `--out`) at
a new directory to keep the tracked results. `RUN_DIR`, `EXTRA`, `STRUCTURE`
and `LADDER` select variants through `run_scal.slurm`:

- **Reference ladder** (`bentheimer128/`): ΔPc = 0.0014, equilibrium gate
  |ΔS_w| < 5e-4 per block, at most 150 blocks per rung.
- **Fine ladder** (`bentheimer128_fine/`): ΔPc = 0.00035 (entry/32) with
  `--stall-n 8` so that the stall window spans the same Pc range;
  `--pre-blocks 5` holds each rung for only 5 blocks while the leg has not yet
  activated (below entry, or just after the reversal); the equilibrium gate is
  |ΔS_w| < 2e-4 per block and Ca_flux ≤ 1e-5, with `--max-blocks 40`:
  `RUN_DIR=demo/scal/bentheimer128_fine EXTRA="--name scal_bent128_fine --dpc
  0.00035 --stall-n 8 --pre-blocks 5 --tol-ca 1e-5 --tol-eq 2e-4 --max-blocks 40
  --frame-blocks 10" sbatch ... demo/scal/run_scal.slurm`.
- **256^3 samples** (`--gpus 4`): the domain is decomposed along z with the
  `shard_map` thin-halo backend (`multigpu.staged_halo_step`). Each rock needs
  its own calibration, because the ladder of one sample does not transfer to
  another. For `test/assets/bentheimer256.npy` (porosity 0.264, an independent
  sample, not a superset of the 128^3 asset) `throat_ladder.py --structure ...
  --out .../ladder256.json` gives r_c = 5.09 voxels, entry Pc = 0.01389,
  pc0 = 0.00694, ΔPc = entry/32 = 0.000434, cap 0.0347. The submit command is
  in the `run_scal.slurm` header.

## Results

Frozen results are tracked per run directory (`metrics.csv`, `pc_steps.csv`,
`run_meta.json`, `report.md`). S_wi is the minimum equilibrium S_w of the
drainage leg and S_or = 1 − max S_w of the imbibition leg.

| run | sample, domain | plate | rungs drain / imb | S_wi | S_or | max plate leak | scrubbed | steps |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `bentheimer128` | bentheimer128, 128×128×147 | 2×2 holes, pitch 4, 3 thick, θ 170° | 10 / 15 | 0.3053 | 0.0021 | 1.058e+03 | 4.611e+04 | 1 372 000 |
| `bentheimer128_fine` | bentheimer128, 128×128×147 | 2×2 holes, pitch 4, 3 thick, θ 170° | 23 / 36 | 0.3436 | 0.0044 | 3.908e+02 | 1.434e+05 | 3 850 000 |
| `bentheimer256_fine` | bent256_0014, 256×256×292 | 5×5 pores, pitch 8, 20 thick, θ 150° | 35 / 58 | 0.3203 | 0.2307 | 2.538e-01 | 1.947e+02 | 4 676 000 |

- The two 128^3 runs use a plate with 2-voxel holes (recorded in
  `run_meta.json`, `extra.plate`), which breaches below the ladder cap: the
  non-wetting phase is produced through the plate and removed by the scrubber.
  The 5-voxel plate of the 256^3 run, the current default of `scal_demo.py`,
  stays sealed (red mass beyond the plate ≤ 0.25 per block). The 128^3 and
  256^3 runs differ in both sample and plate, so their end-point saturations
  are not directly comparable.
- `bentheimer256_fine` runs on `bent256_0014`, a processed 256^3 Bentheimer crop
  (porosity 0.2425 after a radius-3 morphological opening of the pore space and
  filling of face-disconnected pore pockets); provenance and sha256 are in
  `bentheimer256_fine/bent256_0014_geometry.json`. Its ladder is pc0 = 0.00607,
  ΔPc = 0.00038, cap 0.03037; the run used 4x H200.
- The reference run `bentheimer128` was equilibrated on the |ΔS_w| criterion
  alone (its `metrics.csv` has no `ca_flux` column; `--tol-ca 0` selects that
  gate); 1 of its 25 rungs reached the 150-block cap.
- In the fine runs most rungs end at their block cap rather than at the
  Ca_flux ≤ 1e-5 gate: `capped = 1` in `pc_steps.csv` for 56 of 59 rungs in
  `bentheimer128_fine` (45 at the 40-block cap, 11 pre-activation rungs at
  5 blocks) and for 62 of 93 rungs in `bentheimer256_fine` (53 and 9).
- `bentheimer128_periodic_sides/` holds a partial reference-ladder record
  (12 drainage and 10 imbibition rungs) of the same sample with periodic instead
  of sealed y/z faces.

## Outputs

Per run directory: `metrics.csv` (per block: imposed and measured Pc, S_w, the
Ca values, `plate_leak`, `scrubbed`, u_max, phase masses), `pc_steps.csv`
(one row per pressure rung — the Pc–S_w loop), `run_meta.json`, `report.md`;
and, gitignored, 3D frames + GIF, `fields.h5/.xdmf` and checkpoints.

`--plots` additionally writes the analysis curves `sw.svg`, `ca_curve.svg` and
`pc_sw_live.svg` (the loop redrawn every block: completed rungs per leg plus the
current point). They are off by default, so a run writes only its data and renders.

Publication figures (`pc_curve`, `ca_vs_step`, PNG + PDF) are replotted from
the frozen CSVs, also while a run is in progress:

```bash
uv run python demo/scal/publication_plot.py demo/scal/bentheimer128
```
