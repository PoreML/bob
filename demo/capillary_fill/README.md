# capillary_fill — pore-filling events in a single junction

Two displacement experiments in single-junction micro-models, reproduced with bob's
color-gradient solver and compared with Zacharoudiou, Chapman, Boek & Crawshaw,
*J. Fluid Mech.* **824**, 550–573 (2017), [doi:10.1017/jfm.2017.363](https://doi.org/10.1017/jfm.2017.363).
Both junctions have four unequal throats, so the order in which the throats fill is a
direct test of the capillary physics: contact-angle boundary condition, surface tension
and interface dynamics together.

| case | driver | reference | result folder |
|---|---|---|---|
| Spontaneous imbibition, Geometry 3 | `capillary_fill_demo.py` | figure 8 | `output_oh006/` |
| Primary dynamic drainage, Geometry 2 | `drainage_fill_demo.py` | figure 4 | `output_fig4_ca1e-4/` |

Both use the default solver stack (MRT with χ = 0.8, Akai geometric wetting, CSF,
β = 0.99, fp32) at σ = 0.05, on the native 3 µm lattice of the reference. Abbreviations:
WP = wetting phase, NWP = non-wetting phase.

## Spontaneous imbibition (figure 8)

**Question.** Into which downstream throat does the wetting phase go first? Quasi-static
Young–Laplace filling rules predict the narrowest throat (T1, 27 µm). The experiments and
the free-energy LB simulations of the reference show that the WP entering through T3
(60 µm) instead fills the **adjacent throat T2 (47 µm) first**, and the NWP retreats
through the widest throat T4 (100 µm): pore-body geometry, not throat size, selects the
pathway.

**Setup.**
- Geometry (`geometry.py`): stepped-octagon pore body (core 109 × 53 nodes, tiers 83 × 15),
  throat mouths T1 = 9, T2 = 16, T3 = 19, T4 = 35 nodes, 600 µm straight arms to the domain
  faces, 55 µm etch depth (18 nodes). Domain 20 × 483 × 539. T3 inlet on the left, T1 right,
  T2 bottom (y max), T4 top (y = 0).
- Drive: spontaneous, ΔP = 0. All four ports are per-color Zou–He pressure planes at the
  same ambient density — x faces through `bob.bc`, y faces through the demo-local
  `bc_faces.py`, which generalizes the same reconstruction to any face. The three vents
  open onto air, so their colour is anchored to the NWP (`sa_red=0`).
- Fluids: θ = 16° (experimental value), ν = 0.06 / 0.01 (WP / NWP), giving Ohnesorge number
  Oh = 0.059 (reference: 6 × 10⁻²) at viscosity ratio 6. The NWP viscosity sits at the
  stability limit ν ≥ 0.01 for σ = 0.05, so Oh and the reference's viscosity ratio (50)
  cannot both be matched; Oh is matched because it controls the interface morphology,
  while the reference shows the filling sequence to be independent of the viscosity ratio
  (5–500).

**Validation criterion.** PASS = T2 reaches WP majority before both T1 and T4.

**Result.** PASS.

| throat | width | step of WP majority |
|---|---:|---:|
| T2 (adjacent) | 47 µm | 72 500 |
| T1 (narrowest) | 27 µm | 132 500 |
| T4 (widest) | 100 µm | 182 500 |

250 000 steps, about 1.2 h on one GPU (~60 steps/s).

## Primary dynamic drainage (figure 4)

**Question.** Air (NWP) is forced at a constant rate (0.5 µl/min, Ca ≈ 10⁻⁴ in the feeding
throat) into a square pore with throats of 33 / 51 / 65 / 107 µm. It fills the pore body
and leaves through the **widest** downstream throat T4 (107 µm) first — the Young–Laplace
order of the entry pressures, P = 2γ cos θ (1/d + 1/w) — while the 33 µm and 65 µm side
throats stay WP-filled.

**Setup.**
- Geometry (`geometry2.py`): 238 µm square body (79 nodes), throats 11 / 17 / 22 / 36 nodes,
  45 µm etch depth (15 nodes), 450 µm arms. Domain 17 × 379 × 379. NWP enters through the
  51 µm throat (x = 0), the 107 µm throat is straight opposite (x = −1), the 33 µm throat is
  on the y = 0 face and the 65 µm throat on the y = −1 face.
- Drive: rate-controlled piston inlet (`bob.bc.piston_inlet`) behind the sealed x = 0 plane,
  injecting pure NWP at u_in; the inlet pressure floats. The three WP ports are Zou–He
  pressure outlets at ambient density, colour-anchored to the WP (`sa_red=0`).
  Ca = η_w u_in / γ = 10⁻⁴, as in the experiment.
- Fluids: θ = 26° through the WP (bob's red-phase angle is 154°), ν = 0.01 / 0.05
  (NWP / WP), viscosity ratio 5. The experimental ratio (47) is below the ν ≥ 0.01 stability
  limit; at this Ca the viscous pressure drop along an arm is below 1 % of the
  entry-pressure differences, so the ratio does not affect the pathway.

**Validation criterion.** PASS = the T4 mouth reaches NWP majority before the T1 and T3
mouths, and both side throats hold less than 10 % NWP at the end of the run.

**Result.** PASS.

| throat | width | Young–Laplace entry pressure (lu) | step of NWP majority at the mouth | NWP fraction of the arm at the end |
|---|---:|---:|---:|---:|
| T4 | 107 µm | 0.0096 | 4 530 000 | 0.85 |
| T3 | 65 µm | 0.0114 | – | 0.01 |
| T1 | 33 µm | 0.0160 | – | 0.00 |

The inlet over-pressure peaks at 0.0098 lu while the body fills, against a Young–Laplace
T4 entry pressure of 0.0096 lu (2 % apart). 6 785 000 steps, about 17 h on one H200
(~110 steps/s); `--ca 1e-3` shows the same throat sequence in about a tenth of the steps.

## How to run

From the repository root (a GPU is used automatically when available):

```bash
uv run python demo/capillary_fill/capillary_fill_demo.py     # imbibition  -> output_oh006/
uv run python demo/capillary_fill/drainage_fill_demo.py      # drainage    -> output_fig4_ca1e-4/
sbatch demo/capillary_fill/run_fig4.slurm                    # drainage as a resumable SLURM job

uv run python demo/capillary_fill/live_render.py --out <run folder>   # frames + GIF while a run is going (CPU)
uv run python demo/capillary_fill/bc_faces.py                         # boundary-condition self-check
```

Both drivers checkpoint every `--ckpt-every` blocks and continue with `--resume`.

## Outputs

Per run folder (gitignored): `domain.png` / `domain.json` (geometry, probe regions,
parameters), `metrics.csv` (phase fraction per region and, for drainage, inlet
over-pressure and front position), `frames/mid_*.npz` (z-midplane phase field per block),
`checkpoint.npz`, `report.md` (verdict and table), `live/` (rendered frames and GIF).

The saved `frames/mid_*.npz` hold the z-midplane phase field at every block, which is what
the displacement-stage comparison with the reference figures is built from. `live_render.py`
turns them into PNG frames and a GIF.

## Reference

I. Zacharoudiou, E. M. Chapman, E. S. Boek, J. P. Crawshaw, "Pore-filling events in single
junction micro-models with corresponding lattice Boltzmann simulations", *J. Fluid Mech.*
**824**, 550–573 (2017).
