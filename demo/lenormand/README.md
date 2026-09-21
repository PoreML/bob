# Lenormand drainage regimes (3D)

Reproduces the three immiscible-displacement regimes of the **Lenormand, Touboul & Zarcone
(J. Fluid Mech. 1988)** phase diagram — *stable displacement*, *viscous fingering*, *capillary
fingering* — on one geometry, the 256³ synthetic sphere pack, by moving only two control
parameters: the capillary number `Ca` and the viscosity ratio `M`. Driver, solver and boundary
conditions are identical between the runs.

| regime | Ca | M | dominant force | front morphology |
|---|---|---|---|---|
| **stable displacement** | high (preset 1e-3) | ≫ 1 (preset 30) | viscous, favourable | compact, nearly flat front |
| **viscous fingering** | high (preset 1e-3) | ≪ 1 (preset 0.01) | viscous, unfavourable | thin branched fingers (screening) |
| **capillary fingering** | low (preset 1e-5) | preset 30 | capillary | isotropic invasion-percolation clusters |

## Setup

**Control parameters.**

- **Ca** — capillary number, set by the injection speed: `u_in = Ca·σ/ν₀`. With the piston
  inlet the injected volume `u_in·A` per step is exact by construction.
- **M = ν_invading / ν_defending = ν_red / ν_blue** — viscosity ratio, set by the two relaxation
  rates `omega` (invading red) and `omega2` (defending blue), blended per cell by
  `color3d._omega_field`. The baseline viscosity ν₀ = 0.1 is split geometrically,
  ν_red = ν₀·√M and ν_blue = ν₀/√M.

The presets (`REGIMES` in `lenormand_demo.py`) form a rectangle in `(log Ca, log M)`:
stable ↔ viscous isolates the M axis (both Ca = 1e-3); stable ↔ capillary isolates the Ca axis
(both M = 30). The scan/render block is 500 steps at Ca = 1e-3 and 2000 steps at Ca = 1e-5.

Ca = 1e-3 is the validated high-Ca corner: with χ = 0.8 (below) both the stable and the viscous
preset run finite there, whereas Ca = 1e-1 requires `u_in = Ca·σ/ν₀ = 0.05` (Ma ≈ 0.087) and
diverges through compressibility errors with or without the χ spectrum. Two decades still
separate the high-Ca and capillary corners.

**Solver.** The default stack — MRT + Akai wetting + CSF (Eq. 34/35) + `recolor_emag` +
`wall_grad="fluid"` — with `omega`/`omega2` from the M split, `sigma = 0.05`, `beta = 0.7`,
`theta = 135°` on the rock (drainage entry pressure, required for capillary fingering) and
`mrt_chi = 0.8`. float32 by default (`BOB_FP64=1` selects float64).

**MRT χ = 4/5 spectrum (`Params.mrt_chi = 0.8`, `--chi`).** The M = 30 and M = 0.01 splits put
one phase's ω at 1.80 / 1.89, close to the zero-viscosity limit ω = 2, where the default MRT
spectrum leaves several ghost moments relaxing at rate 1.98 (weakly damped). The resulting
ghost-mode instability produces NaNs by step ~2000 on a 64³ crop regardless of the inlet
condition. Relaxing every non-shear moment at `0.8·ω` (Leclaire et al. 2017) damps these modes:
the capillary regime then runs with a flat `u_max ≈ 0.05` and the high-Ca regimes run at
Ca = 1e-3. `mrt_chi = 0.8` is required for the dual-viscosity presets; `--chi 0` restores the
default spectrum.

**Boundary conditions.**

- **Piston inlet** (`drain(inlet="piston")`, the bounce-back velocity BC of MF-LBM): the x = 0
  plane is sealed as the piston face, each colour's counter-stream reflects off it (blue is
  never repainted), and the Ladd kick injects exactly `u_in·A` of red per step. The
  inlet pressure is a free output that builds against the entry barriers.
- **Blue-pinned pressure outlet** (`outlet_sa_red=0.0`): a zero-gradient colour split at the
  outlet is self-referential — the diffusive red tail seeds it and accumulates a spurious red
  pool, giving a Ca-independent false breakthrough. Pinning the outlet colour to the defending
  blue keeps the colour ledger exact.
- **Red reservoir and red-wetting face**: slabs x = 1 … buffer−1 (`--buffer 7`) start pure red
  at t = 0 only (the piston sustains the column), and a per-cell θ field makes the
  face-adjacent slab purely red-wetting (`--theta-face 0`) so the column stays attached to the
  face.

**Geometry.** `test/assets/sphere256.npy` — monodisperse synthetic sphere pack (grain radius
14, porosity 0.258). With `--buffer 7` the domain is 256×256×270 (piston face x = 0, reservoir
x = 1 … 6, rock x = 7 … 262); the float64 state is ~5.5 GB. `--structure` accepts any boolean
volume (True = solid), e.g. a small crop for a local smoke test; `--name` sets the run title.
`demo/imbibition` and `demo/resolution` reuse this driver on other geometries.

## How to run

```bash
# one regime on a local GPU (full 256³ pack):
uv run python demo/lenormand/lenormand_demo.py --regime stable
uv run python demo/lenormand/lenormand_demo.py --regime stable --plots   # also write the analysis curves

# SLURM — the three regimes as parallel one-GPU jobs:
bash demo/lenormand/submit_all.sh
bash demo/lenormand/submit_all.sh capillary        # subset
FRESH=1 REGIME=stable sbatch --job-name=lenormand_stable demo/lenormand/run_lenormand.slurm  # ignore checkpoints
```

Submit SLURM jobs from the repository root; partition, QOS and wall-clock limits in the
`.slurm` files are site-specific. Each run stops at breakthrough (`--steps` is a safety cap);
`--after-breakthrough` / `--after-breakthrough-x` continue toward the saturation plateau. The
capillary regime needs ~1e7 steps at Ca = 1e-5 and exceeds a 3-day wall clock:
`run_lenormand.slurm` resumes from the newest checkpoint, so resubmit the same job after a
timeout. `--gpus N` shards the domain along z (`bob.multigpu`; 256³ runs 1.5× / 2.65× / 4.46×
faster on 2 / 4 / 8 H100 GPUs).

### Faster capillary runs

At (σ = 0.05, ν₀ = 0.1) the Ca = 1e-5 front advances about one layer per 28k steps. Two
independent speedups are available through `submit_tuned.sh` → `run_lenormand_tuned.slurm`
(`SIGMA`, `NU0`, `TAG`, optional `CA`, `GPUS`; output in `output/<regime>_<TAG>`):

1. **Lower ν₀ at fixed σ.** At fixed Ca, `u_in = Ca·σ/ν₀`. σ stays at 0.05: σ = 0.10 and 0.15
   destabilize the run (entry pressures scale with σ, and the spurious-current scale ~σ/ν
   grows). On a 64³ crop the stability floor is ν_blue ≈ 0.005–0.006; the full 256³ pack is
   less tolerant by about one step of the ν₀ ladder — ν₀ = 0.0333 is stable on 64³ but
   diverges at step 2000 on the full pack — so keep at least that margin.
2. **float32.** 2.05× steps/s at 256³ (6.35 → 13 steps/s) and half the memory, at parity with
   float64 on breakthrough step, front morphology and colour ledger.

| output folder | σ | ν₀ | u_in | steps to breakthrough (estimate) |
|---|---|---|---|---|
| `output/capillary` (preset) | 0.05 | 0.10 | 5e-6 | ~1e7 |
| `output/capillary_s005_nu005` | 0.05 | 0.05 | 1e-5 | ~5M |
| `output/capillary_s005_nu004` | 0.05 | 0.04 | 1.25e-5 | ~4M |

## Validation

The regimes must be separable by number, not only by eye. `report.md` gives the breakthrough
saturation, the colour-ledger ratio and the front-morphology descriptors `tip_rough` (front
roughness), `fill_behind_tip` (compactness) and `invaded_frac`. Expected separation:

| metric | stable | viscous | capillary |
|---|---|---|---|
| breakthrough saturation | high (≳ 0.7) | low–mid | lowest |
| `tip_rough` (front roughness) | low | high | high |
| `fill_behind_tip` (compactness) | ≈ 1 | < 1 | < 1 |
| forward bias of fingers | n/a | strong (downstream) | weak (isotropic) |
| colour ledger at breakthrough | 1.00× | 1.00× | 1.00× |

A colour-ledger ratio different from 1.00× indicates a boundary-condition artefact rather than
displacement physics.

**Limitations.** The viscous preset is the marginal case (invading red at ω = 1.89): on a 64³
crop at Ca = 1e-3 its `u_max` ≈ 0.2 is bounded and oscillating, not growing; if a geometry
destabilizes it, reduce the viscosity contrast first (e.g. M = 0.02). The capillary preset
(u_in = 5e-6) needs ~1e7 steps to breakthrough; Ca = 1e-4 (~1e6 steps) is still a decade below
the high-Ca corner. Fingering patterns are stochastic, and each preset is a single realization.

## Outputs

Per regime, in `output/<regime>/` (gitignored):

- `lenormand_<regime>.gif` — PyVista volume render (red invader opaque, rock translucent),
  refreshed every 10 blocks and capped at 300 frames.
- `metrics.csv` — rock-only saturation, front position, colour ledger (`m_red` vs ideal
  `u_in·A·t`), pressure stations, spurious-current channels.
- `fields.h5` + `.xdmf` — 3D fields (phi, p, |u|, u) for ParaView (`--field-every`).
- `report.md` — Ca, M, χ, derived ω/ν, breakthrough step, ledger ratio and front morphology.
- `ckpt_step*_sat*.npz` — resumable checkpoints (every 10 blocks and at exit).

The analysis curves below are written only with `--plots`:

- `saturation.svg` — rock-only saturation against step.
- `ledger_curve.svg` — injected red vs ideal (leak-free reference = 1.0×).
- `pressure_curve.svg` — p_in / p_mid / p_end and the entry-pressure buildup p_in − p_end
  (Haines jumps appear as relaxations).
- `spurious_curve.svg` — max / interface / bulk |u| against `u_in`; `u_max` is also the early
  warning of divergence for the near-limit ω splits.

## References

- Lenormand, Touboul & Zarcone, J. Fluid Mech. (1988) — displacement phase diagram in the
  (Ca, M) plane.
- Leclaire et al., Phys. Rev. E 95, 033306 (2017) — MRT relaxation spectrum tied to χ·ω.
- Akai, Bijeljic & Blunt, Adv. Water Resour. 116 (2018) — wetting boundary condition.
