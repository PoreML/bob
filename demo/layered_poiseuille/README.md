# Layered Poiseuille flow (3D) — viscosity-ratio validation

Validates the default 3D solver stack (`color3d`, MRT + CSF) against the analytic layered
Poiseuille profile at viscosity ratios **M = ν_b/ν_r = 5, 10, 15, 20, 30**. The flow is driven
by per-colour Zou-He pressure boundaries (not a body force), and all ratios are compared in one
combined figure (theory lines + LBM dots, one colour per M).

## Setup

- **Geometry.** Channel `(nz, ny, nx) = (64, 8, 128)`, bounce-back walls on the z faces, y
  periodic. Symmetric three-band layout: viscous blue against both walls, thin red in the
  central band z ∈ [16, 48), so u_x(z) is nearly flat next to the walls with a tall parabola in
  the centre.
- **Viscosities.** The wall fluid is fixed at ν_b = 0.3 (τ = 1.4) and the centre fluid is
  thinned per ratio, ν_r = ν_b/M (down to 0.01 at M = 30). σ = 0 (flat interfaces), β = 0.7, fp64.
- **Drive.** All ratios share the same pressure drop, set so that M = 30 peaks near
  u_max ≈ 0.025; the wall segments of the profiles therefore coincide and the centre peak fans
  upward with M.
- **Boundary conditions.** `zou_he_inlet` at x = 0 feeds each layer its own fluid through a
  per-cell `sa_red` mask at `rho_in = 1 + Δρ`; `zou_he_outlet` at x = −1 holds `rho_out = 1`
  with the zero-gradient colour split, keeping the layers flat at both x ends.
- **Analytic reference.** The theory curves solve d/dz(ν du/dz) = −G on the realized diffuse
  interface (harmonic face viscosities from the measured φ) with the realized pressure gradient
  (linear fit of cs²·ρ(x)).

## How to run

```bash
uv run python demo/layered_poiseuille/layered_demo.py                 # all five ratios
uv run python demo/layered_poiseuille/layered_demo.py --ratios 5 --steps 8000 --block 2000  # quick look
uv run python demo/layered_poiseuille/layered_demo.py --plots          # also write the analysis curves
uv run python demo/layered_poiseuille/publication_plot.py             # publication figure
```

A GPU is used automatically when available. `publication_plot.py` runs the five ratios at the
demo defaults once, caches the steady profiles in `output/profile_pub.csv`, and re-plots from
that file afterwards (delete it to force a re-run).

## Validation and results

Each ratio runs to steady state (relative profile change < 1e-6 per 2000-step block).
**PASS:** maximum relative error against the analytic profile below 8% for every ratio.

Reference run (fp64, single H100):

| M | ν_r | steps to steady | max rel error |
|--:|----:|--------------:|--------------:|
| 5  | 0.060 | 68k  | 1.57% |
| 10 | 0.030 | 84k  | 0.95% |
| 15 | 0.020 | 96k  | 1.69% |
| 20 | 0.015 | 118k | 2.31% |
| 30 | 0.010 | 170k | 2.58% |

All ratios PASS, well inside the 8% gate. ω_r reaches 1.89 at M = 30 (ν_r = 0.01) without
`mrt_chi`: with σ = 0 and flat interfaces the default MRT spectrum stays stable. Convergence
time grows with M as the centre fluid's momentum diffusivity shrinks.

## Outputs

In `output/` (gitignored): `layered.gif` (all ratios developing onto their theory lines),
`report.md` (per-ratio table with PASS/FAIL), `frames/*.png`; from `publication_plot.py`,
`profile_pub.pdf`, `profile_pub.png` (300 dpi) and `profile_pub.csv`. The analysis curve
`profile.svg` (combined end state) is written only with `--plots`.
