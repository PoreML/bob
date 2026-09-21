# Washburn capillary imbibition (3D cylindrical tube)

Validates capillary-driven dynamics — geometric wetting boundary condition,
surface tension, pressure boundaries and viscosity contrast acting together —
against the two-fluid Washburn law. The case reproduces the imbibition test of
Sedahmed & Coelho (Phys. Fluids 36, 092117, 2024), Sec. IV.A: a wetting fluid
spontaneously imbibes a capillary tube and displaces the non-wetting fluid
under capillary suction alone (ΔP = 0), and the meniscus position L(t) is
compared with the analytic law (paper Eq. 47) for six contact angles.

## Setup

- **Geometry**: cylindrical tube of radius r = 5 along x in a 12 x 12 x 101
  domain, with 6-cell open reservoir chambers at both ends. The reservoirs
  contain no solid (periodic in y and z) and are therefore frictionless.
- **Boundary conditions**: per-color Zou-He pressure inlet and outlet at equal
  density (`bc.zou_he_inlet` / `bc.zou_he_outlet`, paper App. B); bounce-back
  tube wall with the geometric wetting BC (akai reorientation).
- **Fluids**: γ = 1/45, viscosity ratio M = ν_nw/ν_w = 1/5 (τ_w = 1.0,
  τ_nw = 0.6), β = 0.95; default solver stack (MRT, CSF surface-tension force
  F = ½σκ∇φ with the paper's Eq. 34/35 solid extrapolation, Eq. 17 recoloring
  `recolor_emag`).
- **Sweep**: imposed wetting-phase contact angles θ = 20°, 30°, 40°, 50°, 60°,
  70°; 30 000 steps each (20 blocks of 1500); float64.

The two-fluid Washburn law balances the capillary pressure against the viscous
drag of both fluid columns,

    a L² + b L = A t + const,   a = (μ_w − μ_nw)/2,   b = μ_nw L_tube,   A = r γ cosθ / 4,

so the rectified variable a L² + b L is linear in t with slope A
(`bob.utils.washburn`).

## How to run

```bash
uv run python demo/washburn/washburn_demo.py                # GPU if available
uv run python demo/washburn/washburn_demo.py --curve-only   # physics + L(t) data/report, no 3D renders
uv run python demo/washburn/washburn_demo.py --plots        # also write the analysis curves
uv run python demo/washburn/publication_plot.py             # paper figure from output/washburn_Lt.csv
```

## Validation criterion

For every angle, the slope A_meas of a linear fit to the rectified L(t) (index
window 10–95 % of the record) is compared with the analytic A. The demo reports
PASS when 0.6 < A_meas/A_analytic < 1.3, R² > 0.97 and L(t) is finite for all
six angles; A_meas/A_analytic = 1 is exact agreement. `report.md` tabulates
A_meas/A_analytic, R² and L_final/r per angle.

## Outputs

In `output/` (gitignored): `washburn_Lt.csv` (the L(t) series per angle),
`imbibition_grid.png` (final 3D state of the six tubes, tube vertical, wall
transparent), `imbibition.gif` (the same grid animated), `report.md`;
`publication_plot.py` adds `washburn_Lt_pub.(pdf|png)`. The analysis curve
`washburn_Lt.png` / `.svg` (L(t): LBM points against the analytic curves, one
color per angle) is written only with `--plots`.

## References

- M. Sedahmed, R. C. V. Coelho, "Wetting and pressure gradient performance in a lattice
  Boltzmann color gradient model", Phys. Fluids 36, 092117 (2024),
  DOI 10.1063/5.0228835 — Sec. IV.A (test case), Eq. 47
  (two-fluid Washburn law), App. B (pressure boundary conditions).
- Leclaire et al. (2017), Eq. 60 — two-fluid Washburn law as implemented in
  `bob.utils.washburn`.
- Akai, Bijeljic & Blunt, Adv. Water Resour. 116 (2018) — wetting boundary
  condition.
