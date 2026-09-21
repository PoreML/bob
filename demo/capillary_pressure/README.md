# Capillary pressure — primary drainage in straight tubes

Capillary-pressure curves (`P_c R/γ` versus wetting saturation `S_w`) for quasi-static primary
drainage of straight capillary tubes, following the pressure-difference protocol of Ramstad et
al. (2009). The simulated entry pressure of each tube is compared with the analytic entry
pressure of its throat: the circular Laplace value, the Princen form for a rectangle, and the
Mason–Morrow (1991) shape-factor form for a voxelized cross-section. All cases run bob's
default solver (MRT + akai + CSF Eq. 34/35 + `recolor_emag` + `wall_grad="fluid"`).

## Setup

All cases share one driver, [`capillary_drainage.py`](capillary_drainage.py); each case is a thin
script that defines its cross-section and calls `run_case(...)`.

| folder | pore | cross-section | R | analytic entry `P_c R/γ` |
|---|---|---|---|---|
| [`circular_tube/`](circular_tube/) | voxelized circle, drawn *r* = 6 | 32×32 | inscribed radius, measured | `(1 + 2√(πG)) cosθ_w` |
| [`square_tube/`](square_tube/) | square, side *a* = 8 | 32×32 | a/2 = 4 | `2 cosθ_w` |
| [`rectangular_tube/`](rectangular_tube/) | rectangle 24×12 | 32×32 | L₂/2 = 6 (inscribed) | `(1 + L₂/L₁) cosθ_w` |

- **Geometry and boundary conditions.** A walled duct (24 cells) fronted at each end by 2
  fixed-density (u = 0) pressure planes and 6 open buffer cells (domain 40×32×32). The inlet
  plane is held at pure non-wetting fluid, ρ = 1. `P_c` is raised by lowering the outlet
  (pure wetting) density, `ρ_out = 1 − P_c·γ/(R·cs²)`, so `P_c = (1 − ρ_out)·cs²`; the inlet is
  never changed.
- **Initial condition.** Red (non-wetting) at ρ_in for `x < x_in`, blue (wetting) at ρ_out
  beyond, so the domain starts in mechanical balance with the boundary conditions.
- **Sweep.** Each case sweeps four contact angles θ = 120, 135, 150, 165° (θ is the oil /
  non-wetting contact angle imposed on `Params`; the water angle is θ_w = 180° − θ = 60, 45, 30,
  15°). For each angle a 20-point `P_c` ladder, refined around that angle's analytic entry, is
  ramped quasi-statically: each pressure step continues from the previous equilibrium.
- **Equilibrium.** The run advances in 2000-step blocks and moves to the next pressure step
  when `|ΔS_w| < 2.5e-4` over a block (capped at 300 blocks; a straight tube drains by a slow
  piston-like sweep that only settles once the front exits). `S_w` is a sharp phase count
  (cells with φ < 0) over the duct pore only, excluding the buffer and pressure planes.
- **Parameters.** γ = σ = 1/45, ω = 1, β = 0.95, fp64.

### Analytic entry pressure

Non-wetting invasion of a straight tube is piston-like through the inscribed circle, so the
analytic entry depends on the throat shape:

- **Square:** the circular Laplace value `P_c R/γ = 2 cosθ_w`.
- **Elongated rectangle:** the inscribed circle only touches the long walls, so the entry is
  the Princen form `(1 + L₂/L₁) cosθ_w`, which reduces to `2 cosθ_w` only for L₁ = L₂.
- **Voxelized circle:** the solver sees the discrete cross-section, so both the inscribed
  radius `R` (distance transform) and the entry coefficient are measured from it with
  `capillary_drainage.throat_coeff`: the Mason–Morrow form `(1 + 2√(πG)) cosθ_w` with shape
  factor `G = A/O²`. The perimeter `O` is a marching-squares contour length, which measures
  curved edges correctly where a voxel staircase over-counts them; the coefficient reduces to
  2 for a perfect circle.

## How to run

```bash
uv run python demo/capillary_pressure/circular_tube/circular_tube.py
uv run python demo/capillary_pressure/square_tube/square_tube.py
uv run python demo/capillary_pressure/rectangular_tube/rectangular_tube.py
uv run python demo/capillary_pressure/circular_tube/circular_tube.py --plots   # also write the analysis curves
```

A GPU is used automatically when available. Rendering dominates the wall-clock time; frames
are written every 500 steps, at most 40 per pressure step.

## Validation

The front sweeps the uniform cross-section piston-like, so each drainage curve is a sharp step
at the entry pressure, and the realized entry tracks each pore's curvature-correct analytic
value closely (square → `2 cosθ_w`; rectangle → Princen `(1 + L₂/L₁) cosθ_w`). The comparison
is graphical — each simulated curve against the dashed analytic entry line of the same colour in
`pc_curve.png`; the driver applies no automated PASS/FAIL gate.

## Outputs

Written to `<case>/output/` (gitignored):

- `pc_curve.png` / `pc_curve.pdf` — overlay of the per-angle `P_c`–`S_w` curves with each
  analytic entry `−coeff·cosθ` as a dashed line in the same colour (publication style).
  This analysis curve is written only with `--plots`.
- `geometry_3d.png`, `geometry_2d.png` — tube walls and cross-section.
- `theta_N/drainage_3d.gif` — PyVista render of the invading (red) phase, smooth-shaded.
- `theta_N/drainage_2d.gif` — mid-plane φ and pressure slices.
- `theta_N/frames_3d/`, `theta_N/frames_2d/` — the full frame sets.

## References

- Ramstad et al. (2009) — lattice-Boltzmann primary drainage with a fixed inlet and a lowered
  outlet pressure (the protocol used here).
- Mason & Morrow (1991) — entry pressure of angular tubes from the shape factor `G = A/O²`.
- Princen — capillary entry pressure of rectangular tubes, `(1 + L₂/L₁) cosθ_w`.
