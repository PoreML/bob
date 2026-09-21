# Wetting — sessile droplets on a flat wall (3D contact-angle sweep)

Validates the geometric wetting boundary condition: a red half-ball placed on
the bottom z-wall relaxes to the contact angle imposed through `Params.theta`
(the red contact angle in degrees), applied with precomputed wall normals
(`color3d.wall_normals`) and the default closed-form reorientation of Akai,
Bijeljic & Blunt (Adv. Water Resour. 116, 2018). Nine imposed angles from 30°
to 150° are swept and the realised angle is measured from the droplet shape.

## Setup

- **Domain**: 60 x 104 x 104 (z, y, x), sealed by solid walls at both z = 0 and
  z = nz−1; x and y are periodic and wide enough that the low-angle puddle does
  not wrap. Walls on both sides of the wall-normal axis are required: a
  one-sided wall under a periodic axis biases the near-wall color gradient and
  pulls the realised angle towards 90°.
- **Initial state**: red half-ball of radius R = 20 resting on the bottom wall,
  blue elsewhere.
- **Parameters**: ω = 1.0, σ = 0.02, β = 0.95, default solver stack (MRT + akai
  reorientation + CSF surface tension + `recolor_emag` + `wall_grad="fluid"`),
  float64. Imposed angles: 30°, 45°, 60°, 75°, 90°, 105°, 120°, 135°, 150°.
- **Run length**: every angle runs the same 20 000 steps; the contact angle
  (`misc.contact_angle`) and a phase-field snapshot are recorded every 200 steps.

## How to run

```bash
uv run python demo/wetting/wetting_demo.py     # GPU if available
uv run python demo/wetting/wetting_demo.py --plots   # also write the analysis curves
```

## Validation criterion and results

The measured angle must increase monotonically with the imposed angle
(`report.md` states PASS/FAIL) and each convergence track must settle on its
imposed target. With the default solver the realised angle tracks the imposed
angle to within about 1–4° over the whole 30°–150° range, with a mild overshoot
at the 150° extreme. The reorientation itself is exact; the residual is a
property of the single-layer geometric boundary condition combined with the
recoloring step.

## Outputs

In `output/` (gitignored): `sessile_grid.png` (3x3 grid of the equilibrated
droplets), `wetting.gif` (the grid animated during relaxation), `metrics.csv`
(imposed and measured angle), `report.md`. The analysis curve `convergence.png`
(measured angle against time step, one track per case, imposed values dashed) is
written only with `--plots`. Renders are 3D PyVista views seen side-on (z up):
red phase opaque, blue hidden, walls translucent, voxel threshold without smoothing.

## References

- Akai, Bijeljic & Blunt, Adv. Water Resour. 116 (2018) — closed-form wetting
  boundary condition.
