# Droplet on a spherical solid (3D)

Validates the geometric wetting boundary condition (default akai reorientation) on a curved 3D
surface, after Akai, Bijeljic & Blunt (Adv. Water Resour. 2018, Fig. 9/10): a red droplet
relaxes on a spherical solid until it matches the analytical sphere-on-sphere solution. With
droplet radius of curvature R1, solid radius R2 and contact angle θ, the centre-to-centre
distance is

```
R3 = sqrt(R1² + R2² − 2·R1·R2·cos θ)
```

## Setup

- **Geometry.** 101³ closed box (bounce-back walls on all faces) with a solid sphere of radius
  R2 = 20 centred at (50, 50, 30); target droplet radius of curvature R1 = 22.5.
- **Initial condition.** A half-shell collar of red coating the upper hemisphere of the solid
  (paper Fig. 10a), volume-matched to the analytical droplet.
- **Parameters.** bob's default solver stack with ω = 1, σ = 0.02, β = 0.7; contact angles
  θ = 60° and 120°; 30 000 relaxation steps per angle; fp64.
- **Solid radius.** The paper text gives the solid radius as "fixed at 40 lattice units", but
  with centre (50, 50, 30) in a 101³ box that solid would extend below the floor and the
  θ = 120° analytical droplet would reach z ≈ 107, outside the domain and unlike Fig. 10, where
  everything sits inside. The figure's proportions match R2 = 20, which is the default here
  (`--R2` overrides).

## How to run

```bash
uv run python demo/droplet_on_sphere/droplet_on_sphere_demo.py   # both angles
./demo/droplet_on_sphere/run_theta060.sh                           # θ = 60 only
./demo/droplet_on_sphere/run_theta120.sh                           # θ = 120 only
uv run python demo/droplet_on_sphere/droplet_on_sphere_demo.py --steps 4000 --block 500  # smoke run
uv run python demo/droplet_on_sphere/droplet_on_sphere_demo.py --plots   # also write the analysis curves
```

The per-angle scripts launch through `scripts/gpu-run.sh` and tag their artifacts (`t060_*`,
`t120_*`) so both can coexist in `output/`.

## Validation and results

A least-squares sphere is fitted to the free red/blue interface (boundary voxels more than 4
cells clear of the solid); R1 and R3 are read from the fit and θ follows from inverting the law
of cosines. **PASS:** θ within 12° of the imposed angle and R1 within 10% of the target, for
every angle.

Reference run (H100):

| imposed θ | θ measured | R1 fit (target 22.5) |
| ---: | ---: | ---: |
| 60° | 60.2° | 22.01 |
| 120° | 126.8° | 21.90 |

Overall: PASS.

## Outputs

In `output/` (gitignored): `t{060,120}_3d.gif` (PyVista render, side-on `xz` camera with z up
so the droplet does not occlude the solid beneath it), `t{060,120}_slice.gif` (midplane y = 50,
analytical circle dotted white), `metrics.csv`, `report.md`, `frames/*.png`. The analysis plot
`plots/comparison.png` is written only with `--plots`.

## References

- Akai, Bijeljic & Blunt, Adv. Water Resour. 116 (2018) — wetting boundary condition for the
  colour-gradient lattice-Boltzmann method; droplet on a spherical solid, Fig. 9/10.
