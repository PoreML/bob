# Laplace law and droplet relaxation (3D)

Three short D3Q19 colour-gradient droplet experiments that validate surface tension and
wetting in bob's default solver against the Young–Laplace law, Δp = 2σ/R:

1. **rounding** — an elongated red box pulls itself into a ball (sphericity → 1);
2. **sessile** — a droplet on the bottom wall relaxes to its contact angle (geometric wetting
   BC; the measured angle tracks the imposed `theta`, default 107°);
3. **laplace** — Δp versus 1/R across droplet radii, scanned over σ = 0.01 … 0.05 (the range
   used by the porous-media cases, which run at σ = 0.05), with one fitted line per σ.

## Setup

- **Rounding.** Periodic 80³ box (`--n`), red box of size n/5 × n/4 × n/2, 5000 steps.
- **Sessile.** Grid scaled from `--n` (nz = 0.65 n, droplet radius 0.225 n), solid wall on the
  bottom z-plane, wall normals from `color3d.wall_normals`, 5000 steps.
- **Laplace scan.** Seven radii evenly spaced in 1/R from R = 18 to R = 7, each in a periodic box
  of side ⌈3R + 8⌉, 2500 steps per droplet, for σ ∈ {0.01, 0.02, 0.03, 0.04, 0.05}. The pressure
  jump is fitted linearly against 1/R; the effective surface tension is σ_eff = slope / 2.
- **Parameters.** ω = 1, σ = 0.02 (rounding / sessile), β = 0.7, fp64.

## How to run

```bash
uv run python demo/laplace/laplace_demo.py                  # all three experiments
uv run python demo/laplace/laplace_demo.py --only laplace   # regenerate only the Laplace scan / plot
uv run python demo/laplace/laplace_demo.py --plots          # also write the analysis curves
uv run python demo/laplace/publication_plot.py              # publication figure from the saved CSV
```

A GPU is used automatically when available. `publication_plot.py` re-plots
`output/plots/laplace_fit.csv` without re-simulating.

## Validation and results

`report.md` records PASS/FAIL for each check:

| check | criterion |
|---|---|
| rounding | final sphericity < 1.15 |
| sessile angle | measured within 12° of the imposed angle |
| Laplace scan | every fit has R² > 0.99 and σ_eff within [0.7, 1.4] × σ |

The Laplace scan gives σ_eff ≈ 1.13 · σ_in with R² ≈ 0.999 at every σ.

## Outputs

In `output/` (gitignored): `rounding.gif`, `sessile.gif` (PyVista 3D renders: red phase opaque,
blue hidden, walls translucent), `plots/laplace_fit.csv`, `metrics.csv`, `report.md`,
`frames/*.png`; from `publication_plot.py`, `plots/laplace_fit_pub.pdf` and
`plots/laplace_fit_pub.png` (300 dpi). The analysis curves `plots/laplace_fit.svg` and
`plots/laplace_fit.png` are written only with `--plots`.
