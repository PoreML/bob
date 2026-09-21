# Physical unit mapping — flip-chip underfill runs

The runs are defined entirely in lattice units (the geometry dataset's `flipchip.toml`
and the staged `*_geometry.json` give every length in cells). This document
fixes a mapping to physical units. It is pure post-processing: no run data
depends on it.

## The three conversion scales

A lattice-Boltzmann run has exactly three independent conversion scales: length
`dx` (m/voxel), time `dt` (s/step) and density `C_ρ` (kg/m³ per lattice unit).
Everything else is a fixed combination — there is no separate freedom for
viscosity, surface tension or pressure:

| quantity            | factor            |
|---------------------|-------------------|
| velocity            | `dx/dt`           |
| kinematic viscosity | `dx²/dt`          |
| dynamic viscosity   | `C_ρ·dx²/dt`      |
| surface tension     | `C_ρ·dx³/dt²`     |
| pressure            | `C_ρ·dx²/dt²`     |

The runs fix the lattice numbers (σ_lat = 0.05, ν_blue = 0.025,
ν_red = 0.025·M, ρ_lat = 1, Zou-He ΔP = 0 wicking — `underfill_demo.py`), so at
most three physical quantities can be matched; the rest are implied and are
checked through dimensionless groups (below).

## Reference material (representative capillary underfill at flow temp ~110 °C)

For a specific product, substitute datasheet values and re-derive with the formulas below.

| property | value |
|---|---|
| resin dynamic viscosity μ | 0.1 Pa·s |
| surface tension σ | 0.035 N/m |
| density ρ | 1700 kg/m³ |
| capillary speed σ/μ | 0.35 m/s |
| air (displaced) μ | 1.8e-5 Pa·s → real M ≈ 5600 |

## Anchors

1. **Length — geometry.** D = 32 cells ↔ 80 µm C4 bump ⇒ **dx = 2.5 µm**.
   Checks: gap 24 c = 60 µm standoff, pitch 64 c = 160 µm, domain
   482×476 c ≈ 1.2×1.2 mm, D range 32–54 c = 80–135 µm bump variation.
   (Alternative: re-anchor per sample so every bump is 80 µm; then dx varies
   1.5–2.5 µm per run.)
2. **Time — match the wicking speed** (red = resin):
   `dt = dx · (σ_lat/μ_lat,red) / (σ/μ)_phys = dx · (2/M) / 0.35` ⇒
   **dt = 14.29/M µs**. This is chosen over matching μ directly because ΔP = 0
   wicking dynamics depend on σ and μ only through σ/μ (plus θ and geometry), so
   fill times and front speeds map correctly.
3. **Density:** **C_ρ = 1700 kg/m³** (only enters inertia/gravity — negligible).

## Per-M conversion table (dx = 2.5 µm)

| M | dt | 1 lat velocity | ν factor dx²/dt | pressure factor | 100k steps |
|---|-----|------|----------|---------|--------|
| 5 | 2.86 µs | 0.875 m/s | 2.19e-6 m²/s | 1.3 kPa | 0.286 s |
| 10 | 1.43 µs | 1.75 m/s | 4.37e-6 m²/s | 5.2 kPa | 0.143 s |
| 20 | 714 ns | 3.50 m/s | 8.75e-6 m²/s | 20.8 kPa | 0.071 s |
| 30 | 476 ns | 5.25 m/s | 1.31e-5 m²/s | 46.9 kPa | 0.048 s |

## Dimensionless comparison

| group | simulation | real | status |
|---|---|---|---|
| contact angle θ | 30–60° | ~15–45° on solder mask | matched (swept) |
| geometry ratios (squish, pitch/D, throats) | exact | — | matched by construction |
| capillary speed σ/μ | matched via dt | 0.35 m/s | matched (anchor) |
| viscosity ratio M | 5–30 | ~5600 | not matched — M is the swept variable |
| front Re | ~0.01 | ~0.002 | both ≪ 1 → inertia negligible |
| Bond number (gravity) | 0 (no g) | ~2e-3 | both ≪ 1 → gravity negligible |

Consequence of the σ/μ anchor: implied μ and σ are each **λ ≈ 5400/M² times smaller**
than the real fluid's (λ ≈ 54 at M = 10) — the *same* λ, so σ/μ is exact. λ only
enters through Re/Oh (negligible, above). **Absolute pressures inherit λ too**:
a lattice Laplace jump maps ~λ× low. If a figure needs physical pressure, scale
by λ or report ΔP in units of 2σcosθ/h.

## Reading the runs in physical units (dx = 2.5 µm)

Three layers: geometry is physical immediately; dynamics need the per-M dt;
below the diffuse-interface width nothing is physical.

**Geometric (no dt needed):** each run is a 1.2 × 1.2 mm window of the die gap.
Voxel volume 15.6 µm³. Across the 32 run geometries: bumps 80–135 µm, standoff
50–115 µm, pitch 127–250 µm — the sample draw reads as manufacturing variation. Void of N
voxels → equivalent spherical diameter d = 2.5 µm · (6N/π)^⅓. Saturation/fill
fraction: dimensionless, unchanged.

**Dynamic (per-M dt = 14.29/M µs):** steps → seconds, front speed ≈ 2 mm/s,
array fill ≈ 0.4 s for every M (all runs map to the same resin; step-count
differences follow from the per-M dt). The physical effect of M is what breaks
this collapse: void counts and air-drag deviations.

**Resolution floor:** interface width 4–5 cells ≈ 10–12 µm. Features below it
are not resolved: voids are meaningful only above ~20 µm equivalent diameter
(~270 voxels, comparable to the acoustic-microscopy detection scale), throats
at ≥ 6 cells = 15 µm, and there are no sub-interface films or menisci
(disconnected sub-scale blobs are recoloring mist, not nucleation).

**Excluded physics (sub-voxel or not in the model) and its effect:**
silica filler (0.1–10 µm) is unresolved → the simulated resin is the homogeneous effective fluid;
the anchored μ is the *filled*-resin viscosity, but filler entrapment/plowing
is out of scope. Newtonian μ: fill shear rates ~30 s⁻¹ sit where datasheet
viscosities are quoted. Wall roughness < 2.5 µm folded into the effective θ.
Air incompressible: real trapped voids compress ~2σ/r ÷ 1 atm ≈ 7 % at
r = 10 µm and can slowly dissolve — entrapment volumes map, later shrinkage
does not. Isothermal: capillary fill (~seconds) finishes before cure kinetics
matter.

## Consistency checks

- **Physical fill time is ~M-independent**: fill steps ∝ M (measured in the runs) and
  dt ∝ 1/M ⇒ every run ≈ 0.43 s to fill the 0.9 mm array — correct, since all
  runs map to the same resin at the same wicking speed. Residual deviations from
  step-linearity in M are the physical effect of M (voids, air drag).
- Front speed maps to ~2 mm/s; Washburn √t from 1 mm to a 10 mm die ⇒ ~45 s —
  inside the reported tens-of-seconds range for capillary underfill.
