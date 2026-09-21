# Capillary-dominated imbibition on sphere packs (M = 1)

Forced imbibition at low capillary number on statistically matched synthetic sphere-pack
geometries of three domain sizes (96³, 128³, 256³), at equal viscosities (M = 1), run past
breakthrough into the saturation plateau. The configuration is the capillary-regime drainage
setup of [`demo/lenormand`](../lenormand/) with one change: the invading (red) phase is made
wetting, contact angle θ = 50° instead of the drainage value 135°. The piston front then
advances by imbibition (film and corner-flow mechanisms) rather than by drainage
(throat-entry-pressure percolation).

## Setup

- **Driver.** `demo/lenormand/lenormand_demo.py` with `--regime capillary --m 1 --sigma 0.05
  --nu0 0.04 --theta 50 --after-breakthrough-x 1.0`.
- **Physics.** Ca = 1e-5, σ = 0.05, ν₀ = 0.04 → u_in = 1.25e-5; piston inlet and blue-pinned
  Zou-He pressure outlet (`outlet_sa_red=0`); MRT with χ = 0.8; fp32; M = 1; θ = 50° on the rock.
  The piston-face slab keeps `theta_face = 0` (perfectly red-wetting face), as in drainage.
- **Run length.** Twice the breakthrough step (`--after-breakthrough-x 1.0`).
- **Geometries.** Nine synthetic sphere-pack samples, three per size, each staged in its own
  run folder together with its provenance JSON. The run folders are not part of the repository
  (`demo/imbibition/sphere/` is gitignored); stage the geometries before submitting:

```
sphere/<size>/<stem>/            # 96, 128, 256 x sphere_0000..0002 = 9 runs
├── <stem>.npy                   # geometry (True = solid)
├── <stem>_geometry.json         # provenance + throat statistics
└── ... run artifacts
```

## How to run

From the repository root:

```bash
bash demo/imbibition/submit_all.sh    # 9 SLURM jobs: 96/128 on 1 GPU, 256 on 4 GPUs (z-sharded)
# or a single run:
SIZE=96 STEM=sphere_0001 GPUS=1 sbatch --job-name=imb_96_1 --gres=gpu:1 \
  --output=demo/imbibition/sphere/96/sphere_0001/slurm-%j.log demo/imbibition/run_imbibition.slurm
```

Partition, QOS and wall-clock limits in `run_imbibition.slurm` are site-specific; adapt them to
your cluster. Jobs resume automatically from the newest checkpoint on resubmission (`FRESH=1`
restarts from step 0). With `--after-breakthrough-x`, a resume that lands past breakthrough
re-anchors the factor at the resumed step (see `lenormand_demo.py --help`); this only matters
when a run times out during the post-breakthrough tail.

## Outputs

Each run folder receives `metrics.csv`, the saturation / pressure / spurious-current / colour-ledger
curves, the 3D GIF, `fields.h5` + `.xdmf` (ParaView), status-named checkpoints, `report.md` and
the SLURM log. The demo has no PASS/FAIL gate; the colour-ledger ratio reported in `report.md`
(1.0 = leak-free injection) is the run-quality check.

## References

- Lenormand, Touboul & Zarcone, J. Fluid Mech. (1988) — displacement regimes; this demo sits in
  the capillary-dominated corner.
