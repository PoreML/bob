# resolution — grid-resolution study of capillary drainage

Runs the same drainage experiment on one sphere-pack geometry at three grid resolutions
(256³ original, 128³ and 64³ downsamples) to quantify how voxel resolution affects the
invasion pattern, breakthrough saturation and capillary pressure of a color-gradient
drainage simulation.

## Setup

The driver is `demo/lenormand/lenormand_demo.py` in its capillary-fingering configuration:
Ca = 10⁻⁵, viscosity ratio M = 30, σ = 0.05, ν₀ = 0.04, θ = 135°, rate-controlled piston
inlet, MRT with χ = 0.8, fp32. Only the geometry changes between the three runs. Each run
stops at breakthrough, writes HDF5 field snapshots every `FIELD_EVERY` blocks, and resumes
from its newest checkpoint when resubmitted.

The geometries come from the resolution-study dataset
(`resolution_0000.npy`, `resolution_0000_ds128.npy`, `resolution_0000_ds64.npy`). Set
`STRUCT_DIR` to their location; the default assumes a `geometry` checkout next to this
repository (`../geometry/data/resolution`).

## How to run

From the repository root:

```bash
bash demo/resolution/submit_all.sh          # submits all three runs
# or a single resolution:
SIZE=128 GPUS=2 sbatch --job-name=res128_ca1e-5 --gres=gpu:2 demo/resolution/run_resolution.slurm
```

GPU counts: 256³ uses 8 GPUs (z-sharded through `bob.multigpu`), 128³ uses 2, 64³ uses 1.
The partition, QOS and resource limits in `run_resolution.slurm` are site-specific.

## Outputs

`demo/resolution/output/res<SIZE>/` (gitignored): the artifacts of the Lenormand driver —
metrics, HDF5 fields and checkpoints — plus the SLURM logs in `demo/resolution/output/`.
See `demo/lenormand/README.md` for the artifact description.
