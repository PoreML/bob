"""Circular capillary tube — multi-theta primary drainage (shared capillary_drainage driver).

Medium resolution: 32 cross-section, central circular pore of drawn radius r=6. The circle is
the textbook Laplace throat (entry = -2 cos theta / R exactly), but the solver sees the voxelized
circle, so both R (inscribed, distance transform) and the entry coeff (Mason-Morrow
1 + 2 sqrt(pi G), which reduces to 2 for a perfect circle) are measured from the discrete
cross-section via `throat_coeff`, so analytic and simulated entry refer to the same geometry.
Sweeps theta in {120,135,150,165}; overlay P_c-S_w plot + per-theta 3D & 2D drainage GIFs.

Usage:  uv run python demo/capillary_pressure/circular_tube/circular_tube.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import capillary_drainage as C

from bob.utils import viz

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plots", action="store_true",
                    help="also write the analysis curves (pc_curve.png, pc_curve.pdf); off by default")
    args = ap.parse_args()
    if args.plots:
        viz.enable_plots(True)
    n, r, duct, buf = 32, 6, 24, 8                          # 2 pressure + 6 buffer
    cross = C.circle_cross(n, r)
    solid = C.build_tube(cross, duct, buf)
    x_duct = (buf, solid.shape[2] - buf)
    R, coeff, G = C.throat_coeff(solid, x_duct)             # measured from the discrete circle
    print(f"voxel circle r={r}: R_inscribed={R:.2f}  G=A/O^2={G:.4f}  "
          f"entry coeff (1+2sqrt(piG))={coeff:.3f}  (perfect circle: R={r}, coeff=2)")
    out_dir = Path("demo/capillary_pressure/circular_tube/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    C.render_geometry(solid, out_dir / "geometry_3d.png",
                      title=f"circular tube walls (r={r}, buffer excluded)")
    C.render_cross_section(cross, R, out_dir / "geometry_2d.png",
                           rf"circular pore  $r={r}$  in  ${n}\times{n}$ cross-section")
    C.run_case(f"circular tube {n} (r={r})", solid, R, x_in=buf, x_duct=x_duct,
               thetas=[120, 135, 150, 165], plot_title="Circular Tube", out_dir=str(out_dir),
               entry_coeff=coeff, entry_form=r"-(1+2\sqrt{\pi G})\cos\theta")
