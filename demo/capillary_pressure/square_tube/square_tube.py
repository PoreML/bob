"""Square capillary tube — multi-theta primary drainage (shared capillary_drainage driver).

Medium resolution: 32 cross-section, central 8x8 square pore (inscribed R = a/2 = 4).
Sweeps theta in {120,135,150,165}; overlay P_c-S_w plot + per-theta 3D & 2D drainage GIFs,
plus geometry illustrations (3D translucent walls + 2D cross-section).

Usage:  uv run python demo/capillary_pressure/square_tube/square_tube.py
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
    n, a, duct, buf = 32, 8, 24, 8                          # 2 pressure + 6 buffer
    R = a / 2.0
    cross = C.square_cross(n, a)
    solid = C.build_tube(cross, duct, buf)
    out_dir = Path("demo/capillary_pressure/square_tube/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    C.render_geometry(solid, out_dir / "geometry_3d.png",
                      title=f"square tube walls (a={a}, buffer excluded)")
    C.render_cross_section(cross, R, out_dir / "geometry_2d.png",
                           rf"square pore  $a={a}$  in  ${n}\times{n}$ cross-section")
    C.run_case(f"square tube {n} (a={a})", solid, R, x_in=buf, x_duct=(buf, solid.shape[2] - buf),
               thetas=[120, 135, 150, 165], plot_title="Square Tube", out_dir=str(out_dir))
