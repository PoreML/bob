"""Rectangular capillary tube — multi-theta primary drainage (shared capillary_drainage driver).

Medium resolution: 32 cross-section, central 24x12 rectangular pore. Non-dimensionalized by the
inscribed radius R = L2/2 = 6 (same convention as the square's a/2). An elongated rectangle's
inscribed circle only touches the long walls, so the drainage entry is the curvature-correct
Princen rectangular form P_c R/gamma = (1 + L2/L1) cos theta_w (= 2 cos theta_w only when L1=L2,
i.e. the square), not the circular 2 cos theta_w. Sweeps theta in {120,135,150,165}; overlay
P_c-S_w plot + per-theta 3D & 2D drainage GIFs.

Usage:  uv run python demo/capillary_pressure/rectangular_tube/rectangular_tube.py
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
    n, L1, L2, duct, buf = 32, 24, 12, 24, 8
    R = L2 / 2.0                                            # inscribed radius; L1>=L2
    coeff = 1.0 + L2 / L1                                   # Princen rect entry = 2(1/L1+1/L2)*R_ins (=2 when square)
    cross = C.rect_cross(n, L1, L2)
    solid = C.build_tube(cross, duct, buf)
    out_dir = Path("demo/capillary_pressure/rectangular_tube/output")
    out_dir.mkdir(parents=True, exist_ok=True)
    C.render_geometry(solid, out_dir / "geometry_3d.png",
                      title=f"rect tube walls ({L1}x{L2}, buffer excluded)")
    C.render_cross_section(cross, R, out_dir / "geometry_2d.png",
                           rf"rect pore  $L_1={L1},\,L_2={L2}$  in  ${n}\times{n}$ cross-section")
    C.run_case(f"rect tube {n} ({L1}x{L2})", solid, R, x_in=buf, x_duct=(buf, solid.shape[2] - buf),
               thetas=[120, 135, 150, 165], plot_title="Rectangular Tube", out_dir=str(out_dir),
               entry_coeff=coeff, entry_form=r"-(1+L_2/L_1)\cos\theta")
