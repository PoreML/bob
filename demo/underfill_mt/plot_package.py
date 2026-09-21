"""Plot the MT6225A package geometry before simulating: ball map from the datasheet
JSON, voxelized gap views at the simulation resolution, and a 3D render of the
solid (substrate + die + truncated solder balls).

Usage:  uv run python demo/underfill_mt/plot_package.py     # artifacts in demo/underfill_mt/output/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from geometry import DX_MM, build_gap_mask, load_package
from matplotlib.patches import Circle, Rectangle

OUT = Path(__file__).resolve().parent / "output"


def ball_map(pkg, path):
    """Datasheet-space plan view: 264 balls to scale, 25 depopulated sites marked."""
    D, E = pkg["body"]["D"], pkg["body"]["E"]
    r = pkg["ball_diameter_b"] / 2.0
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.add_patch(Rectangle((-D / 2, -E / 2), D, E, fill=False, ec="k", lw=1.5, label=f"die {D:g}x{E:g} mm"))
    for b in pkg["balls"]:
        ax.add_patch(Circle((b["x"], b["y"]), r, fc="#b0b0b0", ec="#606060", lw=0.5))
    dep = pkg["depopulated"]
    ax.plot(
        [d["x"] for d in dep],
        [d["y"] for d in dep],
        "x",
        color="crimson",
        ms=7,
        mew=1.5,
        ls="none",
        label=f"depopulated ({len(dep)})",
    )
    ax.annotate(
        "I-type dispense edge",
        xy=(-D / 2, 0),
        xytext=(-D / 2 - 1.6, 0),
        rotation=90,
        ha="center",
        va="center",
        color="tab:red",
        fontsize=11,
    )
    ax.arrow(-D / 2 - 0.55, -1.8, 0, 3.6, width=0.03, color="tab:red", alpha=0.4)
    ax.set_xlim(-D / 2 - 2.0, D / 2 + 0.8)
    ax.set_ylim(-E / 2 - 0.8, E / 2 + 0.8)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm) — flow axis")
    ax.set_ylabel("y (mm)")
    ax.set_title(
        f"{pkg['part']} {pkg['package']} — {pkg['ball_count']} balls, pitch {pkg['ball_pitch_e']:g} mm, "
        f"ball dia {pkg['ball_diameter_b']:g} mm, standoff {pkg['standoff_A1']:g} mm"
    )
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def voxel_views(solid, info, path):
    """Simulation-space views: mid-gap plan slice, x-z section through ball row J
    (y=0), and a zoom showing the truncated-ball cross-sections at dx resolution."""
    nz, ny, nx = solid.shape
    xs, ys, zs = info["xs"], info["ys"], info["zs"]
    ext_xy = (xs[0], xs[-1], ys[0], ys[-1])
    kmid = nz // 2
    jmid = int(np.argmin(np.abs(ys - 0.0)))  # row J (y = 0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), height_ratios=(4, 1.4))
    a = axes[0, 0]
    a.imshow(solid[kmid], origin="lower", extent=ext_xy, cmap="Greys", interpolation="none", aspect="equal")
    a.set_title(f"mid-gap slice z = {zs[kmid]:.3f} mm (solid = dark)")
    a.set_xlabel("x (mm)")
    a.set_ylabel("y (mm)")

    a = axes[0, 1]
    zoom = solid[kmid][(np.abs(ys) < 1.7)][:, (np.abs(xs) < 1.7)]
    a.imshow(zoom, origin="lower", extent=(-1.7, 1.7, -1.7, 1.7), cmap="Greys", interpolation="none", aspect="equal")
    a.set_title(f"center zoom (depopulated G7-L11 region), dx = {info['dx']:g} mm")
    a.set_xlabel("x (mm)")
    a.set_ylabel("y (mm)")

    a = axes[1, 0]
    a.imshow(
        solid[:, jmid, :],
        origin="lower",
        extent=(xs[0], xs[-1], zs[0] - info["dx"] / 2, zs[-1] + info["dx"] / 2),
        cmap="Greys",
        interpolation="none",
        aspect="auto",
    )
    a.set_title(f"x-z section through row J (y = {ys[jmid]:.3f} mm): substrate | gap | die")
    a.set_xlabel("x (mm)")
    a.set_ylabel("z (mm)")

    a = axes[1, 1]
    sel = np.abs(xs) < 1.7
    a.imshow(
        solid[:, jmid, :][:, sel],
        origin="lower",
        extent=(-1.7, 1.7, zs[0] - info["dx"] / 2, zs[-1] + info["dx"] / 2),
        cmap="Greys",
        interpolation="none",
        aspect="auto",
    )
    a.set_title("x-z zoom: truncated balls, 7 fluid layers across the 0.21 mm gap")
    a.set_xlabel("x (mm)")
    a.set_ylabel("z (mm)")
    fig.suptitle(f"Voxelized underfill gap {nz}x{ny}x{nx} (z,y,x) — gap porosity {info['gap_porosity']:.3f}", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def front_camera(shape, dist=2.0, height=0.15):
    """Camera in front of the dispense edge (low x), up = +z: the elevation view
    into the open gap. ``height`` lifts it slightly so the die top edge reads."""
    nz, ny, nx = shape
    c = (nx / 2.0, ny / 2.0, nz / 2.0)
    span = max(nx, ny, nz)
    return [(c[0] - dist * span, c[1], c[2] + height * span), c, (0.0, 0.0, 1.0)]


def face_camera(shape, dist=0.75, height=2.6):
    """Near-face-on view of the square 12x12 face: mostly straight down, tilted
    ~16 deg toward the viewer only (no sideways offset — that would roll the
    square in-plane), so the edges and thickness still read as 3D."""
    nz, ny, nx = shape
    c = (nx / 2.0, ny / 2.0, nz / 2.0)
    span = max(nx, ny, nz)
    return [(c[0], c[1] - dist * span, c[2] + height * span), c, (0.0, 0.0, 1.0)]


def render_3d(solid, path, camera=None, zoom=1.1, window=(1280, 720)):
    """Best-effort PyVista render of the solid alone (all-blue phase field hides fluid)."""
    from bob.utils import viz3d

    rhoN = np.full(solid.shape, -1.0)
    cam = camera if camera is not None else viz3d.flow_camera(solid.shape)
    viz3d.phase_frame(path, rhoN, solid, rock_opacity=0.35, camera=cam, zoom=zoom, window=window)


def main():
    OUT.mkdir(exist_ok=True)
    pkg = load_package()
    solid, info = build_gap_mask(pkg, dx=DX_MM)
    ball_map(pkg, OUT / "package_ball_map.png")
    voxel_views(solid, info, OUT / "package_voxel_views.png")
    try:
        render_3d(solid, OUT / "package_3d.png")
        render_3d(solid, OUT / "package_face.png", camera=face_camera(solid.shape), zoom=1.25, window=(1000, 1000))
        render_3d(solid, OUT / "package_front.png", camera=front_camera(solid.shape), zoom=2.6, window=(1600, 420))
        # front zoom: dispense-edge corner of the gap — first ball rows between the plates
        nz, ny, nx = solid.shape
        crop = solid[:, ny // 2 - 60 : ny // 2 + 60, :110]
        render_3d(
            crop, OUT / "package_front_zoom.png", camera=front_camera(crop.shape, height=0.35), zoom=1.5, window=(1280, 640)
        )
    except Exception as e:  # PyVista/EGL is best-effort on headless nodes
        print(f"[3D render unavailable: {type(e).__name__}: {e}]")
    nz, ny, nx = info["shape"]
    print(f"domain (nz, ny, nx) = {info['shape']}  ({nz * ny * nx / 1e6:.2f} M cells)")
    print(f"gap porosity (chip interior) = {info['gap_porosity']:.4f}   balls = {info['n_balls']}")
    print(f"artifacts: {OUT}/package_ball_map.png, package_voxel_views.png, package_3d.png")


if __name__ == "__main__":
    main()
