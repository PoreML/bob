"""PyVista rendering helpers for 3D demos (off-screen).

One consistent look across the 3D cases: the red (non-wetting) phase rendered
opaque, the wetting blue hidden entirely, solid walls/rock translucent gray.
Voxel threshold rendering (cell data) by default — no smoothing that could
misrepresent the interface thickness.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyvista as pv
from PIL import Image

# Flips to True on the first all-black in-process screenshot (headless EGL +
# CUDA on the same GPU — see _render) so every later frame plots in a clean
# subprocess instead of paying the ~20 s driver stall per frame.
_use_subprocess = False


def _as_grid(rhoN, solid):
    nz, ny, nx = rhoN.shape
    grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1))
    # cell data flattens with x fastest — matches C-order ravel of (z, y, x)
    masked = rhoN if solid is None else np.where(solid, -1.0, rhoN)  # walls never read as red
    grid.cell_data["rhoN"] = masked.ravel()
    if solid is not None:
        grid.cell_data["solid"] = solid.astype(float).ravel()
    return grid


def flow_camera(shape, side=0.75, dist=2.1, height=0.85):
    """Camera for x-axis flow renders: flow runs LEFT -> RIGHT on screen.

    The default iso view puts +x receding to the back-right, so an invasion
    front reads as marching right-back-face -> front-left — backwards. Placing
    the camera on the -y side (screen right = view x up = +x) fixes the
    direction; ``side``/``height`` tilt it toward -x/+z for a 3/4 look-down angle
    (``height`` raises the elevation — looking further down onto the domain).

    Only the *direction* of the offset sets the view angle: ``_render`` calls
    ``reset_camera`` for explicit cameras, so the whole domain is framed
    automatically for any ``shape`` — no per-case distance/zoom tuning.
    ``shape`` is the (nz, ny, nx) lattice shape.
    """
    nz, ny, nx = shape
    center = (nx / 2.0, ny / 2.0, nz / 2.0)
    span = max(nx, ny, nz)
    pos = (center[0] + side * span, center[1] - dist * span, center[2] + height * span)
    return [pos, center, (0.0, 0.0, 1.0)]


def _render(rhoN, solid, window, rock_opacity, text, camera="iso", smooth=False, zoom=1.3, shade=False, font_size=11):
    """Render one phase view and return the screenshot as an (h, w, 3) array.

    ``smooth=False`` (default) keeps the voxel threshold look. Voxel surfaces
    only have six axis-aligned normals, so an axis-aligned camera sees one
    normal everywhere and the lighting goes flat; ``smooth=True`` contours the
    fields into triangulated isosurfaces with continuously varying normals —
    right for cases validating smooth analytical shapes (demo/droplet_on_sphere).

    ``shade=True`` lights the **voxel** threshold for a nicer, more 3D look
    WITHOUT contouring the data (the geometry stays the blocky voxel volume):
    the red surface is extracted, given interpolated (smooth-shaded) normals and
    a specular highlight, and a three-point light kit replaces the flat default
    so the curved silhouette reads with depth. Orthogonal to ``smooth`` —
    ``smooth`` changes the geometry, ``shade`` only changes the lighting.
    """
    kwargs = {
        "window": window, "rock_opacity": rock_opacity, "text": text, "camera": camera,
        "smooth": smooth, "zoom": zoom, "shade": shade, "font_size": font_size,
    }
    global _use_subprocess

    def local():
        return _render_local(rhoN, solid, kwargs)

    def sub():
        return _render_subprocess(rhoN, solid, kwargs)

    # Headless nodes (no X) fall back to vtkEGLRenderWindow; an EGL context
    # created while CUDA is active on the GPU — this process's jax OR another
    # process's — stalls ~20-27 s and screenshots all black (driver interop
    # quirk), and a real frame is never all-black (light background). A fresh
    # process gets a clean EGL context (fast + valid), so after a black shot we
    # plot in a subprocess and stick with it; a local success unsticks. If every
    # attempt comes back black (transiently poisoned GPU), raise — callers treat
    # rendering as best-effort, and a black frame silently saved is worse.
    attempts = (sub, local, sub) if _use_subprocess else (local, sub, local)
    for fn in attempts:
        try:
            shot = fn()
        except (OSError, RuntimeError, subprocess.SubprocessError):
            continue
        if shot.any():
            escalated = fn is sub
            if escalated and not _use_subprocess:
                print("[viz3d] black in-process render (EGL/CUDA clash) — switching to subprocess plotting")
            _use_subprocess = escalated
            return shot
    raise RuntimeError("all render attempts returned black (EGL/CUDA clash) — frame not saved")


def _render_local(rhoN, solid, kw):
    """One plotter pass (see _render for the kwargs)."""
    window, camera = kw["window"], kw["camera"]
    smooth, shade = kw["smooth"], kw["shade"]
    grid = _as_grid(np.asarray(rhoN), None if solid is None else np.asarray(solid))
    pl = pv.Plotter(off_screen=True, window_size=list(window))
    pl.add_mesh(grid.outline(), color="#444444")
    source = grid.cell_data_to_point_data() if smooth else grid
    if solid is not None:
        rock = source.contour([0.5], scalars="solid") if smooth else grid.threshold(0.5, scalars="solid")
        if rock.n_cells:
            pl.add_mesh(rock, color="#8a8a8a", opacity=kw["rock_opacity"], smooth_shading=smooth, show_scalar_bar=False)
    red = source.contour([0.0], scalars="rhoN") if smooth else grid.threshold(0.0, scalars="rhoN")
    if red.n_cells:
        if shade and not smooth:
            red = red.extract_surface()  # surface polydata so smooth_shading can interpolate normals
        lit = smooth or shade
        pl.add_mesh(
            red, color="#d62728", smooth_shading=lit, specular=0.5 if lit else 0.0,
            specular_power=15 if lit else 1.0, show_scalar_bar=False,
        )
    if shade:
        pl.enable_lightkit()  # key/fill/back lights — shaded gradient across the dome, vs flat default
    if kw["text"]:
        pl.add_text(kw["text"], position="upper_left", font_size=kw["font_size"], color="#222222")
    pl.camera_position = camera if isinstance(camera, str) else [tuple(v) for v in camera]
    # String presets ("iso"/"xz"/...) already fit the domain to the viewport; an
    # explicit [pos, focal, up] (e.g. flow_camera) does not, so refit along its
    # view direction — keeps the whole domain in frame for any shape.
    if not isinstance(camera, str):
        pl.reset_camera()
    pl.camera.zoom(kw["zoom"])
    shot = pl.screenshot(return_img=True)
    pl.close()
    return shot


def _render_subprocess(rhoN, solid, kwargs):
    """Plot in a fresh interpreter running this file as a script (imports only
    numpy/pyvista/PIL — no jax, so its EGL context never sees CUDA)."""
    kw = dict(kwargs)
    if not isinstance(kw["camera"], str):
        kw["camera"] = [list(map(float, v)) for v in kw["camera"]]
    with tempfile.TemporaryDirectory() as td:
        payload, out = Path(td) / "payload.npz", Path(td) / "shot.npy"
        arrays = {"rhoN": np.asarray(rhoN)}
        if solid is not None:
            arrays["solid"] = np.asarray(solid)
        np.savez(payload, kwargs=json.dumps(kw), **arrays)
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), str(payload), str(out)],
            check=True, capture_output=True, timeout=600,
        )
        shot = np.load(out)
    if not shot.any():
        raise RuntimeError("subprocess render returned an all-black frame")
    return shot


def phase_frame(
    path: Path, rhoN, solid=None, window=(1024, 768), rock_opacity=0.12, text=None, camera="iso", smooth=False,
    zoom=1.3, shade=False, font_size=11,
):
    """Render one frame to ``path``: red phase (rhoN > 0) opaque, solid (if any)
    translucent, wetting blue hidden. ``camera`` is any PyVista camera_position
    (default isometric; e.g. "xz" looks side-on with z up, or ``flow_camera(shape)``
    for left-to-right flow). ``smooth=True`` swaps voxel thresholds for shaded
    isosurfaces (see ``_render``). ``shade=True`` keeps the raw voxel threshold
    but lights it (interpolated normals + light kit) for a nicer 3D look. ``zoom=1.0``
    keeps the whole domain in frame (the 1.3 default crops in for single-droplet cases).
    ``font_size`` sizes the overlay ``text`` (publication renders want ~18)."""
    Image.fromarray(
        _render(
            rhoN, solid, window, rock_opacity, text, camera=camera, smooth=smooth, zoom=zoom, shade=shade,
            font_size=font_size,
        )
    ).save(path)


def phase_row(path: Path, items, window=(480, 420), rock_opacity=0.25):
    """Side-by-side renders in one image: ``items`` is a list of
    (rhoN, solid_or_None, label) triples — the 3D analogue of a panel row."""
    shots = [_render(rhoN, solid, window, rock_opacity, label) for rhoN, solid, label in items]
    Image.fromarray(np.hstack(shots)).save(path)


if __name__ == "__main__":  # subprocess worker: viz3d.py <payload.npz> <shot.npy>
    _data = np.load(sys.argv[1])
    _kw = json.loads(str(_data["kwargs"]))
    _kw["window"] = tuple(_kw["window"])
    np.save(sys.argv[2], _render_local(_data["rhoN"], _data["solid"] if "solid" in _data.files else None, _kw))
