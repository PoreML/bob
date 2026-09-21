"""Capillary-pressure primary-drainage driver shared by the tube cases in this folder.

For one geometry, sweeps several contact angles; for each angle it ramps the non-dimensional
capillary pressure P_c R/gamma through a 20-point ladder refined around that angle's analytic
entry pressure (`entry_coeff * cos(theta_w)` — entry_coeff = 2 for a circular Laplace throat, or
the curvature-correct Mason-Morrow (1 + 2 sqrt(pi G)) for an angular throat, G = A/O^2 of the
throat; see `throat_coeff`), and records the equilibrium wetting saturation S_w to build the
primary-drainage curve. Outputs, per case:
  * one overlay plot: the per-angle P_c-S_w curves, each colour-coded, with its analytic entry
    drawn as a dashed line in the same colour (math typeset),
  * per angle: a 3D drainage GIF (PyVista front) and a separate 2D GIF (phi + pressure slices).

Method (after Ramstad et al. 2009):
  * Geometry / BCs: a walled `duct`, fronted each end by `press_layers` fixed-pressure planes +
    open buffer cells (pressure BC decoupled from the duct). S_w is measured over the
    duct pore only.
  * Pressure-consistent init: red (NW) at the inlet density rho_in=1, blue (WP) at the outlet
    density rho_out — the domain starts in mechanical balance with the BCs (no t=0 shock).
  * The inlet density is held fixed; P_c is raised by lowering the outlet wetting density
    (P_c = (1 - rho_out) cs^2). Quasi-static: each P_c continues from the previous equilibrium.
  * Equilibrium: run `block_steps`-step blocks (default 2000); advance when |Delta S_w| < `tol`
    (default 2.5e-4) over a block, else run another block (capped at `max_blocks`, generous so a
    straight tube's slow at-entry piston sweep can finish). Frames render at a fixed step cadence
    (`render_every`, default 500) for smooth movies, but no more than `frames_per_pc` per
    pressure step (a slow sweep would otherwise emit thousands of near-identical frames; render
    is the wall-clock bottleneck). Default solver (MRT + akai + CSF + recolor_emag +
    wall_grad="fluid").
"""
from __future__ import annotations

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.lines import Line2D
from scipy import ndimage

from bob import color3d, d3q19
from bob.utils import viz

plt.switch_backend("Agg")
plt.rcParams["mathtext.fontset"] = "cm"

GAMMA = 1.0 / 45
CS2 = float(d3q19.CS2)


# ----------------------------------------------------------------------------- geometry
def build_tube(cross2d, duct, buf):
    """Extrude a 2D (z,y) cross-section along x; open inlet/outlet reservoir slabs of `buf` cells."""
    nz, ny = cross2d.shape
    nx = duct + 2 * buf
    solid = np.broadcast_to(cross2d[:, :, None], (nz, ny, nx)).copy()
    solid[:, :, :buf] = False
    solid[:, :, nx - buf:] = False
    return solid


def square_cross(n, a):
    s = np.ones((n, n), bool)
    lo = (n - a) // 2
    s[lo:lo + a, lo:lo + a] = False
    return s


def circle_cross(n, r):
    """Circular pore of radius r centred in an n x n cross-section (True = solid)."""
    yy, xx = np.indices((n, n)).astype(float)
    c = (n - 1) / 2.0
    return (yy - c) ** 2 + (xx - c) ** 2 > r ** 2


def rect_cross(n, L1, L2):
    s = np.ones((n, n), bool)
    lo1, lo2 = (n - L1) // 2, (n - L2) // 2
    s[lo1:lo1 + L1, lo2:lo2 + L2] = False
    return s


def four_grain_cross(n, Rg):
    yy, xx = np.indices((n, n)).astype(float)
    c = (n - 1) / 2.0
    grains = np.zeros((n, n), bool)
    for cy in (c - Rg, c + Rg):
        for cx in (c - Rg, c + Rg):
            grains |= (xx - cx) ** 2 + (yy - cy) ** 2 <= Rg ** 2
    in_box = (np.abs(xx - c) <= Rg) & (np.abs(yy - c) <= Rg)
    return ~((~grains) & in_box)


# ----------------------------------------------------------------------------- BCs / init
def pressure_inlet_layers(state, rho_in, layers):
    """Fixed-density (u=0) pure non-wetting on the first `layers` x-planes (f_i = w_i*rho_in).
    where-form, not .at[].set: two slice-updates at opposite ends of a multigpu-sharded x
    axis miscompile under GSPMD (jax 0.6.2) — see bc.inlet_reservoir."""
    feq = d3q19.W[:, None, None, None] * rho_in
    slab = jnp.arange(state.fR.shape[3]) < layers
    return color3d.State(jnp.where(slab, feq, state.fR), jnp.where(slab, 0.0, state.fB))


def pressure_outlet_layers(state, rho_out, layers):
    """Fixed-density (u=0) pure wetting on the last `layers` x-planes (f_i = w_i*rho_out)."""
    feq = d3q19.W[:, None, None, None] * rho_out
    nx = state.fR.shape[3]
    slab = jnp.arange(nx) >= nx - layers
    return color3d.State(jnp.where(slab, 0.0, state.fR), jnp.where(slab, feq, state.fB))


def init_state_pressure(solid, x_in, rho_in, rho_out):
    """Pressure-consistent IC: red (NW) at rho_in for x<x_in, blue (WP) at rho_out for x>=x_in,
    equilibrium at u=0 — domain starts in mechanical balance with the fixed-pressure BCs."""
    nz, ny, nx = solid.shape
    xx = np.broadcast_to(np.arange(nx), (nz, ny, nx))
    rhoR = jnp.asarray(np.where(xx < x_in, rho_in, 0.0))
    rhoB = jnp.asarray(np.where(xx >= x_in, rho_out, 0.0))
    W = d3q19.W[:, None, None, None]
    return color3d.State(W * rhoR[None], W * rhoB[None])


def wetting_saturation(state, region):
    """Threshold S_w over the duct pore. phi = (rR - rB)/(rR + rB): cells with phi < 0 are
    wetting (blue); the non-wetting phase (phi > 0) and the phi = 0 interface count as
    non-wetting. S_w = (# strictly-wetting pore cells) / (# pore cells) — a sharp phase
    count, insensitive to the diffuse-interface / corner-film density drift that a b/(r+b)
    density-ratio measure would include."""
    rR, rB = (np.asarray(z) for z in color3d.densities(state))
    phi = (rR - rB) / np.maximum(rR + rB, 1e-12)
    wetting = (phi < 0.0) & region
    return float(wetting.sum() / max(region.sum(), 1e-12))


# ----------------------------------------------------------------------------- rendering
def _fit_title_fs(title, fig_w_in, fs_max=13.0):
    """Monospace title fontsize that fits `fig_w_in` inches (~0.62 em/char), capped at fs_max."""
    return min(fs_max, 0.96 * fig_w_in * 72.0 / (0.62 * max(len(title), 1)))


def tube_camera(shape):
    nz, ny, nx = shape
    c = (nx / 2.0, ny / 2.0, nz / 2.0)
    span = max(nx, ny, nz)
    return [(c[0], c[1] - 1.9 * span, c[2] + 0.5 * span), c, (0.0, 0.0, 1.0)]


def render_3d(solid, rhoN, cam, title, out_png, window=(1100, 460)):
    """PyVista voxel render (threshold geometry, no contouring) with the lighting doing the
    depth work: the red surface is extracted to polydata so smooth_shading interpolates
    normals (as viz3d's `shade` option does), and SSAO ambient occlusion darkens crevices so the
    pore space reads. Grains gray translucent."""
    nz, ny, nx = solid.shape
    grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1))
    grid.cell_data["solid"] = solid.astype(float).ravel()
    grid.cell_data["rhoN"] = np.where(solid, -1.0, rhoN).ravel()
    pl = pv.Plotter(off_screen=True, window_size=list(window))
    pl.set_background("white")
    pl.add_mesh(grid.outline(), color="#333333", line_width=2)
    grains = grid.threshold(0.5, scalars="solid")
    if grains.n_cells:
        pl.add_mesh(grains, color="#9d9d9d", opacity=0.22,
                    specular=0.4, specular_power=22, ambient=0.25, diffuse=0.8)
    red = grid.threshold(0.0, scalars="rhoN")
    if red.n_cells:
        red = red.extract_surface()  # polydata so smooth_shading can interpolate normals
        pl.add_mesh(red, color="#c0392b", opacity=0.9, smooth_shading=True,
                    specular=0.5, specular_power=22, ambient=0.3, diffuse=0.85)
    pl.camera_position = cam
    pl.reset_camera()
    pl.camera.zoom(0.92)
    pl.enable_3_lights()
    try:
        pl.enable_ssao(radius=3.0, bias=0.4)
        pl.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    img = pl.screenshot(return_img=True)
    pl.close()
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.imshow(img)
    ax.set_axis_off()
    ax.set_title(title, fontsize=_fit_title_fs(title, 9.6), family="monospace")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def render_2d(rhoR, rhoB, solid, ymid, title, out_png, p_vmin, p_vmax):
    """2D mid-y (x-z) slice: top = phi color field (red NW / blue WP), bottom = pressure rho*cs^2."""
    rho = rhoR + rhoB
    phi = (rhoR - rhoB) / np.maximum(rho, 1e-12)
    mask = solid[:, ymid, :]
    sphi = np.where(mask, np.nan, phi[:, ymid, :])
    spr = np.where(mask, np.nan, (rho * CS2)[:, ymid, :])
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6))
    c1 = plt.cm.bwr.copy()
    c1.set_bad("0.55")
    im0 = axes[0].imshow(sphi, origin="lower", cmap=c1, vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    axes[0].set_title(r"$\phi$ color field : red = non-wetting,  blue = wetting", fontsize=14)
    fig.colorbar(im0, ax=axes[0], fraction=0.045, pad=0.01)
    c2 = plt.cm.viridis.copy()
    c2.set_bad("0.55")
    im1 = axes[1].imshow(spr, origin="lower", cmap=c2, vmin=p_vmin, vmax=p_vmax, aspect="auto", interpolation="nearest")
    axes[1].set_title(r"pressure  $p = \rho\, c_s^2$", fontsize=14)
    fig.colorbar(im1, ax=axes[1], fraction=0.045, pad=0.01)
    for ax in axes:
        ax.set_xlabel(r"$x$  (inlet $\rightarrow$ outlet)", fontsize=13)
        ax.set_ylabel(r"$z$", fontsize=13)
    fig.suptitle(title, fontsize=_fit_title_fs(title, 10.0, fs_max=14.0), family="monospace")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def render_geometry(solid, out_png, title="grain geometry (buffer excluded)", window=(1000, 780)):
    """Publication render of the solid alone: crop to the solid's x-extent (drops the
    empty buffer/reservoir slabs), marching-cubes surface (cell -> point data, contour at 0.5),
    opaque and smooth-shaded, with SSAO ambient occlusion + specular so the pore openings and
    surface relief read from the lighting."""
    xs = np.where(solid.any(axis=(0, 1)))[0]
    s = solid[:, :, xs.min():xs.max() + 1]
    nz, ny, nx = s.shape
    grid = pv.ImageData(dimensions=(nx + 1, ny + 1, nz + 1))
    # contour a one-cell pore-padded copy (origin shifted to keep alignment) so the isosurface
    # closes where the solid meets the domain faces — otherwise clipped solids render as open shells
    pad = pv.ImageData(dimensions=(nx + 3, ny + 3, nz + 3), origin=(-1.0, -1.0, -1.0))
    pad.cell_data["solid"] = np.pad(s, 1, constant_values=False).astype(float).ravel()
    surf = pad.cell_data_to_point_data().contour([0.5], scalars="solid")
    try:  # light volume-preserving Taubin smooth — takes the edge off the voxel staircase
        surf = surf.smooth_taubin(n_iter=15, pass_band=0.2)
    except Exception:
        pass
    pl = pv.Plotter(off_screen=True, window_size=list(window))
    pl.set_background("white")
    pl.add_mesh(grid.outline(), color="#333333", line_width=2)
    pl.add_mesh(surf, color="#9aa3b2", smooth_shading=True,
                specular=0.5, specular_power=24, ambient=0.28, diffuse=0.85)
    c = (nx / 2.0, ny / 2.0, nz / 2.0)
    span = max(nx, ny, nz)
    pl.camera_position = [(c[0] + 1.1 * span, c[1] - 1.5 * span, c[2] + 0.9 * span), c, (0.0, 0.0, 1.0)]
    pl.reset_camera()
    pl.camera.zoom(1.05)
    pl.enable_3_lights()
    try:
        pl.enable_ssao(radius=4.0, bias=0.4)
        pl.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    img = pl.screenshot(return_img=True)
    pl.close()
    fig, ax = plt.subplots(figsize=(8.0, 6.6))
    ax.imshow(img)
    ax.set_axis_off()
    ax.set_title(title, fontsize=16, family="monospace")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def render_cross_section(cross2d, R, out_png, title):
    """2D illustration of the duct cross-section (z,y): solid gray, pore white.
    Publication fonts (2x). `R` kept in the signature for call-site symmetry."""
    del R
    nz, ny = cross2d.shape
    fig, ax = plt.subplots(figsize=(7.6, 7.2))
    img = np.where(cross2d, 0.62, 1.0)
    ax.imshow(img, origin="lower", cmap="gray", vmin=0.0, vmax=1.0, interpolation="nearest",
              extent=(-0.5, ny - 0.5, -0.5, nz - 0.5))
    ax.set_xlabel(r"$y$  (cells)", fontsize=20)
    ax.set_ylabel(r"$z$  (cells)", fontsize=20)
    ax.tick_params(labelsize=18)
    ax.set_title(title, fontsize=20)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- sweep
def pc_ladder(entry, n=20):
    """`n` non-dim P_c points: a coarse equal baseline + a refined cluster around `entry`."""
    nr = max(6, round(n * 0.4))                            # refined points bracketing the entry
    nb = n - nr
    base = np.linspace(0.4, max(2.8, entry + 1.4), nb)
    refine = np.linspace(max(0.3, entry - 0.4), entry + 0.4, nr)
    return np.unique(np.round(np.concatenate([base, refine]), 3))


def run_theta(name, solid, R, coeff, theta, x_in, x_duct, out_dir, *, beta=0.95, press_layers=2,
              block_steps=2000, render_every=500, max_blocks=300, tol=0.00025, frames_per_pc=40, fps=15):
    """One contact angle: ramp the refined P_c ladder, equilibrate each (block-wise,
    |dS_w|<tol over a `block_steps` block, capped at max_blocks), render a frame at a fixed
    `render_every`-step cadence (decoupled from the blocks) for smooth 3D + 2D movies. A
    straight tube drains by a steady piston sweep, so a point right at the entry can take many
    blocks to settle; `max_blocks` is generous so the sweep actually reaches equilibrium, and
    `frames_per_pc` caps frames per pressure step so one slow sweep does not render thousands of
    near-identical frames (render is the wall-clock bottleneck). The analytic non-dimensional
    entry pressure is `coeff * cos(theta_w)` (coeff = 2 for a circular Laplace throat,
    (1+2*sqrt(pi*G)) for an angular Mason-Morrow throat). Returns (theta, curve, entry)."""
    theta_w = 180.0 - theta
    entry = coeff * np.cos(np.deg2rad(theta_w))
    nz, ny, nx = solid.shape
    ymid = ny // 2
    cam = tube_camera(solid.shape)
    solid_j = jnp.asarray(solid)
    nw = color3d.wall_normals(solid)
    xx = np.broadcast_to(np.arange(nx), (nz, ny, nx))
    region = (~solid) & (xx >= x_duct[0]) & (xx < x_duct[1])
    p = color3d.Params(omega=1.0, sigma=GAMMA, beta=beta, theta=theta)
    pcs = pc_ladder(entry)
    rho_out0 = 1.0 - pcs[0] * GAMMA / (R * CS2)
    state = init_state_pressure(solid, x_in, 1.0, rho_out0)
    p_vmin = (1.0 - pcs[-1] * GAMMA / (R * CS2)) * CS2
    p_vmax = 1.0 * CS2

    def body(s, rout):
        s = color3d.step(s, p, solid=solid_j, nw=nw)
        s = pressure_inlet_layers(s, 1.0, press_layers)
        s = pressure_outlet_layers(s, rout, press_layers)
        return s, None

    run_chunk = jax.jit(lambda s, rout: jax.lax.scan(lambda c, _: body(c, rout), s, None, length=render_every)[0])

    tdir = Path(out_dir) / f"theta_{theta:.0f}"
    d3, d2 = tdir / "frames_3d", tdir / "frames_2d"
    for d in (d3, d2):
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.png"):
            f.unlink()
    f3, f2, curve = [], [], []
    gidx, total = 0, 0
    t0 = time.time()

    def render(pc, step):
        nonlocal gidx
        Sw = wetting_saturation(state, region)
        rR, rB = (np.asarray(z) for z in color3d.densities(state))
        rhoN = (rR - rB) / np.maximum(rR + rB, 1e-12)
        ttl = f"{name}  theta_oil={theta:.0f}   P_c R/g={pc:4.2f}   step={step:6d}   S_w={Sw:5.3f}"
        p3 = d3 / f"f{gidx:04d}.png"
        p2 = d2 / f"f{gidx:04d}.png"
        render_3d(solid, rhoN, cam, ttl, p3)
        render_2d(rR, rB, solid, ymid, ttl, p2, p_vmin, p_vmax)
        f3.append(p3)
        f2.append(p2)
        gidx += 1
        return Sw

    for pc in pcs:
        rho_out = 1.0 - pc * GAMMA / (R * CS2)
        sw_block = wetting_saturation(state, region)
        nblk, npc = 0, 0
        for _b in range(max_blocks):
            for _ in range(max(1, block_steps // render_every)):   # fixed render_every-step cadence ...
                state = run_chunk(state, jnp.float64(rho_out))
                total += render_every
                if npc < frames_per_pc:                            # ... but capped per P_c (a slow at-entry sweep
                    sw = render(pc, total)                         #     would otherwise emit thousands of frames)
                    npc += 1
                else:
                    sw = wetting_saturation(state, region)
            nblk += 1
            if abs(sw - sw_block) < tol:                           # equilibrium checked once per block
                break
            sw_block = sw
        if npc >= frames_per_pc:                                   # always show the settled state
            render(pc, total)
        curve.append((float(pc), sw))
        capped = "  CAPPED@max_blocks" if nblk >= max_blocks else ""
        print(f"    [{name} theta_oil={theta:.0f}] P_c={pc:4.2f}: S_w={sw:.3f}  ({nblk} blocks){capped}")
    def _gif(frames, path):                                       # all frames kept on disk; GIF capped for watchability
        sel = frames
        if len(frames) > 400:
            idx = np.unique(np.linspace(0, len(frames) - 1, 400).round().astype(int))
            sel = [frames[i] for i in idx]
            print(f"    GIF {path.name}: subsampled {len(frames)}->{len(sel)} frames (full set on disk)")
        viz.write_gif(sel, path, fps)

    _gif(f3, tdir / "drainage_3d.gif")
    _gif(f2, tdir / "drainage_2d.gif")
    print(f"  theta={theta:.0f} done: {gidx} frames, {total} steps ({time.time()-t0:.0f}s) -> {tdir}")
    return theta, curve, entry


# marching-squares (mid-point contour) segment length per 2x2 corner code — a far better
# perimeter than the voxel staircase (which over-counts every curved edge): exact for
# axis-aligned edges, ~2*pi*r for a circle. Corner bits TL=1,TR=2,BR=4,BL=8.
_SEG = np.zeros(16)
for _c in (1, 2, 4, 8, 7, 11, 13, 14):
    _SEG[_c] = np.sqrt(2) / 2          # one corner cut -> short diagonal
for _c in (3, 6, 9, 12):
    _SEG[_c] = 1.0                     # straight edge across the cell
for _c in (5, 10):
    _SEG[_c] = np.sqrt(2)              # saddle -> two diagonals


def _contour_perimeter(mask):
    """Mid-point marching-squares contour length of a 2D boolean region."""
    m = mask.astype(np.uint8)
    code = m[:-1, :-1] * 1 + m[:-1, 1:] * 2 + m[1:, 1:] * 4 + m[1:, :-1] * 8
    return float(_SEG[code].sum())


def throat_coeff(solid, x_duct):
    """Analytic entry prefactor (1 + 2*sqrt(pi*G)) and inscribed radius R measured from the
    narrowest pore cross-section in the duct (Mason-Morrow 1991; G = A/O^2 = area/perimeter^2,
    perimeter by marching squares). Reduces to coeff~2 for a circular throat; gives the
    curvature-correct value for an angular throat. R and G describe the discrete throat the
    solver resolves, so the analytic and simulated entry pressures refer to the same geometry.
    Returns (R_inscribed, coeff, G)."""
    lo, hi = x_duct
    areas = [int((~solid[:, :, x]).sum()) for x in range(lo, hi)]
    xt = lo + int(np.argmin(areas))
    pore = ~solid[:, :, xt]
    lbl, _ = ndimage.label(pore)
    main = lbl == np.bincount(lbl[lbl > 0].ravel()).argmax()
    A = int(main.sum())
    perim = _contour_perimeter(main)
    R = float(ndimage.distance_transform_edt(main).max())
    G = A / perim ** 2 if perim else 1.0 / (4 * np.pi)
    return R, 1.0 + 2.0 * np.sqrt(np.pi * G), G


def overlay_plot(results, name, out_png, entry_form):
    """Publication-grade overlay of the per-theta P_c-S_w curves (theta = oil contact angle,
    the angle imposed on Params). Each curve's analytic entry -coeff*cos(theta) is a dashed
    line in the same colour; the entry values are folded into the curve labels and one neutral
    dashed legend entry carries the formula. `name` is the display title (capitalized, no
    domain-size/R clutter — geometry details live in the caption). STIX serif typography
    (matches the mathtext), inward major+minor ticks on a full box, no grid. Writes 300-dpi
    PNG + vector PDF.

    An analysis curve: writes nothing unless plots are enabled (``viz.plots_enabled``,
    i.e. the case script's ``--plots``)."""
    if not viz.plots_enabled():
        return
    rc = {
        "font.family": "STIXGeneral", "mathtext.fontset": "stix",
        "axes.linewidth": 1.4,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 7, "ytick.major.size": 7,
        "xtick.minor.size": 3.5, "ytick.minor.size": 3.5,
        "xtick.major.width": 1.4, "ytick.major.width": 1.4,
        "xtick.minor.width": 1.1, "ytick.minor.width": 1.1,
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(10.0, 7.6))
        colors = plt.cm.viridis(np.linspace(0.05, 0.8, len(results)))
        pcmax = 0.0
        for (theta, curve, entry), c in zip(results, colors):
            sw = [s for _, s in curve]
            pc = [p for p, _ in curve]
            pcmax = max(pcmax, max(pc))
            ax.axhline(entry, color=c, ls="--", lw=1.8, alpha=0.9)
            ax.plot(sw, pc, "o-", color=c, ms=8, lw=2.4, markeredgecolor="white",
                    markeredgewidth=0.9, zorder=3,
                    label=rf"$\theta={theta:.0f}^\circ$   ($P_c R/\gamma = {entry:.2f}$)")
        handles, labels = ax.get_legend_handles_labels()
        handles.append(Line2D([], [], color="0.35", ls="--", lw=1.8))
        labels.append(rf"analytic entry $= {entry_form}$")
        ax.set_xlabel(r"wetting saturation  $S_w$", fontsize=24)
        ax.set_ylabel(r"capillary pressure  $P_c\,R/\gamma$", fontsize=24)
        ax.set_xlim(0.0, 1.02)
        ax.set_ylim(0.0, pcmax + 0.35)
        ax.minorticks_on()
        ax.tick_params(labelsize=21)
        # blank the x-axis origin label — it collides with the y-axis "0.0" at the corner
        ax.xaxis.set_major_formatter(lambda v, _pos: "" if v == 0 else f"{v:.1f}")
        ax.set_title(rf"{name} — Primary Drainage" + "\n"
                     + r"$\theta$ = oil contact angle", fontsize=21, pad=12)
        ax.legend(handles, labels, fontsize=17.5, loc="upper right", frameon=True,
                  framealpha=0.95, edgecolor="0.75", borderpad=0.7, labelspacing=0.45)
        fig.tight_layout()
        fig.savefig(out_png, dpi=300)
        fig.savefig(Path(out_png).with_suffix(".pdf"))
        plt.close(fig)


def run_case(name, solid, R, x_in, x_duct, thetas, out_dir, *,
             entry_coeff=2.0, entry_form=r"-2\cos\theta", plot_title=None, **kw):
    """Full case: run each theta, then the colour-coded overlay plot. `entry_coeff` is the
    analytic non-dim entry prefactor on cos(theta_w) (2 = circular Laplace; pass
    1+2*sqrt(pi*G) from `throat_coeff` for an angular Mason-Morrow throat). `plot_title`
    is the clean capitalized display title for the publication pc_curve (defaults to `name`,
    which stays verbose for logs/frame titles)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== {name}: domain {solid.shape[2]}x{solid.shape[1]}x{solid.shape[0]}, R={R:.2f}, "
          f"entry_coeff={entry_coeff:.3f}, thetas={thetas} ===")
    results = [run_theta(name, solid, R, entry_coeff, th, x_in, x_duct, out, **kw) for th in thetas]
    overlay_plot(results, plot_title or name, out / "pc_curve.png", entry_form)
    print(f"=== {name} DONE -> {out}/pc_steps.csv + per-theta 3D/2D GIFs"
          + (f" + {out}/pc_curve.png" if viz.plots_enabled() else "") + " ===")
    return results
