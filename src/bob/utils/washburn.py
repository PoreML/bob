"""Washburn-law analysis + presentation helpers shared by the 2D/3D demos.

Spontaneous imbibition of a wetting fluid displacing a SECOND fluid of finite
viscosity follows the two-fluid Washburn law (Leclaire et al. 2017, Eq. 60): the
capillary pressure balances the viscous drag of BOTH columns, giving an implicit
quadratic in the meniscus position L(t),

    a L^2 + b L = A t + a L0^2 + b L0,   a = (mu_I - mu_O)/2,  b = mu_O L_tube,
    A = size*sigma*cos(theta) / k        k = 6 (2D slit, size = h) | 4 (3D tube, size = R)

solved as L(t) = (-b + sqrt(b^2 + 4 a (A t + a L0^2 + b L0))) / (2 a). The pure
sqrt(t) law is the mu_O -> 0 limit (then L^2 = (2A/mu_I) t). ``rectify`` maps a
measured L to a L^2 + b L, which is linear in t with slope A — the two-fluid
linearization used for the fit and the cos(theta) scaling.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from bob.utils import viz

# Plot style (shared with demo/laplace): categorical palette in fixed slot order,
# dark text tokens, recessive grid, mathtext.
_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
_INK, _INK2 = "#0b0b0b", "#52514e"
_RC = {
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": _INK2,
    "axes.labelcolor": _INK,
    "text.color": _INK,
    "xtick.color": _INK2,
    "ytick.color": _INK2,
    "axes.grid": True,
    "grid.color": _INK,
    "grid.alpha": 0.10,
    "grid.linewidth": 0.6,
    "mathtext.fontset": "stixsans",
}


def capillary_A(dim, size, sigma, theta_deg):
    """Two-fluid Washburn constant A (geometry x capillary, viscosity-free):
    2D slit aperture h: h*sigma*cos(theta)/6;  3D tube radius R: R*sigma*cos(theta)/4.
    ``gamma = sigma`` (the input surface tension)."""
    c = np.cos(np.deg2rad(theta_deg))
    if dim == 2:
        return float(size * sigma * c / 6.0)
    if dim == 3:
        return float(size * sigma * c / 4.0)
    raise ValueError(f"dim must be 2 or 3, got {dim}")


def two_fluid_Lt(ts, A, mu_I, mu_O, L_tube, L0=0.0):
    """Analytic meniscus position L(t) for the two-fluid Washburn law (Eq. 60).
    a = (mu_I-mu_O)/2, b = mu_O*L_tube; L = (-b + sqrt(b^2 + 4 a (A t + a L0^2 + b L0)))/(2a).
    Falls back to the linear A t / b solution when mu_I == mu_O (a -> 0)."""
    ts = np.asarray(ts, float)
    a = (mu_I - mu_O) / 2.0
    b = mu_O * L_tube
    rhs = A * ts + a * L0**2 + b * L0
    if abs(a) < 1e-12:
        return rhs / b
    return (-b + np.sqrt(np.clip(b**2 + 4.0 * a * rhs, 0.0, None))) / (2.0 * a)


def rectify(Ls, mu_I, mu_O, L_tube):
    """Map L -> a L^2 + b L (a=(mu_I-mu_O)/2, b=mu_O*L_tube), the variable that is
    linear in t (slope A) under the two-fluid Washburn law."""
    L = np.asarray(Ls, float)
    a = (mu_I - mu_O) / 2.0
    b = mu_O * L_tube
    return a * L**2 + b * L


def fit_slope(ts, ys, lo_frac=0.2, hi_frac=0.95):
    """Linear fit y = slope*t + b over the index window [lo_frac, hi_frac] of the
    series (drops the early inertial transient and the near-outlet end).
    Returns (slope, intercept, r2)."""
    ts = np.asarray(ts, float)
    ys = np.asarray(ys, float)
    n = len(ts)
    lo = int(n * lo_frac)
    hi = max(int(n * hi_frac), lo + 2)
    tw, yw = ts[lo:hi], ys[lo:hi]
    slope, intercept = np.polyfit(tw, yw, 1)
    resid = yw - (slope * tw + intercept)
    r2 = 1.0 - resid.var() / yw.var() if yw.var() > 0 else 0.0
    return float(slope), float(intercept), float(r2)


def curve_Lt(path, per_theta, analytic_curves):
    """L vs t, publication style (tall 4x6 in): LBM dots + two-fluid analytic solid
    lines, one categorical color per theta, each curve direct-labeled at its right
    end; a small ink-colored legend explains the dot/line encoding. Also writes the
    LBM series to <path stem>.csv (table view) and an SVG twin.
    ``per_theta``: list of (theta, ts, Ls); ``analytic_curves``: list of (td, Ld)
    aligned to it (each the analytic L sampled on dense times td).

    The CSV is always written; the figure is an analysis curve, so it needs plots
    enabled (``viz.plots_enabled``)."""
    path = Path(path)
    viz.write_csv(
        path.with_suffix(".csv"),
        [{"theta": theta, "t": t, "L": length} for theta, ts, Ls in per_theta for t, length in zip(ts, Ls)],
    )
    if not viz.plots_enabled():
        return
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(4, 6))
        for ((theta, ts, Ls), (td, Ld)), col in zip(zip(per_theta, analytic_curves), _SERIES):
            ax.plot(td, Ld, color=col, lw=2.0, zorder=2)
            ax.plot(ts, Ls, "o", color=col, ms=4.5, mec="white", mew=0.8, zorder=3)
            ax.annotate(
                f"{theta:.0f}\N{DEGREE SIGN}", (float(td[-1]), float(Ld[-1])),
                xytext=(5, 0), textcoords="offset points", va="center", ha="left", color=_INK,
            )
        t_max = max(float(np.max(ts)) for _, ts, _ in per_theta)
        y_max = max(max(float(np.max(Ls)) for _, _, Ls in per_theta), max(float(np.max(Ld)) for _, Ld in analytic_curves))
        ax.set_xlim(0, t_max * 1.14)
        ax.set_ylim(0, y_max * 1.06)
        ax.xaxis.set_major_locator(plt.MultipleLocator(10_000))
        ax.xaxis.set_major_formatter(lambda x, _pos: f"{x / 1000:g}")
        ax.set_xlabel(r"time step  ($\times 10^3$)")
        ax.set_ylabel("penetration L (cells)")
        ax.set_title("Washburn imbibition", pad=10)
        handles = [
            plt.Line2D([], [], color=_INK2, lw=0, marker="o", ms=4.5, mec="white", mew=0.8, label="LBM"),
            plt.Line2D([], [], color=_INK2, lw=2.0, label="two-fluid analytic"),
        ]
        ax.legend(handles=handles, loc="upper left", frameon=False)
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        fig.savefig(path.with_suffix(".svg"))
        plt.close(fig)


def curve_rectified(path, per_theta, mu_I, mu_O, L_tube):
    """(a L^2 + b L) vs t: straight lines under the two-fluid law, slope ~ cos(theta)."""
    series = []
    for i, (theta, ts, Ls) in enumerate(per_theta):
        c = _SERIES[i % len(_SERIES)]
        series.append((list(ts), list(rectify(Ls, mu_I, mu_O, L_tube)), f"theta={theta:.0f}", {"color": c, "lw": 2}))
    viz.line_plot(path, series, "Two-fluid Washburn linearization: a L^2 + b L linear in t", "time step", "a L^2 + b L")


def combined_gif(frame_lists, out, fps):
    """Tile per-theta frame strips side by side (left->right = list order) into
    one GIF. All lists must hold equal-height PNGs (same geometry per angle)."""
    out = Path(out)
    n = min(len(fl) for fl in frame_lists)
    cdir = out.parent / "combined_frames"
    cdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n):
        imgs = [np.asarray(Image.open(fl[i]).convert("RGB")) for fl in frame_lists]
        h = min(im.shape[0] for im in imgs)
        combo = Image.fromarray(np.hstack([im[:h] for im in imgs]))
        p = cdir / f"combo_{i:04d}.png"
        combo.save(p)
        paths.append(p)
    viz.write_gif(paths, out, fps)
