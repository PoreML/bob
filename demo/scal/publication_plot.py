"""Publication figures for a demo/scal run -- replots from frozen CSVs, never re-simulates.

  pc_curve:   the Pc-S_w hysteresis loop (drainage up / imbibition down, arrows),
              measured Pc solid + imposed Pc as faint dashed check.
  ca_vs_step: log-y Ca(t) (interface + bulk) with pressure-step boundaries -- the
              quasi-static evidence (Ca collapses at each equilibrium).
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RC = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.4,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.grid": False,
}


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def pc_curve(run):
    rows = read_csv(run / "pc_steps.csv")
    legs = {"drain": [r for r in rows if r["leg"] == "drain"], "imb": [r for r in rows if r["leg"] == "imb"]}
    c_drain, c_imb = plt.cm.viridis(0.03), plt.cm.viridis(0.55)
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(10, 7.5))
        for leg, colr, lbl in (("drain", c_drain, "drainage"), ("imb", c_imb, "imbibition")):
            if not legs[leg]:
                continue
            sw = np.array([float(r["sw_eq"]) for r in legs[leg]])
            pcm = np.array([float(r["pc_meas"]) for r in legs[leg]])
            pci = np.array([float(r["pc"]) for r in legs[leg]])
            ax.plot(sw, pcm * 100, "-o", color=colr, lw=2.2, ms=9, mec="white", mew=1.2, label=lbl)
            ax.plot(sw, pci * 100, "--", color=colr, lw=1.2, alpha=0.55)
            i = len(sw) // 2
            if len(sw) > 2:
                ax.annotate(
                    "",
                    xy=(sw[i + 1], pcm[i + 1] * 100),
                    xytext=(sw[i], pcm[i] * 100),
                    arrowprops={"arrowstyle": "-|>", "color": colr, "lw": 2},
                )
        ax.plot([], [], "--", color="0.45", lw=1.2, label="imposed $P_c$")
        ax.axhline(0.0, color="0.75", lw=1.0, zorder=0)
        ax.set_xlabel(r"water saturation  $S_w$", fontsize=24)
        ax.set_ylabel(r"$P_c$  ($\times 10^{-2}$ l.u.)", fontsize=24)
        ax.tick_params(labelsize=21)
        ax.set_xlim(0, 1)
        ax.set_title(r"Bentheimer SCAL loop ($\theta$ = oil contact angle $135^\circ$)", fontsize=24)
        ax.legend(fontsize=19, frameon=False)
        fig.tight_layout()
        fig.savefig(run / "pc_curve.png", dpi=300)
        fig.savefig(run / "pc_curve.pdf")
        plt.close(fig)


def ca_vs_step(run):
    rows = read_csv(run / "metrics.csv")
    steps_pc = [int(float(r["step_end"])) for r in read_csv(run / "pc_steps.csv")]
    s = np.array([float(r["step"]) for r in rows])
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(10, 6))
        for key, frac, lbl in (("ca_iface", 0.03, "interface"), ("ca_bulk", 0.55, "bulk")):
            v = np.maximum(np.array([float(r[key]) for r in rows]), 1e-12)
            ax.semilogy(s, v, color=plt.cm.viridis(frac), lw=1.8, label=lbl)
        for x in steps_pc:
            ax.axvline(x, color="0.88", lw=0.6, zorder=0)
        ax.set_xlabel("time step", fontsize=24)
        ax.set_ylabel(r"Ca $= \mu u / \sigma$", fontsize=24)
        ax.tick_params(labelsize=21)
        ax.set_title("Capillary number along the pressure ladder", fontsize=24)
        ax.legend(fontsize=19, frameon=False)
        fig.tight_layout()
        fig.savefig(run / "ca_vs_step.png", dpi=300)
        fig.savefig(run / "ca_vs_step.pdf")
        plt.close(fig)


if __name__ == "__main__":
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("demo/scal/bentheimer128")
    pc_curve(run)
    ca_vs_step(run)
    print(f"wrote {run}/pc_curve.png+pdf and {run}/ca_vs_step.png+pdf")
