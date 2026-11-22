#!/usr/bin/env python3
"""Plot operating curves and detection rates from offline benchmark results."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from locations import OUTPUT

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import beta

R = OUTPUT
SC = ("01", "02", "03")
NAME = {"01": "01-Basic", "02": "02-Semiurban", "03": "03-Rural"}
SCOL = {"01": "#375b80", "02": "#a25320", "03": "#137367"}
INK, GRID = "#273239", "#dbe2e6"
VAR = {"STIR": ("#137367", "o", "-"), "interpolation": ("#a25320", "^", "-"), "moving average": ("#375b80", "s", "-"),
       "plain": ("#737e87", "D", "-")}


def configure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.2, "axes.labelsize": 8.4, "axes.titlesize": 8.4,
        "axes.edgecolor": "#53616a", "axes.labelcolor": INK, "axes.linewidth": 0.65,
        "axes.spines.top": False, "axes.spines.right": False, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.fontsize": 7.6, "pdf.fonttype": 42,
        "xtick.labelsize": 7.4, "ytick.labelsize": 7.4,
        "figure.facecolor": "white", "axes.facecolor": "white"})


def frontier(rows, key_fa="fa", key_det="det"):
    """best detection with at most b false alarms, as a step function over the observed false-alarm counts"""
    pts = sorted({(r[key_fa], r[key_det]) for r in rows})
    best, out = -1, []
    for fa, det in pts:
        if det > best:
            best = det
            out.append((fa, det))
    return out


def fig_opcurve(outdir):
    d = json.loads((R / "opcurve/opcurve.json").read_text())
    se = None
    f = R / "se/se_summary.json"
    if f.exists():
        se = json.loads(f.read_text())
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.9))
    for ax, sc in zip(axes, SC):
        n = d[sc]["n_attacks"]
        proto = d[sc]["protocol"]["detected"]
        for name, (col, mk, ls) in VAR.items():
            rows = d[sc][name]
            fr = frontier(rows)
            xs = [x for x, _ in fr] + [max(1000, fr[-1][0])]
            ys = [y for _, y in fr] + [fr[-1][1]]
            ax.step(xs, ys, where="post", color=col, lw=1.5 if name == "STIR" else 1.1, ls=ls, zorder=3 if name == "STIR" else 2)
            one = [r for r in rows if abs(r["scale"] - 1.0) < 1e-9][0]
            ax.plot([one["fa"]], [one["det"]], marker=mk, ms=6.5, mfc=col if name == "STIR" else "white", mec=col, mew=1.2, ls="none", zorder=5)
        if se and sc in se:
            for name, (col, mk) in (("SE largest normalized residual", ("#7a4b9a", "v")), ("SE objective J", ("#9a7a2e", "P"))):
                if name in se[sc]:
                    rows = se[sc][name]
                    fr = frontier(rows)
                    ax.step([x for x, _ in fr] + [1000], [y for _, y in fr] + [fr[-1][1]], where="post", color=col, lw=1.1, ls="--", zorder=2)
        ax.axhline(proto, color="#9aa5ac", lw=0.7, ls=":")
        ax.set_xscale("symlog", linthresh=1, linscale=0.55)
        ax.set_xlim(-0.15, 1000)
        ax.set_xticks([0, 1, 10, 100, 1000])
        ax.set_xticklabels(["0", "1", "10", "100", "1000"])
        lo = proto - (1.5 if sc == "01" else 2.5)
        ax.set_ylim(lo, n + 0.6)
        ax.set_yticks(range(int(np.ceil(lo)), n + 1, 2 if n < 20 else 3))
        ax.set_title(f"{NAME[sc]} ({n} attacks)", loc="left")
        ax.set_xlabel("False alarms (per alert)", fontsize=8.0)
        ax.grid(axis="y", color=GRID, lw=0.55, alpha=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Attacks detected (per alert)")
    handles = [Line2D([], [], color=VAR[k][0], lw=2, label=k if k != "STIR" else "STIR") for k in VAR]
    if se:
        handles += [Line2D([], [], color="#7a4b9a", lw=1.4, ls="--", label="SE, largest normalized residual"),
                    Line2D([], [], color="#9a7a2e", lw=1.4, ls="--", label="SE, objective $J$")]
    handles += [Line2D([], [], color="#9aa5ac", lw=0.9, ls=":", label="protocol channels only"),
                Line2D([], [], color=INK, marker="o", mfc="white", ls="none", ms=6, label="threshold learned (scale 1)")]
    fig.legend(handles=handles, loc="upper center", ncol=3 if not se else 3, frameon=False, fontsize=7.4, columnspacing=1.0, handlelength=1.8)
    fig.tight_layout(rect=[0, 0, 1, 0.78 if not se else 0.74])
    fig.savefig(outdir / "fig_opcurve.pdf")
    plt.close(fig)


def cp(x, n):
    lo = beta.ppf(0.025, x, n - x + 1) if x else 0.0
    hi = beta.ppf(0.975, x + 1, n - x) if x < n else 1.0
    return lo, hi


def fig_inject(outdir):
    from locations import trial_records
    recs = {sc: trial_records(sc) for sc in SC}
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.7))
    panels = [("scale", "(a) Scaled by $1\\pm\\alpha$", "$\\alpha$ (%)", "a"),
              ("random", "(b) Randomized within $\\pm\\alpha$", "$\\alpha$ (%)", "a"),
              (("freeze", "extrap"), "(c) Frozen / extrapolated", "Duration (s)", "D")]
    for ax, (kind, title, xl, fld) in zip(axes, panels):
        kinds = kind if isinstance(kind, tuple) else (kind,)
        for sc in SC:
            for kd in kinds:
                rr = [r for r in recs[sc] if r["kind"] == kd]
                xs = sorted({r[fld] for r in rr})
                ys, lo, hi = [], [], []
                for x in xs:
                    q = [r for r in rr if r[fld] == x]
                    k = sum(r["detected"] for r in q)
                    l_, h_ = cp(k, len(q))
                    ys.append(100 * k / len(q)); lo.append(100 * l_); hi.append(100 * h_)
                ls = "-" if kd != "extrap" else "--"
                ax.plot(xs, ys, color=SCOL[sc], ls=ls, marker="o" if kd != "extrap" else "s", ms=3.6, lw=1.3)
                ax.fill_between(xs, lo, hi, color=SCOL[sc], alpha=0.10, lw=0)
        if fld == "a":
            ax.set_xscale("log")
            xs = sorted({r["a"] for r in recs["01"] if r["kind"] == kinds[0]})
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{100 * x:g}" for x in xs], rotation=0, fontsize=7.0)
            ax.minorticks_off()
        else:
            ax.set_xticks([120, 300, 500])
        ax.set_ylim(-2, 102)
        ax.set_title(title, loc="left")
        ax.set_xlabel(xl)
        ax.grid(axis="y", color=GRID, lw=0.55, alpha=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Injections detected (%)")
    handles = [Line2D([], [], color=SCOL[s], lw=2, label=NAME[s]) for s in SC] + \
              [Line2D([], [], color=INK, lw=1.2, ls="-", marker="o", ms=3.6, label="frozen"),
               Line2D([], [], color=INK, lw=1.2, ls="--", marker="s", ms=3.6, label="extrapolated")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, fontsize=7.4, columnspacing=1.0, handlelength=1.6)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(outdir / "fig_inject.pdf")
    plt.close(fig)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    configure()
    fig_opcurve(out)
    fig_inject(out)
    print("figures written to", out)
