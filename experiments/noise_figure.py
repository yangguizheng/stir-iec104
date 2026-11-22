#!/usr/bin/env python3
"""Plot detection rates across independent noise realizations."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from locations import OUTPUT

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = OUTPUT
FROZEN = ROOT / "main"
NOISE = ROOT / "noise20"
SCENARIOS = ("01-Basic", "02-Semiurban", "03-Rural")
MARKERS = ("o", "s", "^")
MAIN, INK, GRID = "#137367", "#273239", "#dbe2e6"


def configure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.3, "axes.labelsize": 9.6, "axes.titlesize": 9.5,
        "axes.edgecolor": "#53616a", "axes.labelcolor": INK, "axes.linewidth": 0.65,
        "axes.spines.top": False, "axes.spines.right": False, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.fontsize": 8.7, "pdf.fonttype": 42,
        "figure.facecolor": "white", "axes.facecolor": "white"})


def main(out):
    configure()
    seeds = sorted(int(p.stem[4:]) for p in NOISE.glob("seed*.json"))
    data = {s: json.loads((NOISE / f"seed{s}.json").read_text()) for s in seeds}
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.95), sharey=True)
    levels = (0, 0.5, 1)
    colors = (MAIN, "#526b89")
    for i, (ax, name) in enumerate(zip(axes, SCENARIOS), start=1):
        sc = f"{i:02}"
        p = json.loads((FROZEN / f"phys_{sc}.json").read_text())
        f = json.loads((FROZEN / f"net_{sc}.json").read_text())["peralert"]
        obs = [[[f["detected"]], [p["detected"]]]]
        for sg in ("0.005", "0.01"):
            obs.append([[data[s][sg][sc]["detected"] for s in seeds], [data[s][sg][sc]["phys_detected"] for s in seeds]])
        for j, (label, color, marker) in enumerate(zip(("Complete", "Physical only"), colors, MARKERS)):
            vals = [np.asarray(level[j]) * 100 / p["n_attacks"] for level in obs]
            low, high, med = [v.min() for v in vals], [v.max() for v in vals], [np.median(v) for v in vals]
            ax.fill_between(levels, low, high, color=color, alpha=0.17, linewidth=0)
            ax.plot(levels, med, marker=marker, color=color, linestyle="-" if j == 0 else "--",
                    linewidth=1.9 if j == 0 else 1.45, markersize=5.8, markeredgecolor="white",
                    markeredgewidth=0.75, label=label)
            ax.vlines(levels[1:], low[1:], high[1:], color=color, linewidth=0.75, zorder=4)
        ax.set(title=name, ylim=(0, 105), yticks=(0, 25, 50, 75, 100), xlim=(-0.08, 1.08))
        ax.set_xticks(levels, ("0", "0.5", "1.0"))
        ax.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Attacks detected (%)")
    axes[1].set_xlabel("Measurement noise σ (%)", labelpad=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.55, 1.01), handlelength=2)
    fig.tight_layout(rect=(0, 0, 1, 0.87), w_pad=1.1)
    fig.savefig(Path(out) / "fig_noise_recall.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print("noise figure from", len(seeds), "realizations")


if __name__ == "__main__":
    main(sys.argv[1])
