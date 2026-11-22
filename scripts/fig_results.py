#!/usr/bin/env python3
"""Plot detector ablations, time-window and noise sweeps, and recorded alert traces."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.ticker import MultipleLocator
import numpy as np


SCENARIOS = ("01-Basic", "02-Semiurban", "03-Rural")
SCENARIO_COLORS = ("#375b80", "#a25320", "#137367")
METHOD_COLORS = ("#737e87", "#375b80", "#a25320", "#137367")
MARKERS = ("o", "s", "^")
MAIN = "#137367"
INK = "#273239"
GRID = "#dbe2e6"
ATTACK = "#eef2f3"


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def configure():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9.3,
        "axes.labelsize": 9.6,
        "axes.titlesize": 9.5,
        "axes.edgecolor": "#53616a",
        "axes.labelcolor": INK,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "legend.fontsize": 8.7,
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def grid(ax):
    ax.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)


def save(fig, path):
    fig.savefig(path, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def ablation(results, out):
    methods = ("Protocol only", "Plain residual", "Moving average", "STIR")
    rows = []
    for i in range(1, 4):
        full = load_json(results / "main" / f"net_{i:02}.json")
        protocol = full["peralert_protocol"]
        plain = load_json(results / "ablation" / f"net_{i:02}_plain.json")["peralert"]
        mean = load_json(results / "ablation" / f"net_{i:02}_mean.json")["peralert"]
        rows.append((protocol, plain, mean, full["peralert"]))

    detected = np.array([[run["detected"] for run in row] for row in rows])
    false = np.array([[run["false_alarms"] for run in row] for row in rows])
    data = (detected - detected[:, [0]], false)
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.45), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1.1]})
    x = np.arange(len(SCENARIOS))
    width = 0.18
    for ax, values, ylabel, ticks in zip(
        axes, data,
        ("Net additional detections", "False alarm onsets"),
        (np.arange(0, 9, 2), np.arange(0, 201, 50)),
    ):
        limit = max(values.max() * 1.22, ticks[-1] * 1.08)
        ax.axhspan(0, limit, color=MAIN, alpha=0.022, zorder=0)
        for j, (name, color) in enumerate(zip(methods, METHOD_COLORS)):
            offset = (j - (len(methods) - 1) / 2) * width
            y = values[:, j]
            fill = 0.5 * np.array(to_rgb(color)) + 0.5
            ax.bar(x + offset + 0.015, y, width, color="#bfc5c8", alpha=0.12, zorder=1)
            bars = ax.bar(x + offset, y, width, color=fill, edgecolor=color,
                          linewidth=1.35 if name == "STIR" else 0.95, label=name, zorder=2)
            ax.bar_label(bars, labels=[str(v) for v in y], padding=2.5, fontsize=8.3, color=color)
        ax.set(ylim=(0, limit), ylabel=ylabel, yticks=ticks)
        grid(ax)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.04),
                   ncol=4, frameon=False, handlelength=1.3, columnspacing=1.2)
    axes[1].set_xticks(x, SCENARIOS)
    fig.tight_layout(h_pad=1.25)
    save(fig, out / "fig_ablation_tradeoff.pdf")


def skew(results, out):
    half_widths = (0, 5, 8, 9, 10, 11, 12, 20)
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.35), sharex=True)
    for i, (name, color, marker) in enumerate(zip(SCENARIOS, SCENARIO_COLORS, MARKERS), start=1):
        rows = [load_json(results / "sensitivity" / f"{i:02}_skew{delta}_p30.json") for delta in half_widths]
        data = ([row["false_alarms"] for row in rows],
                [100 * row["detected"] / row["n_attacks"] for row in rows])
        for ax, values in zip(axes, data):
            ax.plot(half_widths, values, marker=marker, color=color,
                    linestyle=("-", "--", "-.")[i - 1], label=name, linewidth=1.75,
                    markersize=6.8 if i == 1 else 5.5, markeredgecolor="white", markeredgewidth=0.8)
    axes[0].set(ylabel="False alarm onsets", ylim=(-8, 162), yticks=(0, 50, 100, 150))
    axes[1].set(ylabel="Attacks detected (%)", xlabel="Window half-width Δ (s)",
                ylim=(0, 85), yticks=(0, 20, 40, 60, 80), xticks=half_widths, xlim=(-0.8, 20.8))
    for ax in axes:
        ax.axvline(10, color="#87969c", linewidth=0.9, ls=":", zorder=0)
        grid(ax)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, 1.04),
                   ncol=3, frameon=False, handlelength=2)
    fig.tight_layout(h_pad=0.95)
    save(fig, out / "fig_skew_sweep.pdf")


def noise(results, out):
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.95), sharey=True)
    levels = (0, 0.5, 1)
    colors = (MAIN, "#526b89")
    for i, (ax, name) in enumerate(zip(axes, SCENARIOS), start=1):
        base = results / "main"
        p = load_json(base / f"phys_{i:02}.json")
        f = load_json(base / f"net_{i:02}.json")["peralert"]
        observations = [[[f["detected"]], [p["detected"]]]]
        for sigma in ("0.005", "0.01"):
            nets, physical = [], []
            for seed in range(3):
                run = results / "noise" / f"seed-{seed}"
                nets.append(load_json(run / f"net_{i:02}_s{sigma}.json")["peralert"]["detected"])
                physical.append(load_json(run / f"phys_{i:02}_s{sigma}.json")["detected"])
            observations.append([nets, physical])
        for j, (label, color, marker) in enumerate(zip(("Complete", "Physical only"), colors, MARKERS)):
            values = [np.asarray(level[j]) * 100 / p["n_attacks"] for level in observations]
            low = [v.min() for v in values]
            high = [v.max() for v in values]
            median = [np.median(v) for v in values]
            ax.fill_between(levels, low, high, color=color, alpha=0.17, linewidth=0)
            ax.plot(levels, median, marker=marker, color=color, linestyle="-" if j == 0 else "--",
                    linewidth=1.9 if j == 0 else 1.45, markersize=5.8, markeredgecolor="white",
                    markeredgewidth=0.75, label=label)
            ax.vlines(levels[1:], low[1:], high[1:], color=color, linewidth=0.75, zorder=4)
        ax.set(title=name, ylim=(0, 105), yticks=(0, 25, 50, 75, 100), xlim=(-0.08, 1.08))
        ax.set_xticks(levels, ("0", "0.5", "1.0"))
        grid(ax)
    axes[0].set_ylabel("Attacks detected (%)")
    axes[1].set_xlabel("Measurement noise σ (%)", labelpad=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.55, 1.01), handlelength=2)
    fig.tight_layout(rect=(0, 0, 1, 0.87), w_pad=1.1)
    save(fig, out / "fig_noise_recall.pdf")


def active_spans(t, active):
    edges = np.diff(np.r_[False, active, False].astype(int))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [(t[a], t[b - 1] + 1 - t[a]) for a, b in zip(starts, ends)]


def event_traces(results, data_dir, out):
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 3.85), sharex=True,
                             gridspec_kw={"height_ratios": (2.0, 1.2)})
    rows = (("CONSIST", "Cmd–status", "#87929b"),
            ("ARP", "ARP binding", "#a25320"),
            ("PHYS", "Physical", MAIN))
    for j, (i, name, family) in enumerate(((2, SCENARIOS[1], "VLINE"), (3, SCENARIOS[2], "KCL"))):
        trace, alarms = axes[:, j]
        base = results / "main"
        labels = data_dir / name / "ipal" / "test" / "events.json"
        event = next(e for e in load_json(labels) if e["id"] == "27")
        with np.load(base / f"scores_{i:02}.npz") as scores, np.load(base / f"alarms_{i:02}.npz") as mask:
            if not np.array_equal(scores["t"], mask["t"]):
                raise ValueError(f"Score and alert timestamps differ for {name}")
            t = scores["t"] - event["start"]
            window = (t >= -36) & (t <= 236)
            for ax in (trace, alarms):
                ax.axvspan(0, event["end"] - event["start"], color=ATTACK, linewidth=0, zorder=0)
                ax.axvline(0, color="#7f8e95", linewidth=0.7, ls=":")
            trace.step(t[window], scores[family][window], where="post", color=MAIN, linewidth=1.3)
            trace.axhline(1, color="#53616a", lw=0.8, ls="--")
            trace.set_yscale("symlog", linthresh=1)
            trace.set(ylim=(-0.04, 140), yticks=(0, 1, 10, 100), yticklabels=("0", "1", "10", "100"),
                      title=f"({chr(97 + j)}) {name}, event 27")
            trace.text(0.04, 0.86, f"{family} score", ha="left", transform=trace.transAxes, color=MAIN)
            for k, (channel, _, color) in enumerate(rows):
                y = 2 - k
                active = mask[channel]
                alarms.broken_barh(active_spans(t[window], active[window]),
                                  (y - 0.22, 0.44), facecolors=color, edgecolors="none")
                if channel != "CONSIST":
                    onset = np.flatnonzero(active & ~np.r_[False, active[:-1]] & (t >= 0) & (t <= 235))
                    if onset.size:
                        when = t[onset[0]]
                        alarms.plot(when, y, marker="|", ms=8, mew=1.25, color=INK)
                        alarms.annotate(f"{when:.1f} s", (when, y), xytext=(0, 5),
                                        textcoords="offset points", ha="center", color=INK,
                                        va="bottom", fontsize=8)
            alarms.set(ylim=(-0.45, 2.55), xlim=(-35, 235))
            alarms.set_yticks([2, 1, 0], [label for _, label, _ in rows] if j == 0 else [""] * 3)
            alarms.set_xticks([-30, 0, 60, 120, 180, 230])
            alarms.tick_params(axis="y", length=0)
            grid(trace)
    axes[0, 0].set_ylabel("Score / threshold")
    fig.supxlabel("Time relative to attack onset (s)", fontsize=9.6)
    fig.tight_layout(rect=(0, 0.055, 1, 1), h_pad=0.4, w_pad=1)
    save(fig, out / "fig_event_traces.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    configure()
    results = args.results_dir
    ablation(results, args.out_dir)
    skew(results, args.out_dir)
    noise(results, args.out_dir)
    event_traces(results, args.data_dir, args.out_dir)


if __name__ == "__main__":
    main()

