#!/usr/bin/env python3
"""Figure: attack windows (one row per attack type), benign operator events, and the alarms of every channel over
the 12 h test runs of the three scenarios.  Rows carry identity; colour is redundant.

usage: python -I fig_timeline.py --alarms A01.npz A02.npz A03.npz --events E01.json E02.json E03.json --out fig.pdf
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TYPES = [("industroyer", "Industroyer", "#2a78d6"), ("drift-off", "Drift-off", "#eda100"),
         ("control-and-freeze", "Control & freeze", "#e34948"), ("arp-spoof", "ARP spoofing DoS", "#008300")]
BENIGN = "#8a8985"
ALARM = "#3b3b38"
CHANNELS = [("PHYS", "Physical"), ("CONSIST", "Cmd–status"), ("CMD", "Cmd origin"), ("ARP", "ARP binding"), ("GAP", "Report gap")]
TITLES = ["(a) 01-Basic", "(b) 02-Semiurban", "(c) 03-Rural (no training data)"]


def spans(mask, t):
    m = np.asarray(mask, bool)
    d = np.diff(np.concatenate([[0], m.astype(int), [0]]))
    on, off = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(t[a], max(t[b - 1] - t[a], 1.0)) for a, b in zip(on, off)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alarms", nargs=3, required=True)
    ap.add_argument("--events", nargs=3, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0})
    rows = [lbl for _, lbl, _ in TYPES] + ["Benign events"] + [lbl for _, lbl in CHANNELS]
    fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.2))
    for ax, ap_, ep, title in zip(axes, args.alarms, args.events, TITLES):
        a = np.load(ap_)
        t0 = float(a["t"][0])
        t = (a["t"] - t0) / 3600.0
        events = json.load(open(ep))
        n = len(rows)
        y = {r: n - 1 - i for i, r in enumerate(rows)}
        for e in events:
            s, w = (e["start"] - t0) / 3600.0, (e["end"] - e["start"]) / 3600.0
            if "benign" in e["id"]:
                ax.broken_barh([(s, max(w, 0.02))], (y["Benign events"] - 0.35, 0.7), color=BENIGN, lw=0)
                continue
            kind = e["description"].split(":")[0]
            for k, lbl, col in TYPES:
                if k == kind:
                    ax.axvspan(s, s + w, color=col, alpha=0.10, lw=0)
                    ax.broken_barh([(s, w)], (y[lbl] - 0.35, 0.7), color=col, lw=0)
        for key, lbl in CHANNELS:
            if key in a.files:
                sp = [(x0, wd / 3600.0) for x0, wd in spans(a[key], a["t"] - t0)]
                sp = [(x0 / 3600.0, max(wd, 0.012)) for x0, wd in sp]
                ax.broken_barh(sp, (y[lbl] - 0.3, 0.6), color=ALARM, lw=0)
        ax.axhline(y[CHANNELS[0][1]] + 0.5, color="0.55", lw=0.6)
        ax.set_yticks([y[r] for r in rows])
        ax.set_yticklabels(rows)
        for lab in ax.get_yticklabels()[:len(TYPES)]:
            lab.set_color("#0b0b0b")
        ax.set_ylim(-0.7, n - 0.3)
        ax.set_xlim(0, t[-1])
        ax.set_xticks(np.arange(0, 12.01, 1))
        ax.grid(axis="x", color="0.85", lw=0.5)
        ax.set_axisbelow(True)
        for sp_ in ("top", "right", "left"):
            ax.spines[sp_].set_visible(False)
        ax.set_title(title, loc="left", fontsize=8.5)
    axes[-1].set_xlabel("Time since start of the test run (h)")
    fig.tight_layout()
    fig.savefig(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
