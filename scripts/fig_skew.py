#!/usr/bin/env python3
"""Plot branch power measurements and plain and interval residuals for bus 61."""
import argparse
import gzip
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

KEYS = ["bus.61:active_power", "line.57:active_power_to", "line.113:active_power_from"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--center", type=float, default=0.0)
    ap.add_argument("--skew", type=int, default=10)
    ap.add_argument("--half", type=int, default=300)
    args = ap.parse_args()
    t, cols = [], {k: [] for k in KEYS}
    with gzip.open(args.state, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            t.append(r["timestamp"])
            for k in KEYS:
                v = r["state"].get(k)
                cols[k].append(np.nan if v is None else float(v))
    t = np.array(t)
    X = {k: np.array(v) for k, v in cols.items()}
    raw = X[KEYS[0]] + X[KEYS[1]] + X[KEYS[2]]
    if args.center == 0.0:
        # largest persistent |raw| in the file
        from numpy.lib.stride_tricks import sliding_window_view
        rm = sliding_window_view(np.concatenate([np.full(29, np.inf), np.abs(raw)]), 30).min(axis=1)
        i0 = int(np.nanargmax(rm))
    else:
        i0 = int(np.argmin(np.abs(t - args.center)))
    k = args.skew
    lo, hi = 0.0, 0.0
    for key in KEYS:
        v = X[key]
        pad = np.concatenate([np.full(k, v[0]), v, np.full(k, v[-1])])
        from numpy.lib.stride_tricks import sliding_window_view
        w = sliding_window_view(pad, 2 * k + 1)
        lo = lo + w.min(axis=1)
        hi = hi + w.max(axis=1)
    tol = np.where(lo > 0, lo, np.where(hi < 0, hi, 0.0))
    sl = slice(max(0, i0 - args.half), min(len(t), i0 + args.half))
    x = t[sl] - t[i0]
    fig, ax = plt.subplots(2, 1, figsize=(6.4, 4.2), sharex=True)
    ax[0].plot(x, X[KEYS[1]][sl] / 1e3, label="line 57, end at bus 61", lw=1.2)
    ax[0].plot(x, -X[KEYS[2]][sl] / 1e3, label="line 113, end at bus 61 (sign flipped)", lw=1.2, ls="--")
    ax[0].set_ylabel("Active power (kW)")
    ax[0].legend(fontsize=8, frameon=False)
    ax[1].plot(x, raw[sl] / 1e3, label="plain balance residual", lw=1.2, color="tab:red")
    ax[1].plot(x, tol[sl] / 1e3, label=f"skew-tolerant residual, $\\Delta$ = {k} s", lw=1.6, color="tab:green")
    ax[1].axhline(0, color="0.6", lw=0.6)
    ax[1].set_ylabel("Residual (kW)")
    ax[1].set_xlabel("Time relative to the largest deviation (s)")
    ax[1].legend(fontsize=8, frameon=False)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out)
    print(f"saved {args.out}; peak |raw| {np.nanmax(np.abs(raw[sl])):.1f} W, peak |tol| {np.nanmax(np.abs(tol[sl])):.1f} W")


if __name__ == "__main__":
    main()
