#!/usr/bin/env python3
"""Cache Sherlock measurement columns for repeated detector evaluations."""
import sys
import time
from pathlib import Path

import numpy as np

from locations import DATA, OUTPUT, CACHE, SCRIPTS
sys.path.insert(0, str(SCRIPTS))
import m5_physics as P  # noqa: E402

CFG = {
    "01": ("01-Basic", "cigre-mv", "n302"),
    "02": ("02-Semiurban", "1-MV-semiurb--2-sw", "n406"),
    "03": ("03-Rural", "1-MV-rural--2-sw", "n402"),
}


def needed_keys(scen):
    d, grid, node = CFG[scen]
    present = P.first_keys(DATA / d / f"test.{node}.state.gz")
    net = P.build_net(grid)
    inv = P.invariants(net)
    inv = {n: t for n, t in inv.items() if all(k in present for k, _ in t)}
    need = {k for terms in inv.values() for k, _ in terms}
    for n in inv:
        if n.startswith("VLINE("):
            li, end = n[len("VLINE(line."):-1].split(",")
            need.add(f"line.{li}:active_power_{end}")
    need &= present
    # commanded versus reported positions (protocol check CONSIST) and the switch positions used by the SE baseline
    need |= {k for k in present if k.endswith((":is_closed", ":closed", ":is_connected", ":connected"))}
    return sorted(need)


def main():
    scen, stream = sys.argv[1], sys.argv[2]
    d, grid, node = CFG[scen]
    src = DATA / d / f"{stream}.{node}.state.gz"
    out = CACHE / f"{scen}_{stream}.npz"
    CACHE.mkdir(parents=True, exist_ok=True)
    keys = needed_keys(scen)
    t0 = time.time()
    t, cols = P.load_columns(src, set(keys))
    np.savez(out, t=t, **cols)
    print(f"{scen} {stream}: {len(keys)} keys, {len(t)} rows, {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
