#!/usr/bin/env python3
"""Evaluate detection counts across a fixed grid of threshold multipliers."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E  # noqa: E402

OUTPUT = E.OUTPUT
OUT = E.OUTPUT / "opcurve"
OUT.mkdir(parents=True, exist_ok=True)
SCALES = [float(2 ** (k / 16)) for k in range(-48, 241)]     # 1/8 ... 32768 in 1/16-octave steps
VARIANTS = {
    "STIR": lambda s: OUTPUT / f"main/scores_{s}.npz",
    "plain": lambda s: OUTPUT / f"ablation/scores_{s}_plain.npz",
    "moving average": lambda s: OUTPUT / f"ablation/scores_{s}_mean.npz",
    "interpolation": lambda s: OUTPUT / f"ablation/scores_{s}_interp.npz",
}


def main():
    res = {}
    for scen in ("01", "02", "03"):
        acc = E.Accountant(scen)
        res[scen] = {"n_attacks": len(acc.attacks), "protocol": acc.evaluate(*E.load_scores(VARIANTS["STIR"](scen)), 1e12)["peralert"]}
        for name, fn in VARIANTS.items():
            tp, sc = E.load_scores(fn(scen))
            rows = []
            for s in SCALES:
                r = acc.evaluate(tp, sc, s)
                rows.append(dict(scale=s, det=r["peralert"]["detected"], fa=r["peralert"]["false_alarms"],
                                 det_merged=r["combined"]["detected"], fa_merged=r["combined"]["false_alarms"],
                                 det_phys=r["phys_only_peralert"]["detected"], fa_phys=r["phys_only_peralert"]["false_alarms"],
                                 ids=r["peralert"]["detected_ids"]))
            res[scen][name] = rows
            one = [x for x in rows if abs(x["scale"] - 1.0) < 1e-9][0]
            zero = [x for x in rows if x["fa"] == 0]
            best0 = max(zero, key=lambda x: (x["det"], -abs(np.log(x["scale"])))) if zero else None
            b0 = "none" if best0 is None else f"{best0['det']} (s={best0['scale']:.3g})"
            print(f"[{scen}] {name:15s} s=1: {one['det']}/{res[scen]['n_attacks']} FA {one['fa']}   "
                  f"best at FA=0: {b0}   protocol only: {res[scen]['protocol']['detected']}", flush=True)
    (OUT / "opcurve.json").write_text(json.dumps(res))


if __name__ == "__main__":
    main()
