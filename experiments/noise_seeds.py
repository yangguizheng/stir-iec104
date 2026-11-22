#!/usr/bin/env python3
"""Evaluate the detector with independent realizations of multiplicative instrument noise."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E  # noqa: E402

OUTPUT = E.OUTPUT
OUT = E.OUTPUT / "noise20"
SIGMAS = ("0.005", "0.01")


def one(scen, seed, sg, acc, tmp, thr_from=None):
    p = E.paths(scen)
    ref = OUTPUT / "main"
    common = ["--thr-floor", "1e-3", "--thr-floor-v", "0.01", "--persist", "30", "--skew", "10",
              "--noise-seed", str(seed), "--noise", sg]
    out = tmp / f"phys_{scen}_seed{seed}_s{sg}.json"
    if scen == "03":
        argv = ["--grid", E.CFG[scen]["grid"], "--test", str(p["test"]), "--events", str(p["events"])] + common + \
               ["--convention-from", str(ref / "phys_01.json"), "--thr-from", str(thr_from)]
    else:
        eps = str(max(0.01, 4 * float(sg)))
        argv = ["--grid", E.CFG[scen]["grid"], "--train", str(p["train"]), "--test", str(p["test"]),
                "--events", str(p["events"])] + common + ["--valid-rel", eps, "--signs-from", str(ref / f"phys_{scen}.json")]
    argv += ["--out", str(out), "--dump-scores", str(out.with_suffix(".npz"))]
    _, ph = E.run_physics(argv)
    tp, sc = E.load_scores(out.with_suffix(".npz"))
    r = acc[scen].evaluate(tp, sc, 1.0)
    row = dict(detected=r["peralert"]["detected"], n_attacks=r["peralert"]["n_attacks"], fa=r["peralert"]["false_alarms"],
              fa_benign=r["peralert"]["false_alarms_benign"], merged=r["combined"]["detected"],
              fa_merged=r["combined"]["false_alarms"], detected_ids=r["peralert"]["detected_ids"],
              per_type=r["peralert"]["per_type"], thresholds=ph["thresholds"], phys_detected=ph["detected"])
    row["physical_false_alarm_seconds"] = [item["duration_s"] for item in ph["false_alarm_sources"]]
    row["physical_false_alarms_benign"] = ph["false_alarms_benign"]
    return row, out


def main():
    first, last = int(sys.argv[1]), int(sys.argv[2])
    E.use_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = OUT / "tmp"
    tmp.mkdir(exist_ok=True)
    acc = {s: E.Accountant(s) for s in ("01", "02", "03")}
    for seed in range(first, last + 1):
        res = {}
        for sg in SIGMAS:
            res[sg] = {}
            r1, out1 = one("01", seed, sg, acc, tmp)
            res[sg]["01"] = r1
            res[sg]["02"], _ = one("02", seed, sg, acc, tmp)
            res[sg]["03"], _ = one("03", seed, sg, acc, tmp, thr_from=out1)
            print(f"seed {seed} sigma {sg}: " + "  ".join(f"{s}: {res[sg][s]['detected']}/{res[sg][s]['n_attacks']} FA {res[sg][s]['fa']}"
                                                          for s in ("01", "02", "03")), flush=True)
        (OUT / f"seed{seed}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
