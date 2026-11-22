#!/usr/bin/env python3
"""Evaluate state-estimation residuals and compare alert counts."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E  # noqa: E402

P = E.P
SE = E.OUTPUT / "se"
NCH = {("01", "train"): 8, ("01", "test"): 8, ("03", "test"): 12}
STATS = {"SE objective J": "J", "SE largest normalized residual": "rN"}
SCALES = [float(2 ** (k / 16)) for k in range(-64, 241)]
W = 30


def load(scen, stream, variant):
    n = NCH[(scen, stream)]
    parts = [np.load(SE / f"se_{scen}_{stream}_{variant}_c{c}.npz") for c in range(n)]
    N = int(parts[-1]["b"])
    out = {k: np.full(N, np.nan) for k in ("J", "dof", "rn")}
    arg = np.full(N, -2, dtype=int)
    for z in parts:
        a, b = int(z["a"]), int(z["b"])
        for k in out:
            out[k][a:b] = z[k]
        arg[a:b] = z["arg"]
    out["arg"] = arg
    out["keys"] = [str(x) for x in parts[0]["meas_keys"]]
    return out


def statistic(d, which):
    if which == "J":
        return d["J"] / d["dof"]
    return d["rn"]


def persisted(s):
    return P.rolling_min(np.nan_to_num(s, nan=0.0), W)


def first_alarm_in(t, alarm, ev):
    starts = np.flatnonzero(alarm & ~np.concatenate([[False], alarm[:-1]]))
    for i in starts:
        if ev["start"] <= t[i] <= ev["end"]:
            return int(i)
    return None


def drift_off_maps(scen):
    """attack id -> set of falsified measurement keys"""
    cfgf = E.DATA / E.CFG[scen]["dir"] / "raw/test/sherlock-config.yml"
    cfg = json.loads(cfgf.read_text())          # the .yml files of Sherlock are JSON
    events = json.loads(E.paths(scen)["events"].read_text())

    def hms(s):
        h, m, x = (int(v) for v in s.split(":"))
        return 3600 * h + 60 * m + x
    sched = {}
    for v in cfg.values():
        if isinstance(v, dict):
            for name, when in v.items():
                if isinstance(when, str) and when.count(":") == 2:
                    sched[name] = hms(when)
    maps = {}
    for pb in cfg.get("playbooks", []):
        if "drift_off" in pb.get("module", ""):
            for c in pb.get("configs", []):
                maps[c["id"]] = {k.replace(".MEASUREMENT.", ":") for k in (c.get("grid_value_map") or {})}
    evs = sorted([e for e in events if e["description"].startswith("drift-off")], key=lambda e: e["start"])
    names = sorted(maps, key=lambda n: sched.get(n, 1e12))
    return {e["id"]: maps[n] for e, n in zip(evs, names)}


def main():
    res = {}
    summary = {}
    for scen in ("01", "03"):
        acc = E.Accountant(scen)
        t = acc.t
        events = acc.events
        res[scen] = {}
        for variant in ("static", "flows"):
            test = load(scen, "test", variant)
            fails = int(np.isnan(test["J"]).sum())
            ref_train = load("01", "train", variant)
            for name, which in STATS.items():
                thr = 1.5 * float(persisted(statistic(ref_train, which)).max())
                s_te = persisted(statistic(test, which))
                scores = {"SE": s_te / thr}
                key = f"{name} ({variant})"
                rows = []
                for sc_ in SCALES:
                    r = acc.evaluate(t, scores, sc_)
                    rows.append(dict(scale=sc_, det=r["peralert"]["detected"], fa=r["peralert"]["false_alarms"],
                                     det_merged=r["combined"]["detected"], fa_merged=r["combined"]["false_alarms"],
                                     det_phys=r["phys_only_peralert"]["detected"], fa_phys=r["phys_only_peralert"]["false_alarms"]))
                one = [r for r in rows if abs(r["scale"] - 1.0) < 1e-9][0]
                # localization of drift-off attacks that the SE statistic alarms on
                alarm = s_te > thr
                loc = dict(alarmed=0, top_hit=0)
                maps = drift_off_maps(scen)
                for ev in events:
                    if ev["id"] in maps:
                        i = first_alarm_in(t, alarm, ev)
                        if i is not None:
                            loc["alarmed"] += 1
                            j = int(test["arg"][i])
                            loc["top_hit"] += bool(j >= 0 and test["keys"][j] in maps[ev["id"]])
                # classical decisions (same persistence)
                if which == "J":
                    cl = persisted((test["J"] > chi2.ppf(0.99, test["dof"])).astype(float)) > 0.5
                else:
                    cl = persisted((test["rn"] > 3.0).astype(float)) > 0.5
                rc = acc.evaluate(t, {"SE": cl.astype(float)}, 0.5)
                res[scen][key] = dict(threshold=thr, nominal=one, failures=fails, localization=loc, rows=rows,
                                      classical=dict(det=rc["peralert"]["detected"], fa=rc["peralert"]["false_alarms"],
                                                     det_phys=rc["phys_only_peralert"]["detected"], fa_phys=rc["phys_only_peralert"]["false_alarms"]),
                                      max_test_ratio_benign_free=float(np.nanmax(s_te / thr)))
                bz = [r for r in rows if r["fa"] == 0]
                best0 = max((r["det"] for r in bz), default=None)
                print(f"[{scen}] {key:50s} thr {thr:.4g}  nominal {one['det']}/{len(acc.attacks)} FA {one['fa']} (SE alone {one['det_phys']} FA {one['fa_phys']}); "
                      f"best at FA=0: {best0}; classical {rc['peralert']['detected']} FA {rc['peralert']['false_alarms']}; "
                      f"localization {loc['top_hit']}/{loc['alarmed']}; failed solves {fails}", flush=True)
    (SE / "se_results.json").write_text(json.dumps(res))
    # compact curve file for the figure: the better topology variant of each statistic, by detections at zero false alarms
    for scen in res:
        summary[scen] = {}
        for name in STATS:
            best = None
            for variant in ("static", "flows"):
                key = f"{name} ({variant})"
                rows = res[scen][key]["rows"]
                zero = max((r["det"] for r in rows if r["fa"] == 0), default=-1)
                if best is None or zero > best[0]:
                    best = (zero, variant, rows)
            summary[scen][name] = best[2]
            summary[scen][name + " variant"] = best[1]
    (SE / "se_summary.json").write_text(json.dumps({s: {k: v for k, v in d.items() if not k.endswith("variant")} for s, d in summary.items()}))
    print("variants used in the figure:", {s: {k: v for k, v in d.items() if k.endswith("variant")} for s, d in summary.items()})


if __name__ == "__main__":
    main()
