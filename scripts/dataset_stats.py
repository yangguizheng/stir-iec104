#!/usr/bin/env python3
"""Table 1 statistics for the three Sherlock scenarios: durations, process values, events, grid sizes, RTUs.

usage: python -I dataset_stats.py SHERLOCK_DIR OUT.json
"""
import gzip
import json
import argparse
from collections import Counter
from pathlib import Path

SC = {"01-Basic": ("n302", True, "cigre-mv"), "02-Semiurban": ("n406", True, "1-MV-semiurb--2-sw"),
      "03-Rural": ("n402", False, "1-MV-rural--2-sw")}


def span(path):
    n, t0, t1, keys = 0, None, None, None
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if t0 is None:
                t0, keys = r["timestamp"], r["state"]
            t1 = r["timestamp"]
            n += 1
    num = sum(1 for v in keys.values() if isinstance(v, (int, float)) and not isinstance(v, bool))
    boo = sum(1 for v in keys.values() if isinstance(v, bool))
    kinds = Counter(k.split(".")[0] for k in keys)
    return {"rows": n, "duration_s": t1 - t0, "values": len(keys), "numeric": num, "boolean": boo, "elements": dict(kinds)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root, out = args.data_dir, args.output
    res = {}
    for sc, (vp, has_train, grid) in SC.items():
        d = root / sc
        r = {"vantage_point": vp, "grid_model": grid}
        r["test"] = span(d / f"test.{vp}.state.gz")
        if has_train:
            r["train"] = span(d / f"train.{vp}.state.gz")
        ev = json.loads((d / "ipal/test/events.json").read_text(encoding="utf-8"))
        r["attack_types"] = dict(Counter(e["description"].split(":")[0] for e in ev if "benign" not in e["id"]))
        r["benign_events"] = [e["description"] for e in ev if "benign" in e["id"]]
        r["attack_durations_s"] = sorted(round(e["end"] - e["start"]) for e in ev if "benign" not in e["id"])
        r["attack_examples"] = [e["description"] for e in ev if "benign" not in e["id"]][:40]
        dpm = d / "raw/test/data-point-map.json"
        if dpm.exists():
            m = json.loads(dpm.read_text(encoding="utf-8"))
            r["data_point_map_type"] = type(m).__name__
            if isinstance(m, dict):
                r["data_point_map_keys"] = list(m.keys())[:10]
                r["data_point_map_len"] = len(m)
        res[sc] = r
        print(sc, json.dumps({k: v for k, v in r.items() if k not in ("attack_examples",)})[:1500], flush=True)
    out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
