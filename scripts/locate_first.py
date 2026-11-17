#!/usr/bin/env python3
"""Check whether the first alarm identifies an invariant containing a falsified measurement."""
import json
import argparse
from pathlib import Path


def hms(s):
    h, m, x = (int(v) for v in s.split(":"))
    return 3600 * h + 60 * m + x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("events", type=Path)
    parser.add_argument("physical_result", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    events = json.loads(args.events.read_text(encoding="utf-8"))
    phys = json.loads(args.physical_result.read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in phys["events"]}
    det = set(phys["detected_ids"])
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
    if len(evs) != len(names) or any(n not in sched for n in names):
        parser.error("drift-off events could not be matched to the supplied scenario configuration")
    stats = {"detected": 0, "top_first": 0, "any_first": 0, "top_window": 0, "n_cand": []}
    for e, n in zip(evs, names):
        r = rows[e["id"]]
        if e["id"] not in det or not r.get("first_alarm"):
            continue
        fa = r["first_alarm"]
        att = maps[n]
        cands = fa["candidates"]
        top_hit = bool(cands) and bool(att & set(cands[0]["terms"]))
        any_hit = any(att & set(c["terms"]) for c in cands)
        win_hit = bool(r.get("top_invariant")) and bool(att & set(r["top_invariant"]["terms"]))
        stats["detected"] += 1
        stats["top_first"] += top_hit
        stats["any_first"] += any_hit
        stats["top_window"] += win_hit
        stats["n_cand"].append(fa["n_candidates"])
        print(f"id={e['id']:>2} {n:<10} first alarm +{fa['t_offset_s']:.0f}s candidates={fa['n_candidates']:>2} "
              f"top={cands[0]['name'] if cands else None} top_hit={top_hit} any_hit={any_hit} window_top_hit={win_hit}")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
