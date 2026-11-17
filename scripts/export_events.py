#!/usr/bin/env python3
"""Export event-level detection and first-alarm localization results to CSV."""
import csv
import json
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("scenario")
    parser.add_argument("events", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rdir, tag, events_path, out = args.result_dir, args.scenario, args.events, args.output
    net = json.loads((rdir / f"net_{tag}.json").read_text(encoding="utf-8"))
    phys = json.loads((rdir / f"phys_{tag}.json").read_text(encoding="utf-8"))
    rows = {r["id"]: r for r in phys["events"]}
    events = json.loads(Path(events_path).read_text(encoding="utf-8"))
    per = set(net["peralert"]["detected_ids"])
    merged = set(net["combined"]["detected_ids"])
    first_ch = net["peralert"].get("first_channel", {})
    physical = set(phys["detected_ids"])
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "start", "end", "duration_s", "detected_per_alert", "first_channel", "detected_merged",
                    "physical_detected", "peak_ELEM", "peak_KCL", "peak_VLINE", "first_alarm_offset_s", "first_alarm_candidates",
                    "first_alarm_top_invariant"])
        for e in sorted(events, key=lambda x: x["start"]):
            r = rows.get(e["id"], {})
            pk = r.get("peak", {})
            fa = r.get("first_alarm") or {}
            cands = fa.get("candidates", [])
            w.writerow([e["id"], e["description"], e["start"], e["end"], round(e["end"] - e["start"]),
                        int(e["id"] in per), first_ch.get(e["id"], ""), int(e["id"] in merged),
                        int(e["id"] in physical),
                        *[("" if pk.get(k) is None else round(pk[k], 3)) for k in ("ELEM", "KCL", "VLINE")],
                        ("" if not fa else round(fa["t_offset_s"], 1)), fa.get("n_candidates", ""),
                        (cands[0]["name"] if cands else "")])
    print("wrote", out)


if __name__ == "__main__":
    main()
