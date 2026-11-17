#!/usr/bin/env python3
"""Configure and score an IPAL SIMPLE baseline on all process-state features."""
import argparse
import gzip
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m5_physics import evaluate  # noqa: E402


def make_config(args):
    with gzip.open(args.state, "rt", encoding="utf-8") as f:
        st = json.loads(next(f))["state"]
    numeric = sorted(k for k, v in st.items() if isinstance(v, (int, float)) and not isinstance(v, bool))
    boolean = sorted(k for k, v in st.items() if isinstance(v, bool))
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    cfg = {}
    defaults = {
        "MinMax": {"threshold": 1.0, "discrete_threshold": 10},
        "Histogram": {"window_size": 100, "threshold": 1.0, "discrete_threshold": 10},
        "SteadyTime": {"threshold": 1.0, "discrete_threshold": 10, "adjust": True},
    }
    for name, extra in defaults.items():
        cfg[name] = {"_type": name, "model-file": str(Path(args.model_dir) / f"model_{name}"),
                     "features": [f"state;{k}" for k in numeric + boolean], "preprocessors": [],
                     "trainon": 1.0, "save-training": None, "allow-none": True, **extra}
    Path(args.out).write_text(json.dumps(cfg, indent=1))
    print(f"SIMPLE: {len(numeric)} numeric and {len(boolean)} boolean values; config -> {args.out}")


def score(args):
    t, alarm = [], []
    with gzip.open(args.output, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            t.append(r["timestamp"])
            alarm.append(bool(r.get("ids")))
    t, alarm = np.array(t), np.array(alarm)
    events = json.loads(Path(args.events).read_text(encoding="utf-8"))
    res = evaluate(t, alarm, events)
    res["alarm_fraction"] = float(alarm.mean())
    Path(args.out).write_text(json.dumps(res, indent=2))
    print(f"detected {res['detected']}/{res['n_attacks']}  FA {res['false_alarms']} (benign {res['false_alarms_benign']})  "
          f"TTD {res['avg_ttd_s']}  alarm fraction {res['alarm_fraction']:.3f}  {res['per_type']}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-config")
    m.add_argument("--state", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--model-dir", required=True)
    s = sub.add_parser("score")
    s.add_argument("--output", required=True)
    s.add_argument("--events", required=True)
    s.add_argument("--out", required=True)
    args = ap.parse_args()
    (make_config if args.cmd == "make-config" else score)(args)


if __name__ == "__main__":
    main()
