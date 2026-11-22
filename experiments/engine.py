#!/usr/bin/env python3
"""Cache process values and account for protocol and physical alert onsets."""
import contextlib
import io
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from locations import DATA, OUTPUT, CACHE, SCRIPTS
sys.path.insert(0, str(SCRIPTS))
import m5_physics as P  # noqa: E402
import m5_netchannels as NET  # noqa: E402

CFG = {
    "01": dict(dir="01-Basic", grid="cigre-mv", node="n302", train=True),
    "02": dict(dir="02-Semiurban", grid="1-MV-semiurb--2-sw", node="n406", train=True),
    "03": dict(dir="03-Rural", grid="1-MV-rural--2-sw", node="n402", train=False),
}
FAMS = ("ELEM", "KCL", "VLINE")

# Cached columns
_cache = {}


def load_cache(scen, stream):
    key = (scen, stream)
    if key not in _cache:
        z = np.load(CACHE / f"{scen}_{stream}.npz")
        _cache[key] = {k: z[k] for k in z.files}
    return _cache[key]


def _patched_load_columns(path, keys):
    p = Path(path)
    scen, stream = p.parent.name[:2], p.name.split(".")[0]
    c = load_cache(scen, stream)
    return c["t"], {k: c[k] for k in sorted(keys)}


def use_cache():
    """m5_physics.main() (and load_columns callers) read the cache from now on."""
    P.load_columns = _patched_load_columns


def run_physics(argv):
    """Run the physical detector in-process; returns (stdout text, parsed output json)."""
    out = Path(argv[argv.index("--out") + 1])
    old = sys.argv
    sys.argv = ["m5_physics"] + list(argv)
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            P.main()
    finally:
        sys.argv = old
    return buf.getvalue(), json.loads(out.read_text())


def paths(scen):
    c = CFG[scen]
    d = DATA / c["dir"]
    return dict(train=d / f"train.{c['node']}.state.gz", test=d / f"test.{c['node']}.state.gz",
                events=d / "ipal/test/events.json", train_events=d / "ipal/train/events.json",
                train_pcap=d / "raw/train/pcap", test_pcap=d / "raw/test/pcap")


# Protocol streams
def protocol_streams(scen, stream="test", init_seconds=600, gap_default=10.5):
    """The protocol-channel alert streams of m5_netchannels.main() for the final configuration.
    stream 'test' = the test run (as in the paper). Returns t, chan (channel -> bool), ent ((channel, entity) -> bool)."""
    p = paths(scen)
    c_te, a_te, f_te = NET.parse(p["test_pcap"])
    if CFG[scen]["train"]:
        c_tr, a_tr, f_tr = NET.parse(p["train_pcap"])
        cmd_src = {s for _, s, _, _ in c_tr}
    else:
        c_tr, a_tr, f_tr = [], [], {}
        import dpkt
        peers = defaultdict(set)
        t_first = min((min(v) for v in f_te.values() if v), default=None)
        t_stop = t_first + init_seconds
        for p_ in sorted(Path(p["test_pcap"]).glob("*.pcap")):
            with open(p_, "rb") as f:
                for ts, buf in dpkt.pcap.Reader(f):
                    if ts > t_stop:
                        break
                    try:
                        eth = dpkt.ethernet.Ethernet(buf)
                        if isinstance(eth.data, dpkt.ip.IP) and isinstance(eth.data.data, dpkt.tcp.TCP) and eth.data.data.dport == 2404:
                            peers[NET.ip(eth.data.src)].add(NET.ip(eth.data.dst))
                    except Exception:
                        continue
        mtu = max(peers, key=lambda k: len(peers[k]))
        cmd_src = {mtu}
    bind = {}
    for _, i_, m_ in a_tr:
        bind.setdefault(i_, set()).add(m_)
    max_gap = {}
    for k, ts in f_tr.items():
        if len(ts) > 10:
            max_gap[k] = float(np.max(np.diff(ts)))
    if not f_tr and gap_default > 0:
        max_gap = {k: gap_default for k in f_te}

    cache = load_cache(scen, "test")
    out = build_streams(cache, c_te, a_te, f_te, cmd_src, bind, max_gap)
    out.update(cmd_src=sorted(cmd_src), max_gap=max_gap)
    return out


def build_streams(cache, c_te, a_te, f_te, cmd_src, bind, max_gap, grace=10, hold=30.0, margin=1.5):
    """Protocol-channel alert streams over the timeline of `cache` (cached columns of one run), given the parsed capture
    of that run (c_te commands, a_te ARP, f_te per-RTU report times) and the learned parameters (cmd_src, bind, max_gap).
    Same rules as m5_netchannels.main()."""
    t = cache["t"]
    keys = [k for k in cache if ":" in k]
    kset = set(keys)
    pairs = []
    for k in keys:
        el, attr = k.split(":")
        if attr == "closed" and f"{el}:is_closed" in kset:
            pairs.append((k, f"{el}:is_closed"))
        if attr == "connected" and f"{el}:is_connected" in kset:
            pairs.append((k, f"{el}:is_connected"))
    n = len(t)
    mism_pair = np.zeros((n, len(pairs)), bool)
    for j, (a_, b_) in enumerate(pairs):
        a, b = cache[a_], cache[b_]
        ok = np.isfinite(a) & np.isfinite(b)
        mism_pair[:, j] = ok & ((a != 0) != (b != 0))
    mism = mism_pair.any(axis=1)
    chan = {k: np.zeros(n, bool) for k in ("CMD", "ARP", "GAP", "CONSIST")}
    ent = defaultdict(lambda: np.zeros(n, bool))
    run = 0
    for i in range(n):
        run = run + 1 if mism[i] else 0
        if run > grace:
            chan["CONSIST"][i] = True
    for j, (a_, b_) in enumerate(pairs):
        if not mism_pair[:, j].any():
            continue
        run = 0
        for i in range(n):
            run = run + 1 if mism_pair[i, j] else 0
            if run > grace:
                ent[("CONSIST", a_.split(":")[0])][i] = True

    def mark(name, a, b, who):
        m = (t >= a) & (t <= b)
        chan[name] |= m
        ent[(name, who)] |= m

    for ts, s, d, tid in c_te:
        if s not in cmd_src:
            mark("CMD", ts, ts + hold, s)
    seen = {k: set(v) for k, v in bind.items()}
    for ts, i_, m_ in a_te:
        if i_ in seen and m_ not in seen[i_]:
            mark("ARP", ts, ts + hold, i_)
        seen.setdefault(i_, set()).add(m_) if i_ not in bind else None
    end = t[-1]
    for k, ts in f_te.items():
        if k not in max_gap:
            continue
        thr = max_gap[k] * margin
        seq = list(ts) + [end + 1e9]
        for a, b in zip(seq[:-1], seq[1:]):
            if b - a > thr:
                mark("GAP", a + thr, min(b, end), k)
    return dict(t=t, chan=dict(chan), ent=dict(ent), pairs=pairs)


def save_protocol(scen):
    out = CACHE / f"proto_{scen}.pkl"
    pr = protocol_streams(scen)
    with open(out, "wb") as f:
        pickle.dump(pr, f, protocol=4)
    return out


def load_protocol(scen):
    with open(CACHE / f"proto_{scen}.pkl", "rb") as f:
        return pickle.load(f)


# Accounting
def _on(v):
    return np.flatnonzero(v & ~np.concatenate([[False], v[:-1]]))


class Accountant:
    """Re-implements the 'combined', 'peralert' and 'peralert_protocol' counts of m5_netchannels.main() for any
    physical scores. The protocol streams are evaluated once; only the physical streams depend on the scores."""

    def __init__(self, scen, events=None, proto=None):
        self.scen = scen
        self.pr = proto or load_protocol(scen)
        self.t = self.pr["t"]
        self.events = events or json.loads(paths(scen)["events"].read_text())
        self.attacks = [e for e in self.events if "benign" not in e["id"]]
        t = self.t
        self.alarm_net = np.zeros(len(t), bool)
        for v in self.pr["chan"].values():
            self.alarm_net |= v
        self.ent_stats = []
        for (ch, who), v in sorted(self.pr["ent"].items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
            if not v.any():
                continue
            rr = P.evaluate(t, v, self.events)
            self.ent_stats.append(dict(ch=ch, who=who, fa=rr["false_alarms"], fab=rr["false_alarms_benign"],
                                       on=t[_on(v)]))
        self.proto_hit = self._hits([s["on"] for s in self.ent_stats])
        self.proto_fa = sum(s["fa"] for s in self.ent_stats)
        self.proto_fab = sum(s["fab"] for s in self.ent_stats)

    def _hits(self, on_lists):
        """event id -> earliest offset of an alert start inside the attack window, over the given alert streams"""
        first = {}
        for on in on_lists:
            for e in self.attacks:
                sel = on[(on >= e["start"]) & (on <= e["end"])]
                if len(sel):
                    d = float(sel.min() - e["start"])
                    if e["id"] not in first or d < first[e["id"]]:
                        first[e["id"]] = d
        return first

    def phys_streams(self, tp, scores, scale):
        t = self.t
        streams = {}
        for f in scores:                       # the three invariant families, or any other score stream (e.g. state estimation)
            streams[f] = np.interp(t, tp, (scores[f] > scale).astype(float)) > 0.5
        return streams

    def evaluate(self, tp, scores, scale=1.0):
        t = self.t
        ph = self.phys_streams(tp, scores, scale)
        phys_any = np.zeros(len(t), bool)
        for v in ph.values():
            phys_any |= v
        # merged (one signal): the original ORs the family scores on the physics timeline before interpolating
        pm = np.zeros(len(tp), bool)
        for f in scores:
            pm |= scores[f] > scale
        phys_on_t = np.interp(t, tp, pm.astype(float)) > 0.5
        comb = P.evaluate(t, phys_on_t | self.alarm_net, self.events)
        phys_only = P.evaluate(t, phys_on_t, self.events)
        fa_p, fab_p, on_p = 0, 0, []
        for v in ph.values():
            if v.any():
                rr = P.evaluate(t, v, self.events)
                fa_p += rr["false_alarms"]
                fab_p += rr["false_alarms_benign"]
                on_p.append(t[_on(v)])
        hit_p = self._hits(on_p)
        hit_all = dict(self.proto_hit)
        for k, d in hit_p.items():
            if k not in hit_all or d < hit_all[k]:
                hit_all[k] = d
        n_streams = len(self.ent_stats) + sum(1 for v in ph.values() if v.any())
        per_type = defaultdict(lambda: [0, 0])
        for e in self.attacks:
            k = e["description"].split(":")[0]
            per_type[k][1] += 1
            per_type[k][0] += e["id"] in hit_all
        return dict(
            peralert=dict(detected=len(hit_all), n_attacks=len(self.attacks), false_alarms=self.proto_fa + fa_p,
                          false_alarms_benign=self.proto_fab + fab_p, detected_ids=sorted(hit_all),
                          per_type={k: f"{a}/{b}" for k, (a, b) in per_type.items()}, alert_streams=n_streams,
                          avg_ttd_s=float(np.mean(list(hit_all.values()))) if hit_all else None),
            combined=dict(detected=comb["detected"], false_alarms=comb["false_alarms"],
                          false_alarms_benign=comb["false_alarms_benign"], detected_ids=comb["detected_ids"],
                          avg_ttd_s=comb["avg_ttd_s"]),
            phys_only_peralert=dict(detected=len(hit_p), false_alarms=fa_p, detected_ids=sorted(hit_p)),
            phys_merged=dict(detected=phys_only["detected"], false_alarms=phys_only["false_alarms"]),
            peralert_protocol=dict(detected=len(self.proto_hit), false_alarms=self.proto_fa),
        )


def load_scores(path):
    z = np.load(path)
    return z["t"], {f: z[f] for f in FAMS if f in z.files}
