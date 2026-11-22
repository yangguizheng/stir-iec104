#!/usr/bin/env python3
"""Evaluate six chronological folds of a benign training recording."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E  # noqa: E402

P, NET = E.P, E.NET
G = 70                                   # guard [s] around the held-out block (persistence 30 + delay 10 + envelope 10 + slack)


def parsed_train(scen):
    f = E.CACHE / f"trainpcap_{scen}.pkl"
    if not f.exists():
        c, a, fr = NET.parse(E.paths(scen)["train_pcap"])
        with open(f, "wb") as fh:
            pickle.dump(dict(c=c, a=a, f=fr), fh, protocol=4)
    with open(f, "rb") as fh:
        return pickle.load(fh)


def block_edges(scen, K):
    cache = E.load_cache(scen, "train")
    t = cache["t"]
    n = len(t)
    evs = json.loads(E.paths(scen)["train_events"].read_text())
    edges = [int(x) for x in np.linspace(0, n, K + 1)]
    for j in range(1, K):
        for ev in evs:
            a = int(np.searchsorted(t, ev["start"] - 5 - G))
            b = int(np.searchsorted(t, ev["end"] + 300 + G))
            if a < edges[j] < b:
                edges[j] = a
    return edges, evs


def rising(v):
    return np.flatnonzero(v & ~np.concatenate([[False], v[:-1]]))


def full_fit(scen):
    """sign conventions, retained equations and thresholds of the complete training run (cached)"""
    f = E.CACHE / f"fullfit_{scen}.pkl"
    if not f.exists():
        out, _, _, kept, signs = physical_fold(scen, 0, 0)
        with open(f, "wb") as fh:
            pickle.dump(dict(out=out, kept=kept, signs=signs), fh, protocol=4)
    with open(f, "rb") as fh:
        return pickle.load(fh)


def physical_fold(scen, a, b, margin=1.5, floor=1e-3, floor_v=0.01, valid_rel=0.01, fixed=None):
    """fixed: dict(kept=..., signs=...) of the complete training run; only the thresholds are then learned on the fit part"""
    P.ALIGN = "interval"
    cfg = E.CFG[scen]
    cols_all = E.load_cache(scen, "train")
    net = P.build_net(cfg["grid"])
    inv = P.invariants(net)
    present = set(cols_all)
    inv = {n: t for n, t in inv.items() if all(k in present for k, _ in t)}
    needed = {k for terms in inv.values() for k, _ in terms}
    for n in inv:
        if n.startswith("VLINE("):
            li, end = n[len("VLINE(line."):-1].split(",")
            needed.add(f"line.{li}:active_power_{end}")
    needed &= present
    cols = {k: cols_all[k] for k in sorted(needed)}
    n_t = len(cols_all["t"])
    M = np.zeros(n_t, bool)
    if b > a:
        M[max(0, a - G):min(n_t, b + G)] = True
    fit = ~M
    ctr = P.Cache(cols, 10)
    if fixed is not None:
        kept, signs = fixed["kept"], fixed["signs"]
        inv = {}
    else:
        signs, _ = P.choose_signs(P.Cache({k: v[fit] for k, v in cols.items()}, 0), inv, with_ident=True)
        kept = {}
    for nme, terms in inv.items():
        r = np.abs(P.residual(ctr, terms, signs[nme]))
        if nme.startswith("VLINE("):
            r = np.abs(P.gate(ctr, nme, r, 1e3))
            ok = float(np.nanpercentile(r[fit], 99.9)) < valid_rel
        else:
            mag = np.zeros(len(r))
            for k, _ in terms:
                mag = mag + np.abs(ctr(k))
            ok = float(np.nanpercentile(r[fit], 99.9) / (np.nanmedian(mag[fit]) + 1e3)) < valid_rel
        if ok:
            kept[nme] = terms
    signs = {n: signs[n] for n in kept}
    st, per_inv = P.scores(ctr, kept, signs, 30, 1e3, keep=True)
    thr = {f: max(float(v[fit].max()) * margin, floor if f != "VLINE" else floor_v) for f, v in st.items()}
    out = dict(thresholds=thr, n_invariants={f: sum(1 for n in kept if P.family(n) == f) for f in E.FAMS})
    blk = {}
    for f, v in st.items():
        al = v > thr[f]
        prev = al[a - 1] if a > 0 else False
        seg = al[a:b]
        starts = np.flatnonzero(seg & ~np.concatenate([[prev], seg[:-1]]))
        blk[f] = dict(false_alarms=int(len(starts)), peak_ratio=float(v[a:b].max() / thr[f]) if b > a else None)
    out["physical"] = blk
    return out, st, per_inv, kept, signs


def protocol_fold(scen, a, b):
    cache = E.load_cache(scen, "train")
    t = cache["t"]
    tr = parsed_train(scen)
    lo, hi = (t[max(0, a - G)], t[min(len(t), b + G) - 1]) if b > a else (np.inf, -np.inf)
    c_fit = [x for x in tr["c"] if not (lo <= x[0] <= hi)]
    a_fit = [x for x in tr["a"] if not (lo <= x[0] <= hi)]
    cmd_src = {s for _, s, _, _ in c_fit}
    bind = {}
    for _, i_, m_ in a_fit:
        bind.setdefault(i_, set()).add(m_)
    max_gap = {}
    for k, ts in tr["f"].items():
        ts = np.asarray(ts)
        if ((ts < lo) | (ts > hi)).sum() > 10:
            parts = [ts[ts < lo], ts[ts > hi]] if b > a else [ts]
            gaps = [np.max(np.diff(p)) for p in parts if len(p) > 1]
            if gaps:
                max_gap[k] = float(max(gaps))
    st = E.build_streams(cache, tr["c"], tr["a"], tr["f"], cmd_src, bind, max_gap)
    res = {}
    for ch, v in st["chan"].items():
        seg = v[a:b]
        prev = v[a - 1] if a > 0 else False
        res[ch] = int(len(np.flatnonzero(seg & ~np.concatenate([[prev], seg[:-1]]))))
    ent_fa = 0
    for (ch, who), v in st["ent"].items():
        seg = v[a:b]
        prev = v[a - 1] if a > 0 else False
        ent_fa += int(len(np.flatnonzero(seg & ~np.concatenate([[prev], seg[:-1]]))))
    return dict(channels_merged=res, entity_streams_false_alarms=ent_fa, n_cmd_sources=len(cmd_src),
                n_gap_rtus=len(max_gap))


def main():
    scen, what = sys.argv[1], sys.argv[2]
    K = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    mode = sys.argv[4] if len(sys.argv) > 4 else "strict"      # strict: signs, retained equations and thresholds re-learned
    if what == "fullfit":
        full_fit(scen)
        print("full fit cached", scen)
        return
    if what == "parse":
        parsed_train(scen)
        print("parsed", scen)
        return
    fold = int(what)
    if fold < 0:                                      # self-test: no held-out block, must reproduce the full training fit
        out, st, _, _, _ = physical_fold(scen, 0, 0)
        ref = json.loads((E.OUTPUT / f"main/phys_{scen}.json").read_text())
        ok = out["thresholds"] == ref["thresholds"] and out["n_invariants"] == ref["n_invariants"]
        print("self-test thresholds", out["thresholds"], "n_inv", out["n_invariants"], "->", "OK" if ok else f"BAD vs {ref['thresholds']} {ref['n_invariants']}")
        pr = protocol_fold(scen, 0, 0)
        tr = parsed_train(scen)
        mg = {k: float(np.max(np.diff(np.asarray(v)))) for k, v in tr["f"].items() if len(v) > 10}
        print("self-test protocol", pr, "| RTUs in the full fit:", len(mg))
        return
    edges, evs = block_edges(scen, K)
    a, b = edges[fold], edges[fold + 1]
    t = E.load_cache(scen, "train")["t"]
    inside = [e["id"] for e in evs if a <= np.searchsorted(t, e["start"]) and np.searchsorted(t, e["end"] + 300) <= b]
    ph, _, _, _, _ = physical_fold(scen, a, b, fixed=None if mode == "strict" else full_fit(scen))
    pr = protocol_fold(scen, a, b)
    res = dict(scenario=scen, fold=fold, K=K, mode=mode, block=[int(a), int(b)], seconds=int(b - a), events=inside, edges=edges, **ph, protocol=pr)
    res["false_alarms_total"] = sum(v["false_alarms"] for v in ph["physical"].values()) + pr["entity_streams_false_alarms"]
    out = E.OUTPUT / ("heldout" if mode == "strict" else "heldout_fixed")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"heldout_{scen}_fold{fold}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
