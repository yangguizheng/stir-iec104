#!/usr/bin/env python3
"""Score switch-independent power and voltage balance invariants on IPAL state streams."""
import argparse
import gzip
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

FAMILIES = ("ELEM", "KCL", "VLINE")
ALIGN = "interval" # "interval" = STIR; "mean" = centred moving average of each term over the same window (baseline)


def first_keys(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return set(json.loads(next(f))["state"].keys())


def load_columns(path, keys):
    """Stream the IPAL state file and keep only the needed keys as float arrays (memory-light)."""
    keys = sorted(keys)
    ts, cols = [], {k: [] for k in keys}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            ts.append(r["timestamp"])
            st = r["state"]
            for k in keys:
                v = st.get(k)
                cols[k].append(np.nan if v is None else float(v))
    return np.array(ts), {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def add_noise(cols, sigma, seed, period=10):
    """Multiplicative Gaussian measurement noise, one draw per report. The state holds a value until the next report;
    a new report is assumed wherever the value changes and, while it does not, every `period` seconds (periodic
    transmission), so the noise never freezes into a constant offset."""
    if sigma <= 0:
        return cols
    import zlib
    out = {}
    for k, v in cols.items():
        rng = np.random.default_rng([seed, zlib.crc32(k.encode())])
        n = len(v)
        changed = np.ones(n, bool)
        changed[1:] = ~((v[1:] == v[:-1]) | (np.isnan(v[1:]) & np.isnan(v[:-1])))
        idx = np.arange(n)
        last = np.maximum.accumulate(np.where(changed, idx, 0))
        new_seg = changed | ((idx - last) % period == 0)
        seg = np.cumsum(new_seg) - 1
        out[k] = v * (1.0 + sigma * rng.standard_normal(int(seg[-1]) + 1)[seg])
    return out


def build_net(grid):
    if grid == "cigre-mv":
        import pandapower.networks as pn
        return pn.create_cigre_network_mv(with_der="all")
    import simbench as sb
    return sb.get_simbench_net(grid)


def coupler_groups(net):
    """Buses tied by bus-bus switches (et == 'b'): KCL is written over the whole group (valid open or closed)."""
    parent = {b: b for b in net.bus.index}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for _, sw in net.switch.iterrows():
        if sw.et == "b":
            a, c = find(sw.bus), find(int(sw.element))
            if a != c:
                parent[a] = c
    groups = {}
    for b in net.bus.index:
        groups.setdefault(find(b), []).append(b)
    return [sorted(g) for g in groups.values() if len(g) > 1]


def invariants(net):
    inv = {}
    grouped = {b for g in coupler_groups(net) for b in g}
    for g in coupler_groups(net):
        for pq, a in (("P", "active_power"), ("Q", "reactive_power")):
            terms = [(f"bus.{b}:{a}", (1.0,)) for b in g]
            for li, row in net.line.iterrows():
                if row.from_bus in g:
                    terms.append((f"line.{li}:{a}_from", (1.0, -1.0)))
                if row.to_bus in g:
                    terms.append((f"line.{li}:{a}_to", (1.0, -1.0)))
            for ti, row in net.trafo.iterrows():
                if row.hv_bus in g:
                    terms.append((f"trafo.{ti}:{a}_hv", (1.0, -1.0)))
                if row.lv_bus in g:
                    terms.append((f"trafo.{ti}:{a}_lv", (1.0, -1.0)))
            inv[f"KCL_{pq}(group.{'+'.join(map(str, g))})"] = terms
    for b in net.bus.index:
        for pq, a in (("P", "active_power"), ("Q", "reactive_power")):
            terms = [(f"bus.{b}:{a}", (1.0,))]
            for li, row in net.line.iterrows():
                if row.from_bus == b:
                    terms.append((f"line.{li}:{a}_from", (1.0, -1.0)))
                if row.to_bus == b:
                    terms.append((f"line.{li}:{a}_to", (1.0, -1.0)))
            for ti, row in net.trafo.iterrows():
                if row.hv_bus == b:
                    terms.append((f"trafo.{ti}:{a}_hv", (1.0, -1.0)))
                if row.lv_bus == b:
                    terms.append((f"trafo.{ti}:{a}_lv", (1.0, -1.0)))
            if len(terms) > 1 and b not in grouped:
                inv[f"KCL_{pq}(bus.{b})"] = terms
            terms = [(f"bus.{b}:{a}", (1.0,))]
            for tbl, name in ((net.load, "load"), (net.sgen, "sgen"), (net.storage, "storage"), (net.ext_grid, "external_grid")):
                for ei, row in tbl.iterrows():
                    if row.bus == b:
                        terms.append((f"{name}.{ei}:{a}", (1.0, -1.0)))
            if len(terms) > 1:
                inv[f"ELEM_{pq}(bus.{b})"] = terms
    for li, row in net.line.iterrows():
        inv[f"VLINE(line.{li},from)"] = [(f"line.{li}:voltage_from", (1.0,)), (f"bus.{row.from_bus}:voltage", (-1.0,))]
        inv[f"VLINE(line.{li},to)"] = [(f"line.{li}:voltage_to", (1.0,)), (f"bus.{row.to_bus}:voltage", (-1.0,))]
    return inv


def family(name):
    return name.split("_")[0].split("(")[0]


class Cache:
    def __init__(self, cols, skew=0):
        self.c, self.skew, self.env = cols, skew, {}

    def __call__(self, key):
        return self.c[key]

    def mean(self, key):
        """centred moving average over [t - skew, t + skew], ignoring missing values (time-alignment baseline)"""
        if not hasattr(self, "_mean"):
            self._mean = {}
        if key not in self._mean:
            v = self.c[key]
            k = self.skew
            ok = np.isfinite(v)
            x = np.where(ok, v, 0.0)
            cs = np.concatenate([[0.0], np.cumsum(x)])
            cn = np.concatenate([[0], np.cumsum(ok)])
            idx = np.arange(len(v))
            lo, hi = np.clip(idx - k, 0, len(v)), np.clip(idx + k + 1, 0, len(v))
            cnt = cn[hi] - cn[lo]
            self._mean[key] = np.where(cnt > 0, (cs[hi] - cs[lo]) / np.maximum(cnt, 1), np.nan)
        return self._mean[key]

    def envelope(self, key):
        """min and max of the series over [t - skew, t + skew] (sampling-time skew between the terms)."""
        if key not in self.env:
            v = self.c[key]
            if self.skew <= 0:
                self.env[key] = (v, v)
            else:
                from numpy.lib.stride_tricks import sliding_window_view
                w = 2 * self.skew + 1
                pad = np.concatenate([np.full(self.skew, v[0]), v, np.full(self.skew, v[-1])])
                win = sliding_window_view(pad, w)
                self.env[key] = (np.nanmin(win, axis=1), np.nanmax(win, axis=1))
        return self.env[key]


def is_voltage(terms):
    return all(":voltage" in k for k, _ in terms)


def residual(cache, terms, signs, normalize_power=False):
    tot, sq = 0.0, 0.0
    lo, hi = 0.0, 0.0
    use_mean = cache.skew > 0 and ALIGN == "mean"
    for (k, _), c in zip(terms, signs):
        v = cache.mean(k) if use_mean else cache(k)
        tot = tot + c * v
        sq = sq + v * v
        if cache.skew > 0 and not use_mean:
            a, b = cache.envelope(k)
            lo = lo + np.where(c > 0, c * a, c * b)
            hi = hi + np.where(c > 0, c * b, c * a)
    if cache.skew > 0 and not use_mean:    # distance of 0 from the interval of values reachable within the skew
        tot = np.where(lo > 0, lo, np.where(hi < 0, hi, 0.0))
    if normalize_power and not is_voltage(terms):
        return tot / (np.sqrt(sq) + 1e3)  # 1 kW regularizer.
    return tot


def gate(cache, name, r, p_gate):
    if not name.startswith("VLINE("):
        return r
    li, end = name[len("VLINE(line."):-1].split(",")
    return np.where(np.abs(cache(f"line.{li}:active_power_{end}")) > p_gate, r, 0.0)


def rolling_min(a, w):
    if w <= 1:
        return a
    from numpy.lib.stride_tricks import sliding_window_view
    return sliding_window_view(np.concatenate([np.full(w - 1, np.inf), a]), w).min(axis=1)


def choose_signs(cache, inv, with_ident=False, ratio=0.5):
    """Per invariant, the sign pattern with the smallest median |plain residual|. A term's sign counts as identifiable
    when flipping it at least doubles that median; otherwise the data cannot tell (e.g. a term that is nearly zero
    under noise)."""
    cache = Cache(cache.c, 0)            # sign conventions from the plain residual
    chosen, ident = {}, {}
    for name, terms in inv.items():
        med = {}
        for signs in itertools.product(*[c for _, c in terms]):
            med[signs] = float(np.nanmedian(np.abs(residual(cache, terms, signs))))
        best = min(med, key=med.get)
        chosen[name] = best
        flags = []
        for i, (_, opts) in enumerate(terms):
            if len(opts) == 1:
                flags.append(True)
                continue
            flip = tuple(-s if j == i else s for j, s in enumerate(best))
            flags.append(med[best] < ratio * med.get(flip, np.inf))
        ident[name] = flags
    return (chosen, ident) if with_ident else chosen


def delay(a, d):
    """causal use of the centred window: the decision for second k is available at k + d"""
    if d <= 0:
        return a
    return np.concatenate([np.zeros(d), a[:-d]])


def term_type(key):
    """measurement type: element class and attribute, e.g. ('line', 'active_power_from')"""
    el, attr = key.split(":")
    return el.split(".")[0], attr


def scores(cache, inv, signs, persist, p_gate, keep=False):
    fam_scores, per_inv = {}, {}
    for f in FAMILIES:
        names = [n for n in inv if family(n) == f]
        if not names:
            continue
        mats = []
        for n in names:
            r = np.abs(gate(cache, n, residual(cache, inv[n], signs[n], normalize_power=True), p_gate))
            mats.append(delay(rolling_min(np.nan_to_num(r, nan=0.0), persist), cache.skew))
        M = np.vstack(mats)
        fam_scores[f] = M.max(axis=0)
        if keep:
            per_inv[f] = (names, M)
    return (fam_scores, per_inv) if keep else fam_scores


def evaluate(t, alarm, events):
    starts = [i for i in range(len(alarm)) if alarm[i] and (i == 0 or not alarm[i - 1])]
    attacks = [e for e in events if "benign" not in e["id"]]
    benign = [e for e in events if "benign" in e["id"]]

    def phase(x):
        for e in attacks:
            if e["start"] <= x <= e["end"]:
                return "attack", e
        for e in attacks:
            if e["end"] < x <= e.get("recovery", e["end"]):
                return "recovery", e
        for e in benign:
            if e["start"] - 5 <= x <= e["end"] + 300:
                return "benign", e
        return "normal", None

    det, fa, fab = {}, 0, 0
    for i in starts:
        ph, e = phase(t[i])
        if ph == "attack" and e["id"] not in det:
            det[e["id"]] = t[i] - e["start"]
        elif ph in ("benign", "normal"):
            fa += 1
            fab += ph == "benign"
    per = defaultdict(lambda: [0, 0])
    for e in attacks:
        k = e["description"].split(":")[0]
        per[k][1] += 1
        per[k][0] += e["id"] in det
    return {"detected": len(det), "n_attacks": len(attacks), "false_alarms": fa, "false_alarms_benign": fab,
            "avg_ttd_s": float(np.mean(list(det.values()))) if det else None,
            "per_type": {k: f"{a}/{b}" for k, (a, b) in per.items()}, "detected_ids": sorted(det)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--train", default="")
    ap.add_argument("--thr-from", default="", help="JSON from a previous run: reuse its family thresholds (transfer)")
    ap.add_argument("--persist", type=int, default=30)
    ap.add_argument("--skew", type=int, default=10, help="tolerated reporting-time skew between terms [s] (interval residual)")
    ap.add_argument("--families", default="", help="comma-separated subset of ELEM,KCL,VLINE")
    ap.add_argument("--quantities", default="", help="P, Q or P,Q: which power-balance invariants to use")
    ap.add_argument("--valid-rel", type=float, default=0.01,
                    help="drop invariants whose training 99.9th-percentile relative residual exceeds this (0 = keep all)")
    ap.add_argument("--p-gate", type=float, default=1e3)
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--thr-floor", type=float, default=1e-3, help="lower bound of the relative power-family thresholds")
    ap.add_argument("--thr-floor-v", type=float, default=0.01, help="lower bound of the line-voltage threshold")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-scores", default="", help="npz with per-second family scores")
    ap.add_argument("--noise", type=float, default=0.0, help="relative std of multiplicative measurement noise, per report")
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--signs-from", default="",
                    help="JSON of a previous run: reuse its sign conventions (the documented conventions of the "
                         "measurement system; used for the noise experiment)")
    ap.add_argument("--convention-from", default="",
                    help="JSON of a run on another grid of the same measurement system: set every sign from its "
                         "per-measurement-type convention, so no data of the monitored grid are used for signs")
    ap.add_argument("--align", choices=["interval", "mean"], default="interval")
    args = ap.parse_args()
    if not args.train and not (args.convention_from and args.thr_from):
        ap.error("provide --train or both --convention-from and --thr-from")
    if args.persist < 1 or args.skew < 0:
        ap.error("--persist must be positive and --skew must be nonnegative")
    if args.thr_floor <= 0 or args.thr_floor_v <= 0:
        ap.error("threshold floors must be positive")
    global ALIGN
    ALIGN = args.align

    present = first_keys(args.test)
    net = build_net(args.grid)
    inv = invariants(net)
    inv = {n: terms for n, terms in inv.items() if all(k in present for k, _ in terms)}
    if args.families:
        keep = set(args.families.split(","))
        inv = {n: t for n, t in inv.items() if family(n) in keep}
    if args.quantities:
        qs = set(args.quantities.split(","))
        inv = {n: t for n, t in inv.items() if n.startswith("VLINE(") or n.split("(")[0].split("_")[-1] in qs}
    needed = {k for terms in inv.values() for k, _ in terms}
    for n in inv:
        if n.startswith("VLINE("):
            li, end = n[len("VLINE(line."):-1].split(",")
            needed.add(f"line.{li}:active_power_{end}")
    needed &= present
    t_te, cols_te = load_columns(args.test, needed)
    cols_te = add_noise(cols_te, args.noise, 2 * args.noise_seed + 1)
    cte = Cache(cols_te, args.skew)
    events = json.loads(Path(args.events).read_text(encoding="utf-8"))
    src = f"buses {len(net.bus)}, lines {len(net.line)}"
    print(f"grid {args.grid} ({src}): invariants {len(inv)}; test {len(t_te)} s, events {len(events)}")

    ctr = None
    if args.train:
        t_tr, cols_tr = load_columns(args.train, needed)
        cols_tr = add_noise(cols_tr, args.noise, 2 * args.noise_seed)
        ctr = Cache(cols_tr, args.skew)
    ident = None
    if ctr is not None:
        signs, ident = choose_signs(ctr, inv, with_ident=True)
    else:
        conv = json.loads(Path(args.convention_from).read_text(encoding="utf-8"))["convention"]
        signs, missing = {}, set()
        for n, terms in inv.items():
            s = []
            for k, opts in terms:
                if len(opts) == 1:
                    s.append(opts[0])
                else:
                    tt = "%s:%s" % term_type(k)
                    if tt not in conv:      # an element's P and Q follow the same direction convention
                        alt = tt.replace("reactive_power", "active_power") if "reactive" in tt else tt.replace("active_power", "reactive_power")
                        if alt in conv:
                            conv[tt] = conv[alt]
                            missing.add(f"{tt} (taken from {alt})")
                        else:
                            missing.add(tt)
                    s.append(conv.get(tt, opts[0]))
            signs[n] = tuple(s)
        print(f"sign conventions from {args.convention_from} by measurement type; types without a convention: {sorted(missing)}")
    convention = {}
    if ident is not None:                                        # per-measurement-type convention of this system
        from collections import Counter
        votes = {}
        for n, terms in inv.items():
            for (k, opts), s, ok in zip(terms, signs[n], ident[n]):
                if len(opts) > 1 and ok:
                    votes.setdefault("%s:%s" % term_type(k), Counter())[s] += 1
        convention = {tt: c.most_common(1)[0][0] for tt, c in votes.items()}
        split = {tt: dict(c) for tt, c in votes.items() if len(c) > 1}
        if split:
            print(f"measurement types with mixed sign choices: {split}")
    if args.signs_from:
        known = json.loads(Path(args.signs_from).read_text(encoding="utf-8"))["signs"]
        signs = {n: tuple(known[n]) if n in known else signs[n] for n in inv}
        print(f"sign conventions taken from {args.signs_from} for {sum(1 for n in inv if n in known)} of {len(inv)} invariants")
    all_signs = {n: list(signs[n]) for n in inv}  # Include candidates before validation.
    if ctr is not None and args.valid_rel > 0:
        kept, dropped = {}, {}
        for n, terms in inv.items():
            r = np.abs(residual(ctr, terms, signs[n]))
            if n.startswith("VLINE("):
                r = np.abs(gate(ctr, n, r, args.p_gate))
                rel = float(np.nanpercentile(r, 99.9))                     # pu
                ok = rel < args.valid_rel
            else:
                mag = np.zeros(len(r))
                for k, _ in terms:
                    mag = mag + np.abs(ctr(k))
                rel = float(np.nanpercentile(r, 99.9) / (np.nanmedian(mag) + 1e3))
                ok = rel < args.valid_rel
            (kept if ok else dropped)[n] = rel
        fam_cnt = lambda d: {f: sum(1 for n in d if family(n) == f) for f in FAMILIES}
        print(f"validity filter (<{args.valid_rel}): kept {fam_cnt(kept)}, dropped {fam_cnt(dropped)}")
        inv = {n: inv[n] for n in kept}
        signs = {n: signs[n] for n in kept}
    if args.thr_from:
        thr = json.loads(Path(args.thr_from).read_text(encoding="utf-8"))["thresholds"]
    else:
        st = scores(ctr, inv, signs, args.persist, args.p_gate)
        thr = {f: max(float(v.max()) * args.margin, args.thr_floor if f != "VLINE" else args.thr_floor_v) for f, v in st.items()}
    ste, per_inv = scores(cte, inv, signs, args.persist, args.p_gate, keep=True)
    alarm = np.zeros(len(t_te), bool)
    for f, v in ste.items():
        alarm |= v > thr[f]
    res = evaluate(t_te, alarm, events)
    # which family/invariant raised each false alarm
    attacks = [e for e in events if "benign" not in e["id"]]
    starts = [i for i in range(len(alarm)) if alarm[i] and (i == 0 or not alarm[i - 1])]
    fa_src = []
    for i in starts:
        t0 = t_te[i]
        if any(e["start"] <= t0 <= e.get("recovery", e["end"]) for e in attacks):
            continue
        j = i
        while j < len(alarm) and alarm[j]:
            j += 1
        fams_hit = [f for f, v in ste.items() if v[i] > thr[f]]
        tops = {}
        for f in fams_hit:
            names, M = per_inv[f]
            q = int(np.argmax(M[:, i]))
            tops[f] = {"invariant": names[q], "ratio": float(M[q, i] / thr[f]), "terms": [k for k, _ in inv[names[q]]]}
        fa_src.append({"t": float(t0), "duration_s": int(j - i), "families": fams_hit, "top": tops})
    res["false_alarm_sources"] = fa_src[:50]
    ev_rows = []
    for e in events:
        m = (t_te >= e["start"]) & (t_te <= e["end"])
        top = None                                   # invariant with the largest score/threshold inside the window
        if m.any():
            for f, (names, M) in per_inv.items():
                R = M[:, m] / thr[f]
                j = int(np.argmax(R.max(axis=1)))
                if top is None or R[j].max() > top[1]:
                    top = (names[j], float(R[j].max()), [k for k, _ in inv[names[j]]])
        first = None                                 # invariants above threshold when the first alarm starts in the window
        idx = np.flatnonzero(m)
        on = [i for i in idx if alarm[i] and (i == 0 or not alarm[i - 1])]
        if on:
            i = on[0]
            cand = []
            for f, (names, M) in per_inv.items():
                for q in np.flatnonzero(M[:, i] > thr[f]):
                    cand.append({"name": names[q], "ratio": float(M[q, i] / thr[f]), "terms": [k for k, _ in inv[names[q]]]})
            cand.sort(key=lambda x: -x["ratio"])
            first = {"t_offset_s": float(t_te[i] - e["start"]), "n_candidates": len(cand), "candidates": cand[:20]}
        ev_rows.append({"id": e["id"], "type": e["description"],
                        "peak": {f: (float((v[m] / thr[f]).max()) if m.any() else None) for f, v in ste.items()},
                        "top_invariant": None if top is None else {"name": top[0], "ratio": top[1], "terms": top[2]},
                        "first_alarm": first})
    out = {"grid": args.grid, "norm": "relative2", "persist": args.persist, "thresholds": thr, **res, "events": ev_rows,
           "signs": all_signs, "convention": convention, "noise": args.noise, "vscore": "pu", "align": ALIGN,
           "skew": args.skew, "n_invariants": {f: sum(1 for n in inv if family(n) == f) for f in FAMILIES}}
    Path(args.out).write_text(json.dumps(out, indent=2))
    if args.dump_scores:
        np.savez_compressed(args.dump_scores, t=t_te, **{f: v / thr[f] for f, v in ste.items()})
    print(f"thresholds {thr}")
    print(f"detected {res['detected']}/{res['n_attacks']}  false alarms {res['false_alarms']} (benign {res['false_alarms_benign']})  "
          f"avg TTD {res['avg_ttd_s']}")
    print("per type:", res["per_type"])
    if res["false_alarm_sources"]:
        from collections import Counter
        print("false alarms by family:", Counter(tuple(x["families"]) for x in res["false_alarm_sources"]),
              "durations:", sorted(x["duration_s"] for x in res["false_alarm_sources"])[:20])


if __name__ == "__main__":
    main()
