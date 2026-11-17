#!/usr/bin/env python3
"""Evaluate IEC 104 command origin, ARP, report gaps, and command–status consistency."""
import argparse
import gzip
import json
import socket
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m5_physics import evaluate  # noqa: E402

CMD_TYPES = set(range(45, 52)) | set(range(58, 65))


def ip(b):
    return socket.inet_ntoa(b)


def mac(b):
    return ":".join(f"{x:02x}" for x in b)


def parse(pcap_dir):
    import dpkt
    cmds, arps, frames = [], [], defaultdict(list)
    for p in sorted(Path(pcap_dir).glob("*.pcap")):
        with open(p, "rb") as f:
            try:
                rd = dpkt.pcap.Reader(f)
            except ValueError:
                f.seek(0)
                rd = dpkt.pcapng.Reader(f)
            for ts, buf in rd:
                try:
                    eth = dpkt.ethernet.Ethernet(buf)
                except Exception:
                    continue
                if isinstance(eth.data, dpkt.arp.ARP):
                    a = eth.data
                    arps.append((ts, ip(a.spa), mac(a.sha)))
                    continue
                if not isinstance(eth.data, dpkt.ip.IP) or not isinstance(eth.data.data, dpkt.tcp.TCP):
                    continue
                ipk, tcp = eth.data, eth.data.data
                if 2404 not in (tcp.sport, tcp.dport) or not tcp.data:
                    continue
                pl, i = bytes(tcp.data), 0
                while i + 2 <= len(pl) and pl[i] == 0x68:
                    ln = pl[i + 1]
                    apdu = pl[i + 2:i + 2 + ln]
                    i += 2 + ln
                    if len(apdu) < 4 or apdu[0] & 1:          # U/S frames carry no ASDU
                        continue
                    asdu = apdu[4:]
                    if len(asdu) < 6:
                        continue
                    tid = asdu[0]
                    if tcp.sport == 2404:
                        frames[ip(ipk.src)].append(ts)
                    if tid in CMD_TYPES and tcp.dport == 2404:
                        cmds.append((ts, ip(ipk.src), ip(ipk.dst), tid))
    for k in frames:
        frames[k] = sorted(set(frames[k]))
    return sorted(cmds), sorted(arps), dict(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-pcaps", default="")
    ap.add_argument("--gap-default", type=float, default=0.0, help="max reporting gap per RTU when no training capture exists")
    ap.add_argument("--test-pcaps", required=True)
    ap.add_argument("--state", required=True, help="test state file (timeline)")
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hold", type=float, default=30.0)
    ap.add_argument("--grace", type=int, default=10, help="seconds a command-status mismatch may last (execution and reporting)")
    ap.add_argument("--margin", type=float, default=1.5)
    ap.add_argument("--phys-scores", default="", help="npz from m5_physics.py --dump-scores (OR with the physics alarms)")
    ap.add_argument("--dump-alarms", default="", help="npz with the per-second alarm of every channel (for timeline figures)")
    ap.add_argument("--init-seconds", type=float, default=600.0,
                    help="without training capture: infer the control centre from the first N seconds of capture only")
    args = ap.parse_args()

    if not args.train_pcaps and (args.init_seconds <= 0 or args.gap_default <= 0):
        ap.error("without --train-pcaps, provide a positive --gap-default and --init-seconds")
    c_te, a_te, f_te = parse(args.test_pcaps)
    if args.train_pcaps:
        c_tr, a_tr, f_tr = parse(args.train_pcaps)
        cmd_src = {s for _, s, _, _ in c_tr}
    else:
        # no training capture: the control centre is the client that holds IEC 104 sessions with the most RTUs
        c_tr, a_tr, f_tr = [], [], {}
        peers = defaultdict(set)
        if not f_te:
            ap.error("no IEC 104 reports in the test captures")
        t_first = min(v[0] for v in f_te.values())
        t_stop = t_first + args.init_seconds
        for p_ in sorted(Path(args.test_pcaps).glob("*.pcap")):
            import dpkt
            with open(p_, "rb") as f:
                for ts, buf in dpkt.pcap.Reader(f):
                    if ts > t_stop:
                        break
                    try:
                        eth = dpkt.ethernet.Ethernet(buf)
                        if isinstance(eth.data, dpkt.ip.IP) and isinstance(eth.data.data, dpkt.tcp.TCP) and eth.data.data.dport == 2404:
                            peers[ip(eth.data.src)].add(ip(eth.data.dst))
                    except Exception:
                        continue
        if not peers:
            ap.error("no IEC 104 peers in the initialization interval")
        mtu = max(peers, key=lambda k: len(peers[k]))
        cmd_src = {mtu}
        window = f"first {args.init_seconds:.0f} s of capture"
        print(f"no training capture: control centre inferred from {window} as {mtu} ({len(peers[mtu])} RTU peers; "
              f"next {sorted((len(v) for v in peers.values()), reverse=True)[1:3]})")
    bind = {}
    for _, i_, m_ in a_tr:
        bind.setdefault(i_, set()).add(m_)
    max_gap = {}
    for k, ts in f_tr.items():
        if len(ts) > 10:
            max_gap[k] = float(np.max(np.diff(ts)))
    if not f_tr and args.gap_default > 0:
        max_gap = {k: args.gap_default for k in f_te}
    gaps = list(max_gap.values()) or [float("nan")]
    print(f"train: {len(c_tr)} commands from {sorted(cmd_src)}; {len(bind)} ARP bindings; {len(max_gap)} RTUs, "
          f"max gap median {np.median(gaps):.1f}s, max {np.max(gaps):.1f}s")
    print(f"test : {len(c_te)} commands from {sorted({s for _, s, _, _ in c_te})}; {len(a_te)} ARP packets")

    # command-status consistency on the control-centre view: commanded configuration vs reported status
    pairs = []
    with gzip.open(args.state, "rt", encoding="utf-8") as f:
        first = json.loads(next(f))["state"]
    for k in first:
        el, attr = k.split(":")
        if attr == "closed" and f"{el}:is_closed" in first:
            pairs.append((k, f"{el}:is_closed"))
        if attr == "connected" and f"{el}:is_connected" in first:
            pairs.append((k, f"{el}:is_connected"))
    t, mism_pair = [], []
    with gzip.open(args.state, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            t.append(r["timestamp"])
            st = r["state"]
            mism_pair.append([st.get(a) is not None and st.get(b) is not None and bool(st.get(a)) != bool(st.get(b))
                              for a, b in pairs])
    t = np.array(t)
    mism_pair = np.array(mism_pair, dtype=bool).reshape(len(t), len(pairs))
    mism = mism_pair.any(axis=1)
    print(f"command-status pairs: {len(pairs)}")
    chan = {k: np.zeros(len(t), bool) for k in ("CMD", "ARP", "GAP", "CONSIST")}
    ent = defaultdict(lambda: np.zeros(len(t), bool))       # (channel, entity) -> alert of that device/source/IP/RTU
    run = 0
    for i in range(len(t)):
        run = run + 1 if mism[i] else 0
        if run > args.grace:
            chan["CONSIST"][i] = True
    for j, (a_, b_) in enumerate(pairs):                     # the same rule kept per device
        if not mism_pair[:, j].any():
            continue
        run = 0
        for i in range(len(t)):
            run = run + 1 if mism_pair[i, j] else 0
            if run > args.grace:
                ent[("CONSIST", a_.split(":")[0])][i] = True

    def mark(name, a, b, who):
        m = (t >= a) & (t <= b)
        chan[name] |= m
        ent[(name, who)] |= m

    for ts, s, d, tid in c_te:
        if s not in cmd_src:
            mark("CMD", ts, ts + args.hold, s)
    seen = {k: set(v) for k, v in bind.items()}
    for ts, i_, m_ in a_te:
        if i_ in seen and m_ not in seen[i_]:
            mark("ARP", ts, ts + args.hold, i_)
        seen.setdefault(i_, set()).add(m_) if i_ not in bind else None
    end = t[-1]
    for k, ts in f_te.items():
        if k not in max_gap:
            continue
        thr = max_gap[k] * args.margin
        seq = list(ts) + [end + 1e9]
        for a, b in zip(seq[:-1], seq[1:]):
            if b - a > thr:
                mark("GAP", a + thr, min(b, end), k)
    events = json.loads(Path(args.events).read_text(encoding="utf-8"))
    out = {"train_command_sources": sorted(cmd_src), "channels": {}}
    alarm_net = np.zeros(len(t), bool)
    for k, v in chan.items():
        r = evaluate(t, v, events)
        out["channels"][k] = r
        alarm_net |= v
        print(f"{k:3s}: detected {r['detected']}/{r['n_attacks']}  FA {r['false_alarms']} (benign {r['false_alarms_benign']})  {r['per_type']}")
    r = evaluate(t, alarm_net, events)
    out["network_union"] = r
    print(f"NET: detected {r['detected']}/{r['n_attacks']}  FA {r['false_alarms']} (benign {r['false_alarms_benign']})  "
          f"TTD {r['avg_ttd_s']}  {r['per_type']}")
    if args.phys_scores:
        d = np.load(args.phys_scores)
        tp = d["t"]
        phys = np.zeros(len(tp), bool)
        for f in ("ELEM", "KCL", "VLINE"):
            if f in d:
                phys |= d[f] > 1.0
        phys_on_t = np.interp(t, tp, phys.astype(float)) > 0.5
        r = evaluate(t, phys_on_t | alarm_net, events)
        out["combined"] = r
        print(f"ALL: detected {r['detected']}/{r['n_attacks']}  FA {r['false_alarms']} (benign {r['false_alarms_benign']})  "
              f"TTD {r['avg_ttd_s']}  {r['per_type']}")
        # channel-wise: every channel raises its own alerts; an attack counts if any channel starts an alert inside it
        per = {k: evaluate(t, v, events) for k, v in chan.items()}
        per["PHYS"] = evaluate(t, phys_on_t, events)
        det = {event_id for result in per.values() for event_id in result["detected_ids"]}
        attacks = [e for e in events if "benign" not in e["id"]]
        by = {}
        for e in attacks:
            k = e["description"].split(":")[0]
            a, b = by.get(k, (0, 0))
            by[k] = (a + (e["id"] in det), b + 1)
        fa = sum(rr["false_alarms"] for rr in per.values())
        fab = sum(rr["false_alarms_benign"] for rr in per.values())
        out["channelwise"] = {"detected": len(det), "n_attacks": len(attacks), "false_alarms": fa, "false_alarms_benign": fab,
                              "per_type": {k: f"{a}/{b}" for k, (a, b) in by.items()}, "detected_ids": sorted(det)}
        print(f"CHW: detected {len(det)}/{len(attacks)}  FA {fa} (benign {fab})  {out['channelwise']['per_type']}")
        # per alert: every (channel, entity) raises its own alert; physical alerts are keyed by invariant family
        for f in ("ELEM", "KCL", "VLINE"):
            if f in d:
                ent[("PHYS", f)] = np.interp(t, tp, (d[f] > 1.0).astype(float)) > 0.5
        first, fa, fab, n_alerts = {}, 0, 0, 0
        for (ch, who), v in ent.items():
            if not v.any():
                continue
            n_alerts += 1
            rr = evaluate(t, v, events)
            fa += rr["false_alarms"]
            fab += rr["false_alarms_benign"]
            on = np.flatnonzero(v & ~np.concatenate([[False], v[:-1]]))
            for e in attacks:
                hit = [t[i] - e["start"] for i in on if e["start"] <= t[i] <= e["end"]]
                if hit and (e["id"] not in first or min(hit) < first[e["id"]][0]):
                    first[e["id"]] = (min(hit), ch)
        by = {}
        for e in attacks:
            k = e["description"].split(":")[0]
            a, b = by.get(k, (0, 0))
            by[k] = (a + (e["id"] in first), b + 1)
        out["peralert"] = {"detected": len(first), "n_attacks": len(attacks), "false_alarms": fa, "false_alarms_benign": fab,
                           "avg_ttd_s": float(np.mean([v[0] for v in first.values()])) if first else None,
                           "per_type": {k: f"{a}/{b}" for k, (a, b) in by.items()}, "detected_ids": sorted(first),
                           "first_channel": {k: v[1] for k, v in first.items()}, "alert_streams": n_alerts}
        print(f"PER: detected {len(first)}/{len(attacks)}  FA {fa} (benign {fab})  TTD {out['peralert']['avg_ttd_s']}  "
              f"{out['peralert']['per_type']}  ({n_alerts} alert streams)")
        # protocol channels only, per alert (same counting, physical alerts excluded)
        firstp, fap = {}, 0
        for (ch, who), v in ent.items():
            if ch == "PHYS" or not v.any():
                continue
            rr = evaluate(t, v, events)
            fap += rr["false_alarms"]
            on = np.flatnonzero(v & ~np.concatenate([[False], v[:-1]]))
            for e in attacks:
                hit = [t[i] - e["start"] for i in on if e["start"] <= t[i] <= e["end"]]
                if hit and (e["id"] not in firstp or min(hit) < firstp[e["id"]]):
                    firstp[e["id"]] = min(hit)
        out["peralert_protocol"] = {"detected": len(firstp), "false_alarms": fap,
                                    "avg_ttd_s": float(np.mean(list(firstp.values()))) if firstp else None}
        # time with an active alert outside attack and recovery windows (alarm burden not captured by the counts)
        inatk = np.zeros(len(t), bool)
        for e in attacks:
            inatk |= (t >= e["start"]) & (t <= e.get("recovery", e["end"]))
        anyp = alarm_net | phys_on_t
        out["alarm_seconds_outside"] = {"all": int((anyp & ~inatk).sum()), "PHYS": int((phys_on_t & ~inatk).sum()),
                                        **{k: int((v & ~inatk).sum()) for k, v in chan.items()},
                                        "outside_window_seconds": int((~inatk).sum())}
        print(f"PROT-PER: detected {len(firstp)}/{len(attacks)}  FA {fap}   outside-window alarm s {out['alarm_seconds_outside']}")
        if args.dump_alarms:
            fams = {f: np.interp(t, tp, (d[f] > 1.0).astype(float)) > 0.5 for f in ("ELEM", "KCL", "VLINE") if f in d}
            np.savez_compressed(args.dump_alarms, t=t, PHYS=phys_on_t, **chan, **{f"PHYS_{f}": v for f, v in fams.items()})
    Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
