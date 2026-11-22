#!/usr/bin/env python3
"""Fit a weighted least-squares state estimator to recorded grid measurements."""
import logging
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import engine as E  # noqa: E402

warnings.filterwarnings("ignore")
logging.getLogger("pandapower").setLevel(logging.CRITICAL)
import pandapower as pp  # noqa: E402
from pandapower.estimation.state_estimation import StateEstimation  # noqa: E402

P = E.P
OUT = E.OUTPUT / "se"
SIG_V, FLOOR_PQ = 0.004, 2e3          # 0.004 p.u.; 2 kW / 2 kvar


def meas_plan(net, present):
    """(type, element_type, element index, side, key, scale to MW/Mvar) for every measurement the estimator accepts"""
    plan = []
    for i in net.bus.index:
        plan.append(("v", "bus", i, None, f"bus.{i}:voltage", 1.0))
        plan.append(("p", "bus", i, None, f"bus.{i}:active_power", 1e-6))
        plan.append(("q", "bus", i, None, f"bus.{i}:reactive_power", 1e-6))
    for i in net.line.index:
        for side in ("from", "to"):
            plan.append(("p", "line", i, side, f"line.{i}:active_power_{side}", 1e-6))
            plan.append(("q", "line", i, side, f"line.{i}:reactive_power_{side}", 1e-6))
    for i in net.trafo.index:
        for side in ("hv", "lv"):
            plan.append(("p", "trafo", i, side, f"trafo.{i}:active_power_{side}", 1e-6))
            plan.append(("q", "trafo", i, side, f"trafo.{i}:reactive_power_{side}", 1e-6))
    return [m for m in plan if m[4] in present]


def sigmas(plan, ref_cols):
    out = []
    for typ, _, _, _, key, sc in plan:
        if typ == "v":
            out.append(SIG_V)
        else:
            full = float(np.nanpercentile(np.abs(ref_cols[key]), 99))
            out.append(max(0.01 * full, FLOOR_PQ) * sc)
    return np.array(out)


def setup(scen, sigma_from=None):
    cfg = E.CFG[scen]
    net = P.build_net(cfg["grid"])
    test = E.load_cache(scen, "test")
    present = set(test)
    plan = meas_plan(net, present)
    if scen == "03":
        ref = {k: test[k][:600] for _, _, _, _, k, _ in plan}
    else:
        tr = E.load_cache(scen, "train")
        ref = {k: tr[k] for _, _, _, _, k, _ in plan}
    sig = sigmas(plan, ref)
    net.measurement.drop(net.measurement.index, inplace=True)
    for (typ, et, el, side, key, sc), s in zip(plan, sig):
        if et == "bus":
            pp.create_measurement(net, typ, "bus", 0.0, float(s), el)
        else:
            pp.create_measurement(net, typ, et, 0.0, float(s), el, side=side)
    # switch bookkeeping: model switch index -> state key of the reported position
    sw_keys = {i: f"switch.{i}:is_closed" for i in net.switch.index if f"switch.{i}:is_closed" in present}
    n_model_sw = len(net.switch)
    # the public models lack the breakers that the recorded grids have; every branch end gets a switch whose position the
    # topology processor sets from the telemetry (an end that reports no flow is open)
    end_sw = {}
    for i in net.line.index:
        end_sw[("line", i, "from")] = pp.create_switch(net, int(net.line.from_bus[i]), int(i), et="l", closed=True)
        end_sw[("line", i, "to")] = pp.create_switch(net, int(net.line.to_bus[i]), int(i), et="l", closed=True)
    for i in net.trafo.index:
        end_sw[("trafo", i, "hv")] = pp.create_switch(net, int(net.trafo.hv_bus[i]), int(i), et="t", closed=True)
        end_sw[("trafo", i, "lv")] = pp.create_switch(net, int(net.trafo.lv_bus[i]), int(i), et="t", closed=True)
    net._end_sw, net._n_model_sw = end_sw, n_model_sw
    return net, plan, sig, sw_keys


def estimate_one(se, net, vals, sw_closed=None):
    net.measurement["value"] = vals
    if sw_closed is not None:
        net.switch["closed"] = sw_closed
    try:
        res = se.estimate(v_start="flat", delta_start="flat")
    except Exception:
        return None
    if not (res["success"] if isinstance(res, dict) else res):
        return None
    sol = se.solver
    r = np.asarray(sol.r).ravel()
    Rinv = np.asarray(sol.R_inv)
    J = float(r @ (Rinv @ r))
    m = len(r)
    n = len(sol.eppci.v) + len(sol.eppci.delta) - 1
    try:
        R = np.linalg.inv(Rinv)
        Om = np.diag(R - sol.H @ np.linalg.solve(sol.Gm, sol.H.T))
        rn = np.abs(r) / np.sqrt(np.abs(Om) + 1e-30)
    except Exception:
        return J, m - n, np.nan, -1
    j = int(np.argmax(rn))
    return J, m - n, float(rn[j]), int(sol.pp_meas_indices[j])


def selftest(scen):
    net, plan, sig, sw_keys = setup(scen)
    pp.runpp(net, calculate_voltage_angles=True)
    vals = []
    for typ, et, el, side, key, sc in plan:
        if et == "bus":
            v = {"v": net.res_bus.vm_pu.at[el], "p": net.res_bus.p_mw.at[el], "q": net.res_bus.q_mvar.at[el]}[typ]
        elif et == "line":
            v = getattr(net, "res_line")[f"{typ}_{side}_{'mw' if typ == 'p' else 'mvar'}"].at[el]
        else:
            v = getattr(net, "res_trafo")[f"{typ}_{side}_{'mw' if typ == 'p' else 'mvar'}"].at[el]
        vals.append(float(v))
    se = StateEstimation(net, tolerance=1e-9, maximum_iterations=50)
    t0 = time.time()
    out = estimate_one(se, net, np.array(vals))
    ref = net.res_bus.vm_pu.values.copy()
    err = float(np.abs(net.res_bus_est.vm_pu.values - ref).max())
    print(f"selftest {scen}: J={out[0]:.3e} dof={out[1]} rN_max={out[2]:.3e} max|dV|={err:.2e} ({time.time() - t0:.2f} s, {len(plan)} measurements)")
    return err < 1e-6 and out[0] < 1e-6


def run_chunk(scen, stream, variant, chunk, nchunks):
    net, plan, sig, sw_keys = setup(scen)
    cache = E.load_cache(scen, stream)
    n = len(cache["t"])
    edges = [int(x) for x in np.linspace(0, n, nchunks + 1)]
    a, b = edges[chunk], edges[chunk + 1]
    X = np.column_stack([cache[k][a:b] * sc for (_, _, _, _, k, sc) in plan])
    SW = None
    base = net.switch["closed"].values.copy()
    sw_index = []
    if variant in ("updated", "flows"):
        keys = list(sw_keys.values())
        SW = np.column_stack([cache[k][a:b] for k in keys]) if keys else None
        sw_index = list(sw_keys.keys())
    flow_cols = None
    if variant == "flows":
        # open end = no active and no reactive flow reported there (1 W)
        flow_cols = []
        pos = {(m[1], m[2], m[3], m[0]): j for j, m in enumerate(plan)}
        for (et, el, side), sw_i in net._end_sw.items():
            jp, jq = pos.get((et, el, side, "p")), pos.get((et, el, side, "q"))
            if jp is not None and jq is not None:
                flow_cols.append((net.switch.index.get_loc(sw_i), jp, jq))
    se = StateEstimation(net, tolerance=1e-6, maximum_iterations=30)
    J = np.full(b - a, np.nan)
    dof = np.full(b - a, np.nan)
    rn = np.full(b - a, np.nan)
    arg = np.full(b - a, -2, dtype=int)
    last = X[0].copy()
    t0 = time.time()
    for i in range(b - a):
        vals = X[i].copy()
        bad = ~np.isfinite(vals)
        vals[bad] = last[bad]
        last = vals
        swc = None
        if SW is not None or flow_cols is not None:
            closed = base.copy()
            if SW is not None:
                for j, idx in enumerate(sw_index):
                    v = SW[i, j]
                    if np.isfinite(v):
                        closed[net.switch.index.get_loc(idx)] = bool(v)
            if flow_cols is not None:
                for loc, jp, jq in flow_cols:
                    if abs(vals[jp]) < 1e-6 and abs(vals[jq]) < 1e-6:
                        closed[loc] = False
            swc = closed
        out = estimate_one(se, net, vals, swc)
        if out is not None:
            J[i], dof[i], rn[i], arg[i] = out
        if i % 600 == 0:
            print(f"  {scen} {stream} {variant} chunk {chunk}: {i}/{b - a}  {time.time() - t0:.0f} s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"se_{scen}_{stream}_{variant}_c{chunk}.npz", a=a, b=b, J=J, dof=dof, rn=rn, arg=arg,
             meas_keys=np.array([k for (_, _, _, _, k, _) in plan]), sigma=sig)
    print(f"{scen} {stream} {variant} chunk {chunk}: {b - a} s, failures {int(np.isnan(J).sum())}, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    if sys.argv[2] == "selftest":
        raise SystemExit(0 if selftest(sys.argv[1]) else 1)
    run_chunk(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
