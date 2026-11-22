#!/usr/bin/env python3
"""Generate LaTeX tables and the sufficient-condition summary from recorded outcomes."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from locations import OUTPUT

from scipy.stats import beta

R = OUTPUT
SC = ("01", "02", "03")


def cp(x, n):
    lo = beta.ppf(0.025, x, n - x + 1) if x else 0.0
    hi = beta.ppf(0.975, x + 1, n - x) if x < n else 1.0
    return lo, hi


def budget_table(out):
    d = json.loads((R / "opcurve/opcurve.json").read_text())
    names = [("STIR", "\\stir{}"), ("interpolation", "Linear interpolation"), ("moving average", "Moving average, $\\pm 10$~s"),
             ("plain", "Plain residual")]
    rows = []
    for key, label in names:
        cells = []
        for sc in SC:
            for b in (0, 10):
                best = max(r["det"] for r in d[sc][key] if r["fa"] <= b)
                cells.append(str(best))
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    n = [d[sc]["n_attacks"] for sc in SC]
    p = [d[sc]["protocol"]["detected"] for sc in SC]
    tex = ("\\begin{table}[!tb]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}\n"
           "\\caption{Attacks detected per alert at the best common scale of the thresholds, for a budget of at most 0 or 10 false alarms. "
           "The scale is chosen on the test labels, so the counts favor every variant. The protocol channels alone detect "
           f"{p[0]}, {p[1]} and {p[2]} attacks on the three grids.}}\\label{{tab:budget}}\n"
           "\\begin{tabular}{@{}lrrrrrr@{}}\\toprule\n"
           f" & \\multicolumn{{2}}{{c}}{{01-Basic ({n[0]})}} & \\multicolumn{{2}}{{c}}{{02-Semiurban ({n[1]})}} & \\multicolumn{{2}}{{c}}{{03-Rural ({n[2]})}} \\\\\n"
           "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
           "Physical channel & 0 & 10 & 0 & 10 & 0 & 10 \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
    (out / "tab_budget.tex").write_text(tex)


def inject_tables(out):
    from locations import trial_records
    recs = {sc: trial_records(sc) for sc in SC}
    settings = []
    for kind, label in (("scale", "Scaled by $1\\pm\\alpha$"), ("random", "Randomized within $\\pm\\alpha$"),
                        ("freeze", "Frozen"), ("extrap", "Extrapolated")):
        levels = sorted({(r["a"], r["D"]) for r in recs["01"] if r["kind"] == kind})
        for a, D in levels:
            tag = f"$\\alpha = {100 * a:g}\\%$" if kind in ("scale", "random") else f"{D}~s"
            settings.append((kind, a, D, label, tag))
    lines, last = [], None
    for kind, a, D, label, tag in settings:
        if last is not None and kind != last:
            lines.append("\\midrule")
        last = kind
        cells = []
        for sc in SC:
            q = [r for r in recs[sc] if r["kind"] == kind and r["a"] == a and r["D"] == D]
            k = sum(r["detected"] for r in q)
            lo, hi = cp(k, len(q))
            cells.append(f"{100 * k / len(q):.1f} [{100 * lo:.0f}, {100 * hi:.0f}]")
        first = (not lines) or lines[-1] == "\\midrule"
        lab = label if first else ""
        lines.append(f"{lab} & {tag} & " + " & ".join(cells) + " \\\\")
    tex = ("\\begin{table}[!tb]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}\n"
           "\\caption{Share of injected falsifications that the physical channel detected, in percent with the exact 95\\% interval, "
           "for 500 injections per setting and grid. Scaling and randomization last 500~s; frozen and extrapolated measurements last "
           "as indicated.}\\label{tab:inject}\n"
           "\\begin{tabular}{@{}llrrr@{}}\\toprule\n"
           "Falsification & Setting & 01-Basic & 02-Semiurban & 03-Rural \\\\\\midrule\n" + "\n".join(lines) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
    (out / "tab_inject.tex").write_text(tex)

    cats = ["bus injection", "branch flow", "element power", "voltage"]
    names = {"bus injection": "Bus injection $P$, $Q$", "branch flow": "Line and transformer flow", "element power": "Load, generator, storage power",
             "voltage": "Bus and line-end voltage"}
    rows = []
    for c in cats:
        cells = []
        for sc in SC:
            q = [r for r in recs[sc] if r["kind"] == "scale" and r["a"] >= 0.05 and r["cat"] == c]
            k = sum(r["detected"] for r in q)
            cells.append(f"{100 * k / len(q):.1f} ({len(q)})")
        rows.append(f"{names[c]} & " + " & ".join(cells) + " \\\\")
    tex = ("\\begin{table}[!tb]\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}\n"
           "\\caption{Share of scaling injections of 5\\% to 50\\% detected, by type of the falsified measurement, in percent with the number "
           "of injections in parentheses.}\\label{tab:inject-cat}\n"
           "\\begin{tabular}{@{}lrrr@{}}\\toprule\n"
           "Falsified measurement & 01-Basic & 02-Semiurban & 03-Rural \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
    (out / "tab_inject_cat.tex").write_text(tex)

    # sufficient condition of Proposition 2
    summ = {}
    for sc in SC:
        rr = [r for r in recs[sc] if r["kind"] == "scale"]
        summ[sc] = dict(n=len(rr), pred=sum(r["pred"] for r in rr), pred_det=sum(1 for r in rr if r["pred"] and r["detected"]),
                        det=sum(r["detected"] for r in rr))
    (out.parent / "inject_bound_summary.json").write_text(json.dumps(summ))
    return summ


def heldout_table(out):
    rows = []
    for sc in ("01", "02"):
        for f in range(6):
            a = json.loads((R / f"heldout_fixed/heldout_{sc}_fold{f}.json").read_text())
            b = json.loads((R / f"heldout/heldout_{sc}_fold{f}.json").read_text())
            pk = a["physical"]
            fa_fixed = a["false_alarms_total"]
            fa_strict = b["false_alarms_total"]
            thr = a["thresholds"]["KCL"]
            ev = ", ".join(str(e["id"] if isinstance(e, dict) else e).split(" ")[0] for e in a["events"])
            grid = "01-Basic" if sc == "01" else "02-Semiurban"
            rows.append(f"{grid if f == 0 else ''} & {f + 1} & {ev} & {thr:.3g} & {pk['ELEM']['peak_ratio']:.3f} & {pk['KCL']['peak_ratio']:.3f} & "
                        f"{pk['VLINE']['peak_ratio']:.3f} & {fa_fixed} & {fa_strict} \\\\")
        if sc == "01":
            rows.append("\\midrule")
    tex = ("\\begin{table}[!ht]\\centering\\footnotesize\\setlength{\\tabcolsep}{3.5pt}\n"
           "\\caption{Blocked cross-validation on the benign training runs. Events are the benign events of the block. The KCL threshold "
           "is learned from the other five blocks; the next three columns give the highest score in the held-out block as a fraction of "
           "the threshold of its family; the last two columns give the false alarms with sign conventions and retained equations of the "
           "complete training run (fixed) and with these re-estimated without the block (strict).}\\label{tab:heldout}\n"
           "\\begin{tabular}{@{}llllrrrrr@{}}\\toprule\n"
           " & Block & Events & $\\theta_{\\mathrm{KCL}}$ & ELEM & KCL & VLINE & Fixed & Strict \\\\\\midrule\n" + "\n".join(rows) +
           "\n\\bottomrule\\end{tabular}\\end{table}\n")
    (out / "tab_heldout.tex").write_text(tex)


def se_table(out):
    f = R / "se/se_results.json"
    if not f.exists():
        raise FileNotFoundError(f)
    d = json.loads(f.read_text())
    frozen = {sc: json.loads((R / f"main/net_{sc}.json").read_text()) for sc in ("01", "03")}
    physf = {sc: json.loads((R / f"main/phys_{sc}.json").read_text()) for sc in ("01", "03")}
    budget = json.loads((R / "opcurve/opcurve.json").read_text())
    n = {"01": 18, "03": 28}
    rows = []
    order = [("SE objective $J$", "SE objective J"), ("SE largest normalized residual", "SE largest normalized residual")]
    for label, key in order:
        for variant, vlabel in (("static", "static"), ("flows", "processed")):
            cells = []
            for sc in ("01", "03"):
                r = d[sc][f"{key} ({variant})"]
                one = r["nominal"]
                zero = max((x["det"] for x in r["rows"] if x["fa"] == 0), default=0)
                cl = r["classical"]
                cells += [f"{one['det_phys']}/{n[sc]}, {one['fa_phys']}", f"{one['det']}, {one['fa']}", f"{zero}", f"{cl['det_phys']}/{n[sc]}, {cl['fa_phys']}"]
            rows.append(f"{label if variant == 'static' else ''} & {vlabel} & " + " & ".join(cells) + " \\\\")
        rows.append("\\midrule" if label == order[0][0] else "")
    stir = []
    for sc in ("01", "03"):
        pf = physf[sc]
        stir += [f"{pf['detected']}/{n[sc]}, {pf['false_alarms']}", f"{frozen[sc]['peralert']['detected']}, {frozen[sc]['peralert']['false_alarms']}",
                 str(max(x["det"] for x in budget[sc]["STIR"] if x["fa"] == 0)), "--"]
    rows.append("\\midrule")
    rows.append("\\stir{} & & " + " & ".join(stir) + " \\\\")
    rows = [r for r in rows if r]
    tex = ("\\begin{table}[!tb]\\centering\\scriptsize\\setlength{\\tabcolsep}{2.4pt}\n"
           "\\caption{State estimation with bad-data detection in place of the physical channel. Alone: attacks detected by the statistic alone "
           "(merged alarm signal) and its false alarms. Complete: attacks detected per alert and false alarms of the complete detector. Best: "
           "attacks detected per alert at zero false alarms for the best threshold scale. Classical: the same as Alone for the classical decision, "
           "a chi-square test at 1\\% for $J$ and a normalized residual above 3. Static keeps the switch positions of the model; processed sets "
           "the branch-end switches from the telemetry.}"
           "\\label{tab:se}\n"
           "\\begin{tabular}{@{}llllllllll@{}}\\toprule\n"
           " & & \\multicolumn{4}{c}{01-Basic} & \\multicolumn{4}{c}{03-Rural} \\\\\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}\n"
           "Statistic & Topology & Alone & Complete & Best & Classical & Alone & Complete & Best & Classical \\\\\\midrule\n" + "\n".join(rows) +
           "\n\\bottomrule\\end{tabular}\\end{table}\n")
    (out / "tab_se.tex").write_text(tex)


if __name__ == "__main__":
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    budget_table(out)
    s = inject_tables(out)
    heldout_table(out)
    se_table(out)
    print("tables written to", out)
    print("bound summary", s)
