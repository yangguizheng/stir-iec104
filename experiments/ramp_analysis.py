# -*- coding: utf-8 -*-
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from locations import DATA, OUTPUT, CACHE, SCRIPTS
sys.path.insert(0, str(SCRIPTS))
import m5_physics as phy

out = OUTPUT / "ramp"
out.mkdir(parents=True, exist_ok=True)
bins, exposures, summary = [], [], {}
factors = np.array([1., .5, .25, 1/7, 1/14])

for index, scenario in enumerate(('01', '02', '03')):
    snapshot = json.loads((OUTPUT / f'main/phys_{scenario}.json').read_text())
    folder = next(DATA.glob(scenario + '-*'))
    test = next(folder.glob('test.*.state.gz'))
    inv = {name: terms for name, terms in phy.invariants(phy.build_net(snapshot['grid'])).items()
           if name in snapshot['signs'] and phy.family(name) in ('ELEM', 'KCL')}
    keys = {key for terms in inv.values() for key, _ in terms}
    if scenario != '03':
        _, train = phy.load_columns(next(folder.glob('train.*.state.gz')), keys)
        cache = phy.Cache(train, 10)
        kept = {}
        for name, terms in inv.items():
            residual = np.abs(phy.residual(cache, terms, snapshot['signs'][name]))
            magnitude = sum(np.abs(train[key]) for key, _ in terms)
            if np.nanpercentile(residual, 99.9) / (np.nanmedian(magnitude)+1000) < .01:
                kept[name] = terms
        inv = kept
        del train, cache
    counts = {family: sum(phy.family(name) == family for name in inv) for family in ('ELEM', 'KCL')}
    assert all(counts[f] == snapshot['n_invariants'][f] for f in counts)
    times, columns = phy.load_columns(test, {k for terms in inv.values() for k, _ in terms})
    benign = np.ones(len(times), bool)
    for event in json.loads((folder/'ipal/test/events.json').read_text()):
        if 'benign' not in event['id']:
            benign &= ~((times >= event['start']) & (times <= event.get('recovery',event['end'])))
    edges = np.flatnonzero(np.diff(np.r_[False, benign, False]))
    samples, plain_counts, stir_count = [], np.zeros(len(factors), int), 0
    for start, end in edges.reshape(-1, 2):
        segment = {k: v[start:end] for k, v in columns.items()}
        plain, stir = phy.Cache(segment), phy.Cache(segment, 10)
        maxima = np.zeros((2, end-start))
        for name, terms in inv.items():
            threshold = snapshot['thresholds'][phy.family(name)]
            z0, z1 = (np.abs(phy.residual(cache, terms, snapshot['signs'][name], 'relative2')) for cache in (plain, stir))
            width = sum(np.subtract(*reversed(stir.envelope(k))) for k, _ in terms)
            denom = np.sqrt(sum(segment[k]**2 for k, _ in terms)) + 1000
            sample = np.column_stack([width/denom, z0, z1]); samples.append(sample[np.isfinite(sample).all(axis=1)])
            if end-start >= 30:
                for j, z in enumerate((z0,z1)):
                    score = phy.rolling_min(np.nan_to_num(z), 30)/threshold
                    score[:29] = 0.0
                    maxima[j] = np.maximum(maxima[j], score)
        plain_counts += np.sum(factors[:, None]*maxima[0] > 1, axis=1)
        stir_count += int(np.count_nonzero(maxima[1] > 1))
    samples = np.concatenate(samples)
    widths = samples[:,0]; positive = widths[widths > 0]
    cut = np.unique(np.quantile(positive, np.linspace(0, 1, 11)))
    indices = np.where(widths == 0, -1, np.searchsorted(cut[1:-1], widths, side='right'))
    for band in np.unique(indices):
        vals = samples[indices==band]; quant = np.quantile(vals[:,1:], [.5,.9,.99,1], axis=0)
        row = [scenario, int(band), len(vals), *np.quantile(vals[:,0],[0,.5,1]), *quant[:,0], *quant[:,1]]
        bins.append(row)
    summary[scenario] = dict(retained=counts, benign_seconds=int(benign.sum()),equation_seconds=len(samples),stir_max=float(samples[:,2].max()),power_thresholds={f:snapshot['thresholds'][f] for f in counts})
    for factor,count in zip(factors,plain_counts): exposures.append([scenario,factor,int(benign.sum()),int(count),int(stir_count),count/benign.sum(),stir_count/benign.sum()])
    del samples, columns, widths, positive

for file,header,rows in [('ramp_bins.csv','scenario,bin,n,u_min,u_median,u_max,plain_median,plain_p90,plain_p99,plain_max,stir_median,stir_p90,stir_p99,stir_max',bins),('exposure.csv','scenario,factor,benign_seconds,plain_seconds,stir_seconds,plain_fraction,stir_fraction',exposures)]:
    with (out/file).open('w',newline='') as stream:
        writer=csv.writer(stream); writer.writerow(header.split(',')); writer.writerows(rows)
(out/'summary.json').write_text(json.dumps(summary,indent=2))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from plot_ramp import draw
draw(OUTPUT)
