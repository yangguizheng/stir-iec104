# -*- coding: utf-8 -*-
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def draw(root):
    out=root/'ramp'
    def rows(name):
        with (out/name).open() as file:return list(csv.DictReader(file))
    bins=rows('ramp_bins.csv'); exposures=rows('exposure.csv'); bottom=1e-9
    fig,axes=plt.subplots(1,2,figsize=(8.9,3.7));colors=['#375b80','#a25320','#137367']
    for s,color in zip(('01','02','03'),colors):
        b=[r for r in bins if r['scenario']==s and int(r['bin'])>=0];e=[r for r in exposures if r['scenario']==s]
        x=[float(r['u_median']) for r in b]; factors=[float(r['factor']) for r in e]
        for field,ls in [('plain','-'),('stir',':')]:
            median=np.maximum(bottom,[float(r[field+'_median']) for r in b]); upper=np.maximum(bottom,[float(r[field+'_p99']) for r in b]);
            axes[0].plot(x,median,ls=ls,lw=1.6,color=color);axes[0].plot(x,upper,ls=ls,lw=.8,color=color,alpha=.8)
            axes[0].fill_between(x,median,upper,color=color,alpha=.1 if field=='plain' else .045,hatch=None if field=='plain' else '///',linewidth=0)
            axes[1].plot(factors,[100*float(r[field+'_fraction']) for r in e],ls=ls,marker='o' if field=='plain' else None,color=color)
    summary=json.loads((out/'summary.json').read_text())
    thresholds={v for scenario in summary.values() for v in scenario['power_thresholds'].values()}
    for threshold in sorted(thresholds):axes[0].axhline(threshold,ls='--',lw=.7,color='#555555')
    axes[0].set(xscale='log',yscale='log',xlabel='Normalized envelope width',ylabel='Relative residual',title='(a) Median to 99th percentile')
    axes[1].set(xscale='log',xlabel='Ramp-rate factor',ylabel='Benign time above threshold (%)',title='(b) After 30 s persistence')
    axes[1].set_xticks([1/14,1/7,1/4,.5,1],labels=['1/14','1/7','1/4','1/2','1']);axes[1].set_ylim(-3,100)
    handles=[Line2D([],[],color=c,lw=3,label=s) for c,s in zip(colors,['01-Basic','02-Semiurban','03-Rural'])]+[Line2D([],[],color='black',ls=ls,label=name) for ls,name in [('-','Plain'),(':','STIR'),('--','Power thresholds')]]
    fig.legend(handles=handles,loc='upper center',ncol=6,frameon=False,fontsize=8.5)
    fig.text(.01,.02,'Panel (a) shows positive-width bins. Values below $10^{-9}$, including zero, are drawn at the axis floor.',fontsize=8)
    fig.tight_layout(rect=[0,.075,1,.91]);fig.savefig(out/'fig_ramp.pdf');plt.close(fig)

