"""Check rural transfer across initialization windows and reference gap bounds."""
import csv
import json
import sys
from functools import lru_cache
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from locations import DATA, OUTPUT, SCRIPTS
sys.path.insert(0, str(SCRIPTS))
import m5_netchannels as net

data = DATA
out = OUTPUT/'transfer'
out.mkdir(exist_ok=True,parents=True)
_, _, frames = net.parse(data/'01-Basic/raw/train/pcap')
gaps = {key:float(np.max(np.diff(times))) for key,times in frames.items() if len(times)>10}
(out/'training_gaps_01.json').write_text(json.dumps(gaps,indent=2))
net.parse = lru_cache(maxsize=1)(net.parse)
rural=data/'03-Rural'
events = rural/'ipal/test/events.json'
rows=[]
for source,gap in [('02',10.5),('01',max(gaps.values()))]:
    for init in [120,300,600,900]:
        path=out/f'net_init{init}_gap{source}.json'
        sys.argv=['m5_netchannels','--gap-default',str(gap),'--init-seconds',str(init),'--test-pcaps',str(rural/'raw/test/pcap'),'--state',str(next(rural.glob('test.*.state.gz'))),'--events',str(events),'--phys-scores',str(OUTPUT/'main/scores_03.npz'),'--out',str(path)]
        net.main()
        result=json.loads(path.read_text()); alert=result['peralert']; prot=result['peralert_protocol']
        if source=='02' and init==600: assert result==json.loads((OUTPUT/'main/net_03.json').read_text())
        rows.append([init,source,gap,','.join(result['train_command_sources']),prot['detected'],alert['detected'],prot['false_alarms'],alert['false_alarms']])
with (out/'summary.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow('init_seconds,gap_source,gap_seconds,control_center,protocol,complete,protocol_fa,complete_fa'.split(','));w.writerows(rows)
