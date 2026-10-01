"""Plot measured occupancy and active native payload fetch calls."""
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(sys.argv[1]).resolve()
summary=json.loads((root/'summary.json').read_text())
fig,axes=plt.subplots(len(summary),2,figsize=(12,8),layout='constrained')
for i,s in enumerate(summary):
    with (root/s['run']/'timeline.csv').open() as f:
        rows=[{k:float(v) for k,v in r.items()} for r in csv.DictReader(f)]
    x=[r['ms'] for r in rows]
    a,b=axes[i]
    a.step(x,[r['used_mib'] for r in rows],where='post',color='#42566c',label='Allocated staging')
    a.fill_between(x,[r['ready_mib'] for r in rows],step='post',alpha=.7,color='#d99036',label='Fetch done, scatter not started')
    a.axhline(2048,color='#999999',linestyle='--',linewidth=1,label='2 GiB capacity')
    a.set_ylim(0,2200);a.set_ylabel('MiB')
    a.set_title(f"{s['run']}: ready wait {s['ready_wait_ms']['mean']:.1f} ms/chunk; mean held {s['time_average_mib']['ready']:.0f} MiB")
    b.step(x,[r['workers'] for r in rows],where='post',color='#9298a0',label='Active get workers')
    b.step(x,[r['payload_fetches'] for r in rows],where='post',color='#177f85',label='Active native payload fetches')
    b.set_ylim(0,17);b.set_ylabel('Concurrent calls')
    b.set_title(f"{s['run']}: 9 GiB retrieve {s['retrieve_ms']:.1f} ms")
    for ax in (a,b):
        ax.set_xlim(0,400);ax.set_xlabel('Time since retrieve pipeline start (ms)');ax.grid(axis='y',alpha=.2)
        ax.legend(loc='upper right',fontsize=7)
fig.suptitle('DAOS GDR staging: completed data waits while other reads continue\n2 GiB staging / 486 MiB windows / 16 workers / DRAM cache OFF',fontsize=13)
fig.savefig(root/'ready_occupancy.png',dpi=170)
fig.savefig(root/'ready_occupancy.svg')
