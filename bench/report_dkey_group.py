"""Validate and visualize the isolated dkey grouping run."""
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(sys.argv[1]).resolve()
plan=json.loads((root/'plan.json').read_text())
status=json.loads((root/'status.json').read_text())
cleanup=json.loads((root/'cleanup.json').read_text())
rows=[json.loads(x) for x in (root/'measurements.jsonl').read_text().splitlines()]
summary=json.loads((root/'summary.json').read_text())
placements=json.loads((root/'placements.json').read_text())
assert status['status']=='completed' and cleanup['punch_and_close_rc']==0
assert len(rows)==plan['layouts']*len(plan['groups'])*len(plan['workers'])*(plan['repeats']+1)
assert all(r['verified'] and r['calls']*r['group']==plan['chunks'] for r in rows)
here=Path(__file__).resolve().parent
assert all(hashlib.sha256((here/f).read_bytes()).hexdigest()==h for f,h in plan['source_sha256'].items())
layout_text=(root/'object_layout.txt').read_text()
layout={int(g):{'rank':int(rank),'target':int(target)} for g,rank,target in
        re.findall(r'grp: (\d+)\s+replica 0 (\d+):(\d+)',layout_text)}
assert len(layout)==16 and len({(v['rank'],v['target']) for v in layout.values()})==16
for case in placements:
    for x in case['placements']:x.update(layout[x['shard']])
(root/'target_placements.json').write_text(json.dumps(placements,indent=2)+'\n')
for s in summary:
    subset=[r for r in rows if r['group']==s['group'] and r['workers']==s['workers'] and r['repeat']>=0]
    assert len(subset)==plan['layouts']*plan['repeats']
    assert all(len([r for r in subset if r['seed']==seed])==plan['repeats'] for seed in range(plan['layouts']))
    s['mean_ms']=statistics.mean(r['ms'] for r in subset)
    s['min_ms']=min(r['ms'] for r in subset)
    s['max_ms']=max(r['ms'] for r in subset)
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
with (root/'summary.csv').open('w') as f:
    writer=csv.DictWriter(f,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
groups=plan['groups']
fig,ax=plt.subplots(1,2,figsize=(11,4.4),layout='constrained')
colors={1:'#858b94',16:'#147d92'}
for w in plan['workers']:
    selected=[s for s in summary if s['workers']==w]
    ax[0].plot(range(len(groups)),[s['median_ms'] for s in selected],marker='o',color=colors[w],label=f'Max {w} concurrent fetches')
    for i,s in enumerate(selected):
        ax[0].scatter([i]*len(s['per_seed_median_ms']),s['per_seed_median_ms'],s=16,color=colors[w],alpha=.5)
ax[0].set_ylabel('Read latency (ms), lower is better')
ax[0].set_title('Same 1.6875 GiB payload in every read')
ax[0].legend(fontsize=9);ax[0].grid(axis='y',alpha=.2)
for seed in range(plan['layouts']):
    values=[next(c['active_shards'] for c in placements if c['seed']==seed and c['group']==g) for g in groups]
    ax[1].plot(range(len(groups)),values,marker='o',label=f'Key layout {seed+1}')
ax[1].set_ylabel('Targets holding the requested chunks')
ax[1].set_title('Larger groups reduce placement diversity')
ax[1].set_ylim(0,17);ax[1].legend(fontsize=9);ax[1].grid(axis='y',alpha=.2)
for a in ax:a.set_xticks(range(len(groups)),groups);a.set_xlabel('18 MiB chunks per dkey / fetch')
fig.suptitle('DAOS + GDR: dkey grouping microbenchmark',fontsize=14)
fig.savefig(root/'comparison.png',dpi=180)
fig.savefig(root/'comparison.svg')
plt.close(fig)
validation=dict(passed=True,full_byte_verified_reads=len(rows),measured_reads=sum(r['repeat']>=0 for r in rows),
                source_hashes_match=True,layout_target_mapping_verified=True,
                cleanup_rc=cleanup['punch_and_close_rc'],test_oid=plan['oid'],production_object_touched=False)
(root/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print(json.dumps({'validation':validation,'summary':summary},indent=2))
