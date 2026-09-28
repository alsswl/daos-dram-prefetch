"""Compact read-only progress report, safe while experiment files are live."""
import json
from pathlib import Path
import re
import sys
import time
from collections import Counter
from staging_mixed_pressure import read_events, latest_sample

p=Path(sys.argv[1])
status=json.loads((p/'status.json').read_text())
case=p/('prefetch_'+status.get('condition','on'))
if not case.exists():
    case=next(p.glob('prefetch_*'))
phases=json.loads((case/'phases.json').read_text()) if (case/'phases.json').exists() else []
jobs=json.loads((case/'jobs.json').read_text()) if (case/'jobs.json').exists() else []
events=read_events(case)
hits=Counter(); queried=0
for e in events:
    if e['event']=='tier_lookup':
        hits[e['tier']]+=e['hit_chunks']
        if e['tier']=='dram':
            queried+=e['queried_chunks']
log=(case/'server.log').read_text() if (case/'server.log').exists() else ''
print(json.dumps(dict(status=status, condition=case.name,
    phase=phases[-1] if phases else None,
    elapsed_phase_s=round((time.time_ns()-phases[-1]['start_ns'])/1e9) if phases else None,
    elapsed_case_s=round((time.time_ns()-phases[0]['start_ns'])/1e9) if phases else None,
    workflows=Counter(j['status'] for j in jobs), finished=len(jobs),
    lookup_requests=sum(e['event']=='tier_lookup' and e['tier']=='dram' for e in events),
    lookup_chunks=queried, hits=hits,
    dram_pct=round(100*hits['dram']/queried,2) if queried else None,
    daos_pct=round(100*hits['daos']/queried,2) if queried else None,
    peak_staging_gib=max((e['used_bytes'] for e in events),default=0)/2**30,
    staging_snapshot=latest_sample(case),
    alloc_failures=sum(e['event'] in ('allocate','batched_allocate') and e['failed'] for e in events),
    critical_log_lines=[l for l in log.splitlines() if re.search(r'ERROR|GPU buffer full|negative: -|Double free',l)][-3:]
),ensure_ascii=False))
