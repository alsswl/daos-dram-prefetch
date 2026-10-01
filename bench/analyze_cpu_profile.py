"""Attribute user-CPU stack samples within measured read intervals only."""
import collections
import json
from pathlib import Path
import re
import sys

root=Path(sys.argv[1])
trials=[json.loads(x) for x in (root/'trials.jsonl').read_text().splitlines()]
counts={};examples={}
for block in (root/'callstacks.txt').read_text().split('\n\n'):
    lines=block.strip().splitlines()
    if not lines:continue
    match=re.match(r'([0-9.]+):',lines[0])
    if not match:continue
    t=int(float(match[1])*1e9)
    r=next((r for r in trials if r['phase'] in ('native','engine') and
            not r['trial'].startswith('r-1') and r['start']<=t<=r['end']),None)
    if r is None:continue
    frames=[]
    for line in lines[1:]:
        match=re.match(r'\s*\S+ (.*) \((.*)\)',line)
        if match:frames.append(match[1])
    if not frames:continue
    lock=any(word in frames[0] for word in ('mutex','spin_lock','lll_lock'))
    if any('cuda_gdrcopy_dev_unregister' in f for f in frames):category='CUDA_memory_unregister_path'
    elif any('cuda_gdrcopy_dev_register' in f for f in frames):category='CUDA_memory_register_path'
    elif any('na_ofi_mem_deregister' in f for f in frames):category='other_OFI_deregister_path'
    elif any('na_ofi_mem_register' in f for f in frames):category='other_OFI_register_path'
    elif lock and any('tse_sched' in f for f in frames):category='TSE_scheduler_lock_path'
    elif lock:category='other_lock_path'
    elif any('tse_sched' in f for f in frames):category='TSE_scheduler_nonlock'
    else:category='other'
    counts.setdefault(r['phase'],collections.Counter())[category]+=1
    examples.setdefault(r['phase'],{}).setdefault(category,frames[:25])
out={phase:dict(samples=sum(c.values()),categories={k:dict(samples=n,percent=100*n/sum(c.values()))
          for k,n in c.most_common()},examples=examples[phase]) for phase,c in counts.items()}
out['notes']=['99Hz user-CPU samples from separate OMP=1 64-worker diagnostic run.',
              'Samples represent CPU execution, not elapsed critical-path time.',
              'Registration-path presence does not establish one physical NIC MR creation per chunk.',
              'Initialization, store, warmup, CPU/GPU correctness checks excluded by monotonic phase timestamps.']
(root/'component_attribution.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v['categories'] for k,v in out.items() if k!='notes'},indent=2))
