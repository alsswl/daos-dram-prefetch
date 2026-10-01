"""Validate entry/return completeness and separate nested MR and lock costs."""
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys

root=Path(sys.argv[1])
read=lambda p:json.loads(p.read_text())
funcs={f['id']:f for f in read(root/'functions.json')}
entries={};returns={};events=defaultdict(list)
for line in (root/'bpf.log').read_text().splitlines():
    if line.startswith('E,'):
        _,tag,func,tid,start,end,result=line.split(',')
        events[int(tag),int(func)].append(dict(tid=int(tid),start=int(start),end=int(end),result=int(result)))
    m=re.match(r'@(entered|left)\[(\d+), (\d+)\]: (\d+)',line)
    if m:(entries if m[1]=='entered' else returns)[int(m[2]),int(m[3])]=int(m[4])
assert entries==returns=={k:len(v) for k,v in events.items()},'Lost or unmatched probe events'
assert not re.search(r'Lost|lost|ERROR', (root/'bpf_driver.log').read_text())
trials={run:[json.loads(x) for x in (root/run/'trials.jsonl').read_text().splitlines()]
        for run in ['full','lock_only','baseline']}
for run in trials:
    status=read(root/run/'status.json');cleanup=read(root/run/'cleanup.json')
    assert status['status']=='completed' and status['all_verified']
    assert cleanup['removed']==512 and cleanup['remaining']==0
lock_rows={}
for run in ['full','lock_only']:
    lock_rows[run]={(r['tag'],r['category']):r for r in
                    (json.loads(x) for x in (root/(run+'_locks.jsonl')).read_text().splitlines()) if 'tag' in r}

def stats(xs):
    xs=sorted(xs)
    return dict(mean=statistics.mean(xs),median=statistics.median(xs),p95=xs[math.ceil(len(xs)*.95)-1],max=max(xs))

def union_ns(es):
    points=[]
    for e in es:points.extend([(e['start'],1),(e['end'],-1)])
    points.sort();active=total=0;last=0
    for t,d in points:
        if active:total+=t-last
        active+=d;last=t
    assert active==0
    return total

per_trial=[]
expected_names={'cuda_gdrcopy_dev_register','cuda_gdrcopy_dev_unregister','rxm_mr_regattr','rxm_mr_close',
                'ibv_reg_dmabuf_mr','ibv_cmd_reg_dmabuf_mr','ibv_cmd_dereg_mr',
                'gdr_pin_buffer','gdr_map','gdr_unmap','gdr_unpin_buffer'}
success_int={'cuda_gdrcopy_dev_register','rxm_mr_regattr','rxm_mr_close','ibv_cmd_reg_dmabuf_mr','ibv_cmd_dereg_mr',
             'gdr_pin_buffer','gdr_map','gdr_unmap','gdr_unpin_buffer'}
for r in trials['full']:
    if r['phase'] not in ['engine','native']:continue
    tag=r['phase_tag'];fs={}
    parent=[]
    for i,f in funcs.items():
        es=events.get((tag,i),[])
        assert len(es)==(512 if f['name'] in expected_names else 0),(tag,f['name'],len(es))
        if not es:continue
        assert all(r['start']<=e['start']<=e['end']<=r['end'] for e in es)
        if f['name'] in success_int:assert all(e['result']==0 for e in es),(tag,f['name'])
        if f['name']=='ibv_reg_dmabuf_mr':assert all(e['result']!=0 for e in es)
        fs[f['name']]=dict(calls=len(es),duration_us=stats([(e['end']-e['start'])/1000 for e in es]),
                          sum_thread_ms=sum(e['end']-e['start'] for e in es)/1e6)
        if f['name'] in ['rxm_mr_regattr','rxm_mr_close']:parent+=es
    locks={}
    for cat in ['scheduler_mutex','cuda_register_spin','cuda_unregister_spin']:
        lr=lock_rows['full'][tag,cat];assert lr['failures']==0
        if cat.startswith('cuda'):assert lr['calls']==512
        locks[cat]=dict(calls=lr['calls'],sum_thread_ms=lr['sum_ns']/1e6,
                        mean_us=lr['sum_ns']/lr['calls']/1000,max_us=lr['max_ns']/1000)
    per_trial.append(dict(tag=tag,trial=r['trial'],phase=r['phase'],workers=int(r['trial'].split('_w')[1]),
                          measured=not r['trial'].startswith('r-1'),wall_ms=r['ms'],functions=fs,locks=locks,
                          parent_register_close_union_ms=union_ns(parent)/1e6))
summary=[]
for w in [16,64]:
    for phase in ['engine','native']:
        ss=[r for r in per_trial if r['measured'] and r['workers']==w and r['phase']==phase]
        assert len(ss)==2
        funcs_mean={f:dict(calls_per_read=512,mean_us=statistics.mean(r['functions'][f]['duration_us']['mean'] for r in ss)) for f in ss[0]['functions']}
        lock_only=[]
        for r in trials['lock_only']:
            if r['trial'].startswith('r-1') or r['phase']!=phase or not r['trial'].endswith('_w'+str(w)):continue
            lock_only.append({cat:lock_rows['lock_only'][r['phase_tag'],cat] for cat in ss[0]['locks']})
        lock_means={cat:dict(mean_calls_per_read=statistics.mean(r[cat]['calls'] for r in lock_only),
                            sum_thread_ms_per_read=statistics.mean(r[cat]['sum_ns']/1e6 for r in lock_only),
                            mean_acquisition_us=sum(r[cat]['sum_ns'] for r in lock_only)/sum(r[cat]['calls'] for r in lock_only)/1000)
                    for cat in ss[0]['locks']}
        timing={run:stats([r['ms'] for r in rs if not r['trial'].startswith('r-1') and r['phase']==phase and r['trial'].endswith('_w'+str(w))]) for run,rs in trials.items()}
        summary.append(dict(workers=w,phase=phase,functions=funcs_mean,locks_from_lock_only=lock_means,
                            wall_ms_controls=timing,parent_register_close_union_ms=statistics.mean(r['parent_register_close_union_ms'] for r in ss)))
(root/'per_trial.json').write_text(json.dumps(per_trial,indent=2)+'\n')
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
validation=dict(passed=True,entry_return_counts_match=True,probe_records=sum(len(v) for v in events.values()),
                all_payload_reads_have_512_successful_NIC_MR_registrations_and_deregistrations=True,
                all_payload_reads_have_512_GDRCopy_pin_map_unmap_unpin=True,all_test_keys_removed=1536,
                notes=['Function duration is entry-to-return elapsed time including waits/preemption.',
                       'Nested function durations must not be summed with their parents.',
                       'Thread-summed lock acquisition time is not critical-path wall time.',
                       'Control runs use identical payload and settings but fresh key namespaces; timing noise remains.',
                       'Lock timers include clock overhead (minimum clock pair measured in lock logs).'])
(root/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print(json.dumps(validation,indent=2))
for r in summary:
    f=r['functions'];l=r['locks_from_lock_only']
    print(r['workers'],r['phase'],'OFI us',round(f['rxm_mr_regattr']['mean_us'],1),round(f['rxm_mr_close']['mean_us'],1),
          'NIC us',round(f['ibv_cmd_reg_dmabuf_mr']['mean_us'],1),round(f['ibv_cmd_dereg_mr']['mean_us'],1),
          'CUDA us',round(f['cuda_gdrcopy_dev_register']['mean_us'],1),round(f['cuda_gdrcopy_dev_unregister']['mean_us'],1),
          'scheduler meanus',round(l['scheduler_mutex']['mean_acquisition_us'],2),
          'controls ms',{k:round(v['median'],1) for k,v in r['wall_ms_controls'].items()})
