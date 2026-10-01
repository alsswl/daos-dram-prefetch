#!/usr/bin/env python3
"""Explain existing 1ms CUDA diagnostic windows, without executing a workload."""
from bisect import bisect_left, bisect_right
from collections import Counter
import gzip
import json
from pathlib import Path
import re

from analyze_gpu_overlap import Intersections, merge
from analyze_prefetch_remaining_latency import api_id
from report_prefetch_timing import stats

ROOT=Path(__file__).resolve().parent
CASE=ROOT/'sharegpt_backoff_1ms_d256_s8_20260930_profile/c16_profile'
OUTPUT=ROOT/'prefetch_remaining_latency_20260930'


def run():
    path=next((CASE/'gpu_trace').glob('*.pt.trace.json.gz'))
    with gzip.open(path,'rt') as f:
        trace=json.load(f)
    events=trace['traceEvents']
    epoch=trace['baseTimeNanoseconds']
    window=next(e for e in events if e.get('cat')=='Trace' and e.get('ph')=='X')
    start,end=window['ts'],window['ts']+window['dur']
    kernels=[e for e in events if e.get('cat')=='kernel']
    compute=[e for e in kernels if re.search(r'gemm|nvjet|cutlass|flash|fmha|attention',e['name'],re.I)
             and not re.search(r'cache|transfer|gather|reshape',e['name'],re.I)]
    steps=sorted([e for e in events if e.get('cat')=='user_annotation' and e['name'].startswith('execute_context_')],key=lambda e:e['ts'])
    step_starts=[e['ts'] for e in steps]
    def spans(rows):
        return Intersections((e['ts'],e['ts']+e['dur']) for e in rows)
    busy,any_kernel,cpu_steps=spans(compute),spans(kernels),spans(steps)
    raw_compute=sorted(compute,key=lambda e:e['ts'])
    compute_starts=[e['ts'] for e in raw_compute]
    cpu_tids={e['tid'] for e in steps}
    syncs=[e for e in events if e.get('cat') in ('cuda_runtime','cuda_driver') and 'Synchronize' in e['name'] and e['tid'] in cpu_tids]
    sync_spans=spans(syncs)
    hosts=[]
    for path in CASE.glob('trace.*.jsonl'):
        for line in path.open():
            e=json.loads(line)
            ts=(e['time_ns']-epoch)/1000
            if start<=ts<=end:
                e['ts']=ts
                hosts.append(e)
    retrieve_start={e['request_id']:e['ts'] for e in hosts if e['event']=='retrieve_start'}
    retrieve_end={e['request_id']:e['ts'] for e in hosts if e['event']=='retrieve_return'}
    retrieve_spans=Intersections((a,retrieve_end[rid]) for rid,a in retrieve_start.items() if rid in retrieve_end)
    reads=json.loads((ROOT/'sharegpt_backoff_1ms_d256_s8_20260930/backoff_profile_analysis/daos_host_spans.json').read_text())
    calls=json.loads((CASE/'warm_profile/replay_calls.json').read_text())
    callmap={c['server_request_id']:c for c in calls}
    rows=[]
    gap_examples=[]
    for r in reads:
        a,b=r['start'],r['end']
        c=callmap[api_id(r['rid'])]
        row=dict(index=c['index'],session=c['session'],turn=c['turn'],chunks=r['chunks'],
            start_ms=(a-start)/1000,read_ms=(b-a)/1000,
            compute_overlap_ms=busy.overlap(a,b)/1000,
            any_kernel_overlap_ms=any_kernel.overlap(a,b)/1000,
            cpu_execute_overlap_ms=cpu_steps.overlap(a,b)/1000,
            any_retrieve_overlap_ms=retrieve_spans.overlap(a,b)/1000,
            mainthread_cuda_sync_overlap_ms=sync_spans.overlap(a,b)/1000,
            ready_to_retrieve_ms=(retrieve_start[r['rid']]-b)/1000 if r['rid'] in retrieve_start else None)
        rows.append(row)
        if row['any_kernel_overlap_ms']==0:
            j=bisect_right([x[1] for x in any_kernel.spans],a)-1
            prev_end=any_kernel.spans[j][1] if j>=0 else None
            nxt=next((x[0] for x in any_kernel.spans if x[0]>=b),None)
            s=bisect_right(step_starts,a)-1
            gap_examples.append(dict(row,prior_step=steps[s]['name'] if s>=0 else None,
                next_step=steps[s+1]['name'] if s+1<len(steps) else None,
                full_kernel_gap_ms=(nxt-prev_end)/1000 if nxt is not None and prev_end is not None else None))
    step_rows=[]
    for s in steps:
        a,b=s['ts'],s['ts']+s['dur']
        i=bisect_left(compute_starts,a)
        first=raw_compute[i]['ts'] if i<len(raw_compute) and raw_compute[i]['ts']<b else None
        step_rows.append(dict(name=s['name'],start_ms=(a-start)/1000,duration_ms=s['dur']/1000,
            recognized_compute_ms=busy.overlap(a,b)/1000,
            cpu_start_to_first_compute_ms=(first-a)/1000 if first else None,
            retrieve_inside_ms=retrieve_spans.overlap(a,b)/1000))
    def summary(rows):
        total=sum(r['read_ms'] for r in rows)
        return dict(requests=len(rows),host_read_ms=stats(r['read_ms'] for r in rows),
            compute_overlap_pct=100*sum(r['compute_overlap_ms'] for r in rows)/total if total else None,
            cpu_execute_overlap_pct=100*sum(r['cpu_execute_overlap_ms'] for r in rows)/total if total else None,
            any_retrieve_overlap_pct=100*sum(r['any_retrieve_overlap_ms'] for r in rows)/total if total else None,
            mainthread_cuda_sync_overlap_pct=100*sum(r['mainthread_cuda_sync_overlap_ms'] for r in rows)/total if total else None,
            ready_to_retrieve_ms=stats(r['ready_to_retrieve_ms'] for r in rows))
    result=dict(all_reads=summary(rows),after_first16=summary([r for r in rows if r['index']>=16]),
        kernel_free_read_examples=gap_examples,
        decode_only_steps=dict(count=sum(s['name'].startswith('execute_context_0(0)') for s in steps),total=len(steps)),
        step_duration_ms=stats(r['duration_ms'] for r in step_rows),
        cpu_start_to_first_compute_ms=stats(r['cpu_start_to_first_compute_ms'] for r in step_rows),
        step_retrieve_inside_ms=stats(r['retrieve_inside_ms'] for r in step_rows),
        limitations=['Profiled first512 steps, host DAOS reads include queues/metadata/completion.',
                     'CPU execution annotations include setup/retrieve, not continuous GPU model compute.',
                     'Same-thread CUDA sync overlap does not identify which stream or source operation without stacks.',
                     'Temporal overlap and cumulative request-ms are not an end-to-end critical path attribution.'])
    (OUTPUT/'compute_windows.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUTPUT/'profile_daos_rows.json').write_text(json.dumps(rows)+'\n')
    (OUTPUT/'profile_step_rows.json').write_text(json.dumps(step_rows)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
