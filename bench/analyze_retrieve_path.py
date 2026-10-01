import json
from pathlib import Path
import statistics
import sys


def stats(xs):
    return dict(mean=statistics.mean(xs),median=statistics.median(xs),min=min(xs),max=max(xs))


def main():
    root=Path(sys.argv[1]);plan=json.loads((root/'plan.json').read_text())
    rs=[json.loads(x) for x in (root/'trials.jsonl').read_text().splitlines()]
    events=[json.loads(x) for x in (root/'events.jsonl').read_text().splitlines()]
    samples=[json.loads(x) for x in (root/'gpu_samples.jsonl').read_text().splitlines()]
    status=json.loads((root/'status.json').read_text());cleanup=json.loads((root/'cleanup.json').read_text())
    assert status['status']=='completed' and status['all_verified'] and cleanup['removed']==512 and cleanup['remaining']==0
    for r in rs:
        selected=[e for e in events if e['trial']==r['trial'] and e['phase']==r['phase']]
        r['worker_count']=int(r['trial'].split('_w')[1]);r['repeat']=int(r['trial'].split('_')[0][1:])
        payload=[e for e in selected if e['kind']=='payload']
        if r['phase'] in ('native','engine'):
            assert len(payload)==512 and sum(e['bytes'] for e in payload)==9*2**30
            if r['phase']=='engine':
                assert sum(e['kind']=='metadata' for e in selected)==512
                assert sum(e['kind']=='scatter' for e in selected)==19
        r['stage_sum_ms']={k:sum(e['end']-e['start'] for e in selected if e['kind']==k)/1e6
                           for k in ['payload','metadata','allocate','scatter']}
        r['stage_calls']={k:sum(e['kind']==k for e in selected) for k in r['stage_sum_ms']}
        points=[(r['start'],None,0),(r['end'],None,0)]
        for e in selected:
            assert r['start']<=e['start']<=e['end']<=r['end']
            points.extend([(e['start'],e['kind'],1),(e['end'],e['kind'],-1)])
        points.sort(key=lambda x:x[0]);active={k:0 for k in r['stage_sum_ms']}
        union={k:0 for k in active};exclusive={k:0 for k in ['payload_without_scatter','payload_with_scatter','scatter_without_payload','metadata_without_payload_or_scatter','other']}
        last=r['start']
        for t,kind,delta in points:
            dt=t-last
            for k,v in active.items():
                if v:union[k]+=dt
            if active['payload']:category='payload_with_scatter' if active['scatter'] else 'payload_without_scatter'
            elif active['scatter']:category='scatter_without_payload'
            elif active['metadata']:category='metadata_without_payload_or_scatter'
            else:category='other'
            exclusive[category]+=dt
            if kind:active[kind]+=delta
            last=t
        assert sum(exclusive.values())==r['end']-r['start'] and all(v==0 for v in active.values())
        r['stage_union_ms']={k:v/1e6 for k,v in union.items()}
        r['wall_partition_ms']={k:v/1e6 for k,v in exclusive.items()}
        r['cpu_core_equivalents']=r['cpu_s']/(r['ms']/1000)
        r['hottest_thread_core_equivalent']=max(t['cpu_s'] for t in r['thread_cpu'])/(r['ms']/1000)
        r['rx_GB_s']=r['counter_delta']['port_rcv_data']*4/(r['counter_interval_ms']/1000)/1e9
        r['rx_fraction_of_link_rate']=r['rx_GB_s']/(plan['link_mbps']/8000)
        r['gib_s']=9/(r['ms']/1000) if r['phase'] in ('engine','native') else None
        ss=[s for s in samples if r['start']<=s['t']<=r['end']]
        r['gpu_util_pct']=stats([s['gpu_pct'] for s in ss]) if ss else None
        r['gpu_memory_util_pct']=stats([s['memory_pct'] for s in ss]) if ss else None
    summary=[]
    for w in plan['workers']:
        out=dict(workers=w)
        for phase in ['cold_lookup','warm_lookup','engine','native']:
            ss=[r for r in rs if r['worker_count']==w and r['phase']==phase and r['repeat']>=0]
            assert len(ss)==plan['repeats']
            out[phase]={k:stats([s[k] for s in ss]) for k in ['ms','cpu_core_equivalents','hottest_thread_core_equivalent','rx_GB_s']}
            out[phase]['wall_partition_mean_ms']={k:statistics.mean(s['wall_partition_ms'][k] for s in ss) for k in ss[0]['wall_partition_ms']}
            out[phase]['stage_union_mean_ms']={k:statistics.mean(s['stage_union_ms'][k] for s in ss) for k in ss[0]['stage_union_ms']}
            out[phase]['stage_sum_mean_ms']={k:statistics.mean(s['stage_sum_ms'][k] for s in ss) for k in ss[0]['stage_sum_ms']}
        summary.append(out)
    (root/'analyzed_trials.json').write_text(json.dumps(rs,indent=2)+'\n')
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (root/'validation.json').write_text(json.dumps(dict(passed=True,trials=len(rs),payload_reads=sum(r['phase'] in ('native','engine') for r in rs),all_engine_destinations_verified=True,all_native_final_scratch_slices_verified=True,cleanup_passed=True),indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
