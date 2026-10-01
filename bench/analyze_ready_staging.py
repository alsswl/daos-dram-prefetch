"""Attribute completed-but-held staging without assuming it is recoverable latency."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]


def stats(xs):
    xs=sorted(xs)
    return dict(mean=statistics.mean(xs),median=statistics.median(xs),
                p95=xs[math.ceil(.95*len(xs))-1],max=max(xs))


def analyze(folder):
    rows=sorted((json.loads(x) for x in (folder/'chunk_lifecycle.jsonl').read_text().splitlines()),key=lambda x:x['t'])
    result=json.loads((folder/'result.json').read_text())
    manifest=json.loads((folder/'manifest.json').read_text())
    plan=json.loads((folder/'measurement_plan.json').read_text())
    cleanup=json.loads((folder/'cleanup.json').read_text())
    assert result['status']=='passed' and result['returned_chunks']==512 and result['capacity_failed_chunks']==0
    assert result['all_returned_gpu_values_match'] and cleanup['removed']==512 and cleanup['remaining']==0
    assert all(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==h for f,h in plan['source_sha256'].items())
    begin,end=rows[0],rows[-1]
    assert begin['event']=='retrieve_start' and end['event']=='retrieve_end'
    start,stop=begin['t'],end['t'];duration=stop-start
    chunk_bytes,limit,depth=begin['chunk_bytes'],begin['window_chunks'],begin['depth']
    chunks={key:dict(key=key,index=i,window=i//limit) for i,key in enumerate(manifest['keys'])}
    nwindows=math.ceil(len(chunks)/limit)
    windows={i:dict(window=i,chunks=min(limit,len(chunks)-i*limit)) for i in range(nwindows)}
    for r in rows:
        event=r['event']
        if 'key' in r:
            c=chunks[r['key']]
            assert event not in c,(event,c['index'])
            c[event]=r['t']
        elif event in ('scatter_start','scatter_done'):
            for key in r['keys']:
                assert event not in chunks[key]
                chunks[key][event]=r['t']
        if 'window' in r:
            w=windows[r['window']]
            assert event not in w
            w[event]=r['t']
    for i,w in windows.items():
        cs=[c for c in chunks.values() if c['window']==i]
        w['last_fetch_done']=max(c['fetch_done'] for c in cs)
        w['last_worker_start']=max(c['worker_start'] for c in cs)
        w['last_fetch_start']=max(c['fetch_start'] for c in cs)
        w['scatter_start']=cs[0]['scatter_start']
        w['scatter_done']=cs[0]['scatter_done']
        assert w['last_fetch_done']<=w['pipeline_load_done']<=w['scatter_start']
        w['ready_to_scatter_ms']=(w['scatter_start']-w['pipeline_load_done'])/1e6
        # Time while this window is complete and an earlier window is not yet
        # released. This is evidence of ordered-window waiting, not guaranteed
        # speedup from an alternative schedule.
        prior_release=max((windows[j]['window_released'] for j in range(i)),default=start)
        w['earlier_window_wait_ms']=max(0,min(w['scatter_start'],prior_release)-w['pipeline_load_done'])/1e6
    for c in chunks.values():
        w=windows[c['window']]
        assert c['allocated']<=c['fetch_start']<=c['fetch_done']<=c['scatter_start']<=c['scatter_done']<=c['freed']
        c.update(fetch_ms=(c['fetch_done']-c['fetch_start'])/1e6,
                 ready_wait_ms=(c['scatter_start']-c['fetch_done'])/1e6,
                 waiting_other_chunks_ms=(w['last_fetch_done']-c['fetch_done'])/1e6,
                 window_finalize_ms=(w['pipeline_load_done']-w['last_fetch_done'])/1e6,
                 window_schedule_wait_ms=w['ready_to_scatter_ms'],
                 ready_to_free_ms=(c['freed']-c['fetch_done'])/1e6)
        c.update(wait_for_last_worker_start_ms=max(0,w['last_worker_start']-c['fetch_done'])/1e6,
                 wait_from_last_worker_to_last_fetch_start_ms=max(0,w['last_fetch_start']-max(c['fetch_done'],w['last_worker_start']))/1e6,
                 wait_after_last_fetch_start_ms=max(0,w['last_fetch_done']-max(c['fetch_done'],w['last_fetch_start']))/1e6)
        assert abs(c['waiting_other_chunks_ms']-sum(c[k] for k in
            ('wait_for_last_worker_start_ms','wait_from_last_worker_to_last_fetch_start_ms','wait_after_last_fetch_start_ms')))<1e-7
    used=ready=copying=post_copy=workers=fetches=0
    pending=set();submitted=0;freed_per_window={i:0 for i in windows}
    areas={k:0 for k in ('used','ready','copying','post_copy','workers','fetches')}
    peaks={k:0 for k in ('used','ready','copying','post_copy')}
    times={k:0 for k in ('depth_full_with_more_windows','next_window_exceeds_budget',
                         'depth_full_ready_present','depth_full_ready_and_no_payload_fetch',
                         'depth_full_ready_and_no_worker','depth_full_ready_and_workers_below_16')}
    timeline=[];last=start
    for r in rows:
        t=r['t'];dt=t-last
        counts=dict(used=used,ready=ready,copying=copying,post_copy=post_copy,workers=workers,fetches=fetches)
        for k,v in counts.items():areas[k]+=v*dt
        more=submitted<nwindows
        full=more and len(pending)>=depth
        reserved=sum(windows[i]['chunks']*chunk_bytes-freed_per_window[i] for i in pending)
        if full:times['depth_full_with_more_windows']+=dt
        if more and begin['capacity']-reserved<windows[submitted]['chunks']*chunk_bytes:
            times['next_window_exceeds_budget']+=dt
        if full and ready:
            times['depth_full_ready_present']+=dt
            if fetches==0:times['depth_full_ready_and_no_payload_fetch']+=dt
            if workers==0:times['depth_full_ready_and_no_worker']+=dt
            if workers<16:times['depth_full_ready_and_workers_below_16']+=dt
        event=r['event']
        if event=='allocated':used+=chunk_bytes
        elif event=='fetch_start':fetches+=1
        elif event=='fetch_done':fetches-=1;ready+=chunk_bytes
        elif event=='scatter_start':
            amount=len(r['keys'])*chunk_bytes;ready-=amount;copying+=amount
        elif event=='scatter_done':
            amount=len(r['keys'])*chunk_bytes;copying-=amount;post_copy+=amount
        elif event=='freed':
            used-=chunk_bytes;post_copy-=chunk_bytes
            freed_per_window[chunks[r['key']]['window']]+=chunk_bytes
        elif event=='worker_start':workers+=1
        elif event=='worker_done':workers-=1
        elif event=='pipeline_submitted':
            assert r['window']==submitted
            pending.add(r['window']);submitted+=1
        elif event=='window_released':pending.remove(r['window'])
        assert min(used,ready,copying,post_copy,workers,fetches)>=0
        for k,v in dict(used=used,ready=ready,copying=copying,post_copy=post_copy).items():peaks[k]=max(peaks[k],v)
        timeline.append(dict(ms=(t-start)/1e6,used_mib=used/2**20,ready_mib=ready/2**20,
                             copying_mib=copying/2**20,post_copy_mib=post_copy/2**20,
                             workers=workers,payload_fetches=fetches,pending_windows=len(pending)))
        last=t
    assert used==ready==copying==post_copy==workers==fetches==0 and not pending
    summed_ready_area=sum((c['scatter_start']-c['fetch_done'])*chunk_bytes for c in chunks.values())
    assert summed_ready_area==areas['ready']
    assert all(abs(c['ready_wait_ms']-sum(c[k] for k in ('waiting_other_chunks_ms','window_finalize_ms','window_schedule_wait_ms')))<1e-7 for c in chunks.values())
    summary=dict(run=folder.name,chunks=len(chunks),windows=nwindows,retrieve_ms=result['retrieve_ms'],
                 pipeline_ms=duration/1e6,fetch_ms=stats([c['fetch_ms'] for c in chunks.values()]),
                 ready_wait_ms=stats([c['ready_wait_ms'] for c in chunks.values()]),
                 ready_to_free_ms=stats([c['ready_to_free_ms'] for c in chunks.values()]),
                 mean_ready_wait_components_ms={k:statistics.mean(c[k] for c in chunks.values()) for k in
                     ('waiting_other_chunks_ms','window_finalize_ms','window_schedule_wait_ms')},
                 mean_within_window_wait_components_ms={k:statistics.mean(c[k] for c in chunks.values()) for k in
                     ('wait_for_last_worker_start_ms','wait_from_last_worker_to_last_fetch_start_ms','wait_after_last_fetch_start_ms')},
                 time_average_mib={k:areas[k]/duration/2**20 for k in peaks},
                 peak_mib={k:v/2**20 for k,v in peaks.items()},
                 ready_fraction_of_allocated_byte_time=areas['ready']/areas['used'],
                 mean_workers=areas['workers']/duration,mean_payload_fetches=areas['fetches']/duration,
                 durations_ms={k:v/1e6 for k,v in times.items()},
                 windows_completed_before_earlier_release=sum(w['earlier_window_wait_ms']>0 for w in windows.values()),
                 max_earlier_window_wait_ms=max(w['earlier_window_wait_ms'] for w in windows.values()),
                 all_bytes_correct=True,cleanup_passed=True,accounting_validated=True)
    (folder/'ready_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (folder/'windows.json').write_text(json.dumps(list(windows.values()),indent=2)+'\n')
    for name,records in [('chunks',list(chunks.values())),('timeline',timeline)]:
        with (folder/(name+'.csv')).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    return summary


if __name__=='__main__':
    root=Path(sys.argv[1]).resolve()
    summaries=[analyze(p) for p in sorted(root.glob('run[0-9]*')) if p.is_dir()]
    assert summaries
    (root/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))
