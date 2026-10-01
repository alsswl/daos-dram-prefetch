#!/usr/bin/env python3
"""Replot an existing trace without rerunning or changing the experiment.

100ms bins integrate observed allocator occupancy as a step function. They
are not interpolated GPU-utilization samples. True lookup-entry timestamps
are absent in these traces; never relabel result notification as lookup start.
"""
import argparse
import bisect
import csv
import hashlib
import json
from pathlib import Path

from report_capacity_matrix import save_chart
from report_prefetch_timing import stats
from staging_mixed_pressure import read_events


def read(path):
    return json.loads(path.read_text())


def dump(path,obj):
    path.write_text(json.dumps(obj,indent=2)+'\n')


def bins(events,start,end,width_ns):
    times=[e['time_ns'] for e in events]
    i=bisect.bisect_left(times,start)
    state=events[i-1]['used_bytes'] if i else 0
    output=[]
    for left in range(start,end,width_ns):
        right=min(end,left+width_ns)
        cursor=left
        peak=state
        area=0
        while i<len(events) and events[i]['time_ns']<right:
            t=events[i]['time_ns']
            area+=state*(t-cursor)
            state=events[i]['used_bytes']
            peak=max(peak,state)
            cursor=t
            i+=1
        area+=state*(right-cursor)
        output.append(dict(start_seconds=(left-start)/1e9,duration_seconds=(right-left)/1e9,
            peak_gib=peak/2**30,time_weighted_mean_gib=area/(right-left)/2**30))
    assert abs(sum(r['duration_seconds'] for r in output)-(end-start)/1e9)<1e-6
    return output


def header(height,title,subtitle):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="1250" height="{height}">',
            f'<rect width="1250" height="{height}" fill="white"/>',
            '<g font-family="sans-serif" font-size="14" fill="#202b3a">',
            f'<text x="70" y="30" font-size="21">{title}</text>',
            f'<text x="70" y="57">{subtitle}</text>']


def axes(svg,top,height,duration,title):
    svg.append(f'<text x="70" y="{top-14}" font-size="16">{title}</text>')
    for value in (0,25,50,75,100):
        y=top+height*(1-value/100)
        svg.extend([f'<line x1="75" x2="1200" y1="{y}" y2="{y}" stroke="#dce3ea"/>',
                    f'<text x="22" y="{y+5}">{value}%</text>'])
    for i in range(7):
        svg.append(f'<text x="{75+1125*i/6}" y="{top+height+25}" text-anchor="middle">{duration*i/6:.2f}s</text>')


def plot_whole(folder,all_bins,capacity):
    svg=header(760,'Qwen3-4B / DAOS demand reads / both prefetch OFF',
        '100ms bins: blue = allocator-observed peak; green = time-weighted mean. Staging capacity 8GiB.')
    for idx,(name,rows) in enumerate(all_bins.items()):
        top=105+idx*310
        duration=sum(r['duration_seconds'] for r in rows)
        axes(svg,top,225,duration,name)
        for key,color in [('peak_gib','#0072b2'),('time_weighted_mean_gib','#009e73')]:
            pts=' '.join(f'{75+1125*r["start_seconds"]/duration:.2f},{top+225*(1-r[key]/capacity):.2f}' for r in rows)
            svg.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.2"/>')
    svg+=['<text x="70" y="738">Occupancy, not GPU compute utilization. Replotted from existing allocation/free events; no new run.</text>','</g></svg>']
    save_chart(folder,'staging_100ms',svg)


def plot_zoom(folder,events,phase,rows,capacity):
    start=phase['start_ns']
    end=min(phase['end_ns'],start+3_000_000_000)
    duration=(end-start)/1e9
    visible=[r for r in rows if start<=r['retrieve_start_ns']<end]
    visible.sort(key=lambda r:r['retrieve_start_ns'])
    height=490+len(visible)*29
    svg=header(height,'Warm: first 3 seconds, allocator-event detail',
        'Top: observed staging occupancy. Bottom: lookup-result notification to retrieve start, then retrieve execution.')
    top=100; h=205
    axes(svg,top,h,duration,'Staging occupancy (no 2-second aggregation)')
    before=[e for e in events if e['time_ns']<=start]
    state=before[-1]['used_bytes'] if before else 0
    points=[(75,top+h*(1-state/(capacity*2**30)))]
    for e in events:
        if not start<=e['time_ns']<=end: continue
        x=75+1125*(e['time_ns']-start)/(end-start)
        points.append((x,top+h*(1-state/(capacity*2**30))))
        state=e['used_bytes']
        points.append((x,top+h*(1-state/(capacity*2**30))))
    points.append((1200,top+h*(1-state/(capacity*2**30))))
    pts=' '.join(f'{x:.2f},{y:.2f}' for x,y in points)
    svg.append(f'<polyline points="{pts}" fill="none" stroke="#0072b2" stroke-width="1.5"/>')
    svg.append('<text x="75" y="363">Gray: result notified, waiting for retrieve | Orange: retrieve API executing</text>')
    for i,r in enumerate(visible):
        y=398+i*29
        def x(t):return 75+1125*(max(start,min(end,t))-start)/(end-start)
        a,b,c=x(r['lookup_notify_ns']),x(r['retrieve_start_ns']),x(r['retrieve_return_ns'])
        svg.append(f'<text x="10" y="{y+6}">#{r["index"]}</text>')
        svg.append(f'<line x1="{a}" x2="{b}" y1="{y}" y2="{y}" stroke="#aab3bf" stroke-width="9"/>')
        svg.append(f'<circle cx="{a}" cy="{y}" r="4" fill="#4b5563"/>')
        svg.append(f'<line x1="{b}" x2="{c}" y1="{y}" y2="{y}" stroke="#e69f00" stroke-width="11"/>')
    svg.extend([f'<text x="70" y="{height-40}">Rows are request indices, not concurrent GPU kernel counts. Lookup API entry was not recorded.</text>',
                f'<text x="70" y="{height-16}">Allocator observations may include store buffers and read-promotion lifetimes, not only the labelled request.</text>',
                '</g></svg>'])
    save_chart(folder,'warm_first3s',svg)


def report(case):
    folder=case.parent/'fine_timeline'
    folder.mkdir(exist_ok=False)
    events=read_events(case)
    all_bins={}
    phase_data={}
    for name in ('cold','warm1'):
        phase=read(case/name/'phase.json')
        b=bins(events,phase['start_ns'],phase['end_ns'],100_000_000)
        all_bins[name]=b
        dump(folder/(name+'_100ms.json'),b)
        calls=read(case/name/'replay_calls.json')
        part=[e for e in events if phase['start_ns']<=e['time_ns']<=phase['end_ns']]
        grouped={}
        for e in part:
            if e.get('request_id') is not None and e['event'] in (
                'early_lookup_notify','retrieve_start','retrieve_return','early_daos_demand_start','early_retrieve_decision'):
                group=grouped.setdefault(e['request_id'],{})
                assert e['event'] not in group
                group[e['event']]=e
        rows=[]
        for c in calls:
            matches=[(rid,g) for rid,g in grouped.items() if rid==c['server_request_id'] or rid.startswith(c['server_request_id']+'-')]
            assert len(matches)==1
            rid,g=matches[0]
            if 'retrieve_start' not in g:
                assert c['cached_tokens']==0
                continue
            def ms(a,b):return (g[b]['monotonic_ns']-g[a]['monotonic_ns'])/1e6
            row=dict(index=c['index'],request_id=rid,
                lookup_notify_ns=g['early_lookup_notify']['time_ns'],
                retrieve_start_ns=g['retrieve_start']['time_ns'],
                retrieve_return_ns=g['retrieve_return']['time_ns'],
                notify_to_retrieve_ms=ms('early_lookup_notify','retrieve_start'),
                retrieve_api_ms=ms('retrieve_start','retrieve_return'),
                daos_demand_resolve_ms=ms('early_daos_demand_start','early_retrieve_decision'))
            assert row['notify_to_retrieve_ms']>=0
            rows.append(row)
        dump(folder/(name+'_request_timing.json'),rows)
        if rows:
            with (folder/(name+'_request_timing.csv')).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
                writer.writeheader();writer.writerows(rows)
        phase_data[name]=dict(total_requests=len(calls),measured_retrieves=len(rows),
            no_retrieve_requests=len(calls)-len(rows),
            notify_to_retrieve_ms=stats(r['notify_to_retrieve_ms'] for r in rows),
            retrieve_api_ms=stats(r['retrieve_api_ms'] for r in rows),
            daos_demand_resolve_ms=stats(r['daos_demand_resolve_ms'] for r in rows),
            time_weighted_mean_staging_gib=sum(r['duration_seconds']*r['time_weighted_mean_gib'] for r in b)/sum(r['duration_seconds'] for r in b))
        if name=='warm1':
            plot_zoom(folder,events,phase,rows,8)
    plot_whole(folder,all_bins,8)
    dump(folder/'summary.json',dict(phases=phase_data,lookup_api_entry_recorded=False,
        notes=['Notify is emitted immediately before sending metadata result, not lookup API start or scheduler receive.',
               'All latency differences use same-process monotonic timestamps.',
               'No cache hits and no retrieve calls in cold; no artificial zero gap is assigned.',
               '100ms occupancy means integrate the last observed allocator state, unlike old sample-count means.',
               'Only already-recorded allocation/free/occupancy observations are used; no new sampling or rerun.'],
        trace_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in case.glob('trace.*.jsonl')}))
    print(json.dumps(phase_data,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',type=Path,required=True)
    report(parser.parse_args().case.resolve())
