"""Postprocess immutable run logs; per-pool occupancy and baseline comparison."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from report_capacity_matrix import save_chart

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

def report(root):
    plan=read(root/'plan.json');case=root/plan['cases'][0]['name']
    assert read(root/'status.json')['status']=='completed'
    window=read(case/'workload.json');start=window['start_ns'];end=window['end_ns']
    width=100_000_000;n=math.ceil((end-start)/width)
    peaks=np.zeros((n,2));areas=np.zeros((n,2));last=start;used=np.zeros(2)
    def add(t):
        nonlocal last
        t=min(t,end)
        while last<t:
            i=(last-start)//width;stop=min(t,start+(i+1)*width)
            areas[i]+=used*(stop-last);peaks[i]=np.maximum(peaks[i],used)
            last=stop
    for path in sorted(case.glob('trace.*.jsonl')):
        for line in path.open():
            e=json.loads(line);t=e['time_ns']
            if t<start:
                used=np.array([e['store_used_bytes'],e['retrieve_used_bytes']],dtype=float)
                continue
            add(t)
            if t>end:break
            used=np.array([e['store_used_bytes'],e['retrieve_used_bytes']],dtype=float)
            i=min(n-1,(t-start)//width);peaks[i]=np.maximum(peaks[i],used)
    add(end)
    durations=np.minimum(width,end-(start+np.arange(n)*width))
    means=areas/durations[:,None]/2**30;peaks/=2**30
    ts=np.arange(n)*.1
    rows=[dict(start_seconds=float(ts[i]),duration_seconds=float(durations[i]/1e9),
               store_peak_gib=float(peaks[i,0]),retrieve_peak_gib=float(peaks[i,1]),
               store_mean_gib=float(means[i,0]),retrieve_mean_gib=float(means[i,1])) for i in range(n)]
    dump(root/'split_occupancy_100ms.json',rows)
    duration=(end-start)/1e9
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1250" height="660">',
         '<rect width="1250" height="660" fill="white"/><g font-family="sans-serif" font-size="14" fill="#243246">',
         '<text x="75" y="28" font-size="20">C8 x S10 x 6 turns | 2GiB split staging | 486MiB windows</text>',
         '<text x="75" y="52">Store 0.5GiB / Retrieve 1.5GiB; up to 3 retrieve windows. Peak and mean in 100ms bins.</text>']
    for i,(name,cap,color) in enumerate(zip(['Store','Retrieve'],[.5,1.5],['#d55e00','#0072b2'])):
        top=100+i*260;h=190
        svg.append(f'<text x="75" y="{top-15}">{name}: reserved {cap} GiB | colored=peak, green=time-weighted mean</text>')
        for fraction in (0,.25,.5,.75,1):
            y=top+h*(1-fraction)
            svg.append(f'<line x1="75" x2="1200" y1="{y}" y2="{y}" stroke="#ddd"/>')
            svg.append(f'<text x="10" y="{y+5}">{cap*fraction:.3f} GiB</text>')
        for values,stroke in ((peaks[:,i],color),(means[:,i],'#009e73')):
            points=' '.join(f'{75+1125*t/duration:.2f},{top+h*(1-v/cap):.2f}' for t,v in zip(ts,values))
            svg.append(f'<polyline points="{points}" fill="none" stroke="{stroke}" stroke-width="0.7"/>')
        for fraction in (0,.25,.5,.75,1):
            svg.append(f'<text x="{75+1125*fraction-20}" y="{top+h+22}">{duration*fraction:.1f}s</text>')
    svg.append('<text x="75" y="640">Occupancy includes DAOS operations and retained DRAM-mirror references; this is not GPU utilization.</text></g></svg>')
    save_chart(root,'split_occupancy',svg)
    names=['realqa_q4_64k_window1024_d256_s2_u0835_resume_fix_20261001',
           'realqa_q4_64k_read500_store500_d256_s1_20261001',root.name]
    comparisons=[]
    for name in names:
        p=ROOT/name;s=read(p/'summary.json');pl=read(p/'plan.json');allturn=s['by_turn'][0]
        comparisons.append(dict(experiment=name,staging_gib=pl['staging_gib'],
            mean_ttft_ms=allturn['ttft_http_ms']['mean'],duration_s=s['duration_s'],
            cached_token_pct=allturn['cached_token_pct'],staging_peak_gib=s['staging_peak_gib'],
            same_input_files=pl['input_sha256']==plan['input_sha256']))
    stats=dict(store_peak_gib=float(peaks[:,0].max()),retrieve_peak_gib=float(peaks[:,1].max()),
               store_time_weighted_mean_gib=float(areas[:,0].sum()/(end-start)/2**30),
               retrieve_time_weighted_mean_gib=float(areas[:,1].sum()/(end-start)/2**30))
    dump(root/'split_comparison.json',dict(results=comparisons,occupancy=stats,
        validation=read(case/'split_policy_check.json'),
        limitations=['One run per configuration; actual generated histories and arrival timing vary.',
                     'Against 1GiB baseline, staging capacity and vLLM KV budget also change.',
                     'Against original 2GiB baseline, both window size and store admission also change.',
                     'Concurrent load intervals include worker queueing and metadata; not an RDMA hardware timeline.']))
    print(json.dumps(dict(results=comparisons,occupancy=stats),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    report(p.parse_args().output.resolve())
