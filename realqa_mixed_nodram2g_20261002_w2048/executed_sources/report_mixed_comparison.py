"""Compare completed mixed-workload trials and render per-purpose occupancy."""
import argparse
import json
from pathlib import Path
import numpy as np
from report_capacity_matrix import save_chart
from report_prefetch_timing import stats

def read(p):return json.loads(p.read_text())
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')

def report(folder):
    comparison_plan=read(folder/'comparison_plan.json')
    configs=comparison_plan['cases'];results=[];series=[];cpu_sizes=[]
    input_hashes=[];schedules=[];all_calls=[]
    for config in configs:
        root=Path(config['root']);plan=read(root/'plan.json');case=root/plan['cases'][0]['name']
        cpu_sizes.append(plan['cpu_gib'])
        assert read(root/'status.json')['status']=='completed'
        summary=read(root/'summary.json');calls=read(case/'calls.json');window=read(case/'workload.json')
        all_calls.append(calls)
        input_hashes.append(plan['input_sha256']);schedules.append(read(root/'mixed_schedule.json'))
        assert len(calls)==480 and all(c['status']=='success' for c in calls)
        expected={(r['session_index'],r['turn']):r for lane in schedules[-1]['lanes'] for r in lane}
        assert all(c['mixed_schedule']==expected[c['session_index'],c['turn']] for c in calls)
        start=window['start_ns'];end=window['end_ns'];width=100_000_000
        n=(end-start+width-1)//width;peak=np.zeros((n,3));area=np.zeros((n,3));used=np.zeros(3)
        last=start;overlap_ns=0
        def advance(t):
            nonlocal last,overlap_ns
            t=min(t,end)
            while last<t:
                i=(last-start)//width;stop=min(t,start+(i+1)*width)
                area[i]+=used*(stop-last);peak[i]=np.maximum(peak[i],used)
                if used[0]>0 and used[1]>0:overlap_ns+=stop-last
                last=stop
        for path in case.glob('trace.*.jsonl'):
            for line in path.open():
                e=json.loads(line);t=e['time_ns'];values=[e['store_used_bytes'],e['retrieve_used_bytes'],e['used_bytes']]
                if t<start:used=np.array(values,dtype=float);continue
                advance(t)
                if t>end:break
                used=np.array(values,dtype=float);i=min(n-1,(t-start)//width);peak[i]=np.maximum(peak[i],used)
        advance(end)
        dur=(end-start)/1e9;peak/=2**30
        durations=np.minimum(width,end-(start+np.arange(n)*width));mean=area/durations[:,None]/2**30
        bins=[dict(seconds=i/10,store_peak_gib=float(peak[i,0]),retrieve_peak_gib=float(peak[i,1]),
                   total_peak_gib=float(peak[i,2]),store_mean_gib=float(mean[i,0]),
                   retrieve_mean_gib=float(mean[i,1])) for i in range(n)]
        dump(root/'mixed_occupancy_100ms.json',bins)
        description=config.get('description', 'one shared 2GiB arena' if not plan.get('split_store_gib')
                               else 'store0.5GiB + retrieve1.5GiB')
        series.append((config['label'],description,dur,peak))
        selected={'all':calls,'new_document':[c for c in calls if c['turn']==0],
                  'followup':[c for c in calls if c['turn']>0],
                  'middle_lane_positions_3_to_50':[c for c in calls if 3<=c['mixed_schedule']['position']<51]}
        timings={k:stats([c['ttft_http_ms'] for c in rows]) for k,rows in selected.items()}
        quarters=[]
        for q in range(4):
            rows=[c for c in calls if q*dur/4 <= (c['http_start_ns']-start)/1e9 < (q+1)*dur/4]
            quarters.append(dict(quarter=q+1,requests=len(rows),new_documents=sum(c['turn']==0 for c in rows),
                                 followups=sum(c['turn']>0 for c in rows)))
        result=dict(condition=config['label'],root=str(root),requests=480,duration_s=dur,ttft_http_ms=timings,
            completion_tokens=sum(c['completion_tokens'] for c in calls),
            input_tokens=sum(c['prompt_tokens'] for c in calls),
            cached_token_pct=summary['by_turn'][0]['cached_token_pct'],
            store_peak_gib=float(peak[:,0].max()),retrieve_peak_gib=float(peak[:,1].max()),
            total_peak_gib=float(peak[:,2].max()),simultaneous_occupancy_s=overlap_ns/1e9,
            simultaneous_occupancy_pct=100*overlap_ns/(end-start),request_mix_by_time_quarter=quarters,
            capacity_validation=read(case/'capacity_policy_check.json'),
            store_validation=read(case/'store_window_policy_check.json'),final_mirror=summary['final']['dram_mirror'])
        if plan.get('dram_disabled'):
            result['no_dram_validation']=read(case/'no_dram_check.json')
        results.append(result)
    assert input_hashes[0]==input_hashes[1] and schedules[0]==schedules[1]
    original={(c['session_index'],c['turn']):c for c in all_calls[0]}
    prompt_matches=sum(c['messages_sha256']==original[c['session_index'],c['turn']]['messages_sha256']
                       for c in all_calls[1])
    delta={k:100*(results[1]['ttft_http_ms'][k]['mean']/results[0]['ttft_http_ms'][k]['mean']-1)
           for k in ('all','new_document','followup','middle_lane_positions_3_to_50')}
    delta['duration']=100*(results[1]['duration_s']/results[0]['duration_s']-1)
    delta_key=comparison_plan.get('delta_key','split_vs_shared_change_pct')
    dump(folder/'comparison.json',dict(results=results,**{delta_key:delta},
        identical_input_and_schedule=True,
        actual_prompt_hash_matches=prompt_matches,actual_prompt_hash_compared=480,
        limitations=['One run per condition; no confidence interval.',
                     'Same planned per-lane sequence, but real generated answers and wall-clock timing vary.',
                     'Concurrent load/copy spans include metadata and worker queueing; not a hardware RDMA timeline.',
                     ('DRAM KV cache and mirror disabled in both conditions; all retrieved chunks validated as DAOS reads.'
                      if cpu_sizes==[0,0] else
                      'DRAM cache and bounded asynchronous mirror enabled; resulting DRAM hit patterns can differ.')]))
    from html import escape
    title=escape(comparison_plan.get('chart_title',
        'Mixed 80-session QA | 2GiB staging | 486MiB windows | capacity-based parallel admission'))
    cpu_label='DRAM KV cache OFF' if cpu_sizes==[0,0] else 'DRAM256GiB'
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="690">',
         '<rect width="1280" height="690" fill="white"/><g font-family="sans-serif" font-size="14" fill="#243246">',
         f'<text x="75" y="30" font-size="20">{title}</text>',
         f'<text x="75" y="55">Orange: store peak / Blue: retrieve peak, 100ms bins. Lookup prefetch OFF; {cpu_label}.</text>']
    for i,(label,description,dur,peak) in enumerate(series):
        top=110+i*270;h=190
        svg.append(f'<text x="75" y="{top-18}">{escape(label)}: {escape(description)}</text>')
        for value in (0,.5,1,1.5,2):
            y=top+h*(1-value/2)
            svg.append(f'<line x1="75" x2="1220" y1="{y}" y2="{y}" stroke="#ddd"/><text x="10" y="{y+5}">{value:.1f} GiB</text>')
        for col,color in ((1,'#0072b2'),(0,'#d55e00')):
            points=' '.join(f'{75+1145*(j/10)/dur:.2f},{top+h*(1-v/2):.2f}' for j,v in enumerate(peak[:,col]))
            svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width=".7"/>')
        for fraction in (0,.25,.5,.75,1):
            svg.append(f'<text x="{75+1145*fraction-20}" y="{top+h+24}">{dur*fraction:.1f}s</text>')
    svg.append('<text x="75" y="660">Per-purpose peaks can occur at different instants; the peak curves must not be added.</text></g></svg>')
    save_chart(folder,'mixed_staging_comparison',svg)
    print(json.dumps(dict(results=[{k:r[k] for k in ('condition','duration_s','cached_token_pct','total_peak_gib')} for r in results],changes=delta),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    report(p.parse_args().output.resolve())
