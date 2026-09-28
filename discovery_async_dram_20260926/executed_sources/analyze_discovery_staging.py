#!/usr/bin/env python3
"""Read recorded DiscoveryBench diagnostics; no model or storage mutations."""
import argparse
from collections import Counter, defaultdict
import csv
import html
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess

from staging_mixed_pressure import read_events


def summarize_case(case, cpu_gib, bin_seconds=5):
    phases = json.loads((case/'phases.json').read_text())
    start = phases[0]['start_ns']
    events = read_events(case)
    end = phases[-1].get('end_ns', events[-1]['time_ns'])
    duration = (end-start)/1e9
    n = max(1, math.ceil(duration/bin_seconds))
    bins = [dict(t_seconds=i*bin_seconds, duration_seconds=min(bin_seconds, duration-i*bin_seconds),
        peak_staging_gib=0., mean_staging_gib=0., peak_ready_gib=0.,
        mean_cpu_gib=0., lookup_chunks=0, dram_chunks=0, daos_chunks=0, miss_chunks=0,
        cpu_staged_requests=0, cpu_unstaged_requests=0, peak_llm_calls=0, mean_llm_calls=0.)
        for i in range(n)]
    def index(ns):
        return min(n-1, max(0, int((ns-start)/1e9/bin_seconds)))
    def integrate(t0, t1, value, field):
        t0, t1 = max(t0,start), min(t1,end)
        while t0 < t1:
            i = index(t0)
            boundary = min(t1, start+int((i+1)*bin_seconds*1e9))
            if boundary <= t0:
                break
            bins[i][field] += value*(boundary-t0)/1e9
            if field == 'mean_staging_gib':
                bins[i]['peak_staging_gib'] = max(bins[i]['peak_staging_gib'],value)
            t0 = boundary
    lookups = {}
    occupancy_seconds = Counter()
    last_sample = None
    last_event = None
    relevant = []
    for e in events:
        if e['time_ns'] > end:
            break
        if last_event:
            integrate(last_event['time_ns'], e['time_ns'], last_event['used_bytes']/2**30, 'mean_staging_gib')
            dt = max(0, min(e['time_ns'], end)-max(last_event['time_ns'], start))/1e9
            used = last_event['used_bytes']/2**30
            occupancy_seconds['empty' if used == 0 else 'nonempty'] += dt
            if used >= 5:
                occupancy_seconds['at_least_5gib'] += dt
            if used >= 9:
                occupancy_seconds['at_least_9gib'] += dt
        last_event = e
        if e['event'] == 'occupancy_sample':
            if last_sample:
                integrate(last_sample['time_ns'], e['time_ns'], last_sample['cpu_hot_bytes']/2**30, 'mean_cpu_gib')
            last_sample = e
        if e['time_ns'] < start:
            continue
        relevant.append(e)
        b = bins[index(e['time_ns'])]
        b['peak_staging_gib'] = max(b['peak_staging_gib'], e['used_bytes']/2**30)
        b['peak_ready_gib'] = max(b['peak_ready_gib'], e['ready_bytes']/2**30)
        if e['event'] == 'tier_lookup':
            row = lookups.setdefault(e['request_id'], dict(time_ns=e['time_ns'], dram=0, daos=0, queried=0))
            row[e['tier']] += e['hit_chunks']
            if e['tier'] == 'dram':
                row['queried'] += e['queried_chunks']
        if e['event'] == 'cpu_get_ready':
            b['cpu_staged_requests' if e['staged_chunks'] else 'cpu_unstaged_requests'] += 1
    if last_event:
        integrate(last_event['time_ns'], end, last_event['used_bytes']/2**30, 'mean_staging_gib')
        dt = max(0, end-max(last_event['time_ns'], start))/1e9
        used = last_event['used_bytes']/2**30
        occupancy_seconds['empty' if used == 0 else 'nonempty'] += dt
        if used >= 5:
            occupancy_seconds['at_least_5gib'] += dt
        if used >= 9:
            occupancy_seconds['at_least_9gib'] += dt
    if last_sample:
        integrate(last_sample['time_ns'], end, last_sample['cpu_hot_bytes']/2**30, 'mean_cpu_gib')
    for row in lookups.values():
        b = bins[index(row['time_ns'])]
        b['lookup_chunks'] += row['queried']
        b['dram_chunks'] += row['dram']
        b['daos_chunks'] += row['daos']
        b['miss_chunks'] += max(0, row['queried']-row['dram']-row['daos'])
    jobs = json.loads((case/'jobs.json').read_text()) if (case/'jobs.json').exists() else []
    calls = []
    first_prompts = []
    agent_issues = Counter()
    for job in jobs:
        agent_log = case/job['folder']/'agent.log'
        if agent_log.exists():
            contents = agent_log.read_text(errors='replace')
            for issue, pattern in {
                'output_parse_error_jobs': r'Could not parse LLM output',
                'context_overflow_jobs': r'maximum context length|longer than the maximum model length',
                'tool_import_error_jobs': r'ImportError|ModuleNotFoundError',
                'scipy_statsmodels_compatibility_jobs': r"cannot import name '_lazywhere'",
                'tool_file_not_found_jobs': r'FileNotFoundError',
                'iteration_limit_jobs': r'iteration limit|time limit',
            }.items():
                agent_issues[issue] += bool(re.search(pattern, contents, re.IGNORECASE))
        path = case/job['folder']/'llm_calls.json'
        if not path.exists():
            continue
        recorded = json.loads(path.read_text())
        if recorded:
            messages = next(iter(recorded.values())).get('messages', [])
            if messages and messages[0]:
                first_prompts.append(messages[0][0])
        for call in recorded.values():
            if 'start_ns' in call:
                call = dict(call)
                call['job_index'] = job['index']
                call['end_ns'] = call.get('end_ns', job['end_ns'])
                calls.append(call)
    points = []
    for call in calls:
        points += [(call['start_ns'], 1), (call['end_ns'], -1)]
    points.sort()
    active, prev = 0, start
    for ns, change in points:
        integrate(prev, ns, active, 'mean_llm_calls')
        # Max active for every time bin overlapped by the interval.
        for i in range(index(max(prev,start)), index(min(ns,end))+1):
            bins[i]['peak_llm_calls'] = max(bins[i]['peak_llm_calls'], active)
        active += change
        prev = ns
    for b in bins:
        for key in ('mean_staging_gib','mean_cpu_gib','mean_llm_calls'):
            b[key] /= max(b['duration_seconds'], 1e-9)
    log = (case/'server.log').read_text()
    requests = re.findall(r'Reqid: ([^,]+), Total tokens (\d+), Inference Engine computed tokens: (\d+), LMCache hit tokens: (\d+), need to load: (\d+)', log)
    prompts = [int(r[1]) for r in requests]
    totals = {k:sum(b[k] for b in bins) for k in ('lookup_chunks','dram_chunks','daos_chunks','miss_chunks',
                                                        'cpu_staged_requests','cpu_unstaged_requests')}
    denominator = max(1, totals['lookup_chunks'])
    stats = dict(condition=case.name, start_ns=start, end_ns=end, elapsed_seconds=duration,
        cpu_gib=cpu_gib, staging_gib=10, workflows=len(jobs),
        python_calls=sum(j.get('python_calls',0) for j in jobs),
        first_prompt_count=len(first_prompts),
        common_first_prompt_prefix_characters=len(os.path.commonprefix(first_prompts)),
        workflow_status=dict(Counter(j['status'] for j in jobs)), forced_workflows=sum(j['forced'] for j in jobs),
        llm_calls=len(calls), llm_call_errors=sum('error' in c for c in calls),
        measured_server_requests=len(requests),
        prompt_tokens_min=min(prompts,default=0), prompt_tokens_max=max(prompts,default=0),
        prompt_tokens_mean=statistics.mean(prompts) if prompts else None,
        mean_client_ttft_ms=statistics.mean(c['ttft_ms'] for c in calls if 'ttft_ms' in c)
            if any('ttft_ms' in c for c in calls) else None,
        peak_staging_gib=max(b['peak_staging_gib'] for b in bins),
        mean_staging_gib=sum(b['mean_staging_gib']*b['duration_seconds'] for b in bins)/duration,
        mean_cpu_gib=sum(b['mean_cpu_gib']*b['duration_seconds'] for b in bins)/duration,
        peak_llm_calls=max(b['peak_llm_calls'] for b in bins),
        mean_llm_calls=sum(b['mean_llm_calls']*b['duration_seconds'] for b in bins)/duration,
        occupancy_time_fraction={k:v/duration for k,v in occupancy_seconds.items()},
        agent_issue_jobs=dict(agent_issues),
        **totals, dram_lookup_ratio=totals['dram_chunks']/denominator,
        cpu_staged_chunks=sum(e['staged_chunks'] for e in relevant if e['event']=='cpu_get_ready'),
        cpu_get_chunks=sum(e['chunks'] for e in relevant if e['event']=='cpu_get_ready'),
        daos_lookup_ratio=totals['daos_chunks']/denominator, miss_lookup_ratio=totals['miss_chunks']/denominator,
        allocator_failures=sum(e['event'] in ('allocate','batched_allocate') and e['failed'] for e in relevant),
        partial_daos_reads=sum(e['event']=='prefetch_ready' and e['chunks']<e['requested_chunks'] for e in relevant),
        gpu_full_log_count=log.count('GPU buffer full'),
        negative_ref_log_count=len(re.findall(r'negative: -|Double free|Double release',log)),
        cpu_store_pressure_log_count=log.count('Local cpu memory under pressure'),
        shutdown_semaphore_warning='leaked semaphore' in log,
        context_overflow_count=len(re.findall(r'maximum context length|longer than the maximum model length',log)),
        engine_dead_count=log.count('EngineDeadError'),
        error_log_lines=[line for line in log.splitlines() if 'ERROR' in line][-25:])
    per_phase=[]
    for phase in phases:
        lo,hi=phase['start_ns'],phase.get('end_ns',end)
        phase_hits=[v for v in lookups.values() if lo<=v['time_ns']<hi]
        q=sum(v['queried'] for v in phase_hits)
        d=sum(v['dram'] for v in phase_hits); remote=sum(v['daos'] for v in phase_hits)
        pe=[e for e in relevant if lo<=e['time_ns']<hi]
        prior=next((e for e in reversed(events) if e['time_ns']<lo), None)
        phase_peak=max((e['used_bytes'] for e in pe),default=0)
        if prior:
            phase_peak=max(phase_peak,prior['used_bytes'])
        gc=[e for e in pe if e['event']=='cpu_get_ready']
        phase_calls=[c for c in calls if lo<=c['start_ns']<hi]
        pjobs=[j for j in jobs if j['phase']==phase['index']]
        per_phase.append(dict(concurrency=phase['concurrency'], elapsed_seconds=(hi-lo)/1e9,
            workflows=len(pjobs), complete=sum(j['status']=='complete' for j in pjobs),
            forced=sum(j['forced'] for j in pjobs), llm_calls=len(phase_calls),
            dram_pct=100*d/max(1,q), daos_pct=100*remote/max(1,q), miss_pct=100*(q-d-remote)/max(1,q),
            peak_staging_gib=phase_peak/2**30,
            cpu_staged_requests=sum(e['staged_chunks']>0 for e in gc),
            cpu_unstaged_requests=sum(e['staged_chunks']==0 for e in gc)))
    stats['phases']=per_phase
    stats['lookup_time_windows']=[]
    for label,lo,hi in [('first_120s',start,min(end,start+120_000_000_000)),
                        ('after_120s',start+120_000_000_000,end)]:
        if hi <= lo:
            continue
        rows=[v for v in lookups.values() if lo<=v['time_ns']<hi]
        q=sum(v['queried'] for v in rows)
        d=sum(v['dram'] for v in rows); remote=sum(v['daos'] for v in rows)
        stats['lookup_time_windows'].append(dict(window=label, requests=len(rows), chunks=q,
            dram_chunks=d,daos_chunks=remote,miss_chunks=q-d-remote,
            dram_pct=100*d/max(1,q),daos_pct=100*remote/max(1,q),miss_pct=100*(q-d-remote)/max(1,q)))
    hit_times=[v['time_ns'] for v in lookups.values() if v['dram']>0]
    stats['last_dram_hit_elapsed_seconds']=(max(hit_times)-start)/1e9 if hit_times else None
    samples=[e for e in relevant if e['event']=='occupancy_sample']
    stats['peak_cpu_gib']=max((e['cpu_hot_bytes']/2**30 for e in samples),default=0)
    mirror_samples=[e for e in samples if e.get('dram_mirror') is not None]
    if mirror_samples:
        stats['dram_mirror_last_sample']=mirror_samples[-1]['dram_mirror']
        stats['dram_mirror_last_sample_ns']=mirror_samples[-1]['time_ns']
    stats['store_gather_devices']=sorted({d for e in relevant if e['event']=='store_gather_start'
                                         for d in e['devices']})
    stats['store_manager_copy_bytes']=sum(e['bytes'] for e in relevant if e['event']=='store_manager_copy')
    with (case/'timeline.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(bins[0]))
        writer.writeheader(); writer.writerows(bins)
    (case/'analysis.json').write_text(json.dumps(stats,indent=2)+'\n')
    # Small request interval file permits independent concurrency checks.
    (case/'llm_intervals.json').write_text(json.dumps([{k:c[k] for k in ('job_index','start_ns','end_ns','ttft_ms','error') if k in c}
                                                     for c in calls],indent=2)+'\n')
    return stats, bins, phases


def plot(folder, results):
    columns = len(results)
    column_width = 900 if columns == 1 else 660
    width, height = columns*column_width+30, 1150
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="white"/>',
        '<g font-family="sans-serif" font-size="12" fill="#222">',
        '<text x="45" y="26" font-size="20">DiscoveryBench real-data agents: KV tier usage and staging over time</text>']
    colors = ['#0072b2','#009e73','#d55e00']
    for col,(stats,bins,phases) in enumerate(results):
        left, w = 60+col*column_width, column_width-80
        duration = stats['elapsed_seconds']
        x = lambda t:left+t/duration*w
        def panel(row, title, ymax, series):
            top, h = 90+row*255, 170
            y = lambda val:top+h-val/ymax*h
            svg.append(f'<text x="{left}" y="{top-18}" font-size="15">{stats["condition"]}: {title}</text>')
            for k in range(5):
                val = ymax*k/4
                svg.append(f'<line x1="{left}" y1="{y(val)}" x2="{left+w}" y2="{y(val)}" stroke="#ddd"/>')
                svg.append(f'<text x="{left-38}" y="{y(val)+4}">{val:.1f}</text>')
                svg.append(f'<text x="{left+k*w/4-10}" y="{top+h+18}">{duration*k/4/60:.1f}</text>')
            for phase in phases:
                xx=x((phase['start_ns']-stats['start_ns'])/1e9)
                svg.append(f'<line x1="{xx}" y1="{top}" x2="{xx}" y2="{top+h}" stroke="#777" stroke-dasharray="4 3"/>')
                svg.append(f'<text x="{xx+3}" y="{top+13}">agents={phase["concurrency"]}</text>')
            for index,(name,values) in enumerate(series):
                # No-query bins have no hit ratio, not a ratio of zero.
                segments = [[]]
                for b,v in zip(bins,values):
                    if v is None:
                        if segments[-1]:
                            segments.append([])
                    else:
                        segments[-1].append(f'{x(b["t_seconds"]):.2f},{y(v):.2f}')
                for segment in segments:
                    if segment:
                        svg.append(f'<polyline points="{" ".join(segment)}" fill="none" stroke="{colors[index]}" stroke-width="1.7"/>')
                xx=left+index*190
                svg.append(f'<text x="{xx}" y="{top+h+43}" fill="{colors[index]}">{html.escape(name)}</text>')
        panel(0,'GPU staging (GiB)',10,[('5s peak',[b['peak_staging_gib'] for b in bins]),
            ('time-weighted mean',[b['mean_staging_gib'] for b in bins])])
        panel(1,'Prefix-lookup chunk fractions (%)',100,
            [(label,[100*b[field]/b['lookup_chunks'] if b['lookup_chunks'] else None for b in bins]) for label,field in
             [('DRAM hit','dram_chunks'),('DAOS hit','daos_chunks'),('miss','miss_chunks')]])
        panel(2,'DRAM cached KV (GiB)',stats['cpu_gib'],[('time-weighted mean',[b['mean_cpu_gib'] for b in bins])])
        panel(3,'In-flight LLM calls (client callbacks)',max(16,stats['peak_llm_calls']),
            [('5s peak',[b['peak_llm_calls'] for b in bins]),('time-weighted mean',[b['mean_llm_calls'] for b in bins])])
    svg += ['<text x="45" y="1132">X axis: elapsed minutes; startup excluded; phase drains included. Instrumented workload, not a matched-request latency benchmark.</text>', '</g></svg>']
    path=folder/'timeline.svg'; path.write_text('\n'.join(svg))
    subprocess.run(['rsvg-convert','-o',str(folder/'timeline.png'),str(path)],check=True)


def main():
    p=argparse.ArgumentParser(); p.add_argument('folder',type=Path); a=p.parse_args()
    folder=a.folder.resolve()
    plan=json.loads((folder/'plan.json').read_text())
    root=Path(__file__).resolve().parent
    source_hashes=plan.get('source_sha256',{})
    inputs={name:digest for task in json.loads((folder/'tasks.json').read_text())
            for name,digest in task['hashes'].items()}
    def mismatches(expected,base):
        return [name for name,digest in expected.items()
                if not (base/name).is_file() or hashlib.sha256((base/name).read_bytes()).hexdigest()!=digest]
    native={case.name:json.loads((case/'native_maps.json').read_text())
            for case in folder.glob('prefetch_*') if (case/'native_maps.json').exists()}
    audit=dict(current_source_mismatches=mismatches(source_hashes,root),
               archived_source_mismatches=mismatches(source_hashes,folder/'executed_sources'),
               dataset_mismatches=mismatches(inputs,root), native_library_maps=native,
               native_maps_equal=all(v==next(iter(native.values())) for v in native.values()) if native else None)
    (folder/'validation.json').write_text(json.dumps(audit,indent=2)+'\n')
    results=[]
    for case in sorted(folder.glob('prefetch_*')):
        if (case/'phases.json').exists():
            results.append(summarize_case(case,plan['args']['cpu_gb']))
    plot(folder,results)
    (folder/'analysis.json').write_text(json.dumps([r[0] for r in results],indent=2)+'\n')
    for stats,_,_ in results:
        print(json.dumps(stats,ensure_ascii=False))


if __name__=='__main__':
    main()
