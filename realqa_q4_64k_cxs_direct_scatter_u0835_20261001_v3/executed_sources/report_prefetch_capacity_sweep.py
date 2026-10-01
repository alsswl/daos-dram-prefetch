#!/usr/bin/env python3
"""Validated cold/warm tables and per-phase staging/hit timelines."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml

from report_cold_warm_prefetch import select_events
from report_prefetch_timing import join, stats
from report_capacity_matrix import save_chart
from staging_mixed_pressure import read_events


def read(path): return json.loads(path.read_text())
def dump(path, value): path.write_text(json.dumps(value,indent=2)+'\n')


def timeline(folder, events, start, end, capacity):
    duration=(end-start)/1e9
    bins=[dict(seconds=i*2,staging=[],dram=0,daos=0,queried=0) for i in range(int(duration//2)+1)]
    for e in events:
        if not start<=e['time_ns']<=end: continue
        b=bins[min(int((e['time_ns']-start)/2e9),len(bins)-1)]
        if e['event']=='occupancy_sample': b['staging'].append(e['used_bytes']/2**30)
        if e['event']=='tier_lookup':
            b[e['tier']]+=e['hit_chunks']
            if e['tier']=='dram': b['queried']+=e['queried_chunks']
    for b in bins:
        values=b.pop('staging')
        b['sampled_peak_gib']=max(values) if values else None
        b['sampled_mean_gib']=sum(values)/len(values) if values else None
        for key in ('dram','daos'):
            b[key+'_hit_pct']=100*b[key]/b['queried'] if b['queried'] else None
    dump(folder/'timeline_bins.json',bins)
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="560">',
         '<rect width="1000" height="560" fill="white"/><g font-family="sans-serif" font-size="13">',
         f'<text x="55" y="28">{folder.parent.name} / {folder.name}: 2-second bins</text>']
    def panel(top,title,series):
        svg.append(f'<text x="60" y="{top-12}">{title}</text>')
        for tick in (0,25,50,75,100):
            y=top+165*(1-tick/100)
            svg.extend([f'<line x1="65" x2="970" y1="{y}" y2="{y}" stroke="#ddd"/>',
                        f'<text x="25" y="{y+4}">{tick}</text>'])
        for key,color,scale in series:
            segment=[]
            def flush():
                if segment:
                    points=' '.join(f'{x:.2f},{y:.2f}' for x,y in segment)
                    svg.append(f'<polyline points="{points}" stroke="{color}" fill="none" stroke-width="1.6"/>')
                    for x,y in segment: svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.5" fill="{color}"/>')
                    segment.clear()
            for b in bins:
                v=b[key]
                if v is None: flush(); continue
                segment.append((65+905*b['seconds']/max(duration,1),top+165*(1-v*scale/100)))
            flush()
        for i in range(5):
            svg.append(f'<text x="{65+905*i/4}" y="{top+187}">{duration*i/4:.0f}s</text>')
    panel(75,'Staging occupancy (%): blue=sampled peak, green=sampled mean',
          [('sampled_peak_gib','#0072b2',100/capacity),('sampled_mean_gib','#009e73',100/capacity)])
    panel(320,'Lookup chunk ratio (%): blue=DRAM, green=DAOS; gaps=no lookup',
          [('dram_hit_pct','#0072b2',1),('daos_hit_pct','#009e73',1)])
    svg+=['<text x="60" y="545">Sampled occupancy, not GPU utilization. Hit ratios use initial CPU-tier candidate chunks.</text></g></svg>']
    save_chart(folder,'staging_hits',svg)


def report(root, partial=False):
    plan=read(root/'plan.json'); records=read(root/'requests.json')
    expected=[(r['index'],r['prompt_sha256']) for r in records]
    if not partial: assert read(root/'status.json')['status']=='completed'
    summaries=[]; details={}; configs=[]; natives=[]; namespaces=[]
    for spec in plan['cases']:
        case=root/spec['name']
        if not (case/'status.json').exists() or read(case/'status.json')['status']!='completed':
            assert partial; continue
        cfg=yaml.safe_load((case/'config.yaml').read_text()); ec=cfg['extra_config']
        assert cfg.pop('max_local_cpu_size')==spec['cpu_gib']
        assert ec.pop('daosgds.gpu_buffer_gb')==spec['staging_gib']
        assert ec.pop('daosgds.dram_prefetch')==spec['prefetch']
        assert ec.pop('daosgds.dram_prefetch_cancel_queued')==spec['cancel_queued']
        assert ec.pop('daosgds.dram_prefetch_early_ready')==spec['early_ready']
        assert ec['daosgds.dram_prefetch_policy']=='capacity'
        namespaces.append(ec.pop('daosgds.object_namespace')); ec.pop('daosgds.root')
        configs.append(cfg); natives.append(read(case/'native_maps.json'))
        events=read_events(case)
        assert len({e['pid'] for e in events})==1
        initial=read(case/'initial_sample.json')
        assert initial['used_bytes']==initial['cpu_hot_bytes']==initial['daos_puts']==0
        seen=set()
        for phase in ('cold','warm'):
            folder=case/phase; calls=read(folder/'replay_calls.json'); window=read(folder/'phase.json')
            assert read(folder/'status.json')['status']=='completed'
            assert len(calls)==256 and not any('error' in c for c in calls)
            assert [(c['index'],c['prompt_sha256']) for c in calls]==expected
            ids={c['server_request_id'] for c in calls}
            assert len(ids)==256 and not ids&seen; seen|=ids
            before,after=read(folder/'initial_sample.json'),read(folder/'final_sample.json')
            assert before==(initial if phase=='cold' else read(case/'cold/final_sample.json'))
            assert after['used_bytes']==after['dram_mirror']['pending_bytes']==after['dram_mirror']['errors']==0
            pa,pb=after['cpu_prefetch'] or {},before['cpu_prefetch'] or {}
            assert pa.get('copy_errors',0)==pa.get('deferred_pending_batches',0)==pa.get('watermark_rejections',0)==0
            counters={k:v-pb.get(k,0) for k,v in pa.items()}
            selected=select_events(events,calls); rows=join(calls,selected)
            assert not any(r['other_failed_chunks'] for r in rows)
            decisions=[e for e in selected if e['event']=='cpu_prefetch_retrieve_decision']
            starts={e['request_id']:e['monotonic_ns'] for e in selected if e['event']=='retrieve_start'}
            ends={e['request_id']:e['monotonic_ns'] for e in selected if e['event']=='retrieve_return'}
            assert all(starts[e['request_id']]<=e['resolve_start_ns']<=e['resolve_end_ns']<=ends[e['request_id']] for e in decisions)
            for decision in ('cancelled_queued','ready_gpu','waited_gpu','capacity_cpu'):
                assert sum(e['decision']==decision for e in decisions)==counters.get('retrieve_'+decision,0)
            candidates=sum(e['queried_chunks'] for e in selected if e['event']=='tier_lookup' and e['tier']=='dram')
            dram=sum(r['dram_lookup_chunks'] for r in rows); daos=sum(r['daos_lookup_chunks'] for r in rows)
            assert 0<=dram+daos<=candidates
            total=sum(r['prompt_tokens'] for r in rows); recompute=sum(r['capacity_recomputed_tokens'] for r in rows)
            scoped=[e for e in events if window['start_ns']<=e['time_ns']<=window['end_ns']]
            entry=dict(**spec,phase=phase,requests=256,ttft_ms=stats(r['ttft_ms'] for r in rows),
                retrieve_ms=stats(r.get('retrieve_ms') for r in rows),queue_ms=stats(r.get('queue_ms') for r in rows),
                computed_tokens=sum(r['computed_prompt_tokens'] for r in rows),input_tokens=total,
                completion_tokens=sum(c['completion_tokens'] for c in calls),
                capacity_recomputed_tokens=recompute,capacity_recompute_input_pct=100*recompute/total,
                capacity_affected_requests=sum(r['capacity_recomputed_tokens']>0 for r in rows),
                dram_hit_pct=100*dram/candidates if candidates else 0,daos_hit_pct=100*daos/candidates if candidates else 0,
                peak_staging_gib=max(e['used_bytes'] for e in scoped)/2**30,
                elapsed_seconds=(window['end_ns']-window['start_ns'])/1e9,counters=counters)
            summaries.append(entry); details[(spec['concurrency'],spec['cpu_gib'],spec['staging_gib'],spec['mode'],phase)]=rows
            dump(folder/'timing_by_request.json',rows); dump(folder/'retrieve_decisions.json',decisions)
            timeline(folder,events,window['start_ns'],window['end_ns'],spec['staging_gib'])
    assert summaries
    assert all(c==configs[0] for c in configs) and all(n==natives[0] for n in natives)
    assert len(namespaces)==len(set(namespaces))
    for name,digest in plan['source_sha256'].items():
        assert hashlib.sha256((root/'executed_sources'/name).read_bytes()).hexdigest()==digest
    paired=[]
    for c in (8,16):
        for d in (8,4,2):
            for s in (8,4):
                for phase in ('cold','warm'):
                    for ref,other in [('off','wait'),('wait','cancel')]:
                        k=(c,d,s)
                        if (*k,ref,phase) not in details or (*k,other,phase) not in details:continue
                        pairs=list(zip(details[(*k,ref,phase)],details[(*k,other,phase)],strict=True))
                        same=[(x,y) for x,y in pairs if x['cached_tokens']==y['cached_tokens']]
                        paired.append(dict(concurrency=c,cpu_gib=d,staging_gib=s,phase=phase,reference=ref,other=other,
                            same_cached_requests=len(same),same_tier_requests=sum(
                                (x['dram_lookup_chunks'],x['daos_lookup_chunks'])==(y['dram_lookup_chunks'],y['daos_lookup_chunks']) for x,y in same)))
    dump(root/'summary.json',summaries); dump(root/'paired_checks.json',paired)
    dump(root/'validation.json',dict(partial=partial,completed_cases=len(configs),
         config_and_native_match=True,inputs_match=True,independent_namespaces=True,
         cold_warm_same_worker=True,buffers_drained=True,decision_boundaries_checked=True))
    lines=['# DRAM/staging/동시성 × 프리페치·대기 취소 비교','',
        'off=DRAM 프리페치 OFF, wait=프리페치 ON/취소 OFF, cancel=프리페치 ON/취소 ON. DAOS 프리페치는 항상 ON.',
        '조건마다 새 프로세스·빈 DRAM·새 namespace에서 cold256 → 같은 캐시 warm256. 각 조건 1회.',
        'cancel에는 조기 준비 알림 변화도 포함된다. 취소 건수 0이면 취소 자체의 효과로 해석하지 않는다.',
        f'완료된 조건 {len(configs)}/36. 부분 보고서={partial}. 단위: TTFT ms, 메모리 GiB.', '']
    for phase in ('warm','cold'):
        lines += [f'## {phase}', '',
          '|동시|DRAM|staging|방식|평균 TTFT|p95|DRAM hit %|DAOS hit %|재계산 입력 %|취소 건수|staging peak|',
          '|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|']
        for e in summaries:
            if e['phase']!=phase:continue
            lines.append(f"|{e['concurrency']}|{e['cpu_gib']}|{e['staging_gib']}|{e['mode']}|{e['ttft_ms']['mean']:.2f}|{e['ttft_ms']['p95']:.2f}|{e['dram_hit_pct']:.2f}|{e['daos_hit_pct']:.2f}|{e['capacity_recompute_input_pct']:.2f}|{e['counters'].get('retrieve_cancelled_queued',0)}|{e['peak_staging_gib']:.3f}|")
    lines += ['', '재계산 비율은 DAOS GPU 할당 실패로 잃은 사용 가능 prefix 토큰 / 전체 입력 토큰이다. 일반 cold miss는 제외한다.',
              'hit 비율은 첫 DRAM 조회 후보 청크를 분모로 쓴다. 시간 그래프는 각 조건 cold/warm/staging_hits.png에 있다.',
              '생성량·rolling 도착 시점·tier별 캐시 배치는 달라질 수 있다. paired_checks.json에서 실제 재사용량도 확인한다.']
    (root/'RESULT_KO.md').write_text('\n'.join(lines)+'\n')
    flat=[]
    for e in summaries:
        r={k:v for k,v in e.items() if not isinstance(v,dict)}
        r.update(ttft_mean_ms=e['ttft_ms']['mean'],ttft_p95_ms=e['ttft_ms']['p95'],
                 cancelled_queued=e['counters'].get('retrieve_cancelled_queued',0),
                 capacity_fallback=e['counters'].get('fallback_requests',0)); flat.append(r)
    with (root/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--partial',action='store_true')
    a=p.parse_args();report(a.folder.resolve(),a.partial)
