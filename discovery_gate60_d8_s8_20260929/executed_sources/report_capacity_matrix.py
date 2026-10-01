#!/usr/bin/env python3
"""Read-only measurement analysis; writes derived matrix reports and SVG/PNG charts."""
import argparse
from collections import defaultdict
import csv
import hashlib
import html
import json
from pathlib import Path
import statistics
import subprocess

import yaml

from analyze_discovery_staging import plot, summarize_case
from staging_mixed_pressure import read_events


def attribute_recomputation(calls, events, chunk=128):
    """Count lost usable prefix only after validating CPU tier and actual reuse.

    GPU-full failed chunk count is NOT the lost-prefix count: successful chunks
    after the first failure are discarded too. Excludes cold/not-found misses.
    """
    by_http = {c['server_request_id']: c for c in calls}
    per_id = defaultdict(lambda: dict(dram_lookup=0,daos_lookup=0,cpu_returned=0,outcomes=[]))
    for e in events:
        if e['event'] not in ('tier_lookup','cpu_get_ready','daos_prefetch_outcome'):
            continue
        rid = e['request_id']
        http_id = rid if rid in by_http else rid.rsplit('-',1)[0]
        if http_id not in by_http:
            raise ValueError(f'Unmapped trace request: {rid}')
        row = per_id[http_id]
        if e['event'] == 'tier_lookup':
            row[e['tier']+'_lookup'] += e['hit_chunks']
        elif e['event'] == 'cpu_get_ready':
            row['cpu_returned'] += e['chunks']
        else:
            row['outcomes'].append(e)
    results = []
    for c in calls:
        r = per_id[c['server_request_id']]
        outcomes = r['outcomes']
        requested = sum(e['requested_chunks'] for e in outcomes)
        returned = sum(e['returned_chunks'] for e in outcomes)
        lost = requested-returned
        cap = lambda chunks: min(c['prompt_tokens']-1,chunks*chunk)
        expected_reuse = cap(r['dram_lookup']+r['daos_lookup'])
        actual_from_trace = cap(r['cpu_returned']+returned)
        issues = []
        if len(outcomes)>1: issues.append('multiple_daos_batches')
        if r['cpu_returned'] != r['dram_lookup']: issues.append('cpu_prefix_changed')
        if requested != r['daos_lookup']: issues.append('daos_lookup_read_mismatch')
        if actual_from_trace != c['cached_tokens']: issues.append('actual_reuse_mismatch')
        first = outcomes[0].get('first_failure') if outcomes else None
        lost_tokens = max(0,expected_reuse-actual_from_trace) if lost else 0
        attributable = lost_tokens if lost and first == 'capacity' and not issues else 0
        results.append(dict(index=c['index'],server_request_id=c['server_request_id'],
            start_ns=c['start_ns'],ttft_ms=c['ttft_ms'],prompt_tokens=c['prompt_tokens'],
            cached_tokens=c['cached_tokens'],computed_prompt_tokens=c['prompt_tokens']-c['cached_tokens'],
            dram_lookup_chunks=r['dram_lookup'],daos_lookup_chunks=r['daos_lookup'],
            daos_requested_chunks=requested,daos_returned_chunks=returned,
            capacity_failed_chunks=sum(e['capacity_failed_chunks'] for e in outcomes),
            other_failed_chunks=sum(e['other_failed_chunks'] for e in outcomes),
            successful_tail_discarded_chunks=sum(e['successful_tail_discarded_chunks'] for e in outcomes),
            prefix_shortfall_tokens=lost_tokens,capacity_recomputed_tokens=attributable,
            unattributed_shortfall_tokens=lost_tokens-attributable,
            first_failure=first,attribution_issues=issues))
    return results


def save_chart(folder, name, svg):
    path = folder/(name+'.svg')
    path.write_text('\n'.join(svg))
    subprocess.run(['rsvg-convert','-o',str(folder/(name+'.png')),str(path)],check=True)


def overview(folder, results, mode='ON'):
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="1030">',
        '<rect width="1500" height="1030" fill="white"/>',
        '<g font-family="sans-serif" font-size="12" fill="#222">',
        f'<text x="30" y="26" font-size="20">DRAM x GPU staging: 16 concurrent requests, DRAM prefetch {mode}</text>',
        '<text x="30" y="50">Staging: blue=5s peak, green=time-weighted mean. Hits: blue=DRAM, green=DAOS, orange=miss.</text>']
    colors = ['#0072b2','#009e73','#d55e00']
    for stats,bins,_ in results:
        col = (8,4,2).index(int(stats['cpu_gib']))
        row = (8,4).index(int(stats['staging_gib']))
        left, top, width, duration = 60+col*495, 100+row*450, 420, stats['elapsed_seconds']
        svg.append(f'<text x="{left}" y="{top-20}" font-size="17">DRAM {stats["cpu_gib"]:g}GiB / staging {stats["staging_gib"]:g}GiB</text>')
        def panel(y_top, title, series):
            height = 135
            svg.append(f'<text x="{left}" y="{y_top-7}">{title}</text>')
            for tick in (0,25,50,75,100):
                y = y_top+height*(1-tick/100)
                svg.append(f'<line x1="{left}" x2="{left+width}" y1="{y}" y2="{y}" stroke="#ddd"/>')
                svg.append(f'<text x="{left-32}" y="{y+4}">{tick}</text>')
            for k in range(5):
                svg.append(f'<text x="{left+k*width/4-10}" y="{y_top+height+17}">{duration*k/4:.0f}</text>')
            for number,values in enumerate(series):
                segments = [[]]
                for b,val in zip(bins,values):
                    if val is None:
                        if segments[-1]: segments.append([])
                        continue
                    segments[-1].append((left+width*b['t_seconds']/duration,y_top+height*(1-val/100)))
                for points in segments:
                    if not points: continue
                    coords = ' '.join(f'{x:.2f},{y:.2f}' for x,y in points)
                    svg.append(f'<polyline points="{coords}" fill="none" stroke="{colors[number]}" stroke-width="1.6"/>')
                    if len(points)==1:
                        x,y=points[0]
                        svg.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.3" fill="{colors[number]}"/>')
        panel(top+12,'GPU staging occupancy (%)',[
            [100*b[k]/stats['staging_gib'] for b in bins] for k in ('peak_staging_gib','mean_staging_gib')])
        panel(top+225,'Lookup chunk fractions (%)',[
            [100*b[k]/b['lookup_chunks'] if b['lookup_chunks'] else None for b in bins]
            for k in ('dram_chunks','daos_chunks','miss_chunks')])
    svg += ['<text x="30" y="1010">X: seconds since first replay wave; each case independently cold. Dots mark isolated lookup bins; gaps mean no lookups.</text>','</g></svg>']
    save_chart(folder,'overview',svg)


def recompute_chart(folder, stats):
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="580">',
        '<rect width="1100" height="580" fill="white"/>',
        '<g font-family="sans-serif" font-size="13" fill="#222">',
        '<text x="40" y="30" font-size="21">Extra prefill due to DAOS prefetch capacity failure</text>',
        '<text x="40" y="55">Lost usable prefix tokens / all input tokens (%). Includes successful tail discarded after first failure.</text>']
    maximum = max(1,max(s['capacity_recompute_input_pct'] for s in stats)*1.2)
    for k in range(6):
        val=maximum*k/5; y=455-val/maximum*350
        svg += [f'<line x1="80" x2="1070" y1="{y}" y2="{y}" stroke="#ddd"/>',
                f'<text x="38" y="{y+4}">{val:.1f}%</text>']
    for i,s in enumerate(stats):
        x=105+i*160; val=s['capacity_recompute_input_pct']; height=val/maximum*350
        svg += [f'<rect x="{x}" y="{455-height}" width="95" height="{height}" fill="#d55e00"/>',
            f'<text x="{x+12}" y="{445-height}">{val:.2f}%</text>',
            f'<text x="{x}" y="485">D{s["cpu_gib"]:g} / S{s["staging_gib"]:g} GiB</text>',
            f'<text x="{x}" y="511">{s["capacity_affected_requests"]}/{s["llm_calls"]} requests</text>']
    svg += ['<text x="40" y="558">D=DRAM capacity; S=GPU staging capacity. Cold misses are excluded from the numerator.</text>','</g></svg>']
    save_chart(folder,'recompute_ratio',svg)


def report(folder):
    plan=json.loads((folder/'plan.json').read_text())
    modes={spec['prefetch'] for spec in plan['cases']}
    assert len(modes)==1, 'Use a separate matrix for each prefetch mode'
    mode='ON' if modes.pop() else 'OFF'
    records=json.loads((folder/'requests.json').read_text())
    all_results, summaries, audit = [], [], {}
    configs, native = [], []
    for spec in plan['cases']:
        case=folder/spec['name']
        assert json.loads((case/'status.json').read_text())['status']=='completed'
        calls=json.loads((case/'replay_calls.json').read_text())
        assert len(calls)==256 and not any('error' in r for r in calls)
        assert [(r['index'],r['prompt_sha256']) for r in calls]==[(r['index'],r['prompt_sha256']) for r in records]
        initial=json.loads((case/'initial_sample.json').read_text())
        assert initial['used_bytes']==initial['cpu_hot_bytes']==initial['daos_puts']==0
        cfg=yaml.safe_load((case/'config.yaml').read_text())
        assert cfg['max_local_cpu_size']==spec['cpu_gib']
        assert cfg['extra_config']['daosgds.gpu_buffer_gb']==spec['staging_gib']
        assert cfg['extra_config']['daosgds.dram_prefetch'] is spec['prefetch']
        assert cfg['extra_config']['daosgds.dram_prefetch_policy']=='capacity'
        cfg.pop('max_local_cpu_size')
        for k in ('daosgds.gpu_buffer_gb','daosgds.root','daosgds.object_namespace'): cfg['extra_config'].pop(k)
        configs.append(cfg)
        native.append(json.loads((case/'native_maps.json').read_text()))
        events=read_events(case)
        attribution=attribute_recomputation(calls,events,plan['chunk_tokens'])
        common_fields=list(attribution[0])
        (case/'recomputation_by_request.json').write_text(json.dumps(attribution,indent=2)+'\n')
        with (case/'recomputation_by_request.csv').open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=common_fields); writer.writeheader(); writer.writerows(attribution)
        stats,bins,phases=summarize_case(case,spec['cpu_gib'])
        final=json.loads((case/'final_sample.json').read_text())
        total=sum(c['prompt_tokens'] for c in calls)
        lost=sum(r['capacity_recomputed_tokens'] for r in attribution)
        daos_tokens=sum(r['daos_requested_chunks']*plan['chunk_tokens'] for r in attribution)
        affected=sum(r['capacity_recomputed_tokens']>0 for r in attribution)
        stats.update(total_prompt_tokens=total,total_completion_tokens=sum(c['completion_tokens'] for c in calls),
            actual_cached_tokens=sum(c['cached_tokens'] for c in calls),
            total_computed_prompt_tokens=sum(r['computed_prompt_tokens'] for r in attribution),
            capacity_recomputed_tokens=lost,capacity_recompute_input_pct=100*lost/total,
            capacity_recompute_daos_pct=100*lost/daos_tokens if daos_tokens else 0,
            capacity_affected_requests=affected,capacity_affected_request_pct=100*affected/len(calls),
            daos_requested_tokens=daos_tokens,
            daos_read_capacity_failed_chunks=sum(r['capacity_failed_chunks'] for r in attribution),
            other_read_failed_chunks=sum(r['other_failed_chunks'] for r in attribution),
            unattributed_shortfall_tokens=sum(r['unattributed_shortfall_tokens'] for r in attribution),
            attribution_issue_requests=sum(bool(r['attribution_issues']) for r in attribution),
            prefetch_counters=final['cpu_prefetch'],backend_alloc_failures=final['daos_alloc_fail'],
            mirror_final=final['dram_mirror'],
            mean_latency_ms=statistics.mean(c['elapsed_seconds']*1000 for c in calls))
        stats['actual_cached_input_pct']=100*stats['actual_cached_tokens']/total
        assert stats['daos_read_capacity_failed_chunks']==stats['gpu_full_log_count'], 'Failure attribution/log mismatch'
        assert not stats['attribution_issue_requests'], 'Request prefix/usage validation failed; inspect JSON'
        assert not stats['unattributed_shortfall_tokens'], 'Unattributed lost prefix; do not claim capacity recomputation'
        assert not stats['other_read_failed_chunks'], 'Non-capacity DAOS read failure occurred'
        assert final['used_bytes']==final['dram_mirror']['pending_bytes']==0
        assert not stats['negative_ref_log_count'] and not final['dram_mirror']['errors']
        (case/'analysis.json').write_text(json.dumps(stats,indent=2)+'\n')
        plot(case,[(stats,bins,phases)],title=f'DRAM {spec["cpu_gib"]}GiB / staging {spec["staging_gib"]}GiB, DRAM prefetch {mode}',
            phase_label='requests',caption='X: elapsed minutes; 256 fixed inputs in waves of16, cold start; no Python execution. Nominal occupancy samples: 20ms.')
        all_results.append((stats,bins,phases)); summaries.append(stats)
        audit[spec['name']]=dict(requests=256,started_empty=True,attribution_issues=0,
            read_capacity_failures_match_log=True,final_staging_empty=True,final_mirror_drained=True)
    root=Path(__file__).resolve().parent
    for label,base in [('archived',folder/'executed_sources'),('current',root)]:
        audit[label+'_source_mismatches']=[n for n,h in plan['source_sha256'].items()
            if hashlib.sha256((base/n).read_bytes()).hexdigest()!=h]
    audit['same_noncapacity_config']=all(c==configs[0] for c in configs)
    audit['same_native_libraries']=all(n==native[0] for n in native)
    audit['requests_hash_matches']=hashlib.sha256((folder/'requests.json').read_bytes()).hexdigest()==plan['requests_sha256']
    audit['same_original_requests']=records==json.loads(Path(plan['source_requests']).read_text())
    assert audit['same_noncapacity_config'] and audit['same_native_libraries'] and audit['requests_hash_matches']
    (folder/'validation.json').write_text(json.dumps(audit,indent=2)+'\n')
    (folder/'analysis.json').write_text(json.dumps(summaries,indent=2)+'\n')
    overview(folder,all_results,mode); recompute_chart(folder,summaries)
    rows=[]
    for s in summaries:
        rows.append(f"| {s['cpu_gib']:g} | {s['staging_gib']:g} | {s['capacity_recomputed_tokens']:,} | "
            f"{s['capacity_recompute_input_pct']:.2f}% | {s['capacity_recompute_daos_pct']:.2f}% | "
            f"{s['capacity_affected_requests']}/256 | {100*s['dram_lookup_ratio']:.2f}% | {100*s['daos_lookup_ratio']:.2f}% | "
            f"{s['peak_staging_gib']:.3f} | {s['mean_client_ttft_ms']:.2f} | {s['elapsed_seconds']:.2f} |")
    text='''# DRAM / GPU staging 용량 6조합 실험

## 조건

Qwen3-14B BF16, object 경로, chunk 128, 동시 요청 16, 같은 DiscoveryBench 기록 입력 256개씩 재생했다. Python 도구는 재실행하지 않았다. DRAM 8/4/2GiB × staging 8/4GiB, 총 6조건 모두 DRAM 프리페치 ON이다.

각 조건마다 새 vLLM 프로세스·빈 DRAM·새 DAOS 이름 공간에서 시작했다. 공용 데이터를 삭제하거나 DAOS/OS 내부 캐시를 비우지는 않았다. DRAM 프리페치는 soft watermark 없이 실제 할당을 시도한다. 물리적 용량, 기존 DAOS serializer 및 비동기 DRAM 승격 큐의 제한은 유지했다. 실행 순서는 D8S8→D8S4→D4S8→D4S4→D2S8→D2S4다.

입력/묶음 구성을 고정했지만 묶음 완료 후 다음 묶음을 보내므로 벽시계 도착 시간까지 동일하지는 않다. EOS를 허용하므로 생성량이 달라질 수 있다. 조건별 1회, 반복 평균/신뢰구간 실험은 아니다.

## 결과

| DRAM(GiB) | staging(GiB) | 실패로 추가 계산한 토큰 | 전체 입력 대비 | DAOS 읽기 예정 토큰 대비 | 영향 요청 | DRAM hit | DAOS hit | staging 최대(GiB) | 평균 TTFT(ms) | 전체 시간(s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

## 재계산 비율의 정의

- **전체 입력 대비 실패 재계산율** = GPU 할당 실패로 잃은 연속 캐시 prefix 토큰 / 전체 입력 토큰. 원래 캐시가 없었던 cold miss, 새 토큰, 생성 토큰은 분자에서 제외한다.
- **DAOS 읽기 예정 토큰 대비** = 같은 분자 / DAOS 프리페치에 요청한 청크 수×128. 요청 실패 비율과 토큰 비율을 혼동하지 않는다.
- 각 요청의 CPU hit/반환 청크, DAOS hit/반환 청크, 실패 원인, 실제 usage.cached_tokens를 대조했다. 검증이 맞지 않는 경우 보고서가 성공으로 완료되지 않도록 했다.
- 첫 청크 실패 뒤에 성공한 청크도 prefix 규칙 때문에 반환하지 않을 수 있다. 따라서 실패한 GPU 할당 수와 재계산 토큰÷128은 같지 않을 수 있다.
- 이것은 미반환 prefix가 모델 입력 계산으로 넘어간 양이다. GPU kernel을 직접 계수한 연산량이나 재계산만의 실행 시간을 뜻하지 않는다. 스케줄러 재선점으로 추가 반복 계산한 양까지 측정하는 지표도 아니다.

## 그래프

- [6조건 시간 그래프](overview.png): 열=DRAM 8/4/2GiB, 행=staging 8/4GiB. 각 칸 위는 staging 점유율, 아래는 DRAM/DAOS/miss 비율이다.
- [실패로 인한 재계산 비율](recompute_ratio.png): 분모는 전체 입력 토큰이다.

staging 그래프의 파랑은 5초 구간 최대, 초록은 시간 가중 평균이다. 높은 최대치를 계속 점유한 것으로 해석하지 않는다. hit 그래프의 파랑=DRAM, 초록=DAOS, 주황=miss이며 조회가 없는 시간은 빈칸이다. hit는 존재 확인 결과라 실제 재사용 성공률과 다르다.

개별 그래프/원시 계측:
'''+ '\n'.join(f"- [{s['condition']} 시간 그래프]({s['condition']}/timeline.png), [요청별 재계산 검증]({s['condition']}/recomputation_by_request.csv)" for s in summaries)+'''

분석 JSON에는 실제 재사용률, 총 생성량, DRAM fallback, DAOS 읽기 실패 청크, 다른 실패 원인, 초기/최종 상태를 함께 저장했다. 코드 스냅샷은 executed_sources, 검증은 validation.json, 실제 설정은 각 조건의 config.yaml에 있다. 원본 discos 및 과거 실험 결과는 수정하지 않았다.
'''
    text=text.replace('총 6조건 모두 DRAM 프리페치 ON이다.',f'총 6조건 모두 DRAM 프리페치 {mode}이다. DAOS 프리페치는 유지했다.')
    if mode=='OFF':
        text=text.replace('DRAM 프리페치는 soft watermark 없이 실제 할당을 시도한다.',
                          'DRAM 프리페치만 비활성화했으며 DRAM 캐시 보관·비동기 쓰기 미러·읽기 승격은 유지했다.')
    (folder/'RESULT_KO.md').write_text(text)
    print('\n'.join(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('folder',type=Path)
    report(p.parse_args().folder.resolve())
