#!/usr/bin/env python3
"""Separate cold/warm HTTP and trace records from each shared live process."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

import yaml

from report_capacity_matrix import save_chart
from report_prefetch_timing import join, stats
from staging_mixed_pressure import read_events


def read(path):
    return json.loads(path.read_text())


def select_events(events, calls):
    """Do not accidentally attribute the other phase's request events."""
    ids = {c['server_request_id'] for c in calls}
    out = []
    for e in events:
        rid = e.get('request_id')
        if rid and (rid in ids or rid.rsplit('-', 1)[0] in ids):
            out.append(e)
    return out


def metric_means(before_path, after_path):
    def values(path):
        return {m.group(1): float(m.group(2)) for m in re.finditer(
            r'^(vllm:\w+)\{[^\n]*\} ([\d.e+-]+)$', path.read_text(), re.M)}
    before, after = values(before_path), values(after_path)
    result = {}
    for k in ('time_to_first_token_seconds', 'request_queue_time_seconds', 'request_prefill_time_seconds'):
        count = after['vllm:'+k+'_count']-before.get('vllm:'+k+'_count', 0)
        total = after['vllm:'+k+'_sum']-before.get('vllm:'+k+'_sum', 0)
        assert count == 256, (k, count)
        result[k] = total/count*1000
    return result


def warm_chart(root, summaries, details):
    warm=sorted([s for s in summaries if s['phase']=='warm'],key=lambda s:(s['concurrency'],s['prefetch']))
    if not warm: return
    keys=['http_to_retrieve_start_ms','retrieve_ms','retrieve_return_to_first_text_ms']
    bars=[]
    for s in warm:
        rows=[r for r in details[s['case']+'/warm'] if all(k in r for k in keys)]
        bars.append((s,[statistics.mean(r[k] for r in rows) for k in keys],len(rows)))
    maximum=max(sum(v) for _,v,_ in bars)*1.2
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="600">',
         '<rect width="1000" height="600" fill="white"/><g font-family="sans-serif" fill="#222">',
         '<text x="30" y="30" font-size="22">Warm TTFT: where the measured time goes</text>',
         '<text x="30" y="57" font-size="13">Qwen3-14B, DRAM8 / staging8 GiB. Same process after cold fill, 256 requests per phase.</text>']
    colors=['#0072b2','#d55e00','#009e73']
    labels=['HTTP start to retrieve start','retrieve API','retrieve return to first text']
    for i,(color,label) in enumerate(zip(colors,labels)):
        x=30+i*325
        svg += [f'<rect x="{x}" y="80" width="12" height="12" fill="{color}"/>',
                f'<text x="{x+18}" y="91" font-size="12">{label}</text>']
    for tick in range(5):
        y=460-tick*310/4
        svg += [f'<line x1="70" x2="970" y1="{y}" y2="{y}" stroke="#ddd"/>',
                f'<text x="20" y="{y+4}" font-size="12">{maximum*tick/4:.0f}</text>']
    svg.append('<text x="15" y="125" font-size="12">ms</text>')
    for i,(s,values,n) in enumerate(bars):
        x=120+i*210; bottom=460
        for val,color in zip(values,colors):
            h=val/maximum*310; bottom-=h
            svg.append(f'<rect x="{x}" y="{bottom}" width="115" height="{h}" fill="{color}"/>')
        svg += [f'<text x="{x+23}" y="{bottom-12}" font-size="16">{sum(values):.1f}</text>',
                f'<text x="{x+3}" y="490" font-size="14">C{s["concurrency"]} {"ON" if s["prefetch"] else "OFF"}</text>',
                f'<text x="{x+3}" y="511" font-size="12">n={n}</text>']
    svg += ['<text x="30" y="550" font-size="12">Host event boundaries, not pure compute/DMA or causal critical-path attribution.</text>',
            '<text x="30" y="573" font-size="12">One trial per condition; generation length and detailed tier placement may differ.</text></g></svg>']
    save_chart(root,'warm_ttft_intervals',svg)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    p.add_argument('--partial', action='store_true')
    a = p.parse_args(); root = a.folder.resolve()
    assert a.partial or read(root/'status.json')['status'] == 'completed'
    plan, records = read(root/'plan.json'), read(root/'requests.json')
    summaries, details, audit, configs, namespaces, natives = [], {}, {}, [], [], []
    for spec in plan['cases']:
        case = root/spec['name']
        if not (case/'initial_sample.json').exists():
            assert a.partial; continue
        initial = read(case/'initial_sample.json')
        assert initial['used_bytes'] == initial['cpu_hot_bytes'] == initial['daos_puts'] == 0
        config = yaml.safe_load((case/'config.yaml').read_text()); extra = config['extra_config']
        assert config['max_local_cpu_size'] == extra['daosgds.gpu_buffer_gb'] == 8
        assert extra.pop('daosgds.dram_prefetch') == spec['prefetch']
        namespaces.append(extra.pop('daosgds.object_namespace')); extra.pop('daosgds.root')
        configs.append(config); natives.append(read(case/'native_maps.json'))
        events = read_events(case)
        assert len({e['pid'] for e in events}) == 1, 'Worker process changed between phases'
        seen_ids = set()
        for phase_name in ('cold', 'warm'):
            phase_dir = case/phase_name
            if not (phase_dir/'status.json').exists() or read(phase_dir/'status.json')['status'] != 'completed':
                assert a.partial; continue
            calls = read(phase_dir/'replay_calls.json')
            assert len(calls) == 256 and not any('error' in c for c in calls)
            assert [(c['index'], c['prompt_sha256']) for c in calls] == [(r['index'], r['prompt_sha256']) for r in records]
            ids = {c['server_request_id'] for c in calls}
            assert len(ids) == 256 and not ids.intersection(seen_ids)
            seen_ids.update(ids)
            phase = read(phase_dir/'phase.json')
            start, end = phase['start_ns'], phase['end_ns']
            assert phase['concurrency'] == spec['concurrency']
            before, after = read(phase_dir/'initial_sample.json'), read(phase_dir/'final_sample.json')
            assert after['used_bytes'] == after['dram_mirror']['pending_bytes'] == after['dram_mirror']['errors'] == 0
            if phase_name == 'cold':
                assert before == initial
            else:
                assert before == read(case/'cold/final_sample.json')
                assert before['cpu_hot_bytes'] > 0 and before['daos_puts'] > 0
            request_events = select_events(events, calls)
            rows = join(calls, request_events)
            assert not any(r['other_failed_chunks'] for r in rows)
            scoped = [e for e in events if start <= e['time_ns'] <= end]
            candidates = sum(e['queried_chunks'] for e in request_events
                             if e['event'] == 'tier_lookup' and e['tier'] == 'dram')
            dram = sum(r['dram_lookup_chunks'] for r in rows)
            daos = sum(r['daos_lookup_chunks'] for r in rows)
            assert 0 <= dram+daos <= candidates
            pf0, pf1 = before['cpu_prefetch'], after['cpu_prefetch']
            pf = {k: v-pf0[k] for k, v in pf1.items()} if pf1 else None
            if pf:
                assert pf['copy_errors'] == pf['watermark_rejections'] == 0
                assert pf['staged_requests'] == sum(r.get('stage_outcome') == 'staged' for r in rows)
                assert pf['fallback_requests'] == sum(r.get('stage_outcome') == 'fallback' for r in rows)
            timings = ['ttft_ms', 'queue_ms', 'reserve_ms', 'copy_ms', 'cpu_get_ms', 'retrieve_ms',
                       'cpu_ready_to_retrieve_ms', 'serializer_wait_ms', 'http_to_retrieve_start_ms',
                       'retrieve_return_to_first_text_ms']
            total_tokens = sum(c['prompt_tokens'] for c in calls)
            s = dict(case=spec['name'], phase=phase_name, concurrency=spec['concurrency'],
                prefetch=spec['prefetch'], requests=256, elapsed_seconds=(end-start)/1e9,
                timings={k: stats(r.get(k) for r in rows) for k in timings},
                first16_ttft=stats(r['ttft_ms'] for r in rows if r['index'] < 16),
                rest240_ttft=stats(r['ttft_ms'] for r in rows if r['index'] >= 16),
                input_tokens=total_tokens, completion_tokens=sum(c['completion_tokens'] for c in calls),
                actual_cached_tokens=sum(c['cached_tokens'] for c in calls),
                computed_input_tokens=sum(r['computed_prompt_tokens'] for r in rows),
                computed_input_tokens_per_request_range=[min(r['computed_prompt_tokens'] for r in rows),
                                                         max(r['computed_prompt_tokens'] for r in rows)],
                capacity_recomputed_tokens=sum(r['capacity_recomputed_tokens'] for r in rows),
                dram_hit_chunks=dram, daos_hit_chunks=daos, queried_chunks=candidates,
                dram_hit_pct=100*dram/candidates, daos_hit_pct=100*daos/candidates,
                lookup_miss_pct=100*(candidates-dram-daos)/candidates,
                peak_staging_gib=max(e['used_bytes'] for e in scoped)/2**30,
                dram_initial_gib=before['cpu_hot_bytes']/2**30, dram_final_gib=after['cpu_hot_bytes']/2**30,
                prefetch_counters_delta=pf, puts_delta=after['daos_puts']-before['daos_puts'],
                mirror_counters_delta={k: v-before['dram_mirror'][k] for k, v in after['dram_mirror'].items()
                                       if k not in ('peak_pending_bytes',)},
                server_mean_ms=metric_means(phase_dir/'metrics_before.txt', phase_dir/'metrics_after.txt'))
            s['actual_cached_input_pct'] = 100*s['actual_cached_tokens']/total_tokens
            s['capacity_recompute_input_pct'] = 100*s['capacity_recomputed_tokens']/total_tokens
            fields = sorted(set().union(*(r.keys() for r in rows)))
            (phase_dir/'timing_by_request.json').write_text(json.dumps(rows, indent=2)+'\n')
            with (phase_dir/'timing_by_request.csv').open('w') as f:
                writer=csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
            summaries.append(s); details[spec['name']+'/'+phase_name] = rows
            audit[spec['name']+'/'+phase_name] = dict(input_validated=True, hit_attribution_validated=True,
                requests=256, started_drained=before['used_bytes'] == 0, finally_drained=True,
                same_worker_pid=next(iter({e['pid'] for e in events})))
    assert summaries, 'No completed phases'
    assert all(c == configs[0] for c in configs) and all(n == natives[0] for n in natives)
    assert len(namespaces) == len(set(namespaces))
    for name, digest in plan['source_sha256'].items():
        assert hashlib.sha256((root/'executed_sources'/name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((Path(__file__).resolve().parent/name).read_bytes()).hexdigest() == digest
    (root/'summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
    (root/'validation.json').write_text(json.dumps(dict(partial=a.partial, phases=audit,
        same_native_and_config=True, unique_namespaces=True, runtime_source_hashes_match=True), indent=2)+'\n')
    paired=[]
    for c in (8,16):
        for phase in ('cold','warm'):
            off,on=f'c{c}_off/{phase}',f'c{c}_on/{phase}'
            if off not in details or on not in details: continue
            pairs=list(zip(details[off],details[on],strict=True))
            same_cached=[(x,y) for x,y in pairs if x['cached_tokens']==y['cached_tokens']]
            same_tiers=[(x,y) for x,y in same_cached if x['dram_lookup_chunks']==y['dram_lookup_chunks']
                        and x['daos_lookup_chunks']==y['daos_lookup_chunks']]
            paired.append(dict(concurrency=c,phase=phase,
                same_cached_requests=len(same_cached),same_tier_hit_requests=len(same_tiers),
                same_tier_ttft_on_minus_off_ms=stats(y['ttft_ms']-x['ttft_ms'] for x,y in same_tiers)))
    (root/'paired_checks.json').write_text(json.dumps(paired,indent=2)+'\n')
    warm_chart(root,summaries,details)
    lines=['# Cold 1회 → 같은 프로세스에서 Warm 1회: DRAM 프리페치 비교','',
        'Qwen3-14B BF16, DRAM/staging 각각 8GiB, 청크128, DAOS object. 각 OFF/ON·동시 요청 조건마다 새 '
        '프로세스·빈 DRAM·독립 DAOS namespace로 시작했다. 동일한 DiscoveryBench 기록 입력 256개를 cold로 실행한 뒤 '
        '저장/복사가 끝나고 staging이 비었음을 확인하고, 프로세스와 DRAM/DAOS 캐시를 유지한 채 같은 256개를 warm으로 재생했다. '
        '원래 Python 도구를 실행하는 full agentic 벤치마크는 아니다.','',
        'rolling 투입: 처음 N개를 함께 보내고 이후 하나가 끝날 때마다 하나를 보충한다. DAOS 프리페치, 새 KV 저장, '
        '비동기 DRAM 저장·읽기 승격, 실제 용량 제한은 모두 유지했다. DRAM 프리페치만 OFF/ON으로 바꿨다. '
        '최대 생성 2048토큰과 기존 EOS/Observation 종료 조건을 유지했으므로 생성량과 정확한 도착 시점은 달라질 수 있다.','',
        '**warm도 DRAM all-hit 또는 ON/OFF 동일 hit를 보장하지 않는다.** 아래 실제 hit와 계산량을 확인해야 한다. '
        '단계별 1회이며 반복 실험의 통계적 우열을 주장하지 않는다. 서버/OS 캐시는 flush하지 않았다.','',
        '| 동시 요청 | 프리페치 | 단계 | 평균 TTFT ms | p95 TTFT ms | 전체 s | DRAM hit % | DAOS hit % | 새로 계산한 입력 토큰 | 생성 토큰 |',
        '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        t=s['timings']['ttft_ms']
        lines.append(f'| {s["concurrency"]} | {"ON" if s["prefetch"] else "OFF"} | {s["phase"]} | '
            f'{t["mean"]:.2f} | {t["p95"]:.2f} | {s["elapsed_seconds"]:.2f} | {s["dram_hit_pct"]:.2f} | '
            f'{s["daos_hit_pct"]:.2f} | {s["computed_input_tokens"]} | {s["completion_tokens"]} |')
    lines += ['', 'hit 비율의 분모는 최초 DRAM tier에서 조회한 전체 후보 청크 수이며 DRAM/DAOS에 동일 분모를 사용했다. '
        '새로 계산한 입력은 HTTP prompt_tokens − cached_tokens로, cold miss와 청크 경계 등을 포함한다. '
        'staging 부족으로 잃은 prefix 재계산과는 별도다. TTFT는 클라이언트의 첫 텍스트 수신 기준이다.', '',
        '## 복사 대기·retrieve·메모리', '',
        '| 조건 | 단계 | 복사 큐 평균 ms | 복사 함수 평균 ms | retrieve 평균 ms | 최대 staging GiB | DRAM 시작 GiB | 공간 부족 fallback | DAOS 실패 재계산 토큰 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    def fmt(value): return '—' if value is None else f'{value:.3f}'
    for s in summaries:
        t=s['timings']; pf=s['prefetch_counters_delta'] or {}
        lines.append(f'| {s["case"]} | {s["phase"]} | {fmt(t["queue_ms"]["mean"])} | '
            f'{fmt(t["copy_ms"]["mean"])} | {fmt(t["retrieve_ms"]["mean"])} | {s["peak_staging_gib"]:.3f} | '
            f'{s["dram_initial_gib"]:.3f} | {pf.get("fallback_requests",0)} | {s["capacity_recomputed_tokens"]} |')
    lines += ['', '복사 함수는 호스트 경과 시간이며 enqueue와 기존 stream 대기를 포함한다. 각 열의 대상 요청 수가 '
        '달라질 수 있으므로 합계를 TTFT로 해석하면 안 된다. 큐·복사 계측 때문에 GPU 동기화를 추가하지 않았다.', '',
        '## ON/OFF 실제 재사용량 일치 여부', '',
        '| 동시 요청 | 단계 | 동일 재사용 토큰 요청 /256 | DRAM·DAOS hit도 같은 요청 /256 |',
        '|---:|---|---:|---:|']
    for r in paired:
        lines.append(f'| {r["concurrency"]} | {r["phase"]} | {r["same_cached_requests"]} | {r["same_tier_hit_requests"]} |')
    lines += ['', '## Warm TTFT가 어느 구간에서 달라졌는가', '',
        '아래는 같은 HTTP 요청의 시작, retrieve API 시작/반환, 첫 텍스트 수신 경계를 연결한 것이다. '
        'retrieve 전 구간은 서버 처리·lookup·비동기 읽기·스케줄 대기를 포함하고, 이후 구간도 모델 계산만이 아니라 '
        '스케줄 및 응답 전달을 포함한다. 인과적인 순수 계산/통신 시간 분해는 아니다.', '',
        '| 조건 | 요청 시작→retrieve 시작 ms | retrieve ms | retrieve 반환→첫 텍스트 ms | TTFT ms | 대상 요청 수 |',
        '|---|---:|---:|---:|---:|---:|']
    for s in summaries:
        if s['phase'] != 'warm': continue
        rows=[r for r in details[s['case']+'/warm'] if all(k in r for k in
            ['http_to_retrieve_start_ms','retrieve_ms','retrieve_return_to_first_text_ms'])]
        if not rows: continue
        means=[statistics.mean(r[k] for r in rows) for k in
               ['http_to_retrieve_start_ms','retrieve_ms','retrieve_return_to_first_text_ms','ttft_ms']]
        lines.append('| '+s['case']+' | '+' | '.join(f'{m:.3f}' for m in means)+f' | {len(rows)} |')
    lines += ['', '[현재 코드의 준비 완료·스케줄링 경로 메모](CONTROL_PATH_NOTES_KO.md)', '']
    if any(s['phase']=='warm' for s in summaries):
        lines += ['![Warm TTFT의 구간별 평균](warm_ttft_intervals.png)', '']
    lines += ['', '서버 지표는 각 단계 앞뒤의 누적 histogram 차이로 계산해 warm 값에 cold가 섞이지 않게 했다. '
        '초기 모델 로딩과 단계 사이 drain은 전체 시간에서 제외했다. cold는 반드시 모든 요청이 miss라는 뜻이 아니라 '
        '출발 상태가 비어 있다는 뜻이다.','',
        '[전체 지표](summary.json) · [요청별 일치 검사](paired_checks.json) · [검증](validation.json)', '']
    if a.partial: lines.insert(2, '**부분 결과: 아직 완료되지 않은 단계는 제외했다.**\n')
    (root/'RESULT_KO.md').write_text('\n'.join(lines))
    print(json.dumps([dict(case=s['case'],phase=s['phase'],ttft=s['timings']['ttft_ms']['mean'],
        dram=s['dram_hit_pct'],daos=s['daos_hit_pct'],computed=s['computed_input_tokens']) for s in summaries],indent=2))


if __name__ == '__main__':
    main()
