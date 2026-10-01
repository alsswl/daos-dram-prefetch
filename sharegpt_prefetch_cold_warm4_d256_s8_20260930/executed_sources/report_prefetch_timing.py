#!/usr/bin/env python3
"""Join HTTP, executor timing, prefetch readiness and retrieve host timestamps."""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import statistics

import yaml

from report_capacity_matrix import attribute_recomputation, save_chart
from staging_mixed_pressure import read_events


def read(path):
    return json.loads(path.read_text())


def stats(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return dict(n=0, mean=None, p50=None, p95=None, maximum=None)
    def q(p):
        x = (len(values)-1)*p
        lo = int(x)
        return values[lo] + (values[min(lo+1, len(values)-1)]-values[lo])*(x-lo)
    return dict(n=len(values), mean=statistics.mean(values), p50=q(.5), p95=q(.95), maximum=values[-1])


def join(calls, events):
    ids = {c['server_request_id'] for c in calls}
    by_id = defaultdict(list)
    for e in events:
        rid = e.get('request_id')
        if not rid:
            continue
        rid = rid if rid in ids else rid.rsplit('-', 1)[0]
        assert rid in ids, f'Unmapped {rid}'
        by_id[rid].append(e)
    attributed = {r['index']: r for r in attribute_recomputation(calls, events, 128)}
    rows = []
    for c in calls:
        row = dict(attributed[c['index']], completion_tokens=c['completion_tokens'])
        assert not row['attribution_issues'] and not row['unattributed_shortfall_tokens']
        group = by_id[c['server_request_id']]
        def one(name):
            matches = [e for e in group if e['event'] == name]
            assert len(matches) <= 1, (c['index'], name, len(matches))
            return matches[0] if matches else None
        lookup = [e for e in group if e['event'] == 'tier_lookup']
        if lookup:
            row['http_to_first_lookup_result_ms'] = (min(e['time_ns'] for e in lookup)-c['start_ns'])/1e6
        start, ready = one('cpu_get_start'), one('cpu_get_ready')
        rs, re = one('retrieve_start'), one('retrieve_return')
        timing = one('cpu_prefetch_timing')
        serializers = [e for e in group if e['event'] == 'serializer_acquired']
        if serializers:
            # Mixed-tier requests can acquire once for CPU and once for DAOS.
            # Their waits may overlap, so do not sum them as wall time.
            row['serializer_wait_ms'] = max(e['wait_ms'] for e in serializers)
        if start and ready:
            row['cpu_get_ms'] = (ready['monotonic_ns']-start['monotonic_ns'])/1e6
            row['cpu_staged_chunks'] = ready['staged_chunks']
        if ready and rs:
            row['cpu_ready_to_retrieve_ms'] = (rs['monotonic_ns']-ready['monotonic_ns'])/1e6
        if rs and re:
            row['retrieve_ms'] = (re['monotonic_ns']-rs['monotonic_ns'])/1e6
            row['http_to_retrieve_start_ms'] = (rs['time_ns']-c['start_ns'])/1e6
            row['retrieve_return_to_first_text_ms'] = c['ttft_ms']-(re['time_ns']-c['start_ns'])/1e6
        if timing:
            row['stage_outcome'] = timing['outcome']
            row['stage_bytes'] = timing['bytes']
            for label, a, b in [('queue_ms', 'queued_ns', 'worker_start_ns'),
                                ('reserve_ms', 'reserve_start_ns', 'reserve_end_ns'),
                                ('copy_ms', 'copy_start_ns', 'copy_end_ns'),
                                ('worker_ms', 'worker_start_ns', 'worker_end_ns')]:
                if a in timing and b in timing:
                    row[label] = (timing[b]-timing[a])/1e6
                    assert row[label] >= 0
            if ready:
                row['worker_end_to_ready_ms'] = (ready['monotonic_ns']-timing['worker_end_ns'])/1e6
            if rs:
                row['worker_end_to_retrieve_ms'] = (rs['monotonic_ns']-timing['worker_end_ns'])/1e6
        rows.append(row)
    return rows


def chart(folder, summaries):
    """Show exact first16/tail contributions, not purported critical-path spans."""
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="860">',
           '<rect width="1200" height="860" fill="white"/>',
           '<g font-family="sans-serif" fill="#222" font-size="13">',
           '<text x="35" y="32" font-size="23">Prefetch diagnosis: identical rolling arrival rule, D8 / S8</text>',
           '<text x="35" y="60">Top: whole-run mean TTFT split into first16 contribution (orange) + remaining240 (blue).</text>',
           '<text x="35" y="82">Contributions are weighted by 16/256 and 240/256. They sum exactly to mean TTFT.</text>']
    ordered = sorted(summaries, key=lambda s: (s['concurrency'], s['repeat'], s['prefetch']))
    maximum = max(s['timings']['ttft_ms']['mean'] for s in ordered)*1.2
    for i in range(5):
        y=365-i*240/4
        svg += [f'<line x1="75" x2="1160" y1="{y}" y2="{y}" stroke="#ddd"/>',
                f'<text x="25" y="{y+4}">{maximum*i/4:.0f}</text>']
    for i, s in enumerate(ordered):
        x=100+i*130
        tail=s['rest240']['mean']*240/256
        first=s['first16']['mean']*16/256
        h1,h2=tail/maximum*240,first/maximum*240
        svg += [f'<rect x="{x}" y="{365-h1}" width="75" height="{h1}" fill="#0072b2"/>',
                f'<rect x="{x}" y="{365-h1-h2}" width="75" height="{h2}" fill="#d55e00"/>',
                f'<text x="{x}" y="{355-h1-h2}">{tail+first:.1f}</text>',
                f'<text x="{x-8}" y="390">C{s["concurrency"]} R{s["repeat"]} {"ON" if s["prefetch"] else "OFF"}</text>']
    svg += ['<text x="35" y="448" font-size="19">ON only: mean executor queue versus host copy interval (ms)</text>',
            '<text x="35" y="473">Blue=queue, orange=copy. Copy includes enqueue + stream wait, not pure DMA.</text>']
    ons = [s for s in ordered if s['prefetch']]
    if ons:
        maximum=max(s['timings'][k]['mean'] for s in ons for k in ['queue_ms','copy_ms'])*1.25
        for i in range(5):
            y=760-i*230/4
            svg += [f'<line x1="75" x2="1160" y1="{y}" y2="{y}" stroke="#ddd"/>',
                    f'<text x="25" y="{y+4}">{maximum*i/4:.1f}</text>']
        for i,s in enumerate(ons):
            for j,k in enumerate(['queue_ms','copy_ms']):
                x=150+i*255+j*75; value=s['timings'][k]['mean']; height=value/maximum*230
                color=['#0072b2','#d55e00'][j]
                svg += [f'<rect x="{x}" y="{760-height}" width="60" height="{height}" fill="{color}"/>',
                        f'<text x="{x}" y="{750-height}">{value:.2f}</text>']
            svg.append(f'<text x="{150+i*255}" y="790">C{s["concurrency"]} R{s["repeat"]} ON</text>')
    svg += ['<text x="35" y="835">Two repetitions; fresh DRAM and namespace each case. First16/tail is a diagnostic split, not controlled cold/warm cohorts.</text>',
            '</g></svg>']
    save_chart(folder, 'timing_diagnosis', svg)


def historical_audit(folder):
    """Keep previous C16 rolling results separate from the new experiment."""
    previous = folder.parent/'prefetch_d8_s8_c16_rolling_repeat3_20260927'
    if not (previous/'plan.json').exists():
        return []
    summaries = []
    for spec in read(previous/'plan.json')['cases']:
        case = previous/spec['name']
        assert read(case/'status.json')['status'] == 'completed'
        calls = read(case/'replay_calls.json')
        assert len(calls) == 256
        summaries.append(dict(name=spec['name'], repeat=spec['repeat'], prefetch=spec['prefetch'],
            source=str(case/'replay_calls.json'),
            source_sha256=hashlib.sha256((case/'replay_calls.json').read_bytes()).hexdigest(),
            mean_ttft_ms=statistics.mean(c['ttft_ms'] for c in calls),
            first16_ms=statistics.mean(c['ttft_ms'] for c in calls if c['index'] < 16),
            rest240_ms=statistics.mean(c['ttft_ms'] for c in calls if c['index'] >= 16),
            first16_computed_tokens=sum(c['prompt_tokens']-c['cached_tokens'] for c in calls if c['index'] < 16),
            rest240_computed_tokens=sum(c['prompt_tokens']-c['cached_tokens'] for c in calls if c['index'] >= 16)))
    (folder/'previous_c16_rolling_audit.json').write_text(json.dumps(summaries, indent=2)+'\n')
    return summaries


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    p.add_argument('--partial', action='store_true')
    a = p.parse_args()
    folder = a.folder.resolve(); plan = read(folder/'plan.json')
    assert a.partial or read(folder/'status.json')['status'] == 'completed'
    records = read(folder/'requests.json')
    summaries, details, configs, natives, namespaces = [], {}, [], [], []
    for spec in plan['cases']:
        case = folder/spec['name']
        if not (case/'status.json').exists() or read(case/'status.json')['status'] != 'completed':
            assert a.partial; continue
        calls = read(case/'replay_calls.json')
        assert len(calls) == 256 and not any('error' in c for c in calls)
        assert [(c['index'], c['prompt_sha256']) for c in calls] == [(r['index'], r['prompt_sha256']) for r in records]
        initial, final = read(case/'initial_sample.json'), read(case/'final_sample.json')
        assert initial['used_bytes'] == initial['cpu_hot_bytes'] == initial['daos_puts'] == 0
        assert final['used_bytes'] == final['dram_mirror']['pending_bytes'] == final['dram_mirror']['errors'] == 0
        config = yaml.safe_load((case/'config.yaml').read_text())
        extra = config['extra_config']
        assert config['max_local_cpu_size'] == extra['daosgds.gpu_buffer_gb'] == 8
        assert extra.pop('daosgds.dram_prefetch') == spec['prefetch']
        namespaces.append(extra.pop('daosgds.object_namespace')); extra.pop('daosgds.root')
        configs.append(config); natives.append(read(case/'native_maps.json'))
        events = read_events(case)
        assert any(e['event'] == 'prefetch_timing_enabled' for e in events)
        rows = join(calls, events)
        pf = final['cpu_prefetch']
        if spec['prefetch']:
            assert len([r for r in rows if r.get('stage_outcome') == 'staged']) == pf['staged_requests']
            assert len([r for r in rows if r.get('stage_outcome') == 'fallback']) == pf['fallback_requests']
            assert pf['copy_errors'] == pf['watermark_rejections'] == 0
        fields = sorted(set().union(*(r.keys() for r in rows)))
        (case/'timing_by_request.json').write_text(json.dumps(rows, indent=2)+'\n')
        with (case/'timing_by_request.csv').open('w') as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        keys = ['ttft_ms', 'queue_ms', 'reserve_ms', 'copy_ms', 'worker_ms', 'cpu_get_ms',
                'worker_end_to_ready_ms', 'cpu_ready_to_retrieve_ms', 'worker_end_to_retrieve_ms',
                'retrieve_ms', 'serializer_wait_ms', 'http_to_first_lookup_result_ms',
                'http_to_retrieve_start_ms', 'retrieve_return_to_first_text_ms']
        phase = read(case/'phases.json')[0]
        summary = dict(**spec, elapsed_seconds=(phase['end_ns']-phase['start_ns'])/1e9,
            timings={k: stats(r.get(k) for r in rows) for k in keys},
            first16_timings={k: stats(r.get(k) for r in rows if r['index'] < 16) for k in keys},
            rest240_timings={k: stats(r.get(k) for r in rows if r['index'] >= 16) for k in keys},
            first16=stats(r['ttft_ms'] for r in rows if r['index'] < 16),
            rest240=stats(r['ttft_ms'] for r in rows if r['index'] >= 16),
            completion_tokens=sum(c['completion_tokens'] for c in calls),
            computed_prompt_tokens=sum(r['computed_prompt_tokens'] for r in rows),
            first16_computed_tokens=sum(r['computed_prompt_tokens'] for r in rows if r['index'] < 16),
            first16_cached_tokens=sum(r['cached_tokens'] for r in rows if r['index'] < 16),
            capacity_recomputed_tokens=sum(r['capacity_recomputed_tokens'] for r in rows),
            peak_staging_gib=max(e['used_bytes'] for e in events)/2**30,
            prefetch_counters=pf,
            mirror_counters=final['dram_mirror'],
            dram_hit_chunks=sum(r['dram_lookup_chunks'] for r in rows),
            daos_hit_chunks=sum(r['daos_lookup_chunks'] for r in rows),
            retrieve_before_cpu_ready=sum(r.get('cpu_ready_to_retrieve_ms', 0) < 0 for r in rows))
        copied = [r for r in rows if r.get('stage_outcome') == 'staged']
        summary['copied_mib_per_request'] = stats(r['stage_bytes']/2**20 for r in copied)
        summary['copy_host_effective_gbps'] = (sum(r['stage_bytes'] for r in copied)/
            sum(r['copy_ms'] for r in copied)/1e6) if copied else None
        metrics = {m.group(1): float(m.group(2)) for m in re.finditer(
            r'^(vllm:\w+)\{[^\n]*\} ([\d.e+-]+)$', (case/'metrics.txt').read_text(), re.M)}
        summary['server_histogram_mean_ms'] = {
            k: metrics['vllm:'+k+'_sum']/metrics['vllm:'+k+'_count']*1000
            for k in ['time_to_first_token_seconds', 'request_queue_time_seconds', 'request_prefill_time_seconds']}
        summaries.append(summary); details[spec['name']] = rows
    assert configs and all(c == configs[0] for c in configs)
    assert all(n == natives[0] for n in natives) and len(namespaces) == len(set(namespaces))
    for n, expected in plan['source_sha256'].items():
        assert hashlib.sha256((folder/'executed_sources'/n).read_bytes()).hexdigest() == expected
        assert hashlib.sha256((Path(__file__).resolve().parent/n).read_bytes()).hexdigest() == expected
    (folder/'timing_validation.json').write_text(json.dumps(dict(
        partial=a.partial, completed_cases=[s['name'] for s in summaries],
        completed_requests=256*len(summaries), same_inputs=True,
        same_config_except_prefetch_and_namespace=True, same_native_maps=True,
        unique_namespaces=True, initially_empty=True, finally_drained=True,
        runtime_source_hashes_match=True, request_hit_attribution_validated=True,
        report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+'\n')
    (folder/'timing_summary.json').write_text(json.dumps(summaries, indent=2)+'\n')
    paired = []
    for repeat in (1, 2):
        for c in (8, 16):
            off, on = f'r{repeat}_c{c}_off', f'r{repeat}_c{c}_on'
            if off not in details or on not in details:
                continue
            off_rows, on_rows = details[off], details[on]
            for label, keep in [('all', lambda x, y: True),
                                ('same_cached_tokens', lambda x, y: x['cached_tokens'] == y['cached_tokens']),
                                ('retrieve_both_same_cached_tokens', lambda x, y: 'retrieve_ms' in x and 'retrieve_ms' in y and x['cached_tokens'] == y['cached_tokens']),
                                ('index_ge_16_same_cached_tokens', lambda x, y: x['index'] >= 16 and x['cached_tokens'] == y['cached_tokens'])]:
                pairs = [(x, y) for x, y in zip(off_rows, on_rows, strict=True) if keep(x, y)]
                paired.append(dict(repeat=repeat, concurrency=c, subset=label, count=len(pairs),
                    on_minus_off={k: stats(y[k]-x[k] for x, y in pairs if k in x and k in y)
                        for k in ['ttft_ms', 'retrieve_ms', 'http_to_retrieve_start_ms',
                                  'retrieve_return_to_first_text_ms']}))
    (folder/'paired_diagnostics.json').write_text(json.dumps(paired, indent=2)+'\n')
    prefixes = [dict(shorter=i, longer=j, shared_chars=len(x['prompt']))
        for i, x in enumerate(records[:16]) for j, y in enumerate(records[:16])
        if i < j and y['prompt'].startswith(x['prompt'])]
    prefix_indices = {r[k] for r in prefixes for k in ('shorter', 'longer')}
    evidence = dict(prefix_pairs_in_first16=prefixes, cases={})
    for name, rows in details.items():
        origin=min(r['start_ns'] for r in rows)
        evidence['cases'][name]=[dict(index=r['index'], cached_tokens=r['cached_tokens'],
            computed_prompt_tokens=r['computed_prompt_tokens'], ttft_ms=r['ttft_ms'],
            first_lookup_result_after_first_http_ms=(r['start_ns']-origin)/1e6+r['http_to_first_lookup_result_ms'])
            for r in rows if r['index'] in prefix_indices]
    (folder/'shared_prefix_evidence.json').write_text(json.dumps(evidence, indent=2)+'\n')
    chart(folder, summaries)
    header = ['# 프리페치 대기열·복사·retrieve 진단', '',
        'Qwen3-14B, DRAM 8GiB / staging 8GiB, chunk128, object 경로. 같은 기록 입력 256개를 rolling 방식으로 투입했다. '
        '각 조건 새 프로세스·빈 DRAM·독립 DAOS namespace. ON/OFF는 DRAM 프리페치만 변경한다. '
        'DAOS 프리페치와 DRAM 비동기 저장/승격은 유지한다. Python 도구 실행 없는 기록 재생이다.', '',
        '조건별 2회, 순서를 반대로 배치했다. 생성 EOS를 유지하여 생성량과 이후 도착 시점·캐시 hit가 완전히 같지는 않다. '
        '첫 16개/나머지 240개는 초기 구간 영향을 보기 위한 보조 분석이며, 나머지가 동일한 warm 상태라는 뜻은 아니다.', '',
        '## 전체 결과', '',
        '| 실행 | 평균 TTFT ms | 첫 16개 ms | 나머지 240개 ms | 전체 s | 생성 토큰 | 공간 부족 fallback | 최대 staging GiB |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        header.append(f'| {s["name"]} | {s["timings"]["ttft_ms"]["mean"]:.2f} | {s["first16"]["mean"]:.2f} | '
                      f'{s["rest240"]["mean"]:.2f} | {s["elapsed_seconds"]:.2f} | {s["completion_tokens"]} | '
                      f'{(s["prefetch_counters"] or {}).get("fallback_requests", 0)} | {s["peak_staging_gib"]:.3f} |')
    header += ['', '## 요청별 시간 계측', '',
        '각 칸은 평균 / p95(ms). 해당 이벤트가 있는 요청만의 통계이므로 서로 다른 열을 더해 TTFT로 해석하면 안 된다.', '',
        '| 실행 | executor 대기 | reserve | H2D copy 호출 | CPU 준비→retrieve | retrieve API |',
        '|---|---:|---:|---:|---:|---:|']
    def fmt(s, k):
        t=s['timings'][k]
        return '—' if not t['n'] else f'{t["mean"]:.3f} / {t["p95"]:.3f}'
    for s in summaries:
        header.append('| '+s['name']+' | '+' | '.join(fmt(s, k) for k in
            ['queue_ms', 'reserve_ms', 'copy_ms', 'cpu_ready_to_retrieve_ms', 'retrieve_ms'])+' |')
    header += ['', '## 서버 측 보조 지표', '',
        '서버 histogram 합계/횟수. 서버 큐 시간은 DRAM 복사 executor 큐와 다른 지표다. '
        '서버 prefill 시간도 순수 GPU 연산 시간으로 해석하지 않는다.', '',
        '| 실행 | 서버 TTFT ms | 서버 큐 ms | 서버 prefill ms | 미재사용 입력 토큰 |',
        '|---|---:|---:|---:|---:|']
    for s in summaries:
        m = s['server_histogram_mean_ms']
        header.append(f'| {s["name"]} | {m["time_to_first_token_seconds"]:.2f} | '
            f'{m["request_queue_time_seconds"]:.2f} | {m["request_prefill_time_seconds"]:.2f} | '
            f'{s["computed_prompt_tokens"]} |')
    header += ['', '## 초기 16개 요청의 실제 캐시 재사용량', '',
        '모든 실행은 빈 DRAM에서 시작했지만, 최초 동시 요청들 사이에서도 먼저 처리된 요청이 저장한 캐시를 '
        '후속 lookup이 재사용한다. 따라서 시작 상태가 같다는 것과 각 요청의 hit가 같다는 것은 다르다. '
        '동일한 클라이언트 입력 목록도 실제 서버 도착·lookup·store 완료 순서를 고정하지는 않는다.', '',
        '| 실행 | 첫 16개 재사용 토큰 | 첫 16개 미재사용 토큰 | 첫 16개 TTFT ms |',
        '|---|---:|---:|---:|']
    for s in summaries:
        header.append(f'| {s["name"]} | {s["first16_cached_tokens"]} | '
            f'{s["first16_computed_tokens"]} | {s["first16"]["mean"]:.2f} |')
    header += ['', '미재사용 토큰에는 정상 cold miss가 포함된다. staging 부족 재계산 토큰과 혼동하면 안 된다. '
        '이 차이는 초기 계산량 차이의 직접적인 근거지만, hit 차이가 생긴 세부 스케줄링 원인을 전부 규명한 것은 아니다.', '']
    if 'r1_c16_on' in details and 'r2_c16_on' in details:
        header += ['### 공유 prefix와 처리 순서의 실제 사례', '',
            '처음 16개 입력에는 같은 문제의 여러 대화 턴이 함께 들어 있다. 예를 들어 요청 0과 5의 '
            '프롬프트 문자열 전체는 더 긴 요청 11과 12의 앞부분이다. 이 기록 재생기는 원래 에이전트의 '
            '턴 간 의존성을 재현하지 않고 각 프롬프트를 독립 요청으로 보낸다. 동시 16개에서는 이 네 요청이 '
            '초기 투입에 모두 포함된다. 동일한 입력 목록·도착 규칙이 서버의 실제 lookup/store 순서를 고정하지는 않는다.', '',
            '| 요청 index | 16개 ON 반복1 캐시 hit 토큰 | 16개 ON 반복2 캐시 hit 토큰 |',
            '|---:|---:|---:|']
        for i in [0, 5, 11, 12]:
            header.append(f'| {i} | {details["r1_c16_on"][i]["cached_tokens"]} | '
                f'{details["r2_c16_on"][i]["cached_tokens"]} |')
        e1 = {r['index']: r for r in evidence['cases']['r1_c16_on']}
        header += ['', f'반복1에서 요청 11의 첫 lookup 결과는 초기 HTTP 투입 약 '
            f'{e1[11]["first_lookup_result_after_first_http_ms"]:.0f}ms 후였고 캐시 hit={e1[11]["cached_tokens"]:,}토큰이었다. '
            f'요청 0은 약 {e1[0]["first_lookup_result_after_first_http_ms"]:.0f}ms 후였으며 '
            f'캐시 hit={e1[0]["cached_tokens"]:,}토큰이었다. 위 표의 반복 간 차이는 복사 대기열 길이만으로 '
            '비교할 수 없을 만큼 입력 계산량이 달라졌다는 직접적인 근거다. 어떤 요청이 각 키를 최초 저장했는지까지 '
            '키 단위로 추적한 것은 아니므로 저장 주체의 인과 연결은 확정하지 않는다.', '',
            '[공유 prefix·요청별 hit·lookup 시점 근거](shared_prefix_evidence.json)', '']
    history = historical_audit(folder)
    if history:
        header += ['', '## 이전 16개 rolling 실험의 평균이 상쇄된 위치', '',
            '아래는 이번 재실험이 아니라 이전 3회 실행을 원본 HTTP 기록으로 다시 계산한 값이다. '
            '계측을 추가한 새 결과와 혼합 평균하지 않았다.', '',
            '| 반복 | OFF 전체 ms | ON 전체 ms | OFF 첫 16개 ms | ON 첫 16개 ms | OFF 나머지 ms | ON 나머지 ms |',
            '|---:|---:|---:|---:|---:|---:|---:|']
        for rep in (1, 2, 3):
            off=next(s for s in history if s['repeat'] == rep and not s['prefetch'])
            on=next(s for s in history if s['repeat'] == rep and s['prefetch'])
            header.append(f'| {rep} | '+ ' | '.join(f'{s[k]:.2f}' for k in
                ['mean_ttft_ms','first16_ms','rest240_ms'] for s in [off,on])+' |')
        header += ['', '3회 모두 나머지 240개 구간에서는 ON의 평균 TTFT가 낮았지만, 첫 16개 구간의 지연이 '
            '전체 평균의 이득을 상쇄했다. 이는 **어느 구간에서 상쇄됐는지**를 보여주며, 초기 지연의 세부 원인이나 '
            '순수 프리페치 인과효과를 확정하는 분석은 아니다. 첫 16개에서도 동시 요청 간 캐시 재사용이 발생하고 그 양이 달랐다.', '']
    header += ['', '## 해석의 경계', '',
        '- executor 대기: submit 직전→worker 함수 진입. 제출/dispatch 오버헤드도 포함한다.',
        '- copy: 기존 H2D 복사 함수의 호스트 경과 시간. enqueue와 stream.synchronize 대기를 포함하며 순수 DMA 시간이 아니다.',
        '- CPU 준비→retrieve: 이미 준비된 데이터가 실제 retrieve API 진입까지 보유된 시간. 양수여도 그만큼 성능이 개선됐다는 뜻은 아니다. '
        '프리페치 완료가 스케줄러의 실행 허용 조건일 수 있어 retrieve 시점 자체가 뒤로 이동할 수 있다.',
        '- retrieve: API 시작→반환. 모델 계산 전체가 아니다. TTFT는 클라이언트에서 첫 텍스트를 받은 시점이다.',
        '- 사건 기록 오버헤드가 존재한다. 추가 GPU 동기화·복사·스케줄링 정책 변경은 없다.',
        '- fallback은 공간 부족 시 기존 DRAM 객체를 반환하는 것으로, cache miss/재계산과 다르다.',
        '- 전체 평균에서 차이가 작다고 프리페치 자체가 동작하지 않았다고 결론 내리지 않는다.', '',
        '[요약 원본](timing_summary.json). 각 실행의 timing_by_request.csv/json에 요청별 근거를 보존했다.', '']
    header += ['![TTFT 초기 구간 기여도와 executor 대기/복사](timing_diagnosis.png)', '']
    if a.partial: header.insert(2, '**부분 결과: 완료된 실행만 표시. 최종 비교가 아님.**\n')
    (folder/'TIMING_RESULT_KO.md').write_text('\n'.join(header))
    print(json.dumps([dict(name=s['name'], ttft=s['timings']['ttft_ms']['mean'],
        queue=s['timings']['queue_ms'], copy=s['timings']['copy_ms'],
        retrieve=s['timings']['retrieve_ms'], fallback=(s['prefetch_counters'] or {}).get('fallback_requests',0))
        for s in summaries], indent=2))


if __name__ == '__main__':
    main()
