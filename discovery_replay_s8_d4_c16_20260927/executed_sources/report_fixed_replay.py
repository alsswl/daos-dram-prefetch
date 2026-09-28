#!/usr/bin/env python3
"""Audit and graph selected fixed DiscoveryBench model-request replay pairs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics

import yaml

from analyze_discovery_staging import summarize_case, plot
from staging_mixed_pressure import read_events


def report(folder):
    plan = json.loads((folder/'plan.json').read_text())
    records = json.loads((folder/'requests.json').read_text())
    concurrencies = list(dict.fromkeys(s['concurrency'] for s in plan['cases']))
    concurrency_text = '·'.join(str(c) for c in concurrencies)
    case_order = ' → '.join(s['name'] for s in plan['cases'])
    results, audit, summaries = {}, {}, []
    for spec in plan['cases']:
        case = folder/spec['name']
        status = json.loads((case/'status.json').read_text())
        if status['status'] != 'completed':
            raise ValueError(f'Incomplete case: {case}')
        calls = json.loads((case/'replay_calls.json').read_text())
        expected = [(r['index'], r['prompt_sha256']) for r in records]
        assert [(r['index'], r['prompt_sha256']) for r in calls] == expected
        assert len(calls) == plan['requests_per_case'] and not any('error' in r for r in calls)
        initial = json.loads((case/'initial_sample.json').read_text())
        assert initial['used_bytes'] == initial['cpu_hot_bytes'] == initial['daos_puts'] == 0
        stats, bins, phases = summarize_case(case, plan['cpu_gib'])
        assert stats['staging_gib'] == plan['staging_gib'], 'Plan/config staging mismatch'
        final = json.loads((case/'final_sample.json').read_text())
        stats.update(concurrency=spec['concurrency'], prefetch=spec['prefetch'],
            actual_cached_tokens=sum(r['cached_tokens'] for r in calls),
            total_prompt_tokens=sum(r['prompt_tokens'] for r in calls),
            total_completion_tokens=sum(r['completion_tokens'] for r in calls),
            mean_latency_ms=statistics.mean(r['elapsed_seconds']*1000 for r in calls),
            prefetch_counters=final.get('cpu_prefetch'),
            daos_allocation_failures=final['daos_alloc_fail'],
            mirror_final=final.get('dram_mirror'))
        # The backend alloc_fail counter includes store allocation, not only GET.
        stats['backend_allocation_failures'] = final['daos_alloc_fail']
        log = (case/'server.log').read_text()
        stats['store_shortfalls'] = [dict(request_id=r, stored_tokens=int(n), requested_tokens=int(total))
            for r,n,total in re.findall(r'\[req_id=([^\]]+)\] Stored (\d+) out of total (\d+) tokens',log)
            if int(n) < int(total)]
        stats['store_admission_warning_chunks'] = [int(n) for n in re.findall(
            r'Local cpu memory under pressure so choosing to store only\s+(\d+) total chunks',log)]
        failures, sample = [], None
        for event in read_events(case):
            if event['event'] == 'occupancy_sample':
                sample = event
            if event['event'] in ('allocate','batched_allocate') and event['failed']:
                failures.append(dict(failure=event, previous_occupancy_sample=sample))
        (case/'allocation_failure_evidence.json').write_text(json.dumps(failures,indent=2)+'\n')
        stats['actual_cached_token_fraction'] = stats['actual_cached_tokens']/stats['total_prompt_tokens']
        results[spec['name']] = (stats, bins, phases)
        summaries.append(stats)
        audit[spec['name']] = dict(input_sequence_equal=True, started_empty=True,
            native_maps=json.loads((case/'native_maps.json').read_text()))
        (case/'analysis.json').write_text(json.dumps(stats, indent=2)+'\n')
    root = Path(__file__).resolve().parent
    audit['archive_hash_mismatches'] = [n for n,h in plan['source_sha256'].items()
        if hashlib.sha256((folder/'executed_sources'/n).read_bytes()).hexdigest() != h]
    audit['current_hash_mismatches'] = [n for n,h in plan['source_sha256'].items()
        if hashlib.sha256((root/n).read_bytes()).hexdigest() != h]
    audit['current_measurement_hash_mismatches'] = [n for n in audit['current_hash_mismatches']
        if n not in ('report_fixed_replay.py','analyze_discovery_staging.py')]
    audit['postprocessing_source_sha256'] = {n:hashlib.sha256((root/n).read_bytes()).hexdigest()
        for n in ('report_fixed_replay.py','analyze_discovery_staging.py')}
    audit['postprocessing_note'] = 'Archived and current source hashes are recorded separately; see mismatch lists.'
    audit['requests_hash_matches'] = hashlib.sha256((folder/'requests.json').read_bytes()).hexdigest() == plan['requests_sha256']
    maps = [audit[s['name']]['native_maps'] for s in plan['cases']]
    audit['native_maps_equal'] = all(m == maps[0] for m in maps)
    rows = []
    comparisons = []
    for concurrency in concurrencies:
        pair = [results[f'c{concurrency:02d}_{mode}'] for mode in ('off','on')]
        chart = folder/f'graphs_c{concurrency:02d}'
        chart.mkdir(exist_ok=True)
        plot(chart, pair, title=f'DiscoveryBench fixed prompt replay: concurrency {concurrency}, OFF vs ON',
             phase_label='requests', caption='X: elapsed minutes (each case starts cold); 5s bins. Same inputs and waves, response-paced arrivals; no Python execution.')
        off, on = [p[0] for p in pair]
        calls_off, calls_on = [json.loads((folder/p[0]['condition']/'replay_calls.json').read_text()) for p in pair]
        cfgs = [yaml.safe_load((folder/p[0]['condition']/'config.yaml').read_text()) for p in pair]
        for cfg in cfgs:
            for key in ('daosgds.dram_prefetch','daosgds.root','daosgds.object_namespace'):
                cfg['extra_config'].pop(key)
        assert cfgs[0] == cfgs[1]
        comparisons.append(dict(concurrency=concurrency,
            completion_token_count_equal=off['total_completion_tokens']==on['total_completion_tokens'],
            equal_output_requests=sum(a['output_sha256']==b['output_sha256'] for a,b in zip(calls_off,calls_on)),
            elapsed_on_minus_off_pct=(on['elapsed_seconds']/off['elapsed_seconds']-1)*100,
            ttft_on_minus_off_pct=(on['mean_client_ttft_ms']/off['mean_client_ttft_ms']-1)*100))
        for st,_,_ in pair:
            counter = st['prefetch_counters'] or {}
            rows.append(f"| {concurrency} | {'ON' if st['prefetch'] else 'OFF'} | {st['llm_calls']} | "
                f"{st['elapsed_seconds']:.2f} | {st['mean_client_ttft_ms']:.2f} | {st['peak_staging_gib']:.3f} | "
                f"{100*st['dram_lookup_ratio']:.2f}% | {100*st['daos_lookup_ratio']:.2f}% | {100*st['miss_lookup_ratio']:.2f}% | "
                f"{counter.get('capacity_rejections',0)} | {st['daos_allocation_failures']} |")
    (folder/'validation.json').write_text(json.dumps(audit,indent=2)+'\n')
    (folder/'comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    (folder/'analysis.json').write_text(json.dumps(summaries,indent=2)+'\n')
    graph_links = '\n'.join(f'- [동시 요청 {c}: OFF/ON](graphs_c{c:02d}/timeline.png)' for c in concurrencies)
    text = f'''# DiscoveryBench 고정 입력 재생: DRAM 프리페치 OFF/ON

## 실험 방법

- Qwen3-14B BF16, 청크 128토큰, DRAM {plan['cpu_gib']:g}GiB, GPU staging {plan['staging_gib']:g}GiB, object 경로.
- 기록된 DiscoveryBench 모델 호출 {len(records)}개의 입력을 시간순으로 선정하여 모든 조건에 동일하게 재생했다. Python 도구는 다시 실행하지 않았다. 전체 에이전트 벤치마크/정답 평가가 아니다.
- 동시 요청 {concurrency_text} × DRAM 프리페치 OFF/ON. 각 조건마다 새 vLLM 프로세스, 빈 DRAM, 새 DAOS 이름 공간으로 시작했다. 공용 캐시 삭제나 서버/OS 캐시 초기화는 하지 않았다.
- {concurrency_text}개씩 함께 보내고 해당 묶음이 모두 끝나면 다음 묶음을 보냈다. 요청 목록/순서/묶음 구성은 고정이지만 벽시계 도착 시각까지 같지는 않다.
- temperature=0, seed=0, thinking OFF, 최대 생성 2048토큰, EOS 허용. 생성량 차이는 별도 기록한다. 응답은 다음 입력에 반영하지 않는다.
- OFF는 DRAM→staging 프리페치만 끈다. DAOS 프리페치, GPU-direct 쓰기, 비동기 DRAM 저장/읽기 승격은 양쪽 모두 켰다.
- ON은 사전 watermark 제한 없이 실제 {plan['staging_gib']:g}GiB 풀에 할당을 시도한다. 실제 공간 부족 시 DRAM 원본으로 fallback한다. DAOS의 기존 serializer 제한은 유지했다.
- 모델 로딩 시간은 제외했다. 매 조건 첫 요청의 cold 영향은 포함했다. 각 조건 1회이며 순서는 {case_order}이다.

## 결과

| 동시 요청 | DRAM 프리페치 | 요청 수 | 전체 시간(s) | 평균 TTFT(ms) | staging 최대(GiB) | DRAM hit | DAOS hit | miss | DRAM 실제 할당 실패 | 백엔드 할당 실패¹ |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

hit 비율은 prefix lookup의 청크 기준이며 분모는 최초 DRAM 조회 청크 수다. 실제 retrieve 성공량과 다를 수 있다. `analysis.json`의 actual_cached_token_fraction은 응답 usage가 보고한 실제 재사용 토큰/입력 토큰 비율이다. 두 분모를 혼동하지 않는다.

¹ 백엔드 할당 실패는 새 캐시 저장용 GPU 할당과 DAOS 읽기용 할당을 함께 센다. 이 수치를 모두 DAOS 읽기 실패로 해석하면 안 된다. `store_shortfalls`는 저장 단계에서 일부 토큰만 저장한 로그, `gpu_full_log_count`와 `partial_daos_reads`는 DAOS 읽기 쪽의 증거다. 일반 `Failed to allocate memory block` 경고에는 CPU LRU 재시도도 섞이므로 GPU 고갈 횟수로 쓰지 않는다.

## 시간 그래프

{graph_links}

각 그림 왼쪽 OFF, 오른쪽 ON. 위부터 staging 사용량(5초 구간 최대/평균), DRAM·DAOS·miss 비율, DRAM 보관량, 응답 중인 모델 요청 수다. 요청이 없는 구간의 hit 비율은 빈칸이다. staging {plan['staging_gib']:g}GiB는 예약된 풀의 크기이며 그래프는 그중 실제 할당량이다.

## 해석 시 주의

이 실험은 동일 입력에 대한 공간 사용/지연 진단이다. 생성 토큰 수, 실제 hit 비율, 완료 순서가 달라지면 실행 시간이 프리페치 효과만을 뜻하지 않는다. `comparison.json`에서 출력 동일성과 생성 토큰 합계를 확인한다. 시간에 따른 DRAM 승격/LRU도 성능의 일부다.

staging 최대치가 설정 용량에 가깝고 실제 할당 실패·fallback·DAOS 부분 읽기가 함께 늘어나는지를 확인해야 공간 경쟁의 증거로 해석할 수 있다. 그런 징후가 없으면 이번 입력은 한계를 드러내지 못한 것이며 경쟁 때문에 느려졌다고 결론 내리지 않는다.

원시 증거: `requests.json`, 각 조건의 `replay_calls.json`, `waves.json`, `config.yaml`, `trace.*.jsonl`, `server.log`, `initial_sample.json`, `final_sample.json`. 코드/입력 해시 및 라이브러리 일치 검사는 `validation.json`에 저장했다.

최초 코드 스냅샷과의 차이 및 실제 후처리 코드 해시는 validation.json에 명시했다. 백엔드 저장용 할당 실패를 DAOS 읽기 실패와 구분하여 해석한다.
'''
    details = ['\n## 실제 생성량 및 재사용량 대조\n',
        '| 동시 요청 | OFF 생성 토큰 | ON 생성 토큰 | 출력 문자열 동일 요청 | OFF 실제 재사용률 | ON 실제 재사용률 |',
        '|---:|---:|---:|---:|---:|---:|']
    for comparison in comparisons:
        c = comparison['concurrency']
        off, on = (results[f'c{c:02d}_{mode}'][0] for mode in ('off','on'))
        details.append(f"| {c} | {off['total_completion_tokens']} | {on['total_completion_tokens']} | "
            f"{comparison['equal_output_requests']}/{len(records)} | "
            f"{100*off['actual_cached_token_fraction']:.2f}% | {100*on['actual_cached_token_fraction']:.2f}% |")
    highest = max(concurrencies)
    off, on = results[f'c{highest:02d}_off'][0], results[f'c{highest:02d}_on'][0]
    high_fraction = on['occupancy_time_fraction'].get('at_least_90pct',0)
    high_seconds = high_fraction*on['elapsed_seconds']
    details += [f'''\n## {highest}개 동시 요청에서 관측한 공간 사용

- ON staging 최대 {on['peak_staging_gib']:.3f}GiB. 용량의 90%({.9*on['staging_gib']:g}GiB) 이상인 시간은 약 {high_seconds:.2f}초(전체의 {100*high_fraction:.2f}%)였다. 최대치만으로 지속적인 포화를 판단하지 않는다.
- DRAM 실제 할당 실패/fallback {(on['prefetch_counters'] or {}).get('capacity_rejections',0)}회. 이미 DRAM에 있던 캐시는 CPU 경로로 사용한다.
- 백엔드 할당 실패 {on['backend_allocation_failures']}회. 저장 입장 제한 경고는 {len(on['store_admission_warning_chunks'])}회이며, 각 시도에서 저장 가능한 청크 수는 {on['store_admission_warning_chunks']}였다. 이는 이번 저장 시도의 일부/전체 생략이며 영구적인 KV 손실이나 추론 실패를 뜻하지 않는다.
- DAOS GPU-buffer-full 읽기 경고 {on['gpu_full_log_count']}회, 부분 DAOS 읽기 {on['partial_daos_reads']}회. 0회라면 DAOS 읽기 방해가 관측된 것은 아니다. 발생했다면 해당 시점의 DRAM/DAOS 점유량과 OFF 결과를 함께 대조해야 한다.
- 전체 시간 OFF {off['elapsed_seconds']:.2f}초 / ON {on['elapsed_seconds']:.2f}초. 1회 실행의 작은 차이를 유의미한 성능 악화로 해석하지 않는다. 평균 TTFT는 OFF {off['mean_client_ttft_ms']:.2f}ms / ON {on['mean_client_ttft_ms']:.2f}ms였다.
- DAOS lookup hit 비율은 약 {100*on['daos_lookup_ratio']:.2f}%였다. 이 워크로드에서 실제 발생한 비율이지 사전에 강제한 비율이 아니다.

용량 도달, fallback, 저장 제한, DAOS 부분 읽기는 서로 다른 증거다. 실제 발생한 항목만으로 해석하며, 용량 도달만으로 전체 성능 악화를 단정하지 않는다.

주의: `Local cpu memory under pressure`는 LMCache의 기존 고정 경고 문구다. 이번 GPU-direct store에서는 GPU allocator를 사용하므로, 이 경고를 DRAM 용량 부족으로 해석하면 안 된다. 실제 GPU 할당 실패 이벤트와 직전 CPU/DAOS ready 점유량은 `c{highest:02d}_on/allocation_failure_evidence.json`에 저장했다.
''']
    (folder/'RESULT_KO.md').write_text(text+'\n'.join(details))
    print('\n'.join(rows), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('folder',type=Path)
    report(p.parse_args().folder.resolve())
