#!/usr/bin/env python3
"""Audit and graph the six fixed DiscoveryBench model-request replay cases."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import yaml

from analyze_discovery_staging import summarize_case, plot


def report(folder):
    plan = json.loads((folder/'plan.json').read_text())
    records = json.loads((folder/'requests.json').read_text())
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
        final = json.loads((case/'final_sample.json').read_text())
        stats.update(concurrency=spec['concurrency'], prefetch=spec['prefetch'],
            actual_cached_tokens=sum(r['cached_tokens'] for r in calls),
            total_prompt_tokens=sum(r['prompt_tokens'] for r in calls),
            total_completion_tokens=sum(r['completion_tokens'] for r in calls),
            mean_latency_ms=statistics.mean(r['elapsed_seconds']*1000 for r in calls),
            prefetch_counters=final.get('cpu_prefetch'),
            daos_allocation_failures=final['daos_alloc_fail'],
            mirror_final=final.get('dram_mirror'))
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
    audit['requests_hash_matches'] = hashlib.sha256((folder/'requests.json').read_bytes()).hexdigest() == plan['requests_sha256']
    maps = [audit[s['name']]['native_maps'] for s in plan['cases']]
    audit['native_maps_equal'] = all(m == maps[0] for m in maps)
    rows = []
    comparisons = []
    for concurrency in (4,8,16):
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
    text = '''# DiscoveryBench 고정 입력 재생: DRAM 프리페치 OFF/ON

## 실험 방법

- Qwen3-14B BF16, 청크 128토큰, DRAM 8GiB, GPU staging 10GiB, object 경로.
- 기록된 DiscoveryBench 모델 호출 256개의 입력을 시간순으로 선정하여 모든 조건에 동일하게 재생했다. Python 도구는 다시 실행하지 않았다. 전체 에이전트 벤치마크/정답 평가가 아니다.
- 동시 요청 4·8·16 × DRAM 프리페치 OFF/ON. 각 조건마다 새 vLLM 프로세스, 빈 DRAM, 새 DAOS 이름 공간으로 시작했다. 공용 캐시 삭제나 서버/OS 캐시 초기화는 하지 않았다.
- 4·8·16개씩 함께 보내고 해당 묶음이 모두 끝나면 다음 묶음을 보냈다. 요청 목록/순서/묶음 구성은 고정이지만 벽시계 도착 시각까지 같지는 않다.
- temperature=0, seed=0, thinking OFF, 최대 생성 2048토큰, EOS 허용. 생성량 차이는 별도 기록한다. 응답은 다음 입력에 반영하지 않는다.
- OFF는 DRAM→staging 프리페치만 끈다. DAOS 프리페치, GPU-direct 쓰기, 비동기 DRAM 저장/읽기 승격은 양쪽 모두 켰다.
- ON은 5GiB 사전 제한 없이 실제 10GiB 풀에 할당을 시도한다. 실제 공간 부족 시 DRAM 원본으로 fallback한다. DAOS의 기존 serializer 제한은 유지했다.
- 모델 로딩 시간은 제외했다. 매 조건 첫 요청의 cold 영향은 포함했다. 각 조건 1회이며 순서는 4 OFF→ON→8 OFF→ON→16 OFF→ON이다.

## 결과

| 동시 요청 | DRAM 프리페치 | 요청 수 | 전체 시간(s) | 평균 TTFT(ms) | staging 최대(GiB) | DRAM hit | DAOS hit | miss | DRAM 실제 할당 실패 | DAOS 할당 실패 |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

hit 비율은 prefix lookup의 청크 기준이며 분모는 최초 DRAM 조회 청크 수다. 실제 retrieve 성공량과 다를 수 있다. `analysis.json`의 actual_cached_token_fraction은 응답 usage가 보고한 실제 재사용 토큰/입력 토큰 비율이다. 두 분모를 혼동하지 않는다.

## 시간 그래프

- [동시 요청 4: OFF/ON](graphs_c04/timeline.png)
- [동시 요청 8: OFF/ON](graphs_c08/timeline.png)
- [동시 요청 16: OFF/ON](graphs_c16/timeline.png)

각 그림 왼쪽 OFF, 오른쪽 ON. 위부터 staging 사용량(5초 구간 최대/평균), DRAM·DAOS·miss 비율, DRAM 보관량, 응답 중인 모델 요청 수다. 요청이 없는 구간의 hit 비율은 빈칸이다. staging 10GiB는 예약된 풀의 크기이며 그래프는 그중 실제 할당량이다.

## 해석 시 주의

이 실험은 동일 입력에 대한 공간 사용/지연 진단이다. 생성 토큰 수, 실제 hit 비율, 완료 순서가 달라지면 실행 시간이 프리페치 효과만을 뜻하지 않는다. `comparison.json`에서 출력 동일성과 생성 토큰 합계를 확인한다. 시간에 따른 DRAM 승격/LRU도 성능의 일부다.

staging 최대치가 10GiB에 가깝고 실제 할당 실패·fallback·DAOS 부분 읽기가 함께 늘어나는지를 확인해야 공간 경쟁의 증거로 해석할 수 있다. 그런 징후가 없으면 이번 입력은 한계를 드러내지 못한 것이며 경쟁 때문에 느려졌다고 결론 내리지 않는다.

원시 증거: `requests.json`, 각 조건의 `replay_calls.json`, `waves.json`, `config.yaml`, `trace.*.jsonl`, `server.log`, `initial_sample.json`, `final_sample.json`. 코드/입력 해시 및 라이브러리 일치 검사는 `validation.json`에 저장했다.
'''
    (folder/'RESULT_KO.md').write_text(text)
    print('\n'.join(rows), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('folder',type=Path)
    report(p.parse_args().folder.resolve())
