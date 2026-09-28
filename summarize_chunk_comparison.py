#!/usr/bin/env python3
"""Validate two completed fixed-context runs and compare by context and mode."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics


def read_run(folder):
    manifest = json.loads((folder / 'manifest.json').read_text())
    status = json.loads((folder / 'status.json').read_text())
    if status['status'] != 'complete' or status['cache_checks'] != 'passed':
        raise ValueError(f'{folder}: run not complete and cache-validated')
    if not status.get('all_output_tokens_equal'):
        raise ValueError(f'{folder}: output token equality failed')
    config = manifest['args']
    groups, audit = {}, []
    for path in sorted(folder.glob('*_server.log')):
        text = path.read_text(errors='replace')
        before, _, after = text.partition('[shutdown]')
        errors = [s for s in before.splitlines() if re.search(r'\bERROR\b', s)]
        audit.append({'log': path.name, 'errors_before_shutdown': errors,
                      'errors_after_shutdown': sum(bool(re.search(r'\bERROR\b', s))
                                                   for s in after.splitlines())})
        if errors:
            raise ValueError(f'{path}: ERROR before shutdown')
    for rep in range(1, config['repeats'] + 1):
        for mode in ('dfs', 'object'):
            text = (folder / f'r{rep}_{mode}_restart_server.log').read_text()
            prefetch = re.findall(r'prefetch\[[^]]+\]: (\d+)/(\d+) objects, '
                                  r'([\d.]+) MiB in ([\d.]+) ms', text)
            contexts = config['context_tokens']
            # Last two passes are measured; earlier entries are warmup checks.
            prefetch = prefetch[-2 * len(contexts):]
            if len(prefetch) != 2 * len(contexts):
                raise ValueError(f'{folder}: missing prefetch records')
            for phase_idx, phase in enumerate(('restart_hit', 'same_process_hit')):
                rows = json.loads((folder / f'r{rep}_{mode}_{phase}_responses.json').read_text())
                if len(rows) != len(contexts):
                    raise ValueError('Missing request samples')
                for idx, row in enumerate(rows):
                    n = row['prompt_tokens']
                    expected = min(n - 1, n // config['chunk_size'] * config['chunk_size'])
                    if n != contexts[idx] or row['cached_tokens'] != expected:
                        raise ValueError('Prompt length or full-prefix cache check failed')
                    p = prefetch[phase_idx * len(contexts) + idx]
                    if int(p[0]) != int(p[1]) or int(p[0]) * config['chunk_size'] != n:
                        raise ValueError('Prefetch record does not match request')
                    row['prefetch_ms'] = float(p[3])
                    groups.setdefault((mode, phase, n), []).append(row)
    return manifest, groups, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('current', type=Path)
    args = parser.parse_args()
    old, before, audit_old = read_run(args.baseline)
    new, after, audit_new = read_run(args.current)
    w1 = json.loads((args.baseline / 'workload.json').read_text())
    w2 = json.loads((args.current / 'workload.json').read_text())
    if w1 != w2:
        raise ValueError('Inputs or warmup changed')
    diffs = {k: [v, new['args'][k]] for k, v in old['args'].items() if v != new['args'][k]}
    if set(diffs) - {'chunk_size', 'output'}:
        raise ValueError(f'Unexpected changed arguments: {diffs}')
    for key in ('run_vllm.sh', 'libdaosgdr.so', 'lmcache_daos/gds_backend.py'):
        if old['source_sha256'][key] != new['source_sha256'][key]:
            raise ValueError(f'Backend/runtime changed: {key}')
    rows = []
    for key in sorted(after):
        mode, phase, context = key
        row = dict(mode=mode, phase=phase, context_tokens=context,
                   baseline_chunk=old['args']['chunk_size'], current_chunk=new['args']['chunk_size'],
                   baseline_samples=len(before[key]), current_samples=len(after[key]))
        for metric in ('ttft_ms', 'e2e_ms', 'prefetch_ms'):
            v1 = [r[metric] for r in before[key]]
            v2 = [r[metric] for r in after[key]]
            for label, values in [('baseline', v1), ('current', v2)]:
                row[f'{metric}_{label}_median'] = statistics.median(values)
                row[f'{metric}_{label}_min'] = min(values)
                row[f'{metric}_{label}_max'] = max(values)
            row[f'{metric}_change_percent'] = (statistics.median(v2) / statistics.median(v1) - 1) * 100
        rows.append(row)
    with (args.current / 'chunk_comparison.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    check = {'identical_inputs_and_warmup': True, 'argument_differences': diffs,
             'identical_backend_and_runtime_hashes': True,
             'full_prefix_hits_and_output_tokens_checked': True,
             'baseline_log_audit': audit_old, 'current_log_audit': audit_new}
    (args.current / 'comparison_validation.json').write_text(json.dumps(check, indent=2) + '\n')
    chunk1, chunk2 = old['args']['chunk_size'], new['args']['chunk_size']
    lines = [f'# 청크 {chunk1} → {chunk2} 비교', '',
             'Qwen3-14B, 고정 입력 8K/16K/31K, 출력 64토큰, GPU staging 10GiB, '
             'I/O·메타 작업자 각 16개, 순차 요청, 각 모드 3회 반복. '
             '입력·워밍업 토큰 ID와 백엔드·런처·shim 해시가 동일함을 확인했다.', '',
             '아래는 새 vLLM 프로세스로 시작한 restart_hit의 중앙값이다. '
             'DAOS 서버와 서버 캐시는 초기화하지 않았다. 단위는 ms, 변화율은 음수일 때 개선이다.', '']
    for metric, title in [('ttft_ms', 'TTFT'), ('e2e_ms', '전체 응답 완료 시간'),
                          ('prefetch_ms', '프리페치 시간')]:
        lines += [f'## {title}', '', f'| 모드 | 입력 | {chunk1} | {chunk2} | 변화 |',
                  '|---|---|---:|---:|---:|']
        for row in rows:
            if row['phase'] != 'restart_hit':
                continue
            lines.append(f"| {row['mode']} | {row['context_tokens']//1024}K | "
                         f"{row[metric+'_baseline_median']:.1f} | {row[metric+'_current_median']:.1f} | "
                         f"{row[metric+'_change_percent']:+.1f}% |")
        lines.append('')
    lines += ['## 검증과 해석 범위', '',
              '- 전체 prefix 재사용, 각 실행 내 모드·패스 간 생성 토큰 ID 일치 검사를 통과했다.',
              '- TTFT는 첫 HTTP 응답 텍스트까지다. 전체 응답 시간은 생성 64토큰을 포함한다. '
              '프리페치는 메타데이터 조회·할당·작업 대기를 포함하며 순수 네트워크 시간은 아니다.',
              '- 256은 이전 날짜 측정값이다. 128과 같은 시점에 교차 측정한 것이 아니므로 '
              '서버 부하·캐시 상태 등 시점 차이가 남는다. 변화 전체를 청크 크기만의 효과로 확정하지 않는다.',
              '- dkey별·타깃별 부하를 계측한 실험이 아니므로 분산 개선이 원인인지 단정하지 않는다.',
              '- 청크 경계 hit 검사가 강화되고 워밍업 토큰 생성의 불필요한 길이 경고를 줄인 '
              '드라이버 변경이 있으나, 실제 입력은 동일하고 양쪽 결과 모두 강화된 기준으로 재검증했다.',
              '- 실패한 첫 실행은 별도 v1 디렉터리에 보존했고 비교에서 제외했다. '
              'v2는 새 namespace로 실행했다. 실험 캐시는 자동 삭제하지 않았다.',
              '- 종료 후 EngineDeadError/semaphore 정리 경고는 측정 중 오류와 구분해 '
              '[comparison_validation.json](comparison_validation.json)에 기록했다.', '',
              '[원자료·범위·same_process_hit 포함](chunk_comparison.csv)', '']
    (args.current / 'CHUNK_COMPARISON_KO.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
