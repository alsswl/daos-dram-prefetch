#!/usr/bin/env python3
"""Group reference-comparison requests by context, never mix 8K/16K/31K."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics

REFERENCE = {8192: (76, 149), 16384: (118, 132), 31744: (198, 218)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    a = parser.parse_args()
    status = json.loads((a.folder / 'status.json').read_text())
    config = json.loads((a.folder / 'manifest.json').read_text())['args']
    if status['status'] != 'complete':
        raise RuntimeError('Incomplete experiment; no accepted comparison')
    audit = []
    for path in sorted(a.folder.glob('*_server.log')):
        content = path.read_text(errors='replace')
        before, _, after = content.partition('[shutdown]')
        errors = [line for line in before.splitlines() if re.search(r'\bERROR\b', line)]
        audit.append({'log': path.name, 'errors_before_shutdown': errors,
                      'error_lines_after_shutdown': sum(bool(re.search(r'\bERROR\b', line))
                                                        for line in after.splitlines()),
                      'semaphore_warning': 'leaked semaphore' in content})
    (a.folder / 'log_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    if any(row['errors_before_shutdown'] for row in audit):
        raise RuntimeError('Pre-shutdown ERROR found; inspect log_audit.json before accepting samples')
    groups = {}
    for path in sorted(a.folder.glob('r*_responses.json')):
        for row in json.loads(path.read_text()):
            if row['phase'] != 'fill':
                n, chunk = row['prompt_tokens'], config['chunk_size']
                expected = min(n - 1, n // chunk * chunk)
                if row['cached_tokens'] != expected:
                    raise RuntimeError(f'{path}: expected full prefix hit {expected}, got {row["cached_tokens"]}')
            key = (row['mode'], row['phase'], row['prompt_tokens'])
            groups.setdefault(key, []).append(row)
    result = []
    for (mode, phase, ctx), samples in sorted(groups.items()):
        values = [r['ttft_ms'] for r in samples]
        low, high = REFERENCE.get(ctx, ('', ''))
        result.append(dict(mode=mode, phase=phase, context_tokens=ctx, samples=len(samples),
                           ttft_ms_median=statistics.median(values),
                           ttft_ms_min=min(values), ttft_ms_max=max(values),
                           ttft_ms_stdev=statistics.stdev(values) if len(values) > 1 else 0,
                           first_token_ms_median=statistics.median(r['first_token_ms'] for r in samples),
                           e2e_ms_median=statistics.median(r['e2e_ms'] for r in samples),
                           cached_tokens_min=min(r['cached_tokens'] for r in samples),
                           cached_tokens_max=max(r['cached_tokens'] for r in samples),
                           reference_dfs_hit_low_ms=low if phase != 'fill' else '',
                           reference_dfs_hit_high_ms=high if phase != 'fill' else ''))
    with (a.folder / 'by_context.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    lines = ['# 글루시스 수치와 현재 E2E 비교', '',
             '문서 §6.2 표 2의 **GDS in-process 콜드 hit TTFT**를 비교 대상으로 사용한다. '
             '문서의 초록에는 일부 범위가 다르므로 상세 표를 기준으로 한다. '
             '공급 문서의 수치는 외부 기준값이며 이 실행에서 측정한 수치가 아니다.', '',
             '아래는 각 반복의 새 vLLM 프로세스에서 측정한 restart_hit다. '
             '서버 스토리지 캐시나 DAOS 서버를 재시작·초기화하지 않았다. '
             f"TTFT는 첫 비어 있지 않은 HTTP 텍스트까지, E2E는 생성 {config['max_tokens']}토큰을 포함한 응답 완료까지다.", '',
             '| 입력 | 글루시스 DFS TTFT | 우리 DFS TTFT 중앙값 (범위) | 우리 object TTFT 중앙값 (범위) | object/DFS |',
             '|---|---:|---:|---:|---:|']
    for ctx, (low, high) in REFERENCE.items():
        pair = {r['mode']: r for r in result if r['context_tokens'] == ctx and r['phase'] == 'restart_hit'}
        if len(pair) != 2:
            continue
        def fmt(r):
            return f"{r['ttft_ms_median']:.1f} ({r['ttft_ms_min']:.1f}–{r['ttft_ms_max']:.1f}) ms"
        ratio = pair['object']['ttft_ms_median'] / pair['dfs']['ttft_ms_median']
        lines.append(f"| {ctx // 1024}K | {low}–{high} ms | {fmt(pair['dfs'])} | {fmt(pair['object'])} | {ratio:.2f}× |")
    lines += ['', f"## 전체 응답 완료 시간 — {config['max_tokens']}토큰 생성 포함", '',
              '| 입력 | DFS E2E 중앙값 | object E2E 중앙값 |', '|---|---:|---:|']
    for ctx in REFERENCE:
        pair = {r['mode']: r for r in result if r['context_tokens'] == ctx and r['phase'] == 'restart_hit'}
        if len(pair) == 2:
            lines.append(f"| {ctx // 1024}K | {pair['dfs']['e2e_ms_median']:.1f} ms | {pair['object']['e2e_ms_median']:.1f} ms |")
    lines += ['', 'object/DFS는 지연시간 비율이다. 1보다 크면 object가 더 느리다. '
              '3회 정도의 작은 표본을 통계적으로 확정된 우열이나 유사성 합격 기준으로 사용하지 않는다.', '',
              '캐시 검사: ' + status['cache_checks'] + '. 출력 토큰 ID 전체 일치: '
              + str(status.get('all_output_tokens_equal')) + '. 출력 텍스트 전체 일치: '
              + str(status['all_output_text_equal']) + '.', '',
              '모든 문맥·단계의 E2E와 TTFT 원자료는 [by_context.csv](by_context.csv)에 있다. '
              '서로 다른 문맥 길이가 섞인 summary.csv 중앙값으로 문서의 길이별 TTFT를 비교하지 않는다.', '',
              '## 재현 조건의 한계', '',
              '- 같은 Qwen3-14B, 청크 256토큰, GPU staging 10GiB, I/O 작업자 16개, '
              '비동기 프리페치를 사용했다. pool은 16타깃이며 기존 DFS 루트 기본 파일 클래스 S16, '
              '청크 4MiB를 조회했다. 신규 데이터 파일은 OC_SX(전체 타깃)를 사용한다.',
              '- 원문 vLLM 0.18과 달리 현재는 0.25.1이다. 원문의 정확한 launcher 및 '
              'bench_persist.py가 로컬에 없어 고정 입력 내용과 모든 실행 인자를 일치시키지는 못했다.',
              '- 로컬 공급 저장소 sweep2.py는 max_tokens=1을 사용한다. 이번 실행은 '
              f"전체 응답 시간도 보기 위해 {config['max_tokens']}토큰을 생성한다. TTFT가 기준 비교 항목이며 "
              '원문 TTFT와 우리 E2E 완료 시간을 직접 비교하면 안 된다.',
              '- 원문과 서버 NVMe 배치·실행 당시 부하·정확한 워밍업이 같은지는 검증하지 않았다. '
              'DFS 전용 연결 프로브는 양쪽 공통 워밍업으로 대체했다.',
              '- DFS 경로는 원본 그대로가 아니라 통합 백엔드의 DFS 모드다. '
              '수치가 비슷해도 전체 측정·구현의 정확성이 증명되지는 않는다. '
              'hit 토큰, 출력 토큰 일치, 실제 로드 라이브러리와 함께 판단한다.',
              '- 반복 문장 입력에 대한 출력 일치 검증이다. 다양한 실제 질의나 KV 바이트 '
              '전체 정합성까지 증명하는 것은 아니다.',
              '- 종료 시 일부 로그에 EngineDeadError/semaphore 정리 경고가 있다. '
              '측정 스트림 완료와 metrics 저장 뒤 SIGTERM 종료 구간에서 발생했다. '
              '종료 전 ERROR 검사는 [log_audit.json](log_audit.json)에 남긴다.',
              '- Part B(149문서/12동시 요청), DRAM 트래픽, full-agentic 워크로드는 이번 범위에 포함하지 않았다.', '',
              '자료: [제공 문서 원문](gluesys_supplied_document.txt), '
              '[로컬 공급 sweep2.py](gluesys_local_sweep2.py), '
              '[실행한 비교 코드](compare_e2e_executed.py).', '']
    (a.folder / 'REFERENCE_COMPARISON_KO.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:18]))


if __name__ == '__main__':
    main()
