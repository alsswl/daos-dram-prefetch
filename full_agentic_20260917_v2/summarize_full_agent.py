"""Summarize the supplied-document full-agentic protocol, not the whole test set."""
import argparse
import csv
import json
from pathlib import Path
import re


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('folder', type=Path)
    a = parser.parse_args()
    status = json.loads((a.folder / 'status.json').read_text())
    if status['status'] != 'complete':
        raise ValueError('Full-agentic run is not complete')
    results = json.loads((a.folder / 'summary.json').read_text())
    equality = []
    for run in ('run1', 'run2'):
        left = json.loads((a.folder / f'dfs_{run}/tool_calls.json').read_text())
        right = json.loads((a.folder / f'object_{run}/tool_calls.json').read_text())
        lhs = next(r for r in results if r['mode'] == 'dfs' and r['run'] == run)
        rhs = next(r for r in results if r['mode'] == 'object' and r['run'] == run)
        equality.append({'run': run, 'code_and_observations_equal':
            [(r['code'], r['output']) for r in left] == [(r['code'], r['output']) for r in right],
            'final_answers_equal': lhs['final_answer'] == rhs['final_answer']})
    (a.folder / 'mode_equivalence.json').write_text(json.dumps(equality, indent=2) + '\n')
    audit, rows = [], []
    for r in results:
        tag = f'{r["mode"]}_{r["run"]}'
        text = (a.folder / f'{tag}_server.log').read_text(errors='replace')
        pre, _, post = text.partition('[shutdown]')
        errors = [s for s in pre.splitlines() if re.search(r'\bERROR\b', s)]
        audit.append({'run': tag, 'errors_before_shutdown': errors,
                      'errors_after_shutdown': sum(bool(re.search(r'\bERROR\b', s)) for s in post.splitlines())})
        if errors:
            raise ValueError(f'{tag}: pre-shutdown errors require review')
        requests = r['cache_requests']
        if len(requests) != r['llm_calls'] or r['ttft_count'] != r['llm_calls']:
            raise ValueError(f'{tag}: callback/metrics/request counts differ')
        prompts = sum(x['prompt_tokens'] for x in requests)
        hits = sum(x['lmcache_hit_tokens'] for x in requests)
        tools = json.loads((a.folder / tag / 'tool_calls.json').read_text())
        calls = json.loads((a.folder / tag / 'llm_calls.json').read_text())
        row = {k: r[k] for k in ('mode', 'run', 'workflow_seconds', 'llm_calls', 'python_calls', 'mean_server_ttft_ms')}
        row.update(prompt_tokens_sum=prompts, lmcache_hit_tokens_sum=hits,
                   hit_ratio=hits / prompts, first_request_hit_tokens=requests[0]['lmcache_hit_tokens'],
                   python_seconds=sum(x['elapsed_seconds'] for x in tools),
                   llm_seconds=sum(x['elapsed_seconds'] for x in calls.values()))
        rows.append(row)
    with (a.folder / 'full_agentic_results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (a.folder / 'log_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    lines = ['# DiscoveryBench full agentic 실행 결과', '',
             '첨부 문서 Part 5.3~5.4의 adventure-travel_0_0 태스크 하나를 '
             '실제 ReAct 에이전트로 실행했다. CSV 읽기 → Python 코드 실행 → 결과를 보고 '
             '다음 분석 → 최종 답변까지 진행했다. 전체 테스트셋 실행이나 정답 채점은 아니다.', '',
             '## 설정과 실행 범위', '',
             '- Qwen/Qwen3-14B BF16, vLLM 0.25.1, LMCache 0.5.2, 청크 128토큰.',
             '- 공통 DAOS/libfabric 설치본, GPU staging 10GiB, I/O·메타 작업자 각 16개.',
             '- 최대 문맥 16384, eager, GPU 메모리 비율 0.75, 내장 prefix caching 비활성.',
             '- 모델 temperature 0, 호출당 출력 상한 2048토큰, 최대 25회 반복. 고정 합성 프롬프트나 padding 없음.',
             '- DFS와 object 각각 새 namespace에서 run1 실행 후 vLLM을 종료하고 같은 namespace로 run2 실행.',
             '- 모델 생성 코드는 네트워크가 없는 별도 Python 컨테이너에서 실행. 원본 CSV는 읽기 전용, '
             '작업 디렉터리만 쓰기 가능. pandas/scikit-learn/statsmodels 사용 가능.',
             '- 문서의 예전 클라이언트·chunk 2048·8GiB/8workers 대신 현재 통합 환경과 chunk 128을 사용했다. '
             '공용 컨테이너를 삭제하지 않았다.', '',
             '## 측정', '',
             '| 모드 | 실행 | 전체 작업 시간 | 모델 호출 | Python 실행 | 평균 서버 TTFT | 첫 요청 hit 토큰 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['mode']} | {r['run']} | {r['workflow_seconds']:.2f}s | {r['llm_calls']} | "
                     f"{r['python_calls']} | {r['mean_server_ttft_ms']:.1f}ms | {r['first_request_hit_tokens']} |")
    lines += ['', '전체 작업 시간은 에이전트의 문제 처리 시작부터 최종 답변까지다. 모델 기동과 '
              'Python 컨테이너 기동은 제외하고, 모델 응답·코드 실행·에이전트 처리·로깅은 포함한다. '
              '평균 TTFT는 각 실행 전후 vLLM metrics의 sum/count 차이로 계산한 **서버 측 평균**이다. '
              '앞선 합성 실험의 클라이언트 첫 텍스트 TTFT 중앙값과 직접 비교하지 않는다.', '',
              '## 실제 데이터와 답변', '',
              '문서가 지정한 CSV는 500행·5열이다. stress_tolerance는 전체 행에서 2로 일정하고, '
              'number_of_trips는 2 또는 3이다. 원본 데이터는 수정하지 않았다. '
              '따라서 stress_tolerance 변화에 따른 관계를 추정하기 어렵다는 답변은 실제 관측 데이터의 '
              '제약과 연결된다. 이 실행은 정답 가설과의 자동 채점을 수행하지 않았다.', '']
    for r in results:
        tag = f'{r["mode"]}_{r["run"]}'
        lines += [f'### {tag} 최종 답변', '', r['final_answer'], '',
                  f'[실행 코드와 결과]({tag}/tool_calls.json) · [모델 호출 기록]({tag}/llm_calls.json)', '']
    lines += ['## 해석 시 주의', '',
              '- 각 모드에서 run1/run2 한 쌍만 실행했다. 동일 조건 반복 성능 실험이 아니며, '
              '에이전트가 선택한 분석과 출력 길이 차이를 함께 봐야 한다.',
              '- run1의 뒤쪽 턴은 앞부분 KV를 재사용할 수 있으므로 run1 전체가 miss-only는 아니다. '
              'run2는 새 vLLM 프로세스이며 DAOS 서버 캐시는 유지된다.',
              '- 최종 답변에 도달한 것은 workflow 완료를 의미한다. 과학적 정답이나 다양한 데이터에서의 '
              '정합성이 검증됐다는 뜻은 아니다.',
              '- DFS/object의 동일 실행 단계끼리 Python 코드·실행 출력·최종 답변이 일치하는지는 '
              '[mode_equivalence.json](mode_equivalence.json)에 기록했다. run1과 run2 사이의 답변 '
              '동일성을 뜻하지 않는다.',
              '- 종료 구간의 경고·오류는 측정 중 오류와 구분해 log_audit.json에 기록했다.',
              '- v1은 측정기 콜백 누락과 포트 재사용 검사 문제로 중단된 진단 실행이다. '
              '최종 결과는 수정한 측정기로 새 namespace에서 수행한 v2이며 v1은 포함하지 않는다.', '',
              '[요약 CSV](full_agentic_results.csv) · [전체 메타데이터 및 캐시 hit 기록](summary.json)', '',
              '## 재실행', '', '```bash', 'cd /root/discos_minji',
              './venv/bin/python3 full_agent_bench.py --chunk-size 128 \\',
              '  --output /root/discos_minji/full_agentic_새_결과_폴더', '```', '',
              '기존 결과 폴더는 덮어쓰지 않는다. 현재 보고서는 v2의 완료 결과이며 새 실행은 '
              '별도 namespace를 만든다. 데이터셋 전체 실행 옵션은 아니다.', '']
    (a.folder / 'FULL_AGENTIC_RESULT_KO.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:32]))


if __name__ == '__main__':
    main()
