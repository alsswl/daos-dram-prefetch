# DiscoveryBench full agentic 반복 실행 결과

DFS와 object 각각 최초 캐시 채우기 1회와 vLLM 재시작 후 동일 문제 재풀이 10회를 수행했다. 총 22회의 에이전트 workflow이며, **10회는 오류 자동 재시도가 아니라 같은 문제를 다시 푸는 횟수**다. DiscoveryBench 전체 테스트셋이나 정답 채점 결과는 아니다.

## 실행 조건과 집계 기준

- 모델 `Qwen/Qwen3-14B`, 청크 128토큰, GPU staging 10GiB, I/O·메타 작업자 16·16개.
- 실제 CSV를 읽고 Python 분석 코드를 실행한 뒤 최종 답변을 작성하는 동일 태스크다.
- 모드별 새 namespace로 시작한다. 이후 vLLM만 매번 재시작하고 DAOS 데이터·서버 캐시는 유지한다.
- 같은 재시작 번호의 DFS/object를 교차 실행하고, 매 단계 두 모드의 선후 순서를 뒤집었다.
- 최초 `fill` 1회는 참고값이며, 아래 `restart_hit` 반복 통계에 포함하지 않는다.
- 전체 작업 시간: 에이전트 시작부터 최종 답변까지. 모델·Python 컨테이너 기동은 제외하고 도구 실행·로깅은 포함한다.
- TTFT: 각 실행 내 모델 호출들의 **서버 측 평균 TTFT**다. 그 실행별 평균들을 다시 비가중 평균·중앙값으로 집계한다. 개별 호출의 p50/p95나 클라이언트 첫 텍스트 TTFT가 아니다.
- 표준편차는 표본 표준편차다. 최초 1회의 표준편차는 계산할 수 없어 —로 표시한다.

## 전체 작업 시간 (초)

| 모드 | 단계 | 횟수 | 평균 | 중앙값 | 표준편차 | 최솟값 | 최댓값 |
|---|---|---:|---:|---:|---:|---:|---:|
| dfs | fill | 1 | 9.176 | 9.176 | — | 9.176 | 9.176 |
| dfs | restart_hit | 10 | 8.171 | 8.132 | 0.18 | 7.872 | 8.527 |
| object | fill | 1 | 9.431 | 9.431 | — | 9.431 | 9.431 |
| object | restart_hit | 10 | 8.201 | 8.191 | 0.08 | 8.071 | 8.326 |

## 실행별 평균 서버 TTFT의 분포 (ms)

| 모드 | 단계 | 횟수 | 평균 | 중앙값 | 표준편차 | 최솟값 | 최댓값 | 호출수 가중 평균 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| dfs | fill | 1 | 117.31 | 117.31 | — | 117.31 | 117.31 | 117.31 |
| dfs | restart_hit | 10 | 87.03 | 87.25 | 2.51 | 82.28 | 90.38 | 87.03 |
| object | fill | 1 | 123.68 | 123.68 | — | 123.68 | 123.68 | 123.68 |
| object | restart_hit | 10 | 92.96 | 93.01 | 3.40 | 87.88 | 99.43 | 92.96 |

## 수행 작업 횟수 분포

`호출 수: 실행 횟수` 형식이다. 예: `3: 10`은 모델을 3번 호출한 workflow가 10회라는 뜻이다.

| 모드 | 단계 | 모델 호출 | Python 실행 | 첫 요청 hit 토큰 |
|---|---|---|---|---|
| dfs | fill | 4: 1 | 3: 1 | 0: 1 |
| dfs | restart_hit | 3: 10 | 2: 10 | 1536: 10 |
| object | fill | 4: 1 | 3: 1 | 0: 1 |
| object | restart_hit | 3: 10 | 2: 10 | 1536: 10 |

## 개별 실행

| 모드 | 실행 | 단계 | 전체 시간 (s) | 평균 TTFT (ms) | 모델 호출 | Python 실행 | 생성 토큰 합계 | 첫 요청 hit |
|---|---|---|---:|---:|---:|---:|---:|---:|
| dfs | [run1](dfs_run1/llm_calls.json) | fill | 9.176 | 117.31 | 4 | 3 | 780 | 0 |
| dfs | [run2](dfs_run2/llm_calls.json) | restart_hit | 8.150 | 90.38 | 3 | 2 | 695 | 1536 |
| dfs | [run3](dfs_run3/llm_calls.json) | restart_hit | 8.088 | 90.09 | 3 | 2 | 695 | 1536 |
| dfs | [run4](dfs_run4/llm_calls.json) | restart_hit | 8.394 | 87.49 | 3 | 2 | 695 | 1536 |
| dfs | [run5](dfs_run5/llm_calls.json) | restart_hit | 8.181 | 82.28 | 3 | 2 | 695 | 1536 |
| dfs | [run6](dfs_run6/llm_calls.json) | restart_hit | 8.114 | 84.28 | 3 | 2 | 695 | 1536 |
| dfs | [run7](dfs_run7/llm_calls.json) | restart_hit | 8.109 | 87.00 | 3 | 2 | 695 | 1536 |
| dfs | [run8](dfs_run8/llm_calls.json) | restart_hit | 8.193 | 86.27 | 3 | 2 | 695 | 1536 |
| dfs | [run9](dfs_run9/llm_calls.json) | restart_hit | 7.872 | 85.77 | 3 | 2 | 695 | 1536 |
| dfs | [run10](dfs_run10/llm_calls.json) | restart_hit | 8.085 | 88.64 | 3 | 2 | 695 | 1536 |
| dfs | [run11](dfs_run11/llm_calls.json) | restart_hit | 8.527 | 88.09 | 3 | 2 | 695 | 1536 |
| object | [run1](object_run1/llm_calls.json) | fill | 9.431 | 123.68 | 4 | 3 | 780 | 0 |
| object | [run2](object_run2/llm_calls.json) | restart_hit | 8.286 | 99.43 | 3 | 2 | 695 | 1536 |
| object | [run3](object_run3/llm_calls.json) | restart_hit | 8.136 | 90.64 | 3 | 2 | 695 | 1536 |
| object | [run4](object_run4/llm_calls.json) | restart_hit | 8.211 | 94.96 | 3 | 2 | 695 | 1536 |
| object | [run5](object_run5/llm_calls.json) | restart_hit | 8.268 | 95.05 | 3 | 2 | 695 | 1536 |
| object | [run6](object_run6/llm_calls.json) | restart_hit | 8.171 | 90.11 | 3 | 2 | 695 | 1536 |
| object | [run7](object_run7/llm_calls.json) | restart_hit | 8.071 | 90.70 | 3 | 2 | 695 | 1536 |
| object | [run8](object_run8/llm_calls.json) | restart_hit | 8.167 | 87.88 | 3 | 2 | 695 | 1536 |
| object | [run9](object_run9/llm_calls.json) | restart_hit | 8.217 | 94.78 | 3 | 2 | 695 | 1536 |
| object | [run10](object_run10/llm_calls.json) | restart_hit | 8.326 | 94.75 | 3 | 2 | 695 | 1536 |
| object | [run11](object_run11/llm_calls.json) | restart_hit | 8.155 | 91.27 | 3 | 2 | 695 | 1536 |

## 검증과 해석 범위

- 실행 횟수, 모델 콜백/캐시 로그/TTFT 카운트, Python 실행 횟수, 최초 miss와 재시작 후 첫 요청 prefix load를 검증했다.
- 동일 실행 번호의 DFS/object 11쌍 중 코드·실행 결과 동일 11쌍, 최종 답변 동일 11쌍. 단계 사이 동일성도 `mode_equivalence.json`에 기록했다. 출력 차이는 자동 실패로 처리하지 않았다.
- 종료 전 경고 332줄, 종료 후 오류 8줄을 별도로 기록했다. 종료 전 오류가 있으면 정상 성능 보고서 생성을 거부한다.
- 캐시가 반복마다 누적된다. 10번의 독립적인 새 cold/warm 쌍 실험이 아니며, 뒤쪽 실행은 앞선 재풀이가 만든 추가 prefix도 재사용할 수 있다.
- 최초 실행도 뒤쪽 모델 호출에서는 KV를 재사용할 수 있어 workflow 전체가 miss-only는 아니다.
- 에이전트가 선택하는 분석 횟수·코드·출력 길이가 달라질 수 있다. 최초/재시작 또는 DFS/object의 전체 시간 차이를 모두 I/O나 캐시 효과로 해석하지 않는다.
- 같은 문제를 여러 번 푼 결과이며 여러 태스크·동시 사용자 부하를 대표하지 않는다. workflow 완료는 과학적 답변 정확성의 검증이 아니다.
- 문서의 태스크 데이터는 stress_tolerance가 일정해 관계 분석이 조기에 끝날 수 있다. 실제 수행한 코드와 최종 답변을 함께 확인해야 한다.

[개별 실행 CSV](full_agentic_repeated_runs.csv) · [단계별 통계 CSV](full_agentic_repeated_statistics.csv) · [검증](repeated_validation.json) · [작업 동일성](mode_equivalence.json) · [로그 감사](log_audit.json) · [원자료](summary.json)

## 재실행

```bash
cd /root/discos_minji
./venv/bin/python3 full_agent_bench.py --chunk-size 128 --restart-runs 10 --interleave-modes \
  --output /root/discos_minji/full_agentic_새_결과_폴더
./venv/bin/python3 summarize_full_agent_repeated.py \
  /root/discos_minji/full_agentic_새_결과_폴더
```

이미 존재하는 결과 폴더는 덮어쓰지 않는다. 새 실행마다 새 캐시 namespace가 생성되며, 공유 DAOS 컨테이너나 이전 결과는 삭제하지 않는다.
