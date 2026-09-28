# DiscoveryBench full agentic 실행 결과

첨부 문서 Part 5.3~5.4의 adventure-travel_0_0 태스크 하나를 실제 ReAct 에이전트로 실행했다. CSV 읽기 → Python 코드 실행 → 결과를 보고 다음 분석 → 최종 답변까지 진행했다. 전체 테스트셋 실행이나 정답 채점은 아니다.

## 설정과 실행 범위

- Qwen/Qwen3-14B BF16, vLLM 0.25.1, LMCache 0.5.2, 청크 128토큰.
- 공통 DAOS/libfabric 설치본, GPU staging 10GiB, I/O·메타 작업자 각 16개.
- 최대 문맥 16384, eager, GPU 메모리 비율 0.75, 내장 prefix caching 비활성.
- 모델 temperature 0, 호출당 출력 상한 2048토큰, 최대 25회 반복. 고정 합성 프롬프트나 padding 없음.
- DFS와 object 각각 새 namespace에서 run1 실행 후 vLLM을 종료하고 같은 namespace로 run2 실행.
- 모델 생성 코드는 네트워크가 없는 별도 Python 컨테이너에서 실행. 원본 CSV는 읽기 전용, 작업 디렉터리만 쓰기 가능. pandas/scikit-learn/statsmodels 사용 가능.
- 문서의 예전 클라이언트·chunk 2048·8GiB/8workers 대신 현재 통합 환경과 chunk 128을 사용했다. 공용 컨테이너를 삭제하지 않았다.

## 측정

| 모드 | 실행 | 전체 작업 시간 | 모델 호출 | Python 실행 | 평균 서버 TTFT | 첫 요청 hit 토큰 |
|---|---|---:|---:|---:|---:|---:|
| dfs | run1 | 9.26s | 4 | 3 | 113.1ms | 0 |
| dfs | run2 | 8.06s | 3 | 2 | 89.7ms | 1536 |
| object | run1 | 9.56s | 4 | 3 | 121.5ms | 0 |
| object | run2 | 8.21s | 3 | 2 | 90.2ms | 1536 |

전체 작업 시간은 에이전트의 문제 처리 시작부터 최종 답변까지다. 모델 기동과 Python 컨테이너 기동은 제외하고, 모델 응답·코드 실행·에이전트 처리·로깅은 포함한다. 평균 TTFT는 각 실행 전후 vLLM metrics의 sum/count 차이로 계산한 **서버 측 평균**이다. 앞선 합성 실험의 클라이언트 첫 텍스트 TTFT 중앙값과 직접 비교하지 않는다.

## 실제 데이터와 답변

문서가 지정한 CSV는 500행·5열이다. stress_tolerance는 전체 행에서 2로 일정하고, number_of_trips는 2 또는 3이다. 원본 데이터는 수정하지 않았다. 따라서 stress_tolerance 변화에 따른 관계를 추정하기 어렵다는 답변은 실제 관측 데이터의 제약과 연결된다. 이 실행은 정답 가설과의 자동 채점을 수행하지 않았다.

### dfs_run1 최종 답변

The dataset provided contains only the value 2 for the 'stress_tolerance' column, which prevents any meaningful analysis of the relationship between stress tolerance and creativity score. This suggests a potential data entry error or incomplete dataset. Therefore, no hypothesis can be formulated based on the current data.

WORKFLOW_SUMMARY:
1. Data Loading: Loaded the dataset from the specified path using Python.
2. Data Inspection: Displayed the first few rows of the dataset to understand its structure and the relevant columns.
3. Data Summary: Calculated summary statistics to understand the distribution of the variables.
4. Data Validation: Checked the unique values in the 'stress_tolerance' column and found that it contains only the value 2, indicating a potential data issue.

[실행 코드와 결과](dfs_run1/tool_calls.json) · [모델 호출 기록](dfs_run1/llm_calls.json)

### dfs_run2 최종 답변

The dataset indicates that stress tolerance is consistently rated as 2 for all travelers, which suggests a potential data entry error or a misinterpretation of the variable's scale. Given this, it is not possible to determine a relationship between stress tolerance and creativity score. However, if we assume that the stress tolerance variable was intended to have a different scale, further analysis would be required. 

WORKFLOW_SUMMARY:
1. Data Loading: Loaded the dataset from the specified path using Python.
2. Data Inspection: Displayed the first few rows of the dataset to understand its structure and the relevant columns.
3. Data Summary: Calculated summary statistics to understand the distribution of the variables.
4. Analysis: Noted that the stress tolerance variable is consistently rated as 2, which prevents analysis of its relationship with creativity score.

Hypothesis: Given the dataset's inconsistency in the stress tolerance variable, a hypothesis cannot be formulated based on the provided data. Further data validation and correction are needed to explore the relationship between stress tolerance and creativity score for travelers with different numbers of adventure travel trips per year.

[실행 코드와 결과](dfs_run2/tool_calls.json) · [모델 호출 기록](dfs_run2/llm_calls.json)

### object_run1 최종 답변

The dataset provided contains only the value 2 for the 'stress_tolerance' column, which prevents any meaningful analysis of the relationship between stress tolerance and creativity score. This suggests a potential data entry error or incomplete dataset. Therefore, no hypothesis can be formulated based on the current data.

WORKFLOW_SUMMARY:
1. Data Loading: Loaded the dataset from the specified path using Python.
2. Data Inspection: Displayed the first few rows of the dataset to understand its structure and the relevant columns.
3. Data Summary: Calculated summary statistics to understand the distribution of the variables.
4. Data Validation: Checked the unique values in the 'stress_tolerance' column and found that it contains only the value 2, indicating a potential data issue.

[실행 코드와 결과](object_run1/tool_calls.json) · [모델 호출 기록](object_run1/llm_calls.json)

### object_run2 최종 답변

The dataset indicates that stress tolerance is consistently rated as 2 for all travelers, which suggests a potential data entry error or a misinterpretation of the variable's scale. Given this, it is not possible to determine a relationship between stress tolerance and creativity score. However, if we assume that the stress tolerance variable was intended to have a different scale, further analysis would be required. 

WORKFLOW_SUMMARY:
1. Data Loading: Loaded the dataset from the specified path using Python.
2. Data Inspection: Displayed the first few rows of the dataset to understand its structure and the relevant columns.
3. Data Summary: Calculated summary statistics to understand the distribution of the variables.
4. Analysis: Noted that the stress tolerance variable is consistently rated as 2, which prevents analysis of its relationship with creativity score.

Hypothesis: Given the dataset's inconsistency in the stress tolerance variable, a hypothesis cannot be formulated based on the provided data. Further data validation and correction are needed to explore the relationship between stress tolerance and creativity score for travelers with different numbers of adventure travel trips per year.

[실행 코드와 결과](object_run2/tool_calls.json) · [모델 호출 기록](object_run2/llm_calls.json)

## 해석 시 주의

- 각 모드에서 run1/run2 한 쌍만 실행했다. 동일 조건 반복 성능 실험이 아니며, 에이전트가 선택한 분석과 출력 길이 차이를 함께 봐야 한다.
- run1의 뒤쪽 턴은 앞부분 KV를 재사용할 수 있으므로 run1 전체가 miss-only는 아니다. run2는 새 vLLM 프로세스이며 DAOS 서버 캐시는 유지된다.
- 최종 답변에 도달한 것은 workflow 완료를 의미한다. 과학적 정답이나 다양한 데이터에서의 정합성이 검증됐다는 뜻은 아니다.
- DFS/object의 동일 실행 단계끼리 Python 코드·실행 출력·최종 답변이 일치하는지는 [mode_equivalence.json](mode_equivalence.json)에 기록했다. run1과 run2 사이의 답변 동일성을 뜻하지 않는다.
- 종료 구간의 경고·오류는 측정 중 오류와 구분해 log_audit.json에 기록했다.
- v1은 측정기 콜백 누락과 포트 재사용 검사 문제로 중단된 진단 실행이다. 최종 결과는 수정한 측정기로 새 namespace에서 수행한 v2이며 v1은 포함하지 않는다.

[요약 CSV](full_agentic_results.csv) · [전체 메타데이터 및 캐시 hit 기록](summary.json)

## 재실행

```bash
cd /root/discos_minji
./venv/bin/python3 full_agent_bench.py --chunk-size 128 \
  --output /root/discos_minji/full_agentic_새_결과_폴더
```

기존 결과 폴더는 덮어쓰지 않는다. 현재 보고서는 v2의 완료 결과이며 새 실행은 별도 namespace를 만든다. 데이터셋 전체 실행 옵션은 아니다.
