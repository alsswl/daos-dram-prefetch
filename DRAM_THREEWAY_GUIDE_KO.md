# DRAM 보관·GPU 프리페치 3조건 비교

완료 결과: [2026-09-22 cold/warm 3조건 실험](dram_threeway_20260922_v1/RESULT_KO.md).

## 비교 목적

동일한 DAOS object GPU-direct 저장 경로에서 다음을 분리해서 비교한다.

| 조건 이름 | DRAM에 KV 보관 | DRAM → GPU staging 프리페치 | warm 읽기 출처 |
|---|---|---|---|
| `daos_only` | OFF | OFF | DAOS → GPU staging |
| `dram` | ON | OFF | DRAM → vLLM KV 페이지 |
| `dram_prefetch` | ON | ON | DRAM → GPU staging → vLLM KV 페이지 |

세 조건 모두 DAOS 저장은 켠다. `daos_only`도 기존 저장 경로의 임시 CPU 객체와 CPU allocator를 사용한다. DRAM 사용량 자체가 0인 조건이 아니라 **DRAM에 재사용 KV를 보관하지 않는 조건**이다.

## cold와 warm을 공정하게 나누는 방법

각 반복·동시성·조건마다 **새 vLLM 프로세스와 새 UUID 캐시 namespace**를 만든다. 다른 조건이나 이전 반복의 캐시를 재사용하지 않는다.

1. 모델을 로드하고 HTTP health를 확인한다. 측정 전 별도 추론 warmup은 하지 않는다.
2. 서로 다른 입력 4개를 처음 요청한다. 이 cold 구간은 모든 요청의 cached tokens가 0이고 캐시 읽기 batch가 0이어야 통과한다.
3. 같은 입력으로 생성 길이 1토큰의 준비 확인 요청을 보낸다. 모든 요청·응답은 `readiness.json`에 남긴다. 이 트래픽은 cold나 warm 성능 통계에 넣지 않는다.
4. 같은 4입력을 3회 재사용하고 warm 성능을 측정한다. 조건에 맞는 읽기 출처와 cached tokens를 검사한다.
5. 해당 서버를 종료한다. 다음 조건은 다시 빈 KV 캐시 영역에서 시작한다.

**warm 요청 자체를 cold라고 부르지 않는다.** 모든 반복은 cold에서 시작하지만, 재사용 효과는 별도의 warm 구간에서 측정한다. 또한 여기서 cold는 모델 입력의 KV 캐시가 없다는 뜻이다. 운영체제 파일 캐시, 모델 파일 캐시, DAOS 서버 내부 캐시, GPU clock/thermal 상태는 초기화하지 않는다. 공용 컨테이너 삭제·서버 재시작·OS cache drop은 수행하지 않는다.

동시성 1에서는 4개 입력을 차례대로 보낸다. 첫 번째만 새 프로세스의 첫 추론이고, 나머지는 새로운 입력의 KV-cold 요청이다. 첫 번째 요청과 나머지 요청의 초기화 비용이 다를 수 있다. 동시성 4는 4개를 동시 제출하지만 실제 vLLM 스케줄링·chunked prefill은 내부에서 수행된다.

## 고정 조건

- Qwen/Qwen3-14B, BF16, H100 NVL, 실제 HTTP streaming 추론.
- 서로 다른 입력 4개 × 4096토큰, 매 요청 64토큰 생성, temperature/seed 0.
- 청크 128토큰, CPU allocator 4GiB, 공통 GPU staging 10GiB.
- DRAM 프리페치 ON 조건의 GPU 입장 기준 5GiB. 다른 조건에 새 정책을 적용하지 않는다.
- DAOS I/O·메타데이터 작업 스레드 각각 16, async loading ON, non-layerwise.
- vLLM prefix caching OFF, eager, GPU memory utilization 0.75, 최대 길이 8192.
- 동시성 1·4 각각 조건별 새 프로세스 3회. 세 조건이 실행 순서의 각 위치에 한 번씩 오도록 순환한다.
- 총 18개 서버 실행, cold 72요청 + warm 216요청. 준비 확인 요청은 별도다.

준비 확인에서 사용한 1토큰 응답이나 최종 64토큰 응답의 생성 내용은 다음 입력에 붙이지 않는다. ON/OFF 사이 모델 출력이 달라도 다음 측정 입력은 동일하게 유지한다. 입력 token ID와 해시를 저장하고, 생성 token ID·해시도 보존한다.

재사용 대상 기본 KV 크기는 `4 × 4096 × 163840 bytes = 2.5GiB`다. 이번 실험에는 별도 warmup 입력이 없으며, 이 working set은 CPU 4GiB에 들어간다. DRAM eviction 정책의 성능을 비교하는 실험은 아니다. full agentic DiscoveryBench도 아니다.

## 실행

```bash
cd /root/discos_minji
./venv/bin/python3 dram_cold_compare.py \
  --repeats 3 --reuse-passes 3 \
  --output /root/discos_minji/dram_threeway_next
```

결과 경로는 없는 새 이름을 사용한다. 기본 transport는 object이며, DFS는 `--transport dfs`로 별도 실행할 수 있다. 현재 수행한 모델 성능 실험의 범위는 object다. `--reuse-passes 0`은 첫 cold 요청만 측정하고 재사용 실험은 하지 않는다. `--dry-run`은 계획과 실행 소스 사본만 저장한다.

완료된 결과의 조건·읽기 출처·소스 해시를 확인하고 통계를 재계산하는 명령:

```bash
./venv/bin/python3 summarize_dram_threeway.py \
  /root/discos_minji/dram_threeway_20260922_v1
```

분석기는 읽기 전용이며 JSON을 표준 출력으로 내보낸다. cold 12개/조건·동시성, warm 36개/조건·동시성을 별도로 집계한다. warm 요청 36개가 독립적인 서버 실험 36회를 뜻하는 것은 아니다. 새 프로세스 반복은 3회이며 프로세스별 평균도 확인해야 한다.

## 측정 해석과 증거

- TTFT: HTTP 요청 시작부터 첫 비어 있지 않은 텍스트까지.
- E2E: 해당 요청의 64토큰 응답이 끝날 때까지. 서버 시작·모델 로딩이나 전체 에이전트 실행 시간은 아니다.
- cold HTTP 시간은 DAOS 비동기 저장 완료 시간과 다르다. 준비 확인의 DRAM hit만으로 DAOS 복사본 commit을 입증하지 않는다. 이전 별도 바이트 일치·재시작 검증과 이번 성능 측정을 구분한다.
- DAOS/CPU prefetch 호출 수는 요청 batch 단위다. 청크별 DAOS API/RPC 개수가 아니다.
- CPU metric 미노출은 `null`이며, CPU 캐시 항목 0개를 뜻하지 않는다.
- 각 case에 실행 명령·유효 YAML·실제 로드 native library 경로·GPU 상태·요청별 원시값·구간 및 전체 로그를 남긴다. 최상위에는 입력, 계획, 소스 사본/해시, 통합 결과와 검증 결과를 남긴다.
- benchmark namespace의 데이터는 재검증용으로 남기며 공용 데이터를 삭제하지 않는다.

기존 실행기·기본 YAML·백엔드 구현을 바꾸지 않고 별도 실험 스크립트를 추가했다. 일반 실행의 GPU 프리페치 기본값 OFF는 유지된다.
