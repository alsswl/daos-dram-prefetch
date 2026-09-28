# DRAM 캐시 선택형 구현 및 ON/OFF 검증 결과

실행일: 2026-09-22. 작업 위치: `/root/discos_minji`.

## 요약

기존 코드·설정을 보존하고 OFF 기준 실험을 먼저 실행했다. 이후 **기본 OFF인 별도 실행기**를 추가해 LMCache의 CPU 캐시 보관 기능을 켜고 끌 수 있도록 했다. 기존 CPU 버퍼를 재사용하며 기존 DAOS 비동기 GPU-direct 저장은 유지한다. 새 GPU→CPU 비동기 복사 엔진이나 DAOS→DRAM 자동 승격을 구현한 것은 아니다.

object 경로의 Qwen3-14B 실제 추론 비교에서는 단일 요청의 평균 TTFT가 약 21% 줄었으나 동시 요청 4개에서는 개선이 없었다. 기본 실행을 ON으로 바꾸지 않았다. **재사용 데이터가 DRAM에 모두 들어가는 제한된 조건의 결과**이며 일반적인 우열로 확정하지 않는다.

## 성능 비교

Qwen3-14B, 입력 4,096토큰 × 서로 다른 4개 입력, 생성 64토큰, 청크 128, CPU pool 4GiB, GPU staging 10GiB, I/O/meta workers 각각 16. vLLM prefix caching은 끄고 비동기 로딩은 켰다. CPU allocator 용량은 ON/OFF 동일하다.

ON→OFF→OFF→ON 순서로 새 서버를 실행했다. 각 서버에서 독립 warmup, 첫 저장, hit 준비 확인 후 동시성 1·4를 각 3회 측정했다. 조건당 2개의 새 서버 × 3회 × 4요청 = 24개 요청이다. 아래는 원시 요청 시간을 합친 평균이다.

| 동시 요청 | OFF 평균 TTFT | ON 평균 TTFT | OFF 평균 E2E | ON 평균 E2E |
|---:|---:|---:|---:|---:|
| 1 | 67.94ms | 53.65ms | 763.07ms | 739.23ms |
| 4 | 136.48ms | 137.60ms | 849.54ms | 848.38ms |

- 동시성 1: TTFT 21.04% 감소, 요청 E2E 3.12% 감소.
- 동시성 4: TTFT는 오히려 0.82% 증가, E2E는 0.14% 감소. 뚜렷한 개선이라고 해석하지 않는다.
- TTFT p95: 동시성 1은 OFF 76.44ms / ON 54.19ms, 동시성 4는 OFF 149.65ms / ON 142.40ms. 조건당 24개뿐이라 꼬리 지연의 일반적인 우열을 주장하기에는 부족하다.
- 측정 요청은 모두 4,095토큰 hit였다. vLLM이 마지막 한 토큰은 계산하며, 실제 프리페치는 청크 단위로 4,096토큰을 다룬다.
- warm 재사용 48요청/모드에서 OFF는 DAOS prefetch batch 48회, ON은 0회였다. API/RPC 개수가 아닌 요청 단위 batch 개수다. OFF가 해당 구간에 읽은 KV payload는 48 × 0.625GiB = 30GiB이며, ON에서는 이 DAOS payload 읽기를 피했다.
- 동일 입력·동시성별 생성 토큰 SHA-256은 ON/OFF 및 두 서버 반복 사이에서 모두 일치했다. 모든 상황에서 동일 출력을 보장한다는 뜻은 아니다.

단일 요청에서 얻은 지연 개선이 동시성 4에서는 유지되지 않았다. 그 원인을 PCIe, GPU 복사, 스케줄링 중 하나로 단정할 수는 없다. 추가 계측이 필요하다.

## 기존 성능으로 돌아오는지 확인

기능 실행기를 만들기 전 원래 `run_vllm.sh`로 기준 OFF를 실행했다. 비교용 자원 설정은 이후 ON/OFF와 동일하다.

| 동시 요청 | 구현 전 원래 실행 OFF | 새 실행기 OFF 평균 |
|---:|---:|---:|
| 1 | TTFT 67.12ms | TTFT 67.94ms |
| 4 | TTFT 137.14ms | TTFT 136.48ms |

기준값과 새 실행기 OFF 값은 가깝다. 엄밀한 회귀 성능 검정은 아니지만, 기능을 끈 상태에서 큰 성능 변화는 관찰되지 않았다.

기존 `run_vllm.sh`, 기본 YAML, `gds_backend.py`, `dfs_binding.py`, `object_binding.py`, `serde_v2.py`, `libdaosgdr.c/.so`, `compare_e2e.py`의 SHA-256이 기준 실행 시점과 동일함을 확인했다. 기존 경로는 기능 구현을 위해 변경하지 않았다.

## 데이터 및 fallback 검증

### 실제 모델 재시작

ON 서버 두 개에서 각각 fill·warm 실험을 마친 뒤 서버를 종료하고 CPU cache가 비어 있는 새 프로세스로 시작했다. 검증 서버는 `kv_consumer`여서 새 저장을 하지 않는다.

- 두 번 모두 4개 입력 전부 4,095토큰 hit.
- 각 재시작 실행에서 DAOS prefetch batch 4회.
- CPU hit만 있었던 것이 아니라 DAOS 복사본도 실제 읽기 가능한 상태였음을 확인했다.
- 재시작 시 첫 실행 효과가 있어 이 지연값은 warm 성능 표에 합치지 않았다.

### 실제 allocator / 저장 manager / DAOS 전체 바이트 검증

`tests/dram_mirror_roundtrip.py`는 실제 StorageManager의 fan-out, CPU MemoryObj, GPU staging, DAOS 쓰기를 사용한다. GPU에서 만든 1MiB FP16 텐서의 바이트를 검증한다. async DAOS put 작업이 완료될 때까지 기다린 후 확인한다.

| 저장 경로 | DRAM 모드 | 저장 후 CPU cache 항목 | CPU 바이트 비교 | DAOS 바이트 비교 |
|---|---|---:|---|---|
| object | OFF | 0 | 보관 안 함 | 일치 |
| object | ON | 1 | 일치 | 일치 |
| dfs | OFF | 0 | 보관 안 함 | 일치 |
| dfs | ON | 1 | 일치 | 일치 |

ON에서는 cache가 **원래 CPU MemoryObj와 동일한 객체를 보관함**도 확인했다. CPU 복사본만 제거한 후 DAOS에서 새 GPU 버퍼로 읽어도 전체 바이트가 일치했다. 테스트에 사용한 새 UUID 데이터 4건만 성공 후 삭제했다. 일반 벤치마크 namespace는 보존했고, DFS 테스트의 빈 디렉터리도 남겼다. 삭제한 테스트 payload는 고정 seed로 재생성할 수 있다.

DFS는 이 저장·읽기 기능 테스트까지 수행했다. **이번 Qwen ON/OFF 성능 표는 object 경로 결과뿐**이며 DFS 추론 성능을 측정했다고 주장하지 않는다.

### 단위 테스트

`PYTHONPATH=/root/discos_minji ./venv/bin/python3 -m pytest -q tests`: **66 passed**. 설정 격리, 기본 OFF, CPU 용량 검증, 상충 환경변수 처리, CPU 객체 참조 수명, 기존 백엔드 테스트를 포함한다. OpenTelemetry deprecation 경고는 있었다.

## 해석할 때의 주의점

1. 전체 워킹셋은 별도 warmup까지 포함해 기본 payload 3.125GiB다. CPU pool 4GiB에 모두 들어간다. 용량 초과·LRU eviction·낮은 재사용률에서 같은 효과를 보장하지 않는다.
2. full agentic DiscoveryBench가 아니라 고정 토큰 입력으로 실제 모델을 실행한 캐시 재사용 진단이다.
3. TTFT에는 로컬 HTTP와 클라이언트 처리 비용도 포함한다. E2E는 생성 64토큰 응답 종료까지이며, DAOS async commit 완료 시간을 의미하지 않는다.
4. 최초 fill 평균 TTFT는 ON 두 실행 약 316.98/329.48ms, OFF 두 실행 약 324.61/329.84ms였다. 저장 측 추가 지연에 큰 차이를 관찰하지 못했으나 표본이 작고 commit 시간을 재지 않았으므로 “저장 비용 0”이라고 할 수 없다.
5. **CPU metric 미노출:** 현재 vLLM `/metrics`는 해당 LMCache worker의 `local_cpu_hot_cache_count`를 노출하지 않는다. 초반 raw JSON의 `cpu_hot_chunks: 0`은 미노출을 0으로 처리했던 관측 코드의 값이지 실제 캐시 개수가 아니다. 해당 필드는 분석에 사용하지 않았다. 이후 스크립트는 미노출을 `null`로 기록한다. 실험 중 바뀐 것은 이 관측 필드 처리뿐이며 캐시·전송 경로는 바꾸지 않았다. 각 실행의 소스 스냅샷을 보존했다.
6. 모델 실행에서 CPU 경로 확인은 활성 계층(CPU와 DAOS만), vLLM prefix caching OFF, 실제 hit 수와 DAOS prefetch 로그를 함께 사용했다. 별도 1MiB 검증에서는 CPU cache 항목과 객체·바이트를 직접 확인했다.
7. 재시작·취소·eviction·과부하의 모든 조합을 검증한 것은 아니다. 특히 기존 DAOS cache를 async GET한 뒤 DRAM으로 자동 승격하는 기능은 없다.
8. 실험 프로세스는 모두 종료했다. 확인 시 GPU 메모리는 4MiB로 복귀했다. API 서버 종료 과정의 multiprocessing semaphore cleanup 경고는 있었으나 측정 구간의 GPU buffer full, double free, 음수 참조, DAOS 실패는 발견하지 못했다.

## 실행과 복귀

[실행·롤백 안내](DRAM_CACHE_GUIDE_KO.md)를 참고한다. 새 실행기의 `--dram on/off`는 **새 프로세스 시작 시** 적용한다. 같은 조건의 성능 비교에서는 CPU/GPU 용량·입력·동시성을 유지한다. 새 실행기를 쓰지 않으면 원래 동작이다. 전역 설정이나 기본 YAML을 ON으로 변경하지 않았다.

## 원시 근거

- [구현 전 기준 요약](dram_baseline_20260922_v1/summary.json), [기준 소스 해시](dram_baseline_20260922_v1/manifest.json)
- [OFF r1](dram_off_20260922_r1/summary.json), [OFF r2](dram_off_20260922_r2/summary.json)
- [ON r1 및 재시작](dram_on_20260922_r1/summary.json), [ON r2 및 재시작](dram_on_20260922_r2/summary.json)
- [object ON 바이트 검사](dram_mirror_object_on_20260922.json), [object OFF](dram_mirror_object_off_20260922.json)
- [DFS ON 바이트 검사](dram_mirror_dfs_on_20260922.json), [DFS OFF](dram_mirror_dfs_off_20260922.json)
- [선택형 실행기](run_dram_cache.py), [성능 실험](dram_cache_bench.py), [바이트 검사](tests/dram_mirror_roundtrip.py)

각 성능 실행 폴더에는 실제 실행 명령, 유효 YAML, native 라이브러리 경로, 고정 입력, 요청별 생성 토큰/해시, 서버 로그와 원시 metrics, 실행 당시 소스 사본이 있다.
