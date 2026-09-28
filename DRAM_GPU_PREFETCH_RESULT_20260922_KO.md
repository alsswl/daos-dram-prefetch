# DRAM hit GPU 프리페치: 구현 및 비교 결과

2026-09-22, `/root/discos_minji`. **DRAM 캐시는 양쪽 모두 ON으로 유지하고, DRAM→GPU staging 프리페치만 OFF/ON으로 비교했다.** 동시 요청 4개에서 평균 TTFT가 137.47ms → 103.37ms로 24.81% 감소했다. 평균 요청 완료 시간은 846.02ms → 812.91ms로 3.91% 감소했다.

이는 제한된 warm-hit 워크로드에서 관측한 개선이다. 전체 벤치마크가 25% 빨라졌다거나 모든 부하에서 개선된다는 뜻은 아니다. 기능의 기본값은 계속 OFF다.

## 1. 실험 조건

| 항목 | 조건 |
|---|---|
| 모델 / 장치 | Qwen/Qwen3-14B, BF16 / H100 NVL |
| 워크로드 | 독립적인 입력 4개 × 4096토큰, 요청당 생성 64토큰 |
| 실행 형태 | vLLM HTTP streaming, 실제 추론; full agentic DiscoveryBench 아님 |
| 저장 경로 | object, DAOS 저장 ON, `discospool/kvcache` |
| CPU / GPU staging | CPU 4GiB / GPU 10GiB, 양쪽 동일 |
| CPU 프리페치 입장 기준 | ON에서 공유 GPU pool 총 할당량 기준 5GiB |
| 청크 / 작업 풀 | 128토큰 / DAOS I/O 16, 메타데이터 16 |
| 기타 | async loading ON, vLLM prefix caching OFF, eager, GPU memory utilization 0.75 |
| 비교 순서 | OFF r1 → ON r1 → ON r2 → OFF r2 |
| 반복 | 조건별 새 서버 2개, 서버별 동시성 1·4 각각 3회 × 4요청 |
| 표본 수 | 조건·동시성별 24요청, warm 측정 합계 96요청 |

각 서버에서 별도 warmup 입력, fill, 준비 확인을 거쳐 warm 구간을 측정했다. 서버 시작·모델 로딩·fill·재시작 검증은 아래 warm 통계에서 제외했다. CPU 캐시 정책은 그대로 유지했다. warmup 포함 기본 KV working set 약 3.125GiB가 CPU 4GiB에 들어가는 조건이다.

TTFT는 HTTP 요청 시작부터 첫 비어 있지 않은 응답 텍스트까지, E2E는 요청당 64토큰 응답 종료까지다. 모델 호출 시간만이 아니라 요청 처리·대기·전송을 포함하지만, 모델 로딩이나 여러 단계 에이전트 실행 전체 시간은 아니다.

네 서버 모두 실제 로드된 native library 경로를 확인했다:

- `/opt/daos-gds-gpu/lib64/libdaos.so.2.8.0`
- `/opt/daos-gds-gpu/prereq/release/mercury/lib64/libmercury.so.2.4.1`
- `/opt/ofi-cuda/lib64/libfabric.so.1.25.0`

## 2. 측정 결과

24개 요청 원시값을 합쳐 평균을 계산했다. P95는 정렬한 24개 값의 95% 위치를 선형 보간했다. 같은 서버·batch의 요청은 서로 독립적인 실험 반복이 아니므로 이 표로 통계적 유의성을 주장하지 않는다.

| 동시 요청 | GPU 프리페치 | 평균 TTFT (ms) | TTFT P95 (ms) | 평균 E2E (ms) |
|---:|---|---:|---:|---:|
| 1 | OFF | 53.79 | 54.41 | 747.73 |
| 1 | ON | 52.62 | 53.27 | 735.18 |
| 4 | OFF | 137.47 | 142.36 | 846.02 |
| 4 | ON | 103.37 | 109.87 | 812.91 |

| 동시 요청 | 평균 TTFT 감소율 | 평균 E2E 감소율 |
|---:|---:|---:|
| 1 | 2.16% | 1.68% |
| 4 | 24.81% | 3.91% |

서버별 TTFT 평균도 같은 방향이었다:

| 조건 | 동시성 1 r1 / r2 (ms) | 동시성 4 r1 / r2 (ms) |
|---|---:|---:|
| OFF | 53.97 / 53.61 | 137.97 / 136.98 |
| ON | 53.05 / 52.20 | 104.15 / 102.59 |

동시성 1의 개선은 작다. 장시간·추가 반복으로 확인하기 전에는 큰 효과로 해석하지 않는다. 예전 DRAM OFF 실험과 수치를 섞지 않고, 이번에 다시 측정한 DRAM ON + 프리페치 OFF를 기준으로 삼았다.

## 3. 왜 달라졌나

| 요청당 평균 구간 | OFF, 동시성 1 | ON, 동시성 1 | OFF, 동시성 4 | ON, 동시성 4 |
|---|---:|---:|---:|---:|
| retrieve 로그 시간 | 13.64ms | 1.29ms | 13.63ms | 1.24ms |
| 별도 CPU staging 복사 batch | 없음 | 12.58ms | 없음 | 12.59ms |

OFF에서는 retrieve가 CPU에 있는 KV를 GPU로 가져오는 일을 맡는다. ON에서는 약 12.6ms의 복사를 미리 수행하고, retrieve는 GPU staging에서 vLLM KV로 옮기는 약 1.2ms 구간이 된다.

**전송 비용이 사라진 것은 아니다.** 모델 실행 경로에서 처리하던 일을 비동기 작업으로 옮겼고, 동시 요청에서 모델 스케줄링과 겹칠 수 있게 한 효과로 해석할 수 있다. 관측한 retrieve 시간 감소와 TTFT 개선은 이 설명에 부합하지만, 정밀 CUDA timeline을 측정한 것은 아니므로 겹침 비율까지 확정하지 않는다. 작업 스레드·복사 stream은 1개라 H2D batch끼리는 순서대로 처리한다.

복사 로그 시간은 worker 큐 대기를 제외하며, retrieve 시간과 단순히 더한 값이 TTFT는 아니다. 로그의 수백 GB/s retrieve throughput도 DAOS 네트워크 대역폭이 아니라 이미 GPU에 올라온 데이터를 소비하는 구간이다.

## 4. 정확성·동작 확인

- 단위 테스트 **72개 통과**. 취소 시 전송 중 버퍼 유지, 부분 할당 실패 정리, 복사 예외 정리, 공유 pool 사용량에 따른 CPU fallback, OFF 복귀를 포함한다.
- **DFS와 object 각각 실제 1MiB 바이트 일치 검증 통과.** CPU 캐시 hit의 GPU staging 결과, 예산을 작게 설정해 강제한 CPU fallback 결과, CPU 캐시 제거 후 DAOS 읽기 결과를 비교했다. GPU 참조 해제 후 공유 pool 할당량이 원래 값으로 복귀하는 것도 확인했다.
- warm 96요청 모두 HTTP cached tokens 4095, 생성 64토큰. 동일 입력·동시성별로 OFF/ON 및 반복 실행의 **생성 토큰 SHA-256이 모두 동일**했다. 임의 입력 전체의 정확성을 보장하는 검사는 아니다.
- OFF warm 48요청: DAOS prefetch batch 0, CPU staging prefetch 0.
- ON warm 48요청: DAOS prefetch batch 0, CPU staging prefetch **48**, CPU fallback **0**. 이는 요청 batch 수이지 청크별 DAOS API/RPC 수가 아니다.
- ON r1 종료 후 새 `kv_consumer` 프로세스에서 4입력을 다시 요청했다. CPU staging prefetch 0, DAOS prefetch batch **4**, 모두 cached tokens 4095로 DAOS 복사본 재사용을 확인했다. 이 재시작의 TTFT는 warm 성능 통계에 넣지 않았다.
- native GDS 백엔드·DFS/object 바인딩·`run_vllm.sh`·기본 YAML·C shim 소스 및 `.so`는 기존 baseline 사본과 바이트가 동일했다.

CPU hot-cache metric은 현재 vLLM `/metrics`에서 노출되지 않아 `null`로 기록했다. 실제 CPU 항목 수가 0이라는 뜻이 아니다. 이번 byte test의 직접 확인과 warm 구간의 읽기 출처를 함께 사용했다.

모든 측정 요청이 끝난 뒤 ON r1 서버 종료 시 Python logging의 `RuntimeError: reentrant call inside <_io.BufferedWriter name='<stdout>'>`가 1회 기록됐다. 로그상 SIGTERM/abort 종료 처리 중 stdout flush에서 발생했으며, 측정 요청이나 KV 복사 오류는 아니었다. 숨기지 않고 원본 `main_server.log`에 보존했다. 이 벤치는 프로세스 그룹 종료 방식이므로 모든 정상 shutdown callback의 수행을 입증하지 않는다. 종료 후 GPU 사용량은 4MiB로 복귀했다.

## 5. 남은 범위

이번 결과는 **object + DRAM warm hit + 동시성 1/4**에 한정한다. DFS 모델 추론의 ON/OFF 성능 비교, DRAM 용량 초과에 따른 교체, 혼합 CPU/DAOS hit, 실제 GPU staging 포화, 장시간 취소·종료 경합은 아직 검증하지 않았다. 실제 GPU fallback 검사는 용량을 채워 터뜨린 실험이 아니라 입장 기준을 의도적으로 낮춘 기능 검사다.

따라서 다음 연구는 이번 프리페치 효과와 분리해서 진행하는 것이 좋다: DRAM보다 큰 working set에서 어떤 KV를 보관할지, 그리고 GPU staging이 부족할 때 어떤 요청을 먼저 올릴지 비교한다. 이번 구현에는 새 보관·우선순위 정책을 넣지 않았다.

## 6. 원본과 사용법

- [OFF r1 요약](dram_gpu_prefetch_off_20260922_r1/summary.json), [OFF r2 요약](dram_gpu_prefetch_off_20260922_r2/summary.json)
- [ON r1 요약](dram_gpu_prefetch_on_20260922_r1/summary.json), [ON r2 요약](dram_gpu_prefetch_on_20260922_r2/summary.json)
- [재시작 검증](dram_gpu_prefetch_on_20260922_r1/restart_hit.json)
- [object byte test](dram_gpu_prefetch_bytes_object_20260922.json), [DFS byte test](dram_gpu_prefetch_bytes_dfs_20260922.json)
- [실행·롤백과 구현 설명](DRAM_GPU_PREFETCH_GUIDE_KO.md)

각 실행 폴더에 `warm_r*_c*.json`의 요청별 원시값, 해당 구간 로그, 전체 서버 로그, 실행 명령, 유효 설정, 실행 소스 사본과 해시가 있다. 바이트 검증에 사용한 UUID 키만 정리했고, 벤치 캐시는 재검증용으로 유지했다. 공용 컨테이너 전체를 삭제하지 않았다.

원복은 프로세스를 재시작하며 **`--dram on --gpu-prefetch off`**를 사용하면 된다. DRAM 보관까지 끄려면 `--dram off --gpu-prefetch off`를 사용한다.
