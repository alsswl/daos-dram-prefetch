# DRAM 보관·GPU staging 프리페치 3조건 비교 결과

2026-09-22. Qwen3-14B, DAOS object GPU-direct 경로, `/root/discos_minji`.

**모든 조건을 새 프로세스·새 KV 캐시 영역에서 시작했다. 첫 실행(cold)과 재사용(warm)은 분리해서 측정했다.** 단일 요청에서는 DRAM 보관으로 평균 TTFT가 약 20% 감소했다. 동시 요청 4개에서는 DRAM 보관만으로 큰 차이가 없었고, DRAM→GPU staging 프리페치를 추가했을 때 평균 TTFT가 DRAM 보관만 한 조건보다 24.08% 감소했다.

이 결과는 DRAM에 작업 데이터가 모두 들어가는 제한된 합성 워크로드의 관측값이다. full agentic DiscoveryBench나 DRAM 용량 초과 실험은 아니다.

## 1. 세 조건

| 표기 | 실제 설정 | 재사용 데이터 경로 |
|---|---|---|
| DAOS만 | DRAM KV 보관 OFF, CPU 프리페치 OFF | DAOS → GPU staging → vLLM KV cache |
| DRAM 보관 | DRAM KV 보관 ON, CPU 프리페치 OFF | DRAM → vLLM KV cache |
| DRAM + 프리페치 | DRAM KV 보관 ON, CPU 프리페치 ON | DRAM → GPU staging → vLLM KV cache |

세 조건 모두 DAOS 저장은 켰다. “DAOS만”도 임시 CPU 버퍼와 CPU allocator를 사용한다. DRAM을 전혀 사용하지 않는다는 뜻이 아니라 **재사용 KV를 DRAM에 보관하지 않는다**는 뜻이다. 캐시 보관·교체 정책과 DAOS 저장 형식은 이번 실험에서 변경하지 않았다.

## 2. 고정 조건과 cold 시작

- 모델: Qwen/Qwen3-14B, BF16, H100 NVL. vLLM 0.25.1, LMCache 0.5.2, PyTorch 2.11.0, transformers 5.17.0.
- 입력: 서로 다른 token ID 입력 4개 × 4096토큰. 출력은 요청당 64토큰, temperature/seed 0. 모델 출력을 다음 입력에 붙이지 않는다.
- 청크 128토큰, CPU allocator 4GiB, 공통 GPU staging 10GiB, 프리페치 ON 입장 기준 5GiB.
- 비레이어별·async loading ON, vLLM prefix caching OFF, eager, GPU memory utilization 0.75, max model length 8192.
- DAOS I/O/meta 작업 풀 각각 16. 모든 실행의 실제 vLLM GPU KV cache 용량은 **268,176토큰**으로 동일했다.
- DAOS pool/container는 `discospool/kvcache`. 조건마다 새 UUID namespace를 사용했고, 이전 캐시를 삭제하거나 재사용하지 않았다.
- 동시성 1·4 각각 세 조건을 새 프로세스로 3회씩 반복했다. 조건 순서를 순환해 각 조건이 첫째·둘째·셋째 위치에 한 번씩 오도록 했다.
- 각 서버: **첫 cold 4요청 → 준비 확인 4요청(각 1토큰 생성) → warm 4요청 × 3회**. 측정 전 추론 warmup은 하지 않았다.
- 총 18개 새 서버, cold 72요청, warm 216요청. 준비 확인 72요청은 별도 원본에 남기고 성능 통계에서는 제외했다.

기본 KV working set은 `4 × 4096 × 163840 bytes = 2.5GiB`로 CPU 4GiB에 들어간다. 따로 warmup 입력을 넣었던 이전 실험의 3.125GiB와 구분한다.

여기서 cold는 **측정 입력의 KV 캐시가 비어 있음**을 뜻한다. OS/모델 파일 캐시·DAOS 내부 캐시·GPU thermal 상태까지 초기화한 실험은 아니다. 동시성 1의 네 요청 중 첫 번째만 프로세스의 첫 추론이며, 나머지는 새로운 입력의 KV-cold 요청이다. 서버 시작·모델 로딩 시간은 아래 HTTP 지연에서 제외했다.

## 3. 첫 실행: cold

모든 cold 요청은 cached tokens **0**, DAOS 읽기 prefetch batch **0**, CPU staging prefetch batch **0**이었다. 아래 각 행은 3개 새 프로세스의 12요청 평균이다.

| 동시 요청 | 조건 | 평균 TTFT (ms) | TTFT P95 (ms) | 평균 E2E (ms) |
|---:|---|---:|---:|---:|
| 1 | DAOS만 | 358.10 | 457.93 | 1030.86 |
| 1 | DRAM 보관 | 345.80 | 393.32 | 1015.32 |
| 1 | DRAM + 프리페치 | 350.57 | 400.38 | 1022.52 |
| 4 | DAOS만 | 1288.80 | 1326.23 | 2021.36 |
| 4 | DRAM 보관 | 1295.14 | 1341.26 | 2027.27 |
| 4 | DRAM + 프리페치 | 1300.80 | 1337.74 | 2030.17 |

이 구간은 모델 prefill·초기 실행·저장 경로를 포함하지만, 재사용 KV 읽기나 CPU staging 프리페치 효과는 포함하지 않는다. 따라서 cold의 작은 차이를 프리페치 효과라고 해석하면 안 된다. 특히 DAOS만/동시성 1의 첫 반복 평균은 375.24ms, 이후 두 반복은 349.11/349.94ms로 초기·시간 변동이 있다. 이상치라고 임의 제거하지 않았다.

## 4. 재사용: warm

모든 warm 요청은 cached tokens **4095**, 입력 4096토큰, 생성 64토큰이었다. vLLM이 마지막 토큰을 계산하므로 HTTP cached tokens는 4095이며, 백엔드는 청크 단위로 4096토큰을 읽을 수 있다.

각 행은 새 프로세스 3개 × 재사용 3회 × 4입력 = **36요청**의 통계다. TTFT는 요청 시작부터 첫 비어 있지 않은 텍스트까지, E2E는 해당 요청의 응답 완료까지다.

| 동시 요청 | 조건 | 평균 TTFT (ms) | TTFT P95 (ms) | 평균 E2E (ms) |
|---:|---|---:|---:|---:|
| 1 | DAOS만 | 65.48 | 74.28 | 737.08 |
| 1 | DRAM 보관 | 52.49 | 52.98 | 722.50 |
| 1 | DRAM + 프리페치 | 51.57 | 52.21 | 724.42 |
| 4 | DAOS만 | 135.90 | 151.80 | 844.57 |
| 4 | DRAM 보관 | 134.35 | 139.46 | 841.98 |
| 4 | DRAM + 프리페치 | 101.99 | 108.80 | 809.13 |

평균값의 감소율(양수는 감소, 음수는 증가):

| 동시 요청 | 변경 | TTFT 감소율 | E2E 감소율 |
|---:|---|---:|---:|
| 1 | DAOS만 → DRAM 보관 | 19.84% | 1.98% |
| 1 | DRAM 보관 → DRAM + 프리페치 | 1.75% | -0.27% |
| 1 | DAOS만 → DRAM + 프리페치 | 21.24% | 1.72% |
| 4 | DAOS만 → DRAM 보관 | 1.14% | 0.31% |
| 4 | DRAM 보관 → DRAM + 프리페치 | 24.08% | 3.90% |
| 4 | DAOS만 → DRAM + 프리페치 | 24.95% | 4.20% |

**단일 요청에서 프리페치 추가의 E2E 개선은 관측되지 않았다.** TTFT는 약 0.92ms 작아졌지만 E2E 평균은 약 1.92ms 커졌다. 작은 차이이므로 일반적인 개선/악화로 단정하지 않는다.

동시성 4의 DRAM 보관만 켠 조건도 DAOS만 조건보다 평균 1.55ms 작을 뿐이다. 아래 DAOS 반복 변동에 비해 작은 차이라 뚜렷한 이득으로 주장하지 않는다.

## 5. 반복별 변동

아래는 프로세스별 warm TTFT 평균이다. 같은 서버 안의 요청 36개를 독립적인 실험 반복 36회로 보지 않는다. 독립적으로 새 프로세스를 띄운 반복은 각 조건·동시성별 3회다. 통계적 유의성이나 보편적인 성능 우위를 주장하지 않는다.

| 동시 요청 | 조건 | 반복 1 (ms) | 반복 2 (ms) | 반복 3 (ms) |
|---:|---|---:|---:|---:|
| 1 | DAOS만 | 68.52 | 64.75 | 63.17 |
| 1 | DRAM 보관 | 52.59 | 52.44 | 52.45 |
| 1 | DRAM + 프리페치 | 51.89 | 51.57 | 51.25 |
| 4 | DAOS만 | 130.08 | 142.19 | 135.43 |
| 4 | DRAM 보관 | 135.96 | 134.62 | 132.46 |
| 4 | DRAM + 프리페치 | 101.52 | 102.24 | 102.23 |

P95는 요청 원시값을 정렬한 뒤 95% 위치를 선형 보간했다. 각 구간의 표본 수가 작으므로 안정적인 운영 환경 tail latency 추정치로 확대 해석하지 않는다.

## 6. 읽기 출처와 비용

warm 구간 합계(동시성 1·4 합산):

| 조건 | DAOS prefetch batch | CPU staging prefetch batch | CPU staging fallback |
|---|---:|---:|---:|
| DAOS만 | 72 | 0 | 0 |
| DRAM 보관 | 0 | 0 | 0 |
| DRAM + 프리페치 | 0 | 72 | 0 |

이는 요청 batch 수이며 DAOS API/RPC 개수와 다르다. CPU 항목 수 metric은 HTTP endpoint에서 미노출이라 `null`로 기록했다. 0개라는 의미가 아니다.

동시성 4에서 요청당 retrieve 로그 평균은 DAOS만 **1.28ms**, DRAM 보관 **13.61ms**, DRAM + 프리페치 **1.23ms**였다. 프리페치 ON에서는 별도 DRAM→GPU 복사 batch가 평균 **12.55ms** 걸렸다. DAOS만의 DAOS prefetch batch 평균은 **42.51ms**였다.

이 값들은 서로 다른 실행 구간이다. 특히 프리페치는 겹쳐 진행될 수 있고, CPU 복사 로그는 worker 큐 대기 시간을 제외하므로 단순 합산해 TTFT를 계산하면 안 된다. 이번 개선은 **DRAM hit의 전송을 앞선 비동기 구간으로 옮긴 효과와 부합**한다. CUDA timeline으로 겹침 비율을 측정한 것은 아니다.

DRAM 조건의 warm DAOS 읽기는 0회이므로 **DRAM 프리페치 ON/OFF 차이를 GPUDirect 전송 성능 개선으로 설명하지 않는다.** DAOS 자체를 제거한 CPU-only 구성이나 staging pool의 10GiB를 vLLM에 되돌려준 구성은 비교하지 않았다.

## 7. 검증 결과와 종료 경고

- 분석기가 18개 실행·서로 다른 namespace·cold miss·warm hit·읽기 출처를 확인했다.
- 유효 YAML은 의도한 DRAM 보관 여부, 프리페치 backend/입장 기준, 새 namespace 외에 동일했다.
- 실제 native library 경로도 모두 동일했다: `libdaos.so.2.8.0`(`/opt/daos-gds-gpu`), Mercury 2.4.1, libfabric 1.25.0(`/opt/ofi-cuda/lib64`).
- 같은 phase(cold/warm)·입력·동시성 안에서 조건·반복 간 **생성 token ID 해시가 모두 동일**했다. 전체 KV payload 바이트를 이번 추론 실험에서 직접 비교한 것은 아니다.
- 실행 소스 사본의 SHA-256이 맞았고, 실행기·기본 YAML·GDS/프리페치 backend·C shim은 이번 실험 동안 수정하지 않았다. `/root/discos`는 수정하지 않았다.
- 관련 단위 테스트 **77개 통과**. 기존 별도 실제 GPU byte test 결과를 새로 수행한 것으로 합산하지 않는다.
- 측정 구간에서 캐시 hit 검증 실패, backend 생성 실패, GPU buffer full, 음수 참조 카운트는 발견하지 않았다. 서버 종료 후 GPU 메모리는 18회 모두 4MiB로 복귀했다.

**종료 경고는 있었다.** 모든 서버의 종료 시 Python resource tracker가 semaphore 정리 경고를 남겼다. `r3_c1_daos_only`에서는 마지막 응답과 metrics 수집 이후 SIGTERM/abort 종료 과정에서 `AsyncLLM output_handler failed / EngineDeadError`가 1회 발생했다. 전체 로그 순서상 엔진 종료 후 output handler가 종료된 엔진을 기다리다가 기록한 오류이며, 해당 cold/warm 요청 실패는 아니었다. 해당 case의 측정값은 다른 case와 같은 기준으로 포함하고 원본 로그 및 분석 결과에 경고를 보존했다. 이 종료 방식으로 정상 shutdown callback 전체가 검증됐다고 주장하지 않는다.

## 8. 한계와 재실행

현재 범위는 **object + 단일 vLLM + DRAM에 모두 들어가는 4개 입력 + 동시성 1/4**다. 서버 배경 부하, 새 dkey namespace에 따른 물리 배치, GPU thermal/clock 상태를 완전히 고정하지 않았다. GPU 상태는 실행 전후 기록했지만 요청 중 상세 프로파일링은 하지 않았다. DRAM 용량 초과·혼합 CPU/DAOS hit·staging 포화·장시간 부하·DFS의 동일 모델 실험은 별도 검증 대상이다.

cold 응답 완료가 DAOS 비동기 저장의 commit 완료 시점을 뜻하지 않는다. 준비 확인의 DRAM hit만으로 DAOS 영속 복사본까지 입증하지 않으며, 그 기능 검증은 기존의 별도 byte/restart 테스트와 구분한다.

```bash
cd /root/discos_minji
./venv/bin/python3 dram_cold_compare.py --repeats 3 --reuse-passes 3 \
  --output /root/discos_minji/dram_threeway_next

./venv/bin/python3 summarize_dram_threeway.py \
  /root/discos_minji/dram_threeway_20260922_v1
```

- [분석 JSON 및 검증 결과](analysis.json)
- [전체 요청 결과 모음](cases.json), [단계별 요약](summary.json), [입력 token ID](prompts.json)
- [실험 계획](plan.json), [실행 소스 해시](source_sha256.json)
- [실행·해석 안내](../DRAM_THREEWAY_GUIDE_KO.md)
- [종료 경고 원본 로그](r3_c1_daos_only/main_server.log)

각 case 폴더에는 `cold.json`, `warm_1/2/3.json`, `readiness.json`, 구간/전체 로그, 유효 설정, native library 경로, GPU 상태가 있다. DAOS 실험 데이터는 새 namespace에 유지했으며 공용 캐시를 삭제하지 않았다. 일반 실행의 프리페치 기본 OFF는 유지된다.
