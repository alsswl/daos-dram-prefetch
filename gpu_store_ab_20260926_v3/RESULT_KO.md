# 새 KV 저장의 DRAM 우회 구현·비교 결과

실행일: 2026-09-26. 최종 비교 디렉터리: `/root/discos_minji/gpu_store_ab_20260926_v3`.

**KV payload의 CPU 임시 버퍼 왕복을 제거했다. 이번 고정 입력 실험에서
저장 준비·제출 시간은 약 10~13배 짧아졌고, cold 요청 전체 시간은 약 4~5% 줄었다.**
이는 네트워크 저장 자체가 10배 빨라졌다는 뜻이 아니다.

## 1. 구현 내용

```text
기존 host_staged
vLLM paged GPU KV → DRAM 임시 KV → GPU staging → DAOS object

추가 gpu_direct
vLLM paged GPU KV → GPU staging → DAOS object
```

LMCache 엔진의 새 KV 객체 할당을 DAOS 백엔드의 GPU allocator로 바꿨다.
기존 GPU connector가 CPU 대신 GPU 객체로 KV를 모으고, StorageManager가
그 객체를 복사 없이 DAOS 백엔드에 전달한다. gather 이후 CUDA 동기화와
비동기 쓰기의 참조 보유·해제를 처리했다. 엔진이 `fmt=None`을 넘기는 경우도
기존 CPU allocator와 같은 비-layerwise 기본 형식을 부여한다.

옵션은 `extra_config.daosgds.store_path: gpu_direct`다. 기본값은 기존 `host_staged`를
유지한다. `local_cpu: false`가 필요하며 이번 비교는 양쪽 모두 DRAM 캐시 보관 OFF다.
키/메타데이터/제어 정보에는 여전히 CPU 메모리를 사용한다. 비활성 CPU allocator의
예약 메모리 제거까지 구현한 것은 아니다. KV의 실제 목적지와 payload 복사 제거를 검증했다.

구현은 `/root/discos_minji` 안의 런타임 패치이며 `/root/discos`, 설치된 LMCache,
vLLM 파일과 서버 코드는 수정하지 않았다. 읽기 경로와 저장 형식은 유지한다.

## 2. 비교 조건

| 항목 | 양쪽 공통 조건 |
|---|---|
| 모델 | Qwen/Qwen3-14B, BF16 |
| GPU | NVIDIA H100 NVL |
| 스택 | 설치된 vLLM 0.25.1 / LMCache 0.5.2 |
| 저장 방식 | object: DFS 우회, `discospool/kvcache` |
| 입력·출력 | 서로 다른 8192토큰 입력 8개, 각 64토큰 생성, temperature=0, seed=0 |
| KV 청크 | 128토큰 = 20MiB, 요청당 64청크 = 1.25GiB |
| GPU staging | 10GiB |
| CPU | 캐시 보관 OFF, 기존 host allocator 예약 설정 8GiB |
| I/O / 메타데이터 스레드 | 각각 16개 |
| 로딩 | DAOS 비동기 로딩 ON, vLLM prefix caching OFF |
| vLLM 스케줄링 | max model len 16384, max_num_seqs 16, chunked prefill 8192토큰 예산 |
| 반복 | 각 방식 3회, 매번 새 프로세스·새 UUID namespace |
| 순서 | GPU→host, host→GPU, GPU→host |
| 워밍업 | 측정 입력과 첫 청크가 다른 입력으로 저장과 읽기 확인, 측정에서 제외 |
| cold / warm | cold 첫 저장 후 동일 입력 warm 재사용. warm은 skip-save로 읽기만 함 |

각 프로세스에서 입력 4개를 순차 처리(c1), 다른 입력 4개를 동시에 제출(c4)했다.
c4는 HTTP 요청 동시 제출을 뜻하며 vLLM이 4개 prefill을 항상 동시에 계산한다는 뜻은 아니다.
각 조건은 3회 × 4요청 = 12요청, 전체 측정은 96요청이다. 모델 시작 시간은 제외했다.

`save_decode_cache: true`를 양쪽에 적용했다. 초기 예비 실험에서 false일 때
chunked prefill 경계의 마지막 프롬프트 청크가 빠져, 기대 저장량이 달라지는 경우를
발견했다. true로 두어 이후 decode 단계에서도 해당 청크를 저장하게 했으며,
실제 저장량이 요청당 정확히 8192토큰인지 검증했다. 생성 64토큰은 추가 128토큰
청크를 만들지 않는다. 예비 실험 v1/v2는 아래 결과에서 모두 제외했다.

실제 로딩된 네이티브 라이브러리는 모든 실행에서 같았다.

- `/opt/daos-gds-gpu/lib64/libdaos.so.2.8.0`
- `/opt/daos-gds-gpu/prereq/release/mercury/lib64/libmercury.so.2.4.1`
- `/opt/ofi-cuda/lib64/libfabric.so.1.25.0`

## 3. cold: 새 캐시 저장 결과

아래는 3회 반복의 평균이다. TTFT는 HTTP 요청 시작부터 첫 생성 token ID 수신까지다.

| 측정 항목 | 기존 DRAM 경유 | GPU 직접 | 시간 감소 |
|---|---:|---:|---:|
| c1: 요청당 KV 저장 준비·제출 | 56.77ms | 5.43ms | 90.4% (10.45배 단축) |
| c4: 요청당 KV 저장 준비·제출 | 71.78ms | 5.60ms | 92.2% (12.82배 단축) |
| c1: 요청당 평균 TTFT | 664.39ms | 613.06ms | 7.7% |
| c1: 요청당 평균 전체 시간 | 1374.27ms | 1316.54ms | 4.2% |
| c4: 요청당 평균 TTFT | 2119.55ms | 2001.50ms | 5.6% |
| c4: 네 요청 모두 완료한 시간 | 3382.32ms | 3199.26ms | 5.4% |

`저장 준비·제출`은 LMCache `Stored ... cost`를 요청별로 합산한 값이다.
객체 할당, KV gather, 필요한 복사/동기화, 비동기 쓰기 제출을 포함하지만
**DAOS 쓰기 완료를 기다린 시간은 아니다.** 따라서 10배를 DAOS 네트워크 성능 향상으로 해석하면 안 된다.

반면 HTTP 종료 외에 실제 DAOS put 개수와 staging 반환도 별도로 확인했다.
이번에는 해당 완료 조건이 HTTP 요청 종료 전 이미 충족되어, 완료를 포함한 batch 시간은
HTTP batch 시간과 사실상 같았다. 이 완료 검사는 5ms 주기 관측에 기반한 것이다.

| cold 네 요청 처리 시간의 반복 범위 | 기존 | GPU 직접 |
|---|---:|---:|
| c1: 네 요청 순차 완료 | 5469.14~5520.49ms | 5249.95~5300.59ms |
| c4: 네 요청 동시 제출 후 완료 | 3379.30~3387.99ms | 3190.11~3209.61ms |

## 4. 실제 DRAM 우회 여부

실험용 probe로 GPU connector가 받는 객체의 device와 StorageManager의 복사를 기록했다.

| cold 측정 전체, 방식별 24요청 | 기존 | GPU 직접 |
|---|---:|---:|
| GPU connector의 KV gather 목적지 | CPU | CUDA |
| gather payload 합계 | 30GiB | 30GiB |
| StorageManager CPU→GPU 복사 | 30GiB | **0GiB** |
| DRAM hot-cache 보관 청크 | 0 | 0 |
| GPU staging 할당 실패 | 0 | 0 |

새 방식은 GPU 객체를 그대로 DAOS GPU I/O에 전달했다. CPU로 fallback하지 않는다.
이는 LMCache payload 경로 관측이며, 모든 하위 네트워크 계층을 PCIe 분석기로 추적했다는 뜻은 아니다.

## 5. warm 읽기와 정확성

| warm 측정 항목 | 기존 | GPU 직접 |
|---|---:|---:|
| c1 평균 TTFT | 95.50ms | 96.40ms |
| c1 요청당 평균 전체 시간 | 790.75ms | 785.69ms |
| c4 평균 TTFT | 227.30ms | 231.64ms |
| c4 네 요청 모두 완료 | 1024.07ms | 1018.37ms |

읽기 경로를 바꾸지 않았으므로 이 작은 차이를 읽기 개선 효과로 주장하지 않는다.

- cold 측정 48요청: 전부 cached tokens 0.
- warm 측정 48요청: 전부 cached tokens 8191. 8192토큰 캐시가 있어도 vLLM이 마지막 한 토큰을 계산한다.
- 각 실행은 워밍업 포함 정확히 576청크 저장 완료, 종료 전 staging 반환과 CPU hot-cache 0 확인.
- 동일 입력의 host/GPU cold 출력: 24쌍 모두 생성 64개 token ID가 일치.
- 동일 입력의 host/GPU warm 출력: 24쌍 모두 일치.
- 각 방식에서 cold/warm 출력: 각각 24쌍 모두 일치.
- 별도 실GPU 검증: 40층 BF16 KV 20MiB의 CPU/gpu gather 결과, DAOS 왕복, paged GPU scatter 결과가 바이트 단위 일치.
- 최종 단위 테스트: **99 passed**, 기존 OpenTelemetry logging deprecation 경고 58개.
- 실행 소스 스냅샷과 현재 파일의 SHA-256 일치 확인. 최종 실행 로그에 쓰기 실패/할당 실패는 없었다.

서버 종료 시 기존 multiprocessing semaphore 정리 경고가 있었다. 데이터 검증은 통과했지만
모든 종료·장애 상황을 검증했다는 의미는 아니다. 검증용 UUID dkey만 제거했고 기존 사용자 데이터는 삭제하지 않았다.

## 6. 주의할 점: GPU staging 점유가 늘 수 있다

| 관측 최대 staging 점유, 세 반복 중 최대 | 기존 | GPU 직접 |
|---|---:|---:|
| cold c1 | 1.25GiB | 1.25GiB |
| cold c4 | 1.64GiB | **3.28GiB** |
| warm c4 | 3.38GiB | 3.71GiB |

새 쓰기는 처음부터 GPU staging에 객체를 확보한다. 복사 왕복은 없어지지만
읽기 프리페치와 쓰기가 같은 10GiB 풀을 공유하므로 staging 압박은 별도 고려해야 한다.
이번 조건에서는 할당 실패가 없었지만 더 큰 요청이나 동시성에서도 안전하다고 단정할 수 없다.
단일 관측 최대값을 일반적인 필요 용량으로 해석하지 않는다.

## 7. 해석 범위

이번 결과는 고정 8192토큰 입력의 쓰기 경로 비교다. full agentic DiscoveryBench를
다시 실행한 결과가 아니다. 개선은 CPU 왕복 제거에 따른 로컬 저장 준비 비용 감소로
설명할 수 있지만, 전체 요청 시간에는 모델 계산/생성, 스케줄링, 통신이 함께 포함된다.

여기서 cold는 새 namespace의 KV cache miss다. 물리 DAOS 서버의 모든 메모리 캐시나
OS 캐시를 비운 것은 아니다. namespace가 달라 dkey별 타깃 배치도 완전히 같다고
보장하지 않는다. 3회 반복·동일 스택·동일 입력으로 비교했지만, 보편적인 4~5% 개선이나
통계적 유의성을 확정한 것은 아니다. 계측 자체의 오버헤드도 양쪽에 포함된다.

## 8. 파일과 재실행

- 원시 측정: [results.json](results.json), [workload.json](workload.json), [plan.json](plan.json)
- 각 `r*_host_staged`, `r*_gpu_direct`: 설정, 서버 로그, native_maps, trace JSONL, 조건별 결과.
- 실행 당시 소스: `executed_sources/`.
- [사용·롤백 안내](../GPU_STORE_GUIDE_KO.md), [GPU 직접 쓰기 설정](../lmcache_config_daosgds_gpu_store.yaml).

```bash
cd /root/discos_minji
./venv/bin/python3 compare_gpu_store.py --output /root/discos_minji/gpu_store_ab_NEW --repeats 3
./venv/bin/python3 analyze_gpu_store.py /root/discos_minji/gpu_store_ab_NEW
```

실험 서버 프로세스는 모두 종료했다. 비교용 캐시는 각 UUID namespace에 남겨두었고,
공용 컨테이너를 삭제하거나 기본 실행 YAML을 변경하지 않았다.
