# GPU staging 프리페치 동시성 진단

실행일: 2026-09-21. 작업 디렉터리: `/root/discos_minji`.

## 결론

프리페치가 GPU staging에 KV를 미리 읽어 놓고, retrieve가 이를 소비한 뒤 공간을 반환하는 것을 실제 allocator 로그에서 확인했다. 요청 수가 늘수록 staging 사용량과 대기가 증가했다. **이번 조건에서는 공간 부족이나 할당 실패는 발생하지 않았다.**

영역 자체는 서버 초기화 때 10GiB를 미리 확보한다. 프리페치마다 새 staging 영역이 만들어지는 것이 아니라, 그 안에서 필요한 청크를 할당하고 반환한다. 따라서 `nvidia-smi`의 전체 GPU 메모리 사용량만으로 내부 점유율을 알아낼 수 없다.

## 실험 조건

| 항목 | 조건 |
|---|---|
| GPU / 모델 | H100 NVL / Qwen/Qwen3-14B |
| 구성 | vLLM 하나 + 프로세스 내 LMCache, MP 모드 아님 |
| 저장 경로 | object, DFS 우회만 측정 |
| 청크 / staging | 128토큰 / 10GiB |
| I/O / 메타데이터 스레드 | 각각 16개 |
| 입력 / 출력 | 서로 다른 고정 입력 8개, 각 8,192토큰 / 각 64토큰 생성 |
| 요청 하나의 프리페치 | 64청크 × 20MiB = 1.25GiB |
| 동시 요청 수 | 1, 2, 4, 8 |
| 반복 | 각 조건 3회, 각 회 8요청 → 조건당 24요청, 총 96요청 |
| 실행 순서 | 1→2→4→8 / 8→4→2→1 / 1→2→4→8 |
| 캐시 조건 | 별도 프로세스에서 fill 후 재시작, 캐시 hit 확인 후 측정 |
| 저장 차단 | 측정 서버 `kv_role=kv_consumer`, 백엔드 `store=false` |
| vLLM 설정 | eager, GPU 메모리 비율 0.75, 내장 prefix caching 끔 |
| 프리페치 | async loading 및 AsyncMultiSerializer 켬 |

DiscoveryBench 문제를 푼 실험이 아니라 **캐시 hit 상황에서 staging 사용과 대기를 관찰하는 고정 입력 진단**이다. 측정 전 별도 warmup 및 8개 입력의 재시작 hit 검증을 수행했다. 스토리지 서버 캐시를 비우지는 않았다. 모든 측정 요청은 8,191토큰 hit였으며, 나머지 한 토큰은 vLLM이 계산한다. 프리페치 자체는 청크 단위이므로 8,192토큰에 해당하는 데이터를 읽는다.

## 결과

최대값은 해당 동시성의 3회 실행 전체 중 최대, TTFT는 24개 요청의 평균이다.

| 동시 요청 | staging 최대 사용량 | 프리페치 실행 대기 최대 | 읽기 완료→retrieve 시작 대기 최대 | 평균 TTFT | 할당 실패 |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.25GiB | 0.08ms | 13.09ms | 95.13ms | 0 |
| 2 | 2.50GiB | 1.06ms | 34.83ms | 139.21ms | 0 |
| 4 | 3.89GiB | 0.44ms | 73.29ms | 219.23ms | 0 |
| 8 | 5.68GiB | 111.33ms | 140.13ms | 381.24ms | 0 |

관찰된 동시 프리페치 coroutine 수는 각각 최대 1, 2, 4, 4개였다. coroutine 수는 DAOS RPC 수나 실제 동시 네트워크 전송 개수가 아니다.

동시성 8에서 프리페치 완료 후 아직 allocator에 반환되지 않은 버퍼는 최대 5GiB였다. 이 수치에는 retrieve가 시작되어 소비 중인 객체도 포함한다. 위 표의 완료→retrieve 시작 대기는 그와 별도로 측정했다. 각 측정 회차가 끝날 때 staging 할당량은 0으로 돌아왔고, 부분 읽기 반환은 없었다.

## 쉽게 해석하면

staging을 대기실이라고 보면 된다.

1. DAOS에서 읽는 동안 대기실의 자리를 사용한다.
2. 읽기를 끝내도 vLLM이 가져가기 전까지 자리는 바로 비워지지 않는다.
3. vLLM이 retrieve에서 데이터를 소비하고 참조를 해제해야 자리가 반환된다.

이번 실행의 serializer는 전체 512청크 중 절반인 256청크, 즉 **동시 실행 예산 5GiB**를 사용한다. 요청 하나가 64청크라서 최대 4개 요청의 프리페치가 실행 구간에 들어갈 수 있었다. 동시 요청 8개에서는 일부 요청이 앞선 프리페치가 끝나기를 기다렸다.

이 예산은 프리페치 coroutine 실행에만 적용된다. 읽기가 끝나면 실행 예산은 반환되지만, GPU 버퍼는 retrieve가 소비할 때까지 남아 있을 수 있다. 그래서 **5GiB 실행 예산과 실제 staging 최대 점유량 5.68GiB는 모순이 아니다.**

다만 10GiB 공간 자체는 남아 있었다. 따라서 이번 결과를 “staging이 꽉 차서 느려졌다”라고 표현하면 안 된다. 확인한 것은 **동시 실행 제한에 따른 대기와, 프리페치 완료 후 소비까지의 체류 시간**이다. TTFT 증가 전체를 이 두 원인만으로 설명할 수도 없다. lookup, I/O, 스케줄링, GPU 복사와 계산, HTTP 처리 등이 함께 포함된다.

## 계측과 검증

- `DAOS_GDS_STAGING_TRACE`를 설정한 측정 프로세스에서만 계측을 활성화했다. 기본 실행에서는 비활성이다.
- GPU allocator 내부 `allocate/free` 및 batch 변형을 기록했다. CUDA의 reserved 메모리가 아니라 실제 pool 할당 바이트를 기록한다.
- serializer의 원래 `run`을 감싸 대기와 실행 시작을 관찰했다. 예산·스케줄링 정책은 바꾸지 않았다.
- prefetch 완료, retrieve 시작·반환을 기록했다. 계측 때문에 추가 CUDA 동기화를 넣지 않았다. retrieve 반환 시각 자체를 별도 CUDA 이벤트로 측정한 GPU copy 완료 시각으로 해석하면 안 된다.
- 실제 로드된 라이브러리는 `/opt/daos-gds-gpu/lib64/libdaos.so.2.8.0`, 해당 DAOS의 Mercury, `/opt/ofi-cuda/lib64/libfabric.so.1.25.0`이었다.
- 컨테이너 속성은 POSIX, HEALTHY, `rd_fac=0`이었다.
- 측정 96요청 전부 캐시 hit 검사를 통과했다. warmup/hit 검증까지 포함하면 측정 프로세스의 prefetch는 105회였다.
- Python 단위 테스트: `53 passed`. OpenTelemetry deprecation 경고는 있었다.
- 완료 후 이 실험이 시작한 서버 프로세스를 종료했으며 GPU 메모리는 4MiB로 돌아왔다. 서버 종료 과정에는 EngineDeadError 및 semaphore cleanup 경고가 남았으나, 측정 요청 오류나 staging 할당 실패는 없었다.
- `/root/discos`와 설치된 LMCache 파일은 수정하지 않았다. 자체 백엔드의 선택적 runtime wrapper만 추가했다.

계측에는 이벤트별 JSON 로그 기록 비용이 있다. 따라서 TTFT는 계측을 켠 상태의 진단값이지, 계측 없는 서비스의 최종 성능 수치가 아니다. 이번에는 object 경로와 한 가지 입력 크기만 측정했으므로 DFS 결과 또는 일반적인 포화 임계값으로 확대 해석하지 않는다.

## 재실행

결과 폴더는 기존에 없는 이름을 지정한다. 매 실행마다 새 UUID 캐시 영역을 만들며 공용 컨테이너나 기존 캐시를 삭제하지 않는다.

```bash
cd /root/discos_minji
./venv/bin/python3 staging_pressure.py \
  --mode object --context-length 8192 --chunk-size 128 \
  --gpu-buffer-gb 10 --repeats 3 \
  --output /root/discos_minji/staging_pressure_next
```

`--dry-run`은 설정·계획 파일만 만들고 모델과 DAOS를 실행하지 않는다. `--mode dfs`도 지원하지만 이번 실험에서는 실행하지 않았다.

이번 실험용 DAOS namespace `minji-staging-9cd9107365744f0a8376f65297e0b05a:`는 보존했다. 입력 9개(별도 warmup 포함)의 기본 KV payload는 약 11.25GiB이며 메타데이터 등은 별도다.

다음 단계에서는 입력 길이 또는 staging 크기 중 하나만 바꿔 공간 부족이 발생하는 조건을 확인할 수 있다. 그 뒤 “읽기 완료 시 예산 반환”과 “retrieve 소비 후 예산 반환” 같은 정책을 비교할 근거를 마련할 수 있다. 현재 결과만으로 후자가 더 빠르다고 주장할 수는 없다.

## 증거 파일

- [회차별 요약](summary.json), [요청별 출력·시간·hit](passes.json)
- [전체 trace 요약](trace_totals.json), [원시 allocator/lifecycle trace](trace.2804789.jsonl)
- [실행 조건·소스 해시](manifest.json), [실제 라이브러리](measure_native_maps.json)
- [측정 서버 로그](measure_server.log), [fill 서버 로그](fill_server.log)
- [실험 스크립트](../staging_pressure.py), [계측 코드](../lmcache_daos/staging_trace.py)
