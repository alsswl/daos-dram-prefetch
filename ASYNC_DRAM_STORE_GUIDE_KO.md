# GPU-direct 저장 + 비동기 DRAM 보관

2026-09-26. 수정 범위는 `/root/discos_minji`다. 원본 `/root/discos`와 설치된
LMCache/vLLM 파일은 수정하지 않는다. 기존 기본 YAML도 유지한다.

## 1. 경로와 기본 전략

```text
vLLM GPU KV → GPU staging ── GDR ──→ DAOS
                  │                  해당 청크 저장 성공
                  └── 그 후 별도 CUDA stream의 D2H ──→ DRAM 캐시
```

DAOS에 보낼 KV가 DRAM을 경유하지 않는다. DRAM 보관을 위해서는 **별도의
GPU→DRAM 복사**가 필요하다. DAOS에서 다시 읽어오는 방식이 아니라 아직
GPU staging에 있는 같은 데이터를 복사한다. CPU 캐시는 복사가 끝난 뒤에만
lookup에 공개하며, 그전에는 기존 DAOS 읽기 경로를 사용한다.

기본 정책은 **DAOS 우선 + 혼잡 시 DRAM 보관 생략 + hit 기반 LRU**다.

1. 새 KV를 GPU staging에 모아 DAOS GPU API로 저장한다.
2. DAOS 쓰기 성공 후 해당 청크를 DRAM 복사 후보로 등록한다. DAOS 실패 시 보관하지 않는다.
3. 별도 worker/stream으로 pinned CPU 버퍼에 복사한다. CPU 공간 확보와 복사는 모델 스레드에서 하지 않는다.
4. 대기·복사 중인 source의 합계가 1GiB를 넘으면 새 DRAM 보관을 생략한다.
5. 복사 worker가 시작할 때 대기 시간이 100ms를 넘은 항목도 생략한다.
6. CPU가 가득 차면 기존 LRU로 비사용 객체를 퇴출한다. 모두 pin되어 공간이 없으면 대기하지 않고 생략한다.
7. DRAM prefix lookup hit에서 LRU 순서를 갱신한다. 설치 버전의 async 경로에서
   누락되었던 갱신을 이 CPU backend 인스턴스에만 적용한다.

1GiB는 새 GPU 풀을 만드는 크기가 아니라, **DRAM 복사가 붙잡을 수 있는 기존
GPU staging source의 총량 상한**이다. 100ms는 복사 시작 전 큐 대기 제한이며
DMA가 시작된 뒤 강제 중단하는 시간 제한은 아니다.

## 2. 왜 먼저 이 정책인가

먼저 비동기 보관 자체의 이득과 복사 비용을 분리해서 측정할 수 있는 기준 구현이 필요하다.
현재 버전은 성공한 모든 새 쓰기를 후보로 삼고, 부하 때문에 필요한 경우에만 생략한다.
재사용 예측 점수나 2-hit admission을 이미 구현한 것은 아니다.

현재 LMCache CPU 조회는 첫 miss에서 멈추는 **연속 prefix** 방식이다.
따라서 청크를 임의로 골라 담으면 DRAM에 데이터가 있어도 lookup hit로 활용하지 못할 수 있다.
다음 연구 단계는 공통 prefix/요청별 연속 구간 단위 admission, 재참조 빈도,
staging 여유와 D2H 비용을 함께 고려하는 정책이 적합하다.

현재의 혼잡 생략이나 LRU eviction도 중간에 빈 청크를 만들 수 있다. 이 구현은
그 문제까지 해결한 prefix-aware 캐시 정책은 아니다. 부분 CPU hit 뒤 나머지는 DAOS를 사용한다.

추가한 `daosgds.dram_promote_on_read: true` 옵션은 **DAOS에서 GPU staging으로 읽은 KV도
DRAM으로 비동기 승격**한다. 옵션을 생략하거나 false로 두면 이전처럼 새 쓰기만 복제한다.
예제 YAML은 true로 설정했다. DAOS에 이미 있어 쓰기를 건너뛰었을 뿐 실제 읽기는 하지 않은
KV를 자동 승격하는 것은 아니다. 승격은 성공한 실제 DAOS GET을 기준으로 한다.

## 3. 설정 및 롤백

완성 설정: [lmcache_config_daosgds_async_dram.yaml](lmcache_config_daosgds_async_dram.yaml).

```yaml
local_cpu: true
max_local_cpu_size: 8
cache_policy: LRU
enable_async_loading: true
use_layerwise: false
extra_config:
  storage_plugin.daosgds.module_path: lmcache_daos.async_dram_backend
  storage_plugin.daosgds.class_name: DaosAsyncDramBackend
  daosgds.store_path: gpu_direct
  daosgds.async_dram: true
  daosgds.dram_promote_on_read: true
  daosgds.dram_mirror_max_pending_gb: 1
  daosgds.dram_mirror_max_age_ms: 100
```

위는 핵심 항목만 표시한 것이며 실제로는 풀, 컨테이너, staging 크기 등의 설정도 필요하다.

```bash
cd /root/discos_minji
env DAOSGDS_TRANSPORT=object \
  LMCACHE_CONFIG_FILE=/root/discos_minji/lmcache_config_daosgds_async_dram.yaml \
  ./run_vllm.sh ./venv/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-14B --max-model-len 16384 \
  --gpu-memory-utilization 0.75 --enforce-eager --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

GPU-direct만 남기고 DRAM 복제를 끄려면 실행 설정을
[lmcache_config_daosgds_gpu_store.yaml](lmcache_config_daosgds_gpu_store.yaml)로 바꾸고
프로세스를 재시작한다. 기본 `DaosGdsBackend`, `local_cpu: false`, `async_dram` 미지정
조합이다. 새 클래스는 async_dram 활성화를 요구하므로 flag 하나만 끄는 방식은 아니다.
DRAM 비동기 보관과 DRAM→GPU 프리페치는 다른 기능이며, 이번 설정은 후자를 켜지 않는다.

읽기 승격만 끄려면 `daosgds.dram_promote_on_read: false`로 바꾸고 재시작한다.
새 쓰기의 비동기 DRAM 복제는 계속 유지된다. 읽기 전용 소비자는 승격을 켠 상태에서
`daosgds.store: false`로 새 DAOS 쓰기를 막을 수도 있다.

## 4. 동기화·소유권

vLLM 경로에서는 GPU connector의 store stream만 완료를 기다린 뒤 DAOS 작업을 제출한다.
준비된 payload에 대해 DAOS worker가 device-wide synchronize를 다시 하지 않게 했다.
그렇지 않으면 같은 GPU의 백그라운드 D2H까지 기다려 비동기 경로를 다시 직렬화할 수 있다.
엔진을 통하지 않는 수동 StorageManager 호출은 안전을 위해 기존 device-wide barrier를 유지한다.

DRAM worker는 source GPU 객체의 별도 참조를 보유한다. D2H 완료/생략 후 참조를 반환하고,
완료된 CPU 객체만 캐시에 등록한다. CPU allocator 종료 전에 worker를 drain한다.
키·메타데이터·제어 정보에는 여전히 CPU 메모리를 사용한다.

비동기라고 자원 비용이 없어지는 것은 아니다. 이전 청크의 D2H가 다른 청크의 DAOS
I/O/추론과 대역폭을 경쟁할 수 있고, staging 반환도 늦어질 수 있다. DMA stream priority가
별도의 대역폭 보장을 제공하는 것도 아니다. 1GiB/100ms는 시작 설정이지 최적값이라는 결론이 아니다.

## 5. 검증 결과

- 단위 테스트: **111 passed**. 예산 초과, 만료, CPU 공간 부족, 복사 실패,
  참조 반환, 종료, prefix hit LRU 갱신, 불필요한 device-wide sync 생략을 확인했다.
- [실 GPU 20MiB 검증](async_dram_bytes_20260926.json): 새 객체 CUDA 할당,
  DAOS 및 DRAM 바이트 일치, DRAM 제거 후에도 DAOS 읽기 성공, staging 반환 확인.
- [예산 초과 검증](async_dram_budget_20260926_v2.json): 20MiB 청크에 대기 예산 10MiB를 주어
  DRAM 복제만 생략해도 DAOS 쓰기·읽기와 바이트 일치가 성공하는 것을 확인했다.
- Qwen3-14B, 입력 8192토큰 4개 동시 요청, 출력 64토큰, 청크128, staging10GiB,
  DRAM8GiB의 [추론 결과](async_dram_inference_20260926/results.json):
  cold에는 양쪽 hit0, 256청크(5GiB) DAOS 저장과 DRAM 복제 완료.
  warm 재실행 두 번에서 각각 DRAM256청크/DAOS0청크 hit. 생성 token ID는 cold와 모두 일치.
  복제 pending source 최대는 200MiB, 복사 오류/할당 실패/예산 생략/만료는 모두0.
- 해당 추론에서 gather 목적지는 전부 CUDA였고, 기존 StorageManager의 CPU→GPU 저장 복사는 없었다.
- DRAM을 3GiB로 줄인 [추론 검증](async_dram_mixed_20260926/results.json)도 통과했다.
  5GiB 전체를 동시에 DRAM에 보관할 수 없는 조건에서 warm 두 번 모두
  네 요청 합계 DRAM128청크/DAOS128청크 hit였고 생성 token ID가 cold와 일치했다.
  이 숫자는 batch 합계이며 각 요청 내부에서 50:50으로 나뉜다는 뜻은 아니다.
  예산 초과/만료/복사 오류/할당 실패는 없었고 복제 pending 최대는 200MiB였다.

이는 기능 검증이지 비동기 DRAM OFF/ON 반복 성능 비교는 아니다. 특히 5GiB가 DRAM8GiB에
모두 들어가므로 warm DRAM100%를 장시간 일반 워크로드의 hit율로 해석하면 안 된다.
모델 추론은 DAOS object 모드로 검증했다. 추가한 읽기 승격의 실제 20MiB 바이트 검증은
DFS/object 모두 통과했다. MP, 다중 GPU, 삭제·취소·장애와 모든 경합 조합까지
검증한 것은 아니다.

실GPU 바이트 검증에서 생성한 UUID 키만 제거했다. 추론 검증의 새 namespace는 남겼으며
공용 컨테이너와 기존 사용자 캐시를 삭제하지 않았다. 테스트 서버는 실행 후 종료한다.

## 6. 재실행

```bash
cd /root/discos_minji
PYTHONPATH=/root/discos_minji ./venv/bin/python3 -m pytest -q tests
env DAOSGDS_TRANSPORT=object ./run_vllm.sh ./venv/bin/python3 \
  tests/async_dram_roundtrip.py --result /root/discos_minji/async_dram_bytes_NEW.json
./venv/bin/python3 check_async_dram_inference.py \
  --output /root/discos_minji/async_dram_inference_NEW
```

## 7. DAOS 읽기 후 비동기 DRAM 승격

```text
DAOS → GPU staging → 기존 retrieve로 즉시 전달
             └────→ 별도 D2H worker/stream → DRAM 캐시
```

`DaosAsyncDramBackend.get_blocking()`에서 성공한 GPU 객체를 받은 뒤 복사만 예약하고
원래 객체를 반환한다. 비동기 batch 프리페치도 이 GET을 작업 풀에서 호출하므로 동일하게 적용된다.
DRAM 복사를 await하거나 완료까지 기다리지 않는다. 승격 큐 등록 오류가 나도 원래의
정상 DAOS 읽기 결과는 반환한다. 현재 지원 대상은 enable_async_loading 경로다.

쓰기 복제와 읽기 승격은 **동일한 1GiB 대기량·100ms 대기시간 예산과 worker를 공유**한다.
이미 DRAM에 있거나 같은 키의 복사가 진행 중이면 중복 복사를 생략한다.
복사가 끝나기 전에 다음 요청이 오면 그 요청은 다시 DAOS hit가 날 수 있다.
실행 취소 또는 batch의 선행 miss 때문에 최종적으로 소비되지 않은 성공 GET도 승격될 수 있다.

`write_copied`와 `read_copied` 통계를 분리해, 새 쓰기로 채워졌는지 실제 DAOS 읽기로
채워졌는지 구분한다. 기능을 켜는 것만으로 DRAM hit가 보장되지는 않는다.
용량 부족·예산 제한·LRU eviction에 따라 승격 생략 또는 이후 퇴출이 가능하다.

실 GPU 검증 결과:

- [object 20MiB](read_promotion_bytes_object_20260926.json)
- [DFS 20MiB](read_promotion_bytes_dfs_20260926.json)

두 경우 모두 DRAM 복사를 테스트용 게이트로 정지시킨 상태에서 async GET이 먼저 반환됐다.
GPU 소비자 참조를 먼저 반환해도 source staging이 유지됐고, 게이트 해제 후 CPU payload가
바이트 단위로 일치했으며 다음 CPU lookup이 hit했다. 승격 OFF도 확인했다.
테스트 게이트를 위해 이 검증에만 큐 대기 제한 10초를 사용했고 실제 예제 기본값은 100ms다.

Qwen3-14B의 [새 프로세스 읽기 승격 검증](read_promotion_inference_20260926/results.json)도 통과했다.
8192토큰 입력 4개를 먼저 GPU-only 프로세스에서 DAOS에 저장한 뒤 프로세스를 종료했다.
그 다음 DRAM이 빈 새 소비자 프로세스에서 DAOS 쓰기를 끄고 같은 입력을 두 번 요청했다.

| 소비자 단계 | DRAM hit 청크 | DAOS hit 청크 | 소비자 DAOS 쓰기 |
|---|---:|---:|---:|
| 첫 읽기 | 0 | 256 | 0 |
| 다음 읽기 | 256 | 0 | 0 |

`read_copied=256`(5GiB), `write_copied=0`이므로 새 쓰기가 아니라 읽기 승격으로 DRAM이 채워졌다.
복사 pending 최대 200MiB, 복사/등록 오류 및 staging 할당 실패 0, 최종 pending 0이었다.
두 읽기의 생성 token ID는 최초 계산 결과와 일치했다. 이 검증은 DRAM8GiB에 5GiB가
모두 들어가는 기능 확인이며, 장시간 hit율이나 ON/OFF 성능 우위를 측정한 것은 아니다.
실행 후 테스트 서버는 모두 종료했다.

```bash
cd /root/discos_minji
env DAOSGDS_TRANSPORT=object ./run_vllm.sh ./venv/bin/python3 \
  tests/async_dram_promotion_roundtrip.py --result /root/discos_minji/read_promotion_NEW.json
./venv/bin/python3 check_read_promotion_inference.py \
  --output /root/discos_minji/read_promotion_inference_NEW
```

## 8. DRAM hit의 GPU staging 프리페치를 함께 켜기

`DaosAsyncDramBackend`에서 다음 옵션을 추가하고 프로세스를 재시작하면,
기존 비동기 DRAM 보관·승격에 `CPUHitPrefetch`를 함께 사용할 수 있다.
옵션 생략 또는 `false`는 기존 OFF 동작이다. 기본 YAML은 OFF를 유지한다.

```yaml
extra_config:
  # 기존 extra_config 항목을 유지하면서 추가
  daosgds.dram_prefetch: true
  daosgds.cpu_prefetch_gpu_gb: 5
```

DRAM hit 후 async GET에서 H2D 복사를 시작하고, 완료한 GPU 객체를 retrieve에 전달한다.
별도 CUDA stream·worker를 사용하며 원본 CPU 참조와 pin을 소비/취소 완료까지 유지한다.
CPU 종료 시에는 H2D와 DRAM mirror의 D2H worker를 모두 정리한 뒤 host allocator를 해제한다.

5GiB는 공유 staging 전체 점유량에 대한 DRAM 프리페치 입장 기준이다. 새 풀을 만들거나
DAOS와 정확히 반씩 나누는 설정이 아니다. 요청 전체를 추가하면 한도를 넘는 경우에는
원래 CPU 객체를 반환하여 retrieve 시점에 처리한다. hit를 miss로 바꾸지 않는다.
이는 현재 점유량을 확인하는 고정 watermark이며 자동 조절 정책은 아니다.

`tests/async_dram_promotion_roundtrip.py --prefetch`로 20MiB 실제 object 데이터의
읽기 승격→DRAM hit→GPU 프리페치 바이트 일치와 예산 부족 CPU fallback을 확인했다.
[검증 결과](async_dram_prefetch_bytes_20260926.json). 전체 단위 테스트는 113개 통과했다.
첫 시도의 테스트 harness는 PinMonitor 초기화 누락으로 중단됐으며 이를 보완한 재실행 결과다.

동일한 장시간 부하는 `discovery_staging_bench.py --conditions async_dram_on`으로 선택한다.
OFF는 `--conditions async_dram`이다. 각 실행은 새 namespace를 사용한다.

## 9. 실제 용량까지 프리페치를 시도하는 진단 모드

```yaml
extra_config:
  daosgds.dram_prefetch: true
  daosgds.dram_prefetch_policy: capacity
```

`capacity`는 DRAM 프리페치의 사전 watermark 입장 제한만 해제한다. GPU staging의
설정 용량 한도는 유지하며 자동 확장하지 않는다. 요청에 필요한 GPU 객체를
실제로 할당하다 실패하면, 해당 요청에서 이미 할당한 객체를 모두 반환하고 CPU
원본을 사용한다. 사용 중인 다른 요청의 객체를 강제로 퇴출하지 않는다.
`watermark_rejections`와 `capacity_rejections`를 구분해 계측한다.
DAOS의 기존 serializer와 DRAM 승격 큐의 예산 제한은 그대로다.

롤백은 `daosgds.dram_prefetch_policy: watermark`(기본값)로 바꾸고 재시작하면 된다.
`daosgds.dram_prefetch: false`이면 DRAM→staging 프리페치 자체를 끈다.
두 옵션 모두 기본 YAML을 변경하지 않고 실험용 YAML에만 적용했다.

실제 GPU 검증은 [결과 JSON](async_dram_prefetch_capacity_bytes_20260926.json)에 있다.
20MiB 승격/H2D 바이트 일치 및 작은 테스트 풀의 실제 공간 부족 시 CPU fallback을
확인했다. 이것은 기능 검증이며 10GiB staging의 성능 실험 결과가 아니다.

## 10. 동일한 모델 입력 256개로 비교하기

```bash
cd /root/discos_minji
./venv/bin/python3 discovery_fixed_replay.py \
  --output /root/discos_minji/discovery_fixed_replay_NEW
./venv/bin/python3 report_fixed_replay.py \
  /root/discos_minji/discovery_fixed_replay_NEW
```

출력 디렉터리는 새 이름이어야 한다. 기존 자료를 덮어쓰지 않는다.
기본 입력은 지난 DiscoveryBench 실행에서 기록된 성공 모델 호출 중 시간순 256개다.
동시 요청 4/8/16 × OFF/ON의 여섯 조건을 각각 새 프로세스·빈 DRAM·새 DAOS
이름 공간에서 실행한다. staging 8GiB, DRAM 8GiB, chunk 128, object 경로다.
2026-09-27부터 다음 실행의 staging 설정을 10→8GiB로 변경했다. 이전 10GiB
실험의 설정·로그·결과는 수정하지 않았다. 재생 스크립트는 기본 YAML의 staging
용량을 사용하며, 분석/그래프는 각 실험에 저장된 config.yaml의 용량을 사용한다.
Python 도구는 재실행하지 않고 기록된 결과가 들어 있는 프롬프트를 재생한다.

각 조건에 정확히 같은 요청 목록을 사용하지만, 한 묶음이 끝나면 다음 묶음을
보내므로 전송 시각까지 동일한 실험은 아니다. 생성 종료 조건을 유지하므로 출력
토큰 수도 달라질 수 있다. 보고서에서 입력 해시·생성량·실제 hit·할당 실패를 함께
확인한다. 요청 오류 시 자동 재시도로 개수를 늘리지 않고 중단하여 실패를 남긴다.
새 이름 공간은 유지되며 공용 컨테이너/서버 캐시는 삭제하지 않는다.

특정 동시 요청 수와 DRAM 용량만 선택할 수도 있다. 예를 들어 현재 YAML의
staging 8GiB를 유지하고 DRAM 4GiB·동시 요청 16개만 OFF/ON 비교하려면:

```bash
./venv/bin/python3 discovery_fixed_replay.py \
  --output /root/discos_minji/discovery_replay_s8_d4_c16_NEW \
  --cpu-gb 4 --concurrency 16
./venv/bin/python3 report_fixed_replay.py \
  /root/discos_minji/discovery_replay_s8_d4_c16_NEW
```

`--cpu-gb`는 해당 실험의 설정에만 적용되며 기본 YAML의 DRAM 용량을 바꾸지 않는다.
`--concurrency 4 8 16`은 세 동시 요청 조건을 모두 선택한다.

## 11. DRAM / staging 6조합과 DAOS 실패 재계산율

```bash
./venv/bin/python3 discovery_capacity_matrix.py \
  --output /root/discos_minji/discovery_capacity_matrix_NEW
./venv/bin/python3 report_capacity_matrix.py \
  /root/discos_minji/discovery_capacity_matrix_NEW
```

DRAM 8/4/2GiB × staging 8/4GiB, 동시 요청 16, 256개 동일 입력을 각각 새 프로세스와
새 DAOS namespace에서 실행한다. 총 6개 모두 DRAM 프리페치 ON이며 `capacity`
정책이므로 soft watermark는 없다. 물리적 GPU 용량과 기존 DAOS serializer,
비동기 DRAM 보관/승격 큐의 제한은 유지한다. 기본 YAML은 변경하지 않는다.

`CapacityProbeBackend`는 실험용 계측 백엔드다. 기존과 같은 병렬 GET·연속 prefix
반환 규칙을 사용하면서 GET 스레드의 실제 GPU 할당 실패를 요청별로 기록한다.
실패 뒤의 성공 청크가 prefix 규칙으로 버려지는 경우까지 포함하여, 가져오지 못한
KV 토큰을 계산한다. CPU 반환량과 응답 `cached_tokens`까지 맞는지 검증한다.

주 지표는 **DAOS의 GPU 공간 부족으로 재사용하지 못한 토큰 / 전체 입력 토큰**이다.
원래 캐시가 없던 cold miss는 분자에서 제외한다. DAOS 읽기 예정 토큰 대비 비율과
영향 받은 요청 비율도 별도로 제공한다. 이는 prefix 재사용 실패로 추가 입력 계산이
필요한 양이며, GPU 커널별 재계산 시간이나 스케줄러 재선점 횟수의 직접 측정은 아니다.

`overview.png`는 6조건 시간별 staging 점유율과 tier hit 비율, `recompute_ratio.png`는
실패 재계산율이다. 각 조건에는 개별 timeline 및 `recomputation_by_request.csv`가 있다.

### 같은 6조합의 DRAM 프리페치 OFF 추가 측정

`--dram-prefetch off`는 DRAM→GPU staging 사전 복사만 끈다. DRAM 보관과
DAOS 비동기 프리페치, 쓰기 미러 및 읽기 승격은 유지한다. 옵션 기본값은 `on`이다.
각 조건은 ON과 마찬가지로 새 프로세스·빈 DRAM·새 DAOS namespace에서 시작한다.

```bash
./venv/bin/python3 discovery_capacity_matrix.py \
  --output /root/discos_minji/discovery_capacity_matrix_OFF_NEW \
  --requests /root/discos_minji/discovery_capacity_matrix_NEW/requests.json \
  --dram-prefetch off
./venv/bin/python3 report_capacity_matrix.py \
  /root/discos_minji/discovery_capacity_matrix_OFF_NEW
./venv/bin/python3 compare_capacity_prefetch.py \
  --on /root/discos_minji/discovery_capacity_matrix_NEW \
  --off /root/discos_minji/discovery_capacity_matrix_OFF_NEW \
  --output /root/discos_minji/capacity_prefetch_COMPARISON_NEW
```

비교 도구는 ON/OFF의 입력·백엔드 코드·실제 native 라이브러리·서버 명령·설정과
빈 초기 상태를 검증한다. 토글·namespace 이외에 설정 차이가 있으면 중단한다.
기존 ON 결과를 보존한 채 `comparison.png`, `timeline_comparison.png`와 보고서를
새 비교 디렉터리에 쓴다. 한 번씩 측정한 결과이므로 반복 평균이나 신뢰구간은 아니다.

### DRAM 8 / staging 8 / 동시 요청 8의 반복 비교

```bash
./venv/bin/python3 repeat_prefetch_c8.py \
  --output /root/discos_minji/prefetch_d8_s8_c8_REPEAT_NEW
./venv/bin/python3 report_prefetch_c8.py \
  /root/discos_minji/prefetch_d8_s8_c8_REPEAT_NEW
```

같은 256개 입력을 매번 8개씩 보내며 OFF→ON→ON→OFF→OFF→ON 순서로 각각
새 프로세스·빈 DRAM·새 namespace에서 3회씩 측정한다. vLLM의 max-num-seqs는
기존 16을 유지하고 클라이언트의 동시 요청 수만 8로 바꾼다. 보고서는 프로세스별
평균 TTFT의 평균·표본 표준편차, 처리량, 재계산율, 개별 staging/hit 그래프를 제공한다.
생성량·캐시 상태는 실행에 따라 달라질 수 있으며 3회 반복만으로 통계적 유의성이나
보편적인 우위를 주장하지 않는다. 기본 서비스 설정과 과거 결과는 변경하지 않는다.

### 묶음 대신 동시 요청을 유지하는 rolling 방식

```bash
./venv/bin/python3 repeat_prefetch_c8.py --arrival-mode rolling \
  --output /root/discos_minji/prefetch_d8_s8_c8_ROLLING_NEW
./venv/bin/python3 report_prefetch_c8.py \
  /root/discos_minji/prefetch_d8_s8_c8_ROLLING_NEW
```

첫 8개를 함께 시작하고, 하나가 끝나면 아직 끝나지 않은 다른 요청을 기다리지 않고
다음 입력을 넣는다. 총 입력은 그대로 256개, 최대 동시 요청은 8개다. 마지막 요청들을
마무리할 때는 새 입력이 없어 동시성이 감소한다. `--arrival-mode waves`가 기존 방식이며
기본값이다. 모델·백엔드·캐시 정책은 바꾸지 않는다. 요청 payload와 TTFT 측정은 기존
`replay_one`을 그대로 사용한다. 완료 순서·벽시계 도착 시각은 서버 응답에 따라 달라진다.

rolling 실행 전 UUID 테스트 키 하나로 20MiB 실제 GPU 저장/읽기를 검증하고 테스트 키만
정리한다. 실패하면 모델 서버를 시작하지 않는다. 실행 중 DAOS/LMCache 저장 오류도
주기적으로 검사하며, HTTP가 성공하더라도 저장 오류가 나면 새 요청 투입을 중단하고
이미 진행 중인 요청 기록을 남긴다. 저장 공간 부족을 GPU staging 고갈로 집계하지 않는다.

2026-09-27 사전 검사에서 DAOS rank1/tag6의 `DER_NOSPACE(-1007)`가 재현됐다.
따라서 rolling 성능 실험은 아직 실행하지 않았다. 기존 실험 캐시 삭제는 별도 승인 없이
수행하지 않는다. 공간 확보 후 위 명령을 새 출력 디렉터리로 실행한다. 설정/소스 준비만
확인하려면 `--dry-run`을 추가한다. dry-run에는 모델 호출이나 DAOS 접근이 없다.

동시 요청 상한은 `--concurrency 8`(기본) 또는 `--concurrency 16`으로 선택한다.
예를 들어 DRAM/staging 각각 8GiB, 최대 16개를 유지하며 개별 완료마다 다음 요청을
투입하는 OFF/ON 반복 비교는 다음과 같다.

```bash
./venv/bin/python3 repeat_prefetch_c8.py --arrival-mode rolling --concurrency 16 \
  --output /root/discos_minji/prefetch_d8_s8_c16_ROLLING_NEW
./venv/bin/python3 report_prefetch_c8.py \
  /root/discos_minji/prefetch_d8_s8_c16_ROLLING_NEW
```

2026-09-27 사용자가 서버 캐시 정리를 알린 뒤 재조회한 풀 표시 free는 937GB였다.
별도 UUID 키의 20MiB GPU 왕복·전체 바이트 비교가 통과하여 이후 rolling 실험을
시작했다. 사전 검증 키만 자동 정리하며, 실행기에는 과거 실험 캐시 삭제 기능이 없다.
