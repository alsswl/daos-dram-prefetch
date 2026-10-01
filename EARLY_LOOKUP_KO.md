# 존재 확인을 먼저 알리는 프리페치 (실험용)

기존 동작은 변경하지 않고 별도 `EarlyLookupBackend`를 추가했다.
진행 중인 `sharegpt_prefetch_cold_warm4_d256_s8_20260930` 실험은 기존 방식 그대로다.
설치된 LMCache/vLLM 파일과 `/root/discos` 원본은 수정하지 않는다.

## 흐름

1. 비동기 lookup으로 DRAM/DAOS의 연속 prefix hit를 확인한다.
2. DRAM hit의 pin과 읽기 참조를 확보한다. 이때 GPU payload 이동은 없다.
3. 취소 가능한 요청 계획을 등록한 뒤, 존재하는 토큰 수를 스케줄러에 보낸다.
4. 알림 전송 뒤 DRAM H2D 및 DAOS GPU 읽기를 백그라운드에서 시작한다.
5. retrieve는 다음 상태에 따라 실제 데이터를 확보한 뒤 모델 KV로 전달한다.

| 상태 | DRAM hit | DAOS hit |
|---|---|---|
| 완료 | 준비된 GPU staging 또는 CPU fallback 사용 | 준비된 GPU staging 사용 |
| 시작 전 대기 | 작업 취소 후 원래 DRAM 데이터 사용 | 작업 취소 후 retrieve에서 동기 읽기 |
| 전송 중 | 해당 복사 완료까지 대기 | 해당 읽기 완료까지 대기 |

작업 시작/취소는 `Future`의 원자적 상태 전환으로 결정한다. 이미 실행 중인 읽기를
다시 제출하지 않는다. 취소한 작업의 큐 항목은 실행기가 나중에 건너뛰므로, 물리적인
큐 항목 제거와 다르다. 취소된 요청의 GPU/CPU 버퍼는 진행 중인 DMA 완료 후 해제한다.
사용되지 않은 prefix 뒤쪽 객체도 이 요청이 소유한 pin/ref만 해제한다.

## 선택 및 원복

설정 예시: `lmcache_config_daosgds_early_lookup.yaml`.

```yaml
enable_async_loading: true
extra_config:
  storage_plugin.daosgds.module_path: lmcache_daos.early_lookup_backend
  storage_plugin.daosgds.class_name: EarlyLookupBackend
  daosgds.early_lookup: true
  daosgds.dram_prefetch: true
  daosgds.dram_prefetch_early_ready: false
  daosgds.dram_prefetch_cancel_queued: false
```

기존 동작으로 돌아가려면 `daosgds.early_lookup: false`로 바꾸고 프로세스를 재시작한다.
DRAM 프리페치만 끄려면 `daosgds.dram_prefetch: false`로 설정한다.
기존 `dram_prefetch_early_ready/cancel_queued` 기능과 동시 사용하지 않는다.
GPU-direct 저장, 비동기 DRAM 보관·읽기 승격, CPU LRU, DAOS 청크 I/O 작업 풀은 유지한다.
DAOS의 speculative 읽기는 기존 AsyncMultiSerializer 예산을 사용하며,
retrieve가 대기 작업을 가져가는 경우에만 그 speculative 대기를 건너뛴다.

## 주의·검증 범위

- 이 옵션은 비동기 lookup, LocalCPUBackend와 단일 DAOS backend,
  non-layerwise 구성을 대상으로 한다. 추적용 `DAOS_GDS_STAGING_TRACE` 설정이 필요하다.
- LOADING 이벤트 DONE은 이 경로에서 **요청 계획 준비 완료**이다. 새 retrieve hook이
  실제 payload 완료를 별도로 확인하며, 미완료 데이터를 모델이 읽지 못하게 한다.
- 알린 KV를 실제로 가져오지 못하면 예외로 중단한다. 용량 부족·오브젝트 삭제 등의 경우
  자동으로 재계산하도록 스케줄러를 되돌리는 기능은 이번 변경에 포함하지 않는다.
- 청크를 조금씩 모델 KV로 보내는 windowed streaming을 구현한 것은 아니다.
  8GiB staging으로 모든 10GiB hit 요청을 처리할 수 있다고 보장하지 않는다.
- 원래 async lookup client의 backoff(기본10ms), 모델·요청 크기·동시성은 변경하지 않는다.
  조기 알림이 TTFT 개선을 보장하지 않는다.
- `early_lookup_notify`, `early_payload_start/ready`, `early_retrieve_decision`으로
  알림/전송/대기·요청 시 읽기 전환을 구분해 기록한다. 기존 완료-first 전용 집계기를
  그대로 적용하면 CPU-ready 이벤트 의미가 달라 잘못 집계할 수 있다.
- 상태 머신·가짜 I/O 통합 테스트는 `tests/test_early_lookup.py`에 있다.
  실제 GPU·vLLM 검증 결과가 나오기 전에는 성능이나 실환경 안정성을 확인했다고 주장하지 않는다.
