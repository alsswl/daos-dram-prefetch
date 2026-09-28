# 대기 중 DRAM 프리페치 취소: 실제 비교 결과

2026-09-27. 모든 6조건, cold/warm 총 3,072개 HTTP 요청 완료. 조건별 반복은 1회다.

## 결론

**새 방식은 이번 warm에서 기존 방식보다 평균 TTFT가 5.6~5.9% 낮았지만,
대기열 취소가 cold/warm 전체에서 0건이었다. 따라서 이 감소를 취소 기능의 효과라고
주장할 수 없다.** 조기 준비 알림만 적용한 대조군도 비슷한 값을 보였다.
조기 알림에 따른 실행 시점/serializer 예산 반환 변화와 실행 간 변동이 포함되어 있다.
조기 알림만의 인과적인 개선율 역시 반복 검증 없이 확정하지 않는다.

## 조건과 실행 방법

- Qwen3-14B BF16, DAOS object, chunk128, DRAM8GiB / staging8GiB.
- 동시 요청 8/16, 각 기존 wait / 새 cancel / 조기 알림만 하는 early_wait.
- **모든 방식에서 DRAM 프리페치와 DAOS 프리페치는 ON.** OFF는 대기 취소 새 기능 OFF를 뜻한다.
- 각 조건마다 새 프로세스, 빈 DRAM, 독립된 새 DAOS namespace.
- 같은 256개 입력 cold → staging·비동기 저장/복사 drain → 같은 프로세스/캐시 warm256.
- 완료 시 다음 요청을 넣는 rolling 방식. 큐 지연이나 ON 전용 대기 시간은 넣지 않았다.
- DiscoveryBench 기록 입력 재생이며 Python tool/full-agentic 실행은 아니다.
- 실행 순서: 8 cancel → 8 early_wait → 8 wait → 16 cancel → 16 early_wait → 16 wait.
- 공용 캐시는 삭제하지 않았고 서버/OS 캐시를 flush하지 않았다. 사전 GPU 왕복 테스트의 UUID 키 하나만 자체 정리했다.

실행 명령:

```bash
cd /root/discos_minji
./venv/bin/python3 -u compare_queued_prefetch.py \
  --output /root/discos_minji/queued_prefetch_experiment_20260927 \
  --concurrency 8 16 --repeats 1 --with-early-wait
```

이 폴더는 완료된 결과이므로 재실행할 때는 새 출력 폴더 이름을 사용한다.

## Warm TTFT

|동시 요청|기존 방식: 새 기능 OFF|새 방식: ON|조기 알림만 하는 대조군|기존 대비 새 방식|
|---:|---:|---:|---:|---:|
|8|121.62ms|114.82ms|114.45ms|−5.59%|
|16|163.01ms|153.41ms|156.17ms|−5.89%|

|동시 요청|기존 p95|새 방식 p95|대조군 p95|
|---:|---:|---:|---:|
|8|181.16ms|166.24ms|162.30ms|
|16|426.46ms|393.70ms|398.92ms|

Warm의 요청별 실제 재사용 토큰 수는 세 방식 간 256/256 모두 일치했다.
새로 계산한 입력 토큰 합계도 각 warm 17,097로 같았다. 단, DRAM/DAOS 배치까지 완전히 같지는 않다.
새 방식 vs 기존의 tier별 hit 일치는 8개에서 253/256, 16개에서 245/256이다.
실제 DRAM hit 청크 비율은 warm에서 92.41~92.61%, DAOS는 나머지 약 7.4~7.6%였다.
Cold에는 실제 재사용량 차이가 남아 있으므로 성능 원인 해석은 warm 중심으로 한다.

## 새 방식 ON에서 retrieve 때 실제로 무엇을 했나

|동시 요청·warm|대기열 취소 후 DRAM|이미 GPU 준비 완료|worker 완료 대기|공간 확보 실패로 DRAM|
|---|---:|---:|---:|---:|
|8|0|256|0|0|
|16|0|252|1|3|

16개의 3건은 대기 취소가 아니라 기존에도 있던 용량 부족 fallback이다.
모든 12개 cold/warm 단계에서 DAOS 할당 실패 때문에 잃은 prefix/재계산 토큰은 0이었다.
실제 GPU staging peak는 8 ON warm 4.160GiB, 16 ON warm 7.988GiB였다.

8 ON warm의 실행된 worker 평균 큐 대기는 0.905ms, 16 ON warm은 2.261ms였다.
큐 대기가 있었다는 것과 retrieve 시점에도 큐에 있었다는 것은 다르다.
이 실행에서는 retrieve가 해당 데이터를 소비할 때 거의 모두 staging 복사가 끝나 있었다.
따라서 취소로 없앨 수 있는 대기가 거의 드러나지 않았으며, **취소 정책 자체는 이 워크로드에서
실제 분기를 타지 않아 end-to-end 효과를 검증하지 못했다.** 취소 분기의 검증은 앞서 작성한 단위 테스트 범위다.

취소/대조군의 모든 결정 이벤트는 로그상 retrieve 시작과 반환 사이에서 발생했다.
8 cancel 506건, 8 early_wait 504건, 16 cancel 508건, 16 early_wait 503건을 확인했다.
Cold에서 DRAM hit가 없는 요청은 CPU 프리페치 결정 이벤트 자체가 없으므로 총 HTTP 수와 다르다.

## 검증과 한계

- 입력 hash, 설정(스위치/namespace 제외), native 라이브러리 일치 검증 통과.
- 각 조건의 DRAM 초기 상태와 cold/warm 사이 동일 worker 확인.
- GPU staging, deferred batch, 비동기 DRAM mirror가 모두 drain됨. copy_errors/mirror errors 0.
- 각 조건 정상 완료, 스크립트 종료 코드 0. 종료 후 관련 vLLM/벤치 및 GPU compute process 없음.
- 사전 DAOS20MiB GPU 왕복 payload/metadata 일치. 새 정책의 모델 KV 전체 바이트 비교를 수행한 것은 아님.
- 조건별 1회이며 생성 토큰 수와 완료 기반 도착 시각은 다르다. 통계적 유의성/일반적인 처리량 개선 주장은 하지 않는다.
- 현재 측정은 DRAM 프리페치 ON/OFF 비교가 아니라, **DRAM 프리페치를 켠 상태에서 대기 취소 방식 비교**다.

## 원본 결과

- [상세 표](RESULT_KO.md)
- [CSV](summary.csv), [JSON](summary.json)
- [요청별 재사용 일치](paired_checks.json), [검증](validation.json)
- 각 실행 폴더의 cold/warm `retrieve_decisions.json`, `timing_by_request.json`, `replay_calls.json`
- 각 실행 폴더의 `trace.*.jsonl`, `server.log`, `config.yaml`, `native_maps.json`

다음에 취소 기능의 효과를 평가하려면 더 큰 평균 TTFT가 아니라 **retrieve 시점에 아직
queued 상태인 요청이 실제로 생기는 부하**를 찾아야 한다. 이번 결과만으로 큐 취소가 항상
필요 없다고 일반화하거나, 새 방식의 5.6~5.9% 차이를 취소 덕분이라고 해석하지 않는다.
