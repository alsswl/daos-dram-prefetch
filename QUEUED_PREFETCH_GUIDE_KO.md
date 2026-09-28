# DRAM 프리페치 대기 취소: 구현 및 비교 실험

## 무엇이 달라지는가

기존 방식은 CPU hit 프리페치 worker가 staging 복사를 끝낼 때까지 기다린 뒤
LMCache의 비동기 get 결과를 반환한다. worker는 하나이므로 앞 요청이 복사 중이면
뒤 요청은 대기한다. 오래 기다린다는 이유로 DRAM으로 전환하지 않는다.

새 옵션은 DRAM 객체의 참조/핀을 유지하는 지연 객체를 먼저 반환한다. DRAM 데이터는
사용할 수 있으므로 LMCache가 준비 완료를 알릴 수 있다. DAOS와 섞인 요청은 기존처럼
DAOS 읽기도 완료해야 준비 완료가 된다. **GPU 복사가 완료됐다고 거짓으로 알리는 것이
아니라, 사용할 수 있는 CPU 데이터를 먼저 제공하는 방식**이다.

실제 결정은 `LMCacheEngine.retrieve()` 내부의 `_async_process_tokens_internal` 진입,
즉 완료 이벤트를 소비하기 직전에 한다. HTTP 도착 시점이나 scheduler poll 시점이 아니다.

|그때의 상태|동작|
|---|---|
|executor 대기 중|Future.cancel() 성공 → staging 할당/복사 없이 CPU 원본 사용|
|worker가 이미 시작함|취소하지 않고 완료 대기 → staging 사용, 할당 실패이면 CPU 사용|
|이미 복사 완료|staging 사용|
|공간 확보 실패|CPU 원본 사용, 캐시 miss나 재계산으로 취급하지 않음|

취소와 worker 시작의 경쟁은 concurrent Future의 원자적 상태 전이로 결정한다.
단순히 `running()`을 확인한 뒤 취소하는 경쟁 조건을 만들지 않는다. 취소한 큐 항목은
물리적으로 큐 내부에서 제거하지 않아도 executor가 건너뛰므로 데이터 복사를 실행하지 않는다.
worker가 시작했지만 아직 DMA를 제출하기 전인 경우도 안전하게 **진행 중**으로 취급한다.

원본 CPU 참조는 소비 완료와 DMA 완료까지 유지한다. 요청 취소/미사용 청크 정리는
복사 중인 버퍼를 재사용하지 않도록 지연한다. 실행 중 복사를 강제로 취소하거나 CPU와
GPU 경로로 같은 요청을 중복 복사하는 정책은 없다. 복사 오류는 숨기지 않는다.

## 설정과 롤백

`DaosAsyncDramBackend` 및 이를 상속하는 실험용 backend에서 사용한다.
기존 YAML, `/root/discos`, 설치된 LMCache/vLLM 파일은 변경하지 않았다.

```yaml
extra_config:
  daosgds.dram_prefetch: true
  daosgds.dram_prefetch_cancel_queued: true
  daosgds.dram_prefetch_early_ready: false
```

`cancel_queued: true`이면 early readiness도 자동 활성화된다.
원래 동작으로 되돌리려면 **cancel_queued와 early_ready를 모두 false로 하고 프로세스를 재시작**한다.
이 두 키는 기본 false이므로 기존 실험 동작은 유지된다. GPU-direct store, DAOS prefetch,
DRAM 저장/읽기 승격, 한 개의 DRAM copy worker/stream은 그대로다.

## 정확하게 비교할 세 가지 방식

|스크립트 모드|DRAM 프리페치|준비 알림|대기 중 retrieve 도착|
|---|---|---|---|
|wait (새 기능 OFF)|ON|복사 완료 후|그 전에 준비 알림을 못 받으므로 기존처럼 기다림|
|cancel (새 기능 ON)|ON|CPU 사용 가능 시|대기 작업 취소 후 CPU 사용|
|early_wait (선택 대조군)|ON|CPU 사용 가능 시|취소하지 않고 기다림|

**wait와 cancel만 비교하면 준비 알림 시점 변경과 큐 취소 효과가 함께 측정된다.**
`early_wait`와 `cancel`을 비교해야 조기 준비 알림을 공통 조건으로 놓고 취소 효과를 볼 수 있다.
조기 반환은 serializer coroutine이 예산을 반납하는 시점도 바꾼다. 실제 GPU arena 용량 제한은
유지하지만, 이것 역시 wait 대비 변경점이다. early_wait 대조군은 이 동작도 공유한다.

## 실험 준비/실행

기존 256개 DiscoveryBench 기록 입력을 동일하게 재생한다. full agentic/Python 실행이 아니다.
Qwen3-14B, chunk128, object 경로, DRAM8GiB/staging8GiB, soft watermark 없는 실제 용량 제한이다.
각 조건은 새 프로세스·빈 DRAM·새 DAOS namespace로 시작하여 cold256 → 저장/복사 drain →
**동일 프로세스/캐시로 warm256**을 실행한다. 조건마다 초기화하지만 cold와 warm 사이에는 비우지 않는다.
공용 DAOS 데이터 삭제나 서버/OS 캐시 flush는 하지 않는다.

먼저 실행 계획만 생성 (GPU/DAOS 작업 없음, 출력 폴더는 새 이름):

```bash
cd /root/discos_minji
./venv/bin/python3 compare_queued_prefetch.py \
  --output /root/discos_minji/queued_prefetch_plan \
  --concurrency 8 16 --repeats 3 --with-early-wait --dry-run
```

실제 비교 실행 (이 문서 작성 작업에서는 실행하지 않음):

```bash
cd /root/discos_minji
./venv/bin/python3 -u compare_queued_prefetch.py \
  --output /root/discos_minji/queued_prefetch_results \
  --concurrency 8 16 --repeats 3 --with-early-wait
```

위 명령은 3모드 × 2동시성 × 3반복 = 18프로세스이며, 각 cold/warm256개씩 총 9,216요청이다.
`--with-early-wait`를 빼면 기존/새 방식 2모드만 비교한다. 첫 점검은 `--concurrency 8 --repeats 1`로
줄일 수 있다. 무거운 실행은 사용자가 명시적으로 시작해야 한다.

한 요청이 끝나면 다음을 넣는 rolling 방식이다. 인위적인 sleep, 큐 막기, ON 전용 준비 시간,
캐시 주입, worker 수 변경은 벤치에 없다. 실제 출력 길이와 완료 기반 도착 시각은 달라질 수 있다.
테스트에서만 경쟁 조건 재현용으로 worker를 막는다. 벤치에는 그 코드가 들어가지 않는다.

## 확인할 결과

- `RESULT_KO.md`, `summary.csv/json`: 평균/p95 TTFT, 반복 간 변동, 실제 hit 및 새 계산량.
- `paired_checks.json`: 요청별 재사용 토큰 수/DRAM·DAOS hit 일치 여부.
- 각 cold/warm의 `retrieve_decisions.json`: `cancelled_queued`, `waited_gpu`, `ready_gpu`, `capacity_cpu`.
- `timing_by_request.json`, trace JSONL: retrieve, 실제 worker 대기/복사, staging 점유와 해제.
- `validation.json`: 입력/설정/라이브러리, namespace 독립성, phase 간 동일 worker와 drain.

취소된 작업에는 worker 시작/복사 시간이 없다. 실행된 작업만의 평균 큐 대기와 섞어서
"평균 큐 대기가 줄었으니 빨라졌다"고 해석하면 안 된다. 취소 요청 수·바이트와 TTFT를 함께 본다.
취소가 0건이면 이 부하에서는 취소 효과를 확인하지 못한 것이며, 결과를 유리하게 만들기 위해
인위적으로 큐를 지연시키지 않는다. wait 모드의 retrieve 결정 카운터 0은 미계측이라는 뜻이다.

## 검증 범위

CPU 모의 객체로 큐 대기 취소, 이미 진행 중인 복사 대기, 완료 결과 선택, 할당 실패,
요청 취소, 부분 해제, 복사 오류, 시작/취소 경쟁 및 런타임 소비 hook을 테스트한다.
벤치 계획/설정도 dry-run과 테스트로 검증한다. **새 정책의 실제 vLLM/DAOS end-to-end 성능 및
GPU 데이터 정합성은 별도 실행 전에는 검증 완료로 간주하지 않는다.**
