# 복사 완료와 스케줄 가능 시점: 코드 확인 메모

아래는 현재 설치된 코드의 읽기 확인이다. 설정이나 라이브러리를 수정한 실험 결과가 아니다.

## 확인된 실행 구조

1. CPU 프리페치 ON에서는 `CPUHitPrefetch.get()`이 worker의 H2D 복사 완료를 기다린 뒤 GPU 객체를 반환한다. OFF에서는 CPU backend가 DRAM 객체를 반환한다.
2. LMCache의 비동기 조회 경로는 관련 tier의 get 작업들을 모은 `all_done` 완료 callback에서 스케줄러에 읽을 수 있는 토큰 수를 알린다.
3. vLLM adapter가 아직 완료되지 않은 조회에 대해 `None`을 반환하면, scheduler는 그 요청을 현재 스케줄에서 제외하고 다음 기회로 넘긴다.
4. 따라서 ON은 retrieve 내부의 복사를 줄이는 대신, 요청을 스케줄 가능 상태로 만드는 시점을 늦출 수도 있다. 다른 요청의 계산과 겹칠 여지가 있으면 이득이지만, 실제 이득의 크기는 두 구간을 함께 봐야 한다.

코드 위치:

- [CPU 프리페치 worker 완료 대기](/root/discos_minji/lmcache_daos/dram_prefetch_backend.py:184)
- [모든 tier get 완료 뒤 스케줄러 응답](/root/discos_minji/venv/lib/python3.12/site-packages/lmcache/v1/storage_backend/storage_manager.py:651)
- [get task들을 gather하고 callback 등록](/root/discos_minji/venv/lib/python3.12/site-packages/lmcache/v1/storage_backend/storage_manager.py:798)
- [조회 미완료이면 None 반환](/root/discos_minji/venv/lib/python3.12/site-packages/lmcache/integration/vllm/vllm_v1_adapter.py:1434)
- [None인 요청을 현재 스케줄에서 제외](/root/discos_minji/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py:744)

## 추가 지연 후보: lookup backoff

현재 `LMCacheAsyncLookupClient`의 기본 `lookup_backoff_time`은 0.01초다. 이번 생성 YAML에는 이를 덮어쓰는 항목이 없고, 실제 실행 로그는 이 async lookup client 사용을 보여준다.

- 최초 lookup을 전송한 뒤 이 시간만큼 sleep한다.
- 조회가 아직 완료되지 않은 상태를 확인할 때도 sleep한다. 해당 분기는 client lock 안에 있다.
- [기본값 및 pending 확인](/root/discos_minji/venv/lib/python3.12/site-packages/lmcache/v1/lookup_client/lmcache_async_lookup_client.py:153)
- [최초 lookup 전송 후 sleep](/root/discos_minji/venv/lib/python3.12/site-packages/lmcache/v1/lookup_client/lmcache_async_lookup_client.py:220)

이는 실제 존재하는 제어 경로의 지연 요인이다. 하지만 이번 계측은 요청별 pending poll 횟수와 스케줄러 runnable 전환을 직접 기록하지 않았다. 따라서 **ON/OFF 차이 중 몇 ms가 이 backoff 때문인지 확정할 수 없으며**, 단순히 동시 요청 수에 10ms를 곱해 총 지연으로 계산하면 안 된다. 이 값을 바꿔 유리한 결과를 만드는 실험은 수행하지 않았다.

`ready→retrieve` 간격도 프리페치로 숨긴 시간과 같지 않다. ready 시점 자체가 스케줄 시점을 결정하는 조건일 수 있고, 그 이후 스케줄 대기도 포함되기 때문이다. 이번 결과는 host 계측이며 순수 DMA 시간이나 인과적인 critical-path 분해가 아니다.
