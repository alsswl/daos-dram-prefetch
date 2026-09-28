# 작업자 2개 실험 요약

2026-09-28. Qwen3-14B BF16, DRAM 8GiB / GPU staging 8GiB, 청크128, 동시 요청16 / max-num-seqs16. DRAM·DAOS 프리페치 ON, 대기 중 취소 OFF, 조기 준비 알림 OFF.

작업자 1개와 2개를 같은 수정본으로 새로 측정했다. 각 조건은 독립 프로세스·빈 DRAM·새 DAOS namespace에서 cold256 → 저장 완료 확인 → warm256으로 실행했다. 총 1,024개 요청을 모두 완료했다. 공용 설정이나 `/root/discos`는 수정하지 않았다.

## 결과

|Warm 지표|작업자1 / stream1|작업자2 / stream2|
|---|---:|---:|
|평균 TTFT|163.67ms|162.66ms|
|p95 TTFT|428.93ms|415.33ms|
|평균 작업 대기|1.758ms|0.319ms|
|평균 copy 호출 시간|18.396ms|19.264ms|
|평균 retrieve 호출 시간|2.810ms|2.728ms|
|staging 평균 점유|0.150GiB|0.156GiB|
|staging 최대 점유|7.988GiB|7.988GiB|
|DRAM lookup hit|92.36%|92.49%|
|DAOS lookup hit|7.64%|7.51%|
|DRAM hit 청크 중 staging 경유|98.77%|98.84%|
|DRAM staging 할당 실패 후 CPU 경로 fallback|4회|2회|
|DAOS staging 실패로 인한 추가 재계산|0토큰|0토큰|
|전체 입력 중 실제 계산한 토큰|17,097|17,097|
|생성 토큰|69,608|67,504|
|전체 완료 시간|90.42초|87.47초|

작업 대기는 약 81.9% 줄었지만 평균 TTFT는 0.61%만 감소했다. 이 조건에서는 대기열을 병렬 처리하는 것만으로 큰 TTFT 개선이 관측되지는 않았다. copy 호출 시간은 약 4.7% 증가했다. 이 시간에는 CPU 측 제출과 stream 완료 대기가 포함되므로, 이 수치 하나로 GPU 대역폭 경쟁이 원인이라고 단정하지 않는다.

두 작업 스레드가 서로 다른 CUDA stream을 사용하는 것과 CPU 측 작업·copy 호출 구간이 최대 2개까지 겹친 것을 기록했다. 단일 작업자에서는 최대 1개였다. 이 계측은 GPU DMA가 실제로 동시에 수행됐음을 증명하는 GPU 타임라인은 아니다.

warm의 256개 요청 모두 실제 재사용 토큰 수가 같았다. tier별 hit 청크 수까지 같은 요청은 249개였다. 따라서 재계산량 차이는 없지만 저장 위치·도착 시각까지 완전히 같은 실험은 아니다.

cold 평균 TTFT는 392.98 → 351.25ms였지만, 직접 계산한 입력 토큰도 170,822 → 154,182로 달랐다. cold의 차이를 순수 병렬 복사 효과라고 설명하면 안 된다. warm 전체 시간도 생성 토큰 수가 달라 단순 처리량 개선율로 확정하지 않는다. 각 조건 1회 측정이므로 작은 차이는 반복 검증이 필요하다.

## 구현과 검증

- 대기열을 물리적으로 2개로 분할한 것은 아니다. **공통 작업 대기열 + 작업자 2개 + 작업자별 독립 CUDA stream**이다.
- 원래 8GiB staging allocator를 공유한다. 요청별 reserve는 기존 lock으로 보호하며, 물리적 할당 실패 시 CPU fallback을 유지한다. watermark·DAOS 전용 예약·추가 지연은 넣지 않았다.
- 각 작업자는 자기 stream의 복사 완료를 기다린 뒤 결과를 공개한다. 취소·오류 시 복사 중인 버퍼를 먼저 해제하지 않는다. 통계 갱신도 lock으로 보호한다.
- 단위 테스트 58개 통과. 별도 사전 검사에서 독립 stream 2개로 64MiB 복사 6회를 수행하고 전체 바이트 일치를 확인했다. 해당 사전 검사의 동시 시작 barrier는 실제 벤치마크에는 없다.
- DAOS 20MiB GPU 왕복 검사 통과. 모든 단계의 시작 전 staging=0, 종료 후 staging와 mirror 작업 정리 완료 확인.
- 두 실행의 모델 KV 공간은 동일한 40.92GiB / 268,176토큰이었다. CUDA OOM은 로그에서 발견되지 않았고 실험용 vLLM 프로세스는 종료했다.

설정은 `extra_config` 아래 `daosgds.dram_prefetch_workers: 2`. 기본값은 1이며, 1로 바꾸거나 이 항목을 제거하고 재시작하면 원복된다. 실험별 YAML에만 설정했다.

## 자료

- [전체 결과 및 설정](RESULT_KO.md), [세부 수치](summary.json), [검증](validation.json), [재사용량 대조](paired_checks.json).
- [작업자1 warm 그래프](w1/warm/staging_hits.png), [작업자2 warm 그래프](w2/warm/staging_hits.png).
- [실행 스크립트](../compare_prefetch_workers.py), [GPU 복사 검증](gpu_copy_preflight.log).

실험 종료 후 DAOS NVMe 여유 공간은 약 277.66GB, 최소 타깃 여유는 14.39GB였다. 새 캐시는 보존했으며 추가 실험 전에는 용량을 다시 확인해야 한다.
