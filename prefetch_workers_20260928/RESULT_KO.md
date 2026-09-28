# DRAM 프리페치 작업자 1개 vs 2개

Qwen3-14B BF16, DRAM/staging 각 8GiB, 청크128, 동시 요청16, max-num-seqs16.
두 조건 모두 DRAM/DAOS 프리페치 ON, 취소 OFF, 조기 준비 알림 OFF, 물리적 staging 용량만 적용.
공통 대기열 + 작업자별 독립 CUDA stream. 대기열을 물리적으로 2개로 나누지는 않았다.
각 조건은 새 프로세스·빈 DRAM·새 DAOS namespace에서 cold256 → 저장 완료 확인 → warm256.
고정 DiscoveryBench 입력 재생이며 full agentic 도구 실행은 아니다. 조건당 1회, rolling 도착 시간·생성량은 달라질 수 있다.

|단계|작업자|평균 TTFT ms|p95 ms|작업 대기 ms|copy 호출 ms|staging 평균/최대 GiB|DRAM hit %|DAOS hit %|추가 재계산 토큰|
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
|cold|1|392.98|2889.48|0.613|16.787|0.178/4.922|89.12|1.03|0|
|warm|1|163.67|428.93|1.758|18.396|0.150/7.988|92.36|7.64|0|
|cold|2|351.25|2713.82|0.341|17.851|0.152/6.777|90.22|1.00|0|
|warm|2|162.66|415.33|0.319|19.264|0.156/7.988|92.49|7.51|0|

copy 시간은 CPU 측 제출+stream 완료 대기이며 순수 GPU DMA 시간이 아니다. 작업 중첩 역시 CPU 타임스탬프 기준이다.
그래프의 0초 점은 시작 순간이 아니라 첫 0~2초 구간 통계다. 각 단계의 시작 직전 staging=0을 검증했다.

## 시간 그래프

- [작업자1 cold](w1/cold/staging_hits.png), [작업자1 warm](w1/warm/staging_hits.png)
- [작업자2 cold](w2/cold/staging_hits.png), [작업자2 warm](w2/warm/staging_hits.png)

[세부 수치](summary.json) · [요청별 재사용량 대조](paired_checks.json) · [검증 결과](validation.json)

## 설정과 원복

`extra_config`의 `daosgds.dram_prefetch_workers: 2`로 활성화한다. `1`로 바꾸거나 항목을 제거하고 프로세스를 재시작하면 원복된다. 기본값은 1이며 공용 YAML은 바꾸지 않았다.
