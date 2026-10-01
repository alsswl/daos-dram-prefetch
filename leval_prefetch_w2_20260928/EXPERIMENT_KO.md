# L-Eval DRAM 프리페치 작업자 2개 재실험

2026-09-28 사용자 요청: 기존 실행을 멈추고 DRAM 프리페치 작업자 2개로 같은 실험 재실행.

기존 `/root/discos_minji/leval_prefetch_20260928`은 이미 12조건 완료 후 정상 종료되어 있었다. 실행 중인 GPU 작업은 없었고, 기존 12조건 모두 해당 DAOS namespace의 KV 정리 완료를 확인했다. 기존 측정값·로그·그래프는 수정하지 않는다.

## 변경 사항

`daosgds.dram_prefetch_workers: 1 → 2`만 변경한다. 공유 작업 대기열을 작업자 2개가 소비하며 각 작업자는 자기 CUDA 복사 stream을 사용한다. OFF 조건에서는 프리페치 executor를 생성하지 않는다. DAOS I/O/meta 작업자 설정, DRAM 보관·승격 정책, staging allocator와 용량은 변경하지 않는다.

실험 실행기에는 worker 수를 plan에 저장하고 재개 시 같은 값으로 복원하는 기능을 추가했다. 저장 계층과 프리페치 구현 코드 자체는 변경하지 않았다. 기본값은 기존과 같은 1이다.

## 유지되는 조건

- Qwen/Qwen3-14B BF16, thinking OFF, max-model-len 32768, max-num-seqs 16.
- DRAM 8GiB / GPU staging 8GiB / 청크128.
- 동시 요청 8/16 × DRAM 프리페치 OFF/ON × 3회 = 12조건.
- ON: early_ready ON, queued cancellation OFF, 점유율 watermark 전략 없음. DAOS 프리페치는 항상 ON.
- L-Eval 원본 문서18개, 문서당 서로 다른 질문4개 = 조건당72회, 총864회.
- 기존 `requests.json`과 `documents.json`을 바이트 단위로 그대로 복사했다. 요청 SHA256: `b9d36a176907bd78abc368bf250ce69111db4b0e79a9b7c654cc2d748624c331`.
- 본문 길이4402~6875토큰, 최대 전체 입력6967토큰, 출력 상한1024토큰, temperature0, seed0, 정상 EOS.
- 동일한 고정 입력·투입 순서, 완료되는 자리마다 다음 요청을 넣는 rolling 방식. 실제 도착 시각과 출력 길이는 결과에 따라 달라질 수 있다.
- 조건마다 새 프로세스·빈 DRAM/staging·새 DAOS namespace. 완료 후 해당 namespace의 KV만 정리한다.
- 이전의 1-worker ON 결과와 비교할 수 있도록 OFF 대조군도 다시 측정한다. 실행 시점 차이와 생성·재사용량 차이를 함께 확인해야 한다.

## 검증·기록

시작 전 테스트 200개 통과. 1-worker와 2-worker의 생성 설정은 worker 수를 제외하고 완전히 같은지 검증했다. 기존 입력·문서 해시 일치와 모든 이전 실험 namespace의 정리 영수증을 확인했다.

- 진행: `status.json`, `supervision/heartbeat.json`
- 요약: `RESULT_KO.md`, `summary.json`
- 조건별: `staging_hits.png/svg`, `summary.json`, `requests_summary.json`, `timing_summary.json`
- 작업 대기·복사 시간은 호스트 측 측정 구간으로 순수 DMA 시간이나 GPU 계산과의 중첩을 직접 증명하지 않는다.
- 서비스: root 사용자 systemd의 `discos-minji-leval-w2.service`.
- 기존 supervisor의 제한적인 중단 복구를 사용하며, OOM·코드/계측 오류는 무한 재시도하지 않는다.
