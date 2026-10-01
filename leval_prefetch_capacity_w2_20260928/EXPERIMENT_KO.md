# L-Eval DRAM·staging 용량과 프리페치 취소 비교

사용자 요청(2026-09-28): DRAM 4/8/16GiB, 동시 요청 8/16, staging 4/8GiB, 프리페치 OFF/ON/ON+대기열 취소 비교.

## 실험 규모

| 변수 | 값 |
|---|---|
| DRAM | 4 / 8 / 16 GiB |
| GPU staging | 4 / 8 GiB |
| 동시 요청 | 8 / 16 |
| DRAM 프리페치 | OFF / ON(wait) / ON+대기 취소(cancel) |
| DRAM 프리페치 작업자 | 2개, 작업자별 CUDA stream 1개 |
| 반복 | 각 조합 3회 |

36개 고유 조합 × 3회 = 총108실행. 실행당72요청, 총7776요청이다. 각 (동시 수, DRAM, staging) 셀의 OFF/ON/cancel 실행 순서는 3회 반복 동안 회전시켜 각 모드가 첫째·둘째·셋째에 한 번씩 배치되게 했다.

## 고정 워크로드와 모델

- L-Eval 원본 문서18개 × 문서별 서로 다른 질문4개. 기존 작업자2개 실험의 입력·문서 파일을 바이트 그대로 복사했다.
- 요청 SHA256: `b9d36a176907bd78abc368bf250ce69111db4b0e79a9b7c654cc2d748624c331`.
- 원본 commit `cd34b050269148aed75acbbe4a599873ad0f37e9`, 논문 QA 11문서·금융 QA 6문서·계약서 QA 1문서.
- 문서 본문4402~6875토큰, 최대 전체 입력6967토큰. 출력 최대1024토큰, 정상 EOS, temperature0, seed0, thinking OFF.
- Qwen/Qwen3-14B BF16, max-model-len32768, max-num-seqs16, vLLM 내부 prefix cache OFF, LMCache 청크128.
- 원문 절단·패딩·질문 복제·최소 출력 강제 없음. 생성 답변을 다음 입력에 붙이지 않는다.
- 고정된 요청 순서를 사용하되 완료되는 자리마다 다음 요청을 넣는 rolling 방식이다. 같은 입력 목록이지만 실제 도착 시각 및 생성량이 고정된 실험은 아니다.
- 문서 본문 KV 전체의 추정 크기는 약14.95GiB다. 큰 DRAM 조건에서 DAOS hit가 적거나 없어질 수 있으며 실제 관측 결과로 기록한다.

## 모드 정의

- OFF: DRAM 프리페치 executor 없음. DAOS 비동기 프리페치는 유지한다.
- ON(wait): DRAM 프리페치 ON, early_ready ON, queued cancellation OFF.
- ON+취소(cancel): DRAM 프리페치 ON, early_ready ON, queued cancellation ON.
- wait와 cancel은 취소 플래그 이외 설정이 동일하다. retrieve가 왔을 때 아직 시작하지 않은 DRAM 복사 작업을 취소할 수 있으며 이미 실행 중인 복사를 중단하는 기능은 아니다.
- staging 점유율 watermark 등 사전 예약 비율 제한 전략 없이 기존 capacity 정책을 사용한다. 실제 공간 확보 실패 처리는 기존 구현 그대로다.
- DAOS I/O/meta 작업 풀, GPU staging allocator, DRAM 저장·읽기 승격 정책은 이전 실험과 동일하다.

## 캐시 격리와 정리

모든 모드·용량·반복마다 새 vLLM 프로세스와 새 UUID DAOS namespace를 사용한다. 측정 시작 시 DRAM 사용량, staging 사용량, DAOS 쓰기가 0인지 확인한다. 실행 중에는 정상적인 저장·승격·퇴출에 따라 캐시 상태가 변한다. 별도 warm 반복이나 원하는 hit 비율을 위한 강제 데이터 배치는 없다.

완료 후 정확히 해당 namespace의 키만 목록·manifest 검증 후 삭제한다. 풀·컨테이너·공용 OID 전체를 삭제하지 않는다. 이전 실험의 로그·그래프·결과는 유지하며 새 실험에 합산하지 않는다. 시작 전 직전 작업자2개 실험의 12개 namespace가 모두 정리됐음을 확인했다.

## 측정과 결과

- 평균/분포 TTFT, 전체72요청 처리 시간, 입력·생성·재사용·새로 계산한 토큰 수.
- DRAM/DAOS lookup 청크 hit 비율, 시간별 staging 점유율 그래프.
- staging 용량 부족으로 인한 DAOS 재계산 토큰 및 비율. 귀속 검증이 불확실하면 0으로 만들지 않고 별도 표시한다.
- DRAM 프리페치 공간 확보 실패·실제 queued cancellation 건수·retrieve 복사 대기 건수.
- 작업 대기·복사·retrieve 호스트 구간. 순수 DMA 시간이나 GPU 계산 중첩의 직접 증거로 해석하지 않는다.
- `RESULT_KO.md`: 조합별 평균 및 각 실행 그래프 링크.
- `aggregate.json/csv`: 완료된 반복의 조합별 평균, 최대 staging 점유율, 재계산·취소 합계.
- 조건별 `summary.json`, `requests_summary.json`, `timing_summary.json`, `staging_hits.png/svg`.
- `status.json`, `supervision/heartbeat.json`: 진행 상태.

서비스: root 사용자 systemd `discos-minji-leval-capacity.service`. 일시적인 연결·실행기 중단은 제한 재시도를 사용한다. 부분 시도와 로그를 보존하고 전용 KV를 정리한 뒤 재시도하며, OOM·계측·코드 오류는 조사 필요 상태로 남긴다. 사용자 systemd 서비스는 재부팅 후 자동 실행을 보장하지 않는다.

시작 전 테스트202개 통과. 108실행 그리드·3회 순서 회전·모드별 설정 차이·실제 staging 용량을 사용한 점유율 계산·원본 입력 해시 일치를 검증했다.
