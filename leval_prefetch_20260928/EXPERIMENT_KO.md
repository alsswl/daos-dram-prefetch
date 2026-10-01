# L-Eval 문서 QA 기반 DRAM 프리페치 비교

2026-09-28 사용자 승인으로 이전 글쓰기 실험을 중단하고 시작한다. 공식 전체 L-Eval 또는 모델 품질 채점이 아닌, 공개 데이터 기반의 시스템 성능 실험이다.

## 입력 선정

- 원본: OpenLMLab/LEval, commit `cd34b050269148aed75acbbe4a599873ad0f37e9`.
- scientific_qa, financial_qa, legal_contract_qa, narrative_qa 원본 파일을 조사했다.
- Qwen3-14B 토크나이저 기준 본문 4096~8192토큰, 서로 다른 원본 질문이 4개 이상인 문서를 선정했다.
- 중복 본문은 한 번만 사용하며, 문서마다 중복 질문을 제외하고 원본 순서상 앞의 질문 4개를 사용한다. 질문 원본 인덱스·참고 답변·파일 SHA256·제외 이유를 보존한다.
- 선정 결과: 논문 QA 11개, 금융 QA 6개, 계약서 QA 1개 = 18문서. 원문 절단·패딩·질문 복제 없음.
- 본문 길이 4402~6875토큰. 채팅 형식을 포함한 최대 요청 입력 6967토큰, 출력 상한 1024토큰. 32768 컨텍스트 한도 안에 여유 있게 들어간다.
- 현재 Qwen3-14B BF16의 토큰당 KV 163840바이트 기준 본문 KV 합계 약 14.95GiB. 이는 추정 작업 집합 크기이며 실제 resident 메모리 측정값은 아니다.

## 요청 방식

`동일한 공통 지시문 + 원본 문서 + 서로 다른 질문` 형식이다. 질문은 문서 뒤에 둔다. 앞에서 생성한 답변을 다음 입력에 붙이지 않으므로 OFF/ON의 입력 텍스트·토큰 수를 동일하게 유지한다.

총 72개 요청을 고정 seed 20260928로 섞으며 문서별 질문 순서는 보존한다. 이것은 데이터셋이 제공한 실제 서비스 도착 기록이 아니라, 실험에서 정의한 고정 투입 목록이다. 동시 실행 상한을 8 또는 16으로 두고 완료되는 자리마다 다음 요청을 넣는다. 단계별 일괄 대기는 없다. 같은 문서의 독립 질문이 동시에 진행될 수 있으며 최초 저장 전 중복 계산도 결과에 포함한다.

동일 동시 요청 조건의 OFF/ON에 같은 목록·투입 순서를 사용하지만, 완료 속도에 따라 실제 도착 시각은 달라진다. 따라서 고정 wall-clock arrival 실험이라고 해석하지 않는다.

## 비교 조건

- Qwen/Qwen3-14B BF16, thinking OFF, vLLM 내부 prefix cache OFF, max-num-seqs 16.
- DRAM 8GiB / GPU staging 8GiB / LMCache 청크128 / DRAM 복사 작업자1.
- 동시 요청 8/16 × DRAM 프리페치 OFF/ON × 3회 = 12조건, 총 864회 호출.
- ON은 early_ready ON, queued cancellation OFF. 점유율 watermark 전략 없음. DAOS 프리페치는 두 방식 모두 ON.
- temperature 0, seed 0, 출력 최대1024토큰, 정상 EOS. 최소 출력·ignore_eos·별도 stop 문자열 없음.
- 출력 토큰 수·출력 상한 도달 수를 기록한다. 입력이 같아도 생성 결과가 항상 동일하다고 가정하지 않는다.
- 각 조건마다 새 프로세스, 빈 DRAM/staging, 새 DAOS namespace로 시작. 사전 warmup 요청이나 강제 DRAM/DAOS 배치 없음.
- 실험 중 정상 저장·읽기 승격·퇴출이 발생한다. hit 비율을 목표값으로 강제하지 않는다.
- 반복2는 ON→OFF, 반복1/3은 OFF→ON 순서다. 조건별 DAOS KV는 종료 후 해당 namespace만 삭제한다.

## 측정과 해석

- TTFT, 72회 전체 처리 시간, 생성·입력·재사용 토큰 수.
- DRAM/DAOS hit 비율, 시간에 따른 staging 점유, 용량 부족으로 인한 재계산.
- retrieve 호스트 구간, DRAM 프리페치 작업 대기·복사 구간, 준비 완료부터 retrieve까지 간격.
- `timing_requests.json`과 `timing_summary.json`의 복사 시간은 호스트 enqueue/동기화 구간을 포함하며 순수 DMA 시간이 아니다. retrieve 시간도 전송만의 시간이 아니다.
- 준비 완료가 retrieve보다 빠르다는 사실만으로 GPU 계산과의 실제 중첩 또는 순수 절감 시간을 확정하지 않는다. 정확한 장치 중첩 증명이 필요하면 별도 CUDA 타임라인 분석이 필요하다.
- 원본 입력 재사용이 목적이며 공식 품질 점수는 계산하지 않는다. 참고 정답과 실제 생성 출력은 보존한다.

## 결과 위치와 관리

- 진행: `status.json`, `supervision/heartbeat.json`
- 요약: `RESULT_KO.md`, `summary.json`
- 조건별: `staging_hits.png/svg`, `requests_summary.json`, `summary.json`, `timing_summary.json`
- 서비스: root 사용자 systemd의 `discos-minji-leval.service`.
- 일시적 연결/실행기 중단은 기존 supervisor의 제한 재시도를 사용한다. 실패 시도를 보존하고 안전 확인 후 해당 KV만 정리한다. OOM·계측·코드 오류는 무한 재시도하지 않고 조사 필요 상태로 기록한다.

시작 전 단위 테스트 198개 통과. 기존 EQ-Bench 완료 10조건 및 중단 시도 로그는 보존하고, 중단 namespace의 KV 2865개만 삭제했다. 다른 54225개 키는 그대로 보존했다.
