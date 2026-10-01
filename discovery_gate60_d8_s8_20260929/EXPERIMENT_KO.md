# DiscoveryBench 60% staging 점유율 정책 비교

## 구현

`daosgds.dram_prefetch_stop_occupancy_ratio: 0.6`으로 활성화한다.
미지정 또는 `null`이면 기존 경로로 돌아간다. 공유 설정 파일은 수정하지 않았으며 실험별 설정만 사용한다.

- DRAM 프리페치 작업자가 staging을 할당하기 직전 전체 사용량을 검사한다.
- 전체 사용량이 8GiB의 60%(4.8GiB) 이상이면 해당 배치의 DRAM→staging 복사를 생략한다.
- 원본 DRAM 객체의 참조·pin은 유지하고 retrieve가 기존 DRAM 경로로 소비한다. 캐시 miss를 만들지 않는다.
- DRAM/DAOS/쓰기 버퍼 등 공유 staging 할당을 모두 포함한 사용량이다.
- 검사와 할당은 같은 allocator lock에서 수행한다. 대기열 진입 당시가 아니라 실제 할당 시점 기준이다.
- 이미 시작한 복사나 DAOS 프리페치는 중단하지 않는다. 점유율이 내려가면 다음 요청은 다시 허용한다.
- 현재 점유율이 60% 미만이면 큰 배치 하나가 60%를 넘길 수 있다. DAOS 할당도 제한하지 않으므로 전체 점유율의 엄격한 상한은 아니다.
- 기존 physical-capacity fallback은 유지한다. 워터마크 예상 점유율 제한과는 별개의 정책이다.

## 실험

이전 `prefetch_cold_warm_d8_s8_20260927`의 입력 파일을 바이트 그대로 복사했다.
Qwen3-14B, 청크128, DRAM/staging 각각 8GiB, object 경로, 작업자1, 조기 준비 알림·대기열 취소 OFF.

동시 요청 8/16 × OFF/ON/gate60 = 6조건이다. 각 조건에서 cold 256개 후 같은 프로세스·캐시로 warm 256개를 실행한다.
총 3,072요청. 조건별 1회이며 cold/warm은 독립 반복이 아니다.

- 조건마다 새 프로세스·빈 DRAM/staging·독립 DAOS namespace로 시작한다.
- cold→warm 사이에는 staging과 비동기 복사가 모두 비었는지 확인하되 DRAM/DAOS 캐시는 유지한다.
- rolling 요청 투입, 원래 EOS/Observation 종료, 최대 생성 2048토큰을 유지한다.
- DiscoveryBench 기록 입력 재생이며 Python 도구를 실제 실행하는 full agentic 벤치마크는 아니다.
- 각 조건의 서버가 종료된 뒤 해당 namespace만 목록·manifest 검증을 거쳐 삭제한다. 기존 캐시·로그는 보존한다.
- 현재 코드의 ON/gate60을 정책 비교의 주 대조군으로 사용한다. 이전 관측값은 별도 참고 표로 제공한다.
- TTFT, hit, 새 입력 계산량, 생성량, 정책 차단 횟수, staging 그래프와 용량 실패 재계산을 기록한다.

## 파일

- `status.json`: 전체 진행 상황
- `service.log`: 진행 및 오류 기록
- `RESULT_KO.md`: 조건 완료 시 갱신되는 비교 결과
- 각 조건의 `cold/staging_hits.png`, `warm/staging_hits.png`: 점유율·hit 그래프
- `paired_checks.json`: 현재 ON과 다른 모드의 요청별 캐시 재사용량 일치 여부
- `plan.json`, `executed_sources/`: 입력·실행 소스 고정 기록

일반 실험 설정은 변경하지 않았다. 정책을 끄려면 해당 실험 설정의 ratio를 `null`로 설정하고 프로세스를 재시작한다.
