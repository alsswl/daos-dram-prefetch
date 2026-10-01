# 세 가지 읽기 방식 비교

기존 완료 결과: `/root/discos_minji/sharegpt_demand_d256_s8_20260930_v2`.

| 실행 | DRAM 프리페치 | DAOS 프리페치 | lookup/read 방식 |
|---|---|---|---|
| 기존 `c16_demand` | OFF | OFF | 동기 lookup, retrieve에서 데이터 읽기 |
| 이번 `c16_off` | OFF | ON | 비동기 lookup·DAOS 읽기 |
| 이번 `c16_on` | ON | ON | 비동기 lookup·DAOS 읽기·DRAM staging 복사 |

이번 폴더의 OFF/ON은 **DRAM 프리페치**의 구분이다. `c16_off`도 DAOS 프리페치는 켜져 있다.

공통: Qwen3-14B BF16, DRAM256GiB, staging8GiB, 동시 요청16, max-num-seqs16,
max-model-len16384, 청크128, DRAM 복사 작업자1, 대기열 취소OFF, 점유율 조기차단 없음.
동일한 ShareGPT421개 대화의 앞4턴, 단계당1,684개 요청. 출력 상한과 원본 답변을 사용하는
입력 이력도 동일하다. 요청 파일 SHA256:
`e4732c4742a1abb671c121c82a6ef3efbed496075ee07f47e9f25d38593ce016`.

각 조건: 새 프로세스·빈 DRAM·빈 staging·새 DAOS namespace에서 cold1회 실행 후,
동일한 캐시를 유지하면서 warm4회. 두 조건은 순차 실행한다. warm 사이에는 캐시를 지우지
않으며, 조건이 바뀔 때 새 캐시로 시작한다. 완전히 종료된 조건의 전용 namespace만 삭제한다.
OS/DAOS 서버 캐시는 강제로 초기화하지 않는다.

기존 demand와 이번 prefetch 조건은 lookup/스케줄링 프로토콜도 다르므로,
세 조건 간 시간 차이를 순수 DMA 속도 차이라고 해석하지 않는다.
TTFT, 전체 시간, 실제 DRAM/DAOS hit, 입력 재사용·재계산·생성 토큰 수와 staging 점유를
함께 비교한다. warm4회는 캐시를 유지한 연속 실행이며 독립 cold-start 반복이 아니다.
