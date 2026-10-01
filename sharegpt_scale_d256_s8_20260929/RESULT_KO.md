# ShareGPT DRAM256 / staging8 / C16 결과

각 조건은 빈 캐시에서 시작하는 원본 이력 재생 1회. 별도 warm 측정은 없음. DAOS 프리페치는 항상 ON.

|정책|요청|전체 s|평균 TTFT ms|p95 ms|DRAM hit %|DAOS hit %|입력 재사용 %|최대 staging GiB|최대 DRAM GiB|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|[c16_off](c16_off/staging_hits.png)|1684|515.17|260.20|587.08|26.50|34.75|60.12|3.770|256.00|
|[c16_on](c16_on/staging_hits.png)|1684|509.12|258.16|608.80|26.49|34.76|60.12|3.516|256.00|

## 그래프

- c16_off: [staging·hit 비율](c16_off/staging_hits.png), [DRAM 점유](c16_off/dram_residency.png)
- c16_on: [staging·hit 비율](c16_on/staging_hits.png), [DRAM 점유](c16_on/dram_residency.png)

같은 요청별 cached_tokens: 1684/1684. 같은 생성 결과: 710/1684.
TTFT 차이는 캐시 재사용량·계산량·출력 차이와 함께 해석해야 한다. 각 조건1회 예비실험이다.

## 점유율과 해석


[OFF/ON 점유율 비교 그림](staging_comparison.png)

|항목|OFF|ON|
|---|---:|---:|
|할당 이벤트 기준 최대 staging 점유율|47.12%|43.95%|
|20ms 샘플 평균 staging 점유율|0.67%|0.88%|
|staging 할당 실패에 귀속된 재계산 토큰|0|0|
|새로 계산한 입력 토큰|2340145|2340145|
|생성 토큰|314679|314810|

평균 TTFT 감소율은 0.78%, 전체 시간 감소율은 1.18%이다. p95 TTFT는 ON이 더 길었다. 각 조건1회라 성능 우열이나 통계적 유의성을 확정하지 않는다.

DRAM 256GiB는 두 조건 모두 실제로 찼다. 하지만 staging이 60% 이상 찬 샘플은 두 조건 모두 없고, DAOS 용량 실패 재계산 및 ON의 DRAM 프리페치 capacity fallback도 0이다. 이 실행에서 지속적인 staging 공간 병목은 관측되지 않았다. 짧은 복사·작업 대기 등 다른 병목이 없다는 뜻은 아니다.

시간별 hit 그래프에서 초반은 신규 계산, 중간은 DRAM 재사용, 뒤쪽은 DAOS 재사용 비중이 커진다. 전체 요청의 hit 비율을 매 순간 동일한 혼합 비율로 해석하면 안 된다. DRAM이 가득 찬 시점과 뒤쪽 DAOS hit 증가가 함께 관측되며, 순환 재방문과 LRU 교체의 영향이 시사된다.

ON/OFF의 요청별 cached_tokens는 1,684개 모두 같고 새 계산 토큰 수도 같다. 다만 생성 문자열은 710개만 일치했다. 이 실험은 입력을 원본 이력으로 고정해 생성 차이가 다음 입력을 바꾸지 않도록 했다.

모델 시작·256GiB 메모리 초기화·조건 사이 공간 회수 대기 시간은 HTTP 요청 측정에서 제외했다. OS/서버 캐시를 강제로 비운 실험은 아니며, cold는 빈 LMCache DRAM과 새로운 DAOS namespace를 뜻한다.

[공간 회수 대기 기록](SPACE_RECOVERY_KO.md). 사전 검증388개, OFF19,853개, ON19,855개 실험 KV 키만 삭제했다. KV payload 별도 백업은 없으며, 다른 namespace 키54,225개와 모든 로그·그래프는 보존했다.
