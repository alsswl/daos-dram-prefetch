# 존재 확인 먼저 알림: 결과

Qwen3-14B / DRAM256GiB / staging8GiB / C16. DRAM·DAOS 프리페치 ON.

|단계|기존 TTFT ms|조기 알림 TTFT ms|변화 %|전체 s|DRAM hit %|DAOS hit %|입력 재사용 %|peak staging GiB|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|[cold](c16_early/cold/staging_hits.png)|260.01|257.29|-1.05|511.81|26.56|34.68|60.12|4.141|
|[warm1](c16_early/warm1/staging_hits.png)|114.94|113.97|-0.84|390.52|37.87|62.13|98.16|5.488|
|[warm2](c16_early/warm2/staging_hits.png)|115.99|114.47|-1.31|392.73|39.29|60.71|98.16|5.957|
|[warm3](c16_early/warm3/staging_hits.png)|115.81|115.27|-0.46|394.30|39.74|60.26|98.16|5.957|
|[warm4](c16_early/warm4/staging_hits.png)|116.33|114.94|-1.19|395.30|39.87|60.13|98.16|5.234|

음수 변화는 TTFT 감소이다. 같은 입력/출력상한을 재생하지만 실제 캐시 배치·생성량·도착 시각은 달라질 수 있다. paired_checks.json을 함께 확인한다. warm4회는 캐시를 유지한 연속 재사용이며 독립 시행이 아니다.
hit 비율은 최초 lookup 후보 청크 기준. staging은 읽기·쓰기 합계. 그래프는20ms 샘플의2초 구간 최대/평균이며 표의 event peak와 다를 수 있다.
retrieve_decisions.json의 ready/wait_running/queued_to_demand는 배치 수이지 청크 비율이 아니다. 기존 완료-first 전용 타이밍 집계기를 재사용하지 않는다.
