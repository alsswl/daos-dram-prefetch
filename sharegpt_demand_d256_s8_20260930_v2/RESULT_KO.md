# ShareGPT cold / warm 비교

Qwen3-14B, DRAM256GiB / staging8GiB / 동시요청16. DRAM·DAOS 프리페치 모두 OFF, retrieve 시 동기 읽기.

|정책|단계|요청|전체 s|평균 TTFT ms|p95 ms|DRAM hit %|DAOS hit %|입력 재사용 %|최대 staging GiB|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|c16_demand|[cold](c16_demand/cold/staging_hits.png)|1684|513.47|238.44|625.80|26.58|34.66|60.12|1.230|
|c16_demand|[warm1](c16_demand/warm1/staging_hits.png)|1684|400.38|107.19|219.55|37.95|62.05|98.16|1.230|
|c16_demand|[warm2](c16_demand/warm2/staging_hits.png)|1684|402.37|106.72|203.79|39.39|60.61|98.16|1.230|
|c16_demand|[warm3](c16_demand/warm3/staging_hits.png)|1684|403.34|107.92|205.43|39.97|60.03|98.16|1.230|
|c16_demand|[warm4](c16_demand/warm4/staging_hits.png)|1684|405.03|105.59|212.34|40.03|59.97|98.16|1.230|

warm은 해당 조건의 cold 완료 후 같은 프로세스·DRAM·DAOS 캐시를 유지한 재생이다. 각 단계 사이 staging과 비동기 쓰기만 drain한다. warm 반복 사이에도 캐시를 초기화하지 않는다. DRAM/DAOS 배치는 실측하며 전부 DRAM hit라고 가정하지 않는다.
hit 비율은 최초 lookup 후보 청크 기준, 입력 재사용률은 서버 cached_tokens 기준이다. 그래프는20ms 샘플의2초 구간 최대/평균이며 짧은 event peak는 표에만 잡힐 수 있다. staging 점유에는 DAOS/DRAM 읽기와 쓰기가 모두 포함된다.
warm 반복은 캐시를 유지한 연속 재사용이며 독립 시행이 아니다. 도착 시각·생성량·계층별 캐시 배치는 다를 수 있다. 입력/캐시/출력 일치를 함께 확인한다. 시작·초기화·drain·삭제 시간은 요청 처리 시간에서 제외한다. OS/서버 캐시를 강제로 비우지 않는다.
