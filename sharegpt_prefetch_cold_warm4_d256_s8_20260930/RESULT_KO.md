# ShareGPT cold / warm 비교

Qwen3-14B, DRAM256GiB / staging8GiB / 동시요청16. DRAM 프리페치만 OFF/ON.

|정책|단계|요청|전체 s|평균 TTFT ms|p95 ms|DRAM hit %|DAOS hit %|입력 재사용 %|최대 staging GiB|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|c16_off|[cold](c16_off/cold/staging_hits.png)|1684|515.73|260.00|595.87|26.49|34.76|60.12|3.398|
|c16_off|[warm1](c16_off/warm1/staging_hits.png)|1684|399.11|121.90|224.48|37.80|62.20|98.16|4.316|
|c16_off|[warm2](c16_off/warm2/staging_hits.png)|1684|402.52|121.79|225.43|39.48|60.52|98.16|4.004|
|c16_off|[warm3](c16_off/warm3/staging_hits.png)|1684|403.41|122.17|219.38|39.65|60.35|98.16|4.473|
|c16_off|[warm4](c16_off/warm4/staging_hits.png)|1684|404.28|120.94|206.94|40.05|59.95|98.16|4.004|
|c16_on|[cold](c16_on/cold/staging_hits.png)|1684|509.61|260.01|595.37|26.49|34.76|60.12|4.141|
|c16_on|[warm1](c16_on/warm1/staging_hits.png)|1684|384.18|114.94|210.48|37.82|62.18|98.16|4.961|
|c16_on|[warm2](c16_on/warm2/staging_hits.png)|1684|386.97|115.99|210.52|39.39|60.61|98.16|5.137|
|c16_on|[warm3](c16_on/warm3/staging_hits.png)|1684|388.25|115.81|200.91|39.77|60.23|98.16|5.957|
|c16_on|[warm4](c16_on/warm4/staging_hits.png)|1684|390.45|116.33|207.84|39.86|60.14|98.16|5.957|

```json
{
  "cold": {
    "requests": 1684,
    "same_cached_requests": 1684,
    "same_output_requests": 732
  },
  "warm1": {
    "requests": 1684,
    "same_cached_requests": 1684,
    "same_output_requests": 966
  },
  "warm2": {
    "requests": 1684,
    "same_cached_requests": 1684,
    "same_output_requests": 960
  },
  "warm3": {
    "requests": 1684,
    "same_cached_requests": 1684,
    "same_output_requests": 975
  },
  "warm4": {
    "requests": 1684,
    "same_cached_requests": 1684,
    "same_output_requests": 968
  }
}
```

warm은 해당 조건의 cold 완료 후 같은 프로세스·DRAM·DAOS 캐시를 유지한 재생이다. 각 단계 사이 staging과 비동기 쓰기만 drain한다. warm 반복 사이에도 캐시를 초기화하지 않는다. DRAM/DAOS 배치는 실측하며 전부 DRAM hit라고 가정하지 않는다.
hit 비율은 최초 lookup 후보 청크 기준, 입력 재사용률은 서버 cached_tokens 기준이다. 그래프는20ms 샘플의2초 구간 최대/평균이며 짧은 event peak는 표에만 잡힐 수 있다. staging 점유에는 DAOS/DRAM 읽기와 쓰기가 모두 포함된다.
warm 반복은 캐시를 유지한 연속 재사용이며 독립 시행이 아니다. 도착 시각·생성량·계층별 캐시 배치는 다를 수 있다. 입력/캐시/출력 일치를 함께 확인한다. 시작·초기화·drain·삭제 시간은 요청 처리 시간에서 제외한다. OS/서버 캐시를 강제로 비우지 않는다.
