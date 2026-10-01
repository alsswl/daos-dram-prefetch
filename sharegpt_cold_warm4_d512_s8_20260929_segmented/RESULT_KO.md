# ShareGPT cold / warm 비교

Qwen3-14B, DRAM512GiB / staging8GiB / 동시요청16. DRAM 프리페치만 OFF/ON.

|정책|단계|요청|전체 s|평균 TTFT ms|p95 ms|DRAM hit %|DAOS hit %|입력 재사용 %|최대 staging GiB|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|c16_off|[cold](c16_off/cold/staging_hits.png)|1684|512.03|260.30|646.62|61.24|0.00|60.12|1.230|
|c16_off|[warm1](c16_off/warm1/staging_hits.png)|1684|420.74|124.93|205.26|100.00|0.00|98.16|0.059|
|c16_off|[warm2](c16_off/warm2/staging_hits.png)|1684|425.28|127.39|211.44|100.00|0.00|98.16|0.059|
|c16_off|[warm3](c16_off/warm3/staging_hits.png)|1684|429.61|129.96|208.22|100.00|0.00|98.16|0.039|
|c16_off|[warm4](c16_off/warm4/staging_hits.png)|1684|423.36|129.90|229.50|100.00|0.00|98.16|0.039|
|c16_on|[cold](c16_on/cold/staging_hits.png)|1684|505.81|258.60|615.56|61.24|0.00|60.12|3.672|
|c16_on|[warm1](c16_on/warm1/staging_hits.png)|1684|402.17|111.82|171.57|100.00|0.00|98.16|5.391|
|c16_on|[warm2](c16_on/warm2/staging_hits.png)|1684|404.06|111.68|168.63|100.00|0.00|98.16|5.039|
|c16_on|[warm3](c16_on/warm3/staging_hits.png)|1684|411.53|113.01|165.51|100.00|0.00|98.16|5.391|
|c16_on|[warm4](c16_on/warm4/staging_hits.png)|1684|413.22|114.06|169.29|100.00|0.00|98.16|5.391|

```json
{
  "cold": {
    "requests": 1684,
    "same_cached_requests": 1684,
    "same_output_requests": 745
  },
  "warm1": {
    "requests": 1684,
    "same_cached_requests": 1684,
    "same_output_requests": 971
  },
  "warm2": {
    "requests": 1684,
    "same_cached_requests": 1684,
    "same_output_requests": 998
  },
  "warm3": {
    "requests": 1684,
    "same_cached_requests": 1684,
    "same_output_requests": 958
  },
  "warm4": {
    "requests": 1684,
    "same_cached_requests": 1684,
    "same_output_requests": 952
  }
}
```

warm은 해당 조건의 cold 완료 후 같은 프로세스·DRAM·DAOS 캐시를 유지한 재생이다. 각 단계 사이 staging과 비동기 쓰기만 drain한다. warm 반복 사이에도 캐시를 초기화하지 않는다. DRAM/DAOS 배치는 실측하며 전부 DRAM hit라고 가정하지 않는다.
hit 비율은 최초 lookup 후보 청크 기준, 입력 재사용률은 서버 cached_tokens 기준이다. 그래프는20ms 샘플의2초 구간 최대/평균이며 짧은 event peak는 표에만 잡힐 수 있다. staging 점유에는 DAOS/DRAM 읽기와 쓰기가 모두 포함된다.
warm 반복은 캐시를 유지한 연속 재사용이며 독립 시행이 아니다. 도착 시각·생성량·계층별 캐시 배치는 다를 수 있다. 입력/캐시/출력 일치를 함께 확인한다. 시작·초기화·drain·삭제 시간은 요청 처리 시간에서 제외한다. OS/서버 캐시를 강제로 비우지 않는다.
