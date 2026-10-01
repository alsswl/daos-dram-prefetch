# ShareGPT cold / warm 비교

Qwen3-14B, DRAM256GiB / staging8GiB / 동시요청16. DRAM 프리페치만 OFF/ON.

|정책|단계|요청|전체 s|평균 TTFT ms|p95 ms|DRAM hit %|DAOS hit %|입력 재사용 %|최대 staging GiB|
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
|c16_off|[cold](c16_off/cold/staging_hits.png)|1684|527.58|257.02|574.97|26.53|34.72|60.12|3.262|
|c16_off|[warm](c16_off/warm/staging_hits.png)|1684|405.22|121.83|220.06|37.92|62.08|98.16|4.043|
|c16_on|[cold](c16_on/cold/staging_hits.png)|1684|508.11|257.86|595.63|26.49|34.76|60.12|3.672|
|c16_on|[warm](c16_on/warm/staging_hits.png)|1684|381.06|115.29|217.55|37.84|62.16|98.16|5.137|

```json
{
  "cold": {
    "requests": 1684,
    "same_cached_requests": 1684,
    "same_output_requests": 743
  },
  "warm": {
    "requests": 1684,
    "same_cached_requests": 1684,
    "same_output_requests": 975
  }
}
```

warm은 해당 조건의 cold 완료 후 같은 프로세스·DRAM·DAOS 캐시를 유지한 재생이다. 두 단계 사이 staging과 비동기 쓰기만 drain한다. 워크로드가 DRAM보다 커서 warm도 DAOS hit가 발생할 수 있다.
hit 비율은 최초 lookup 후보 청크 기준, 입력 재사용률은 서버 cached_tokens 기준이다. 그래프는20ms 샘플의2초 구간 최대/평균이며 짧은 event peak는 표에만 잡힐 수 있다. staging 점유에는 DAOS/DRAM 읽기와 쓰기가 모두 포함된다.
각 조건1회이며 도착 시각·생성량·계층별 캐시 배치는 다를 수 있다. 입력/캐시/출력 일치를 함께 확인한다. 시작·초기화·drain·삭제 시간은 요청 처리 시간에서 제외한다. OS/서버 캐시를 강제로 비우지 않는다.

## 한눈에 보기

[통합 이미지](OVERVIEW.png) · [한글 대시보드](OVERVIEW.html)

왼쪽 OFF / 오른쪽 ON, 위쪽 cold / 아래쪽 warm. 결과 표와 네 조건의 점유율·hit 그래프를 한 장에 모았다.
