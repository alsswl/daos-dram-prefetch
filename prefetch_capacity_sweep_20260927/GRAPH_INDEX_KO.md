# 시간별 staging 점유율·hit 그래프

각 링크에는 위쪽 staging 점유율, 아래쪽 DRAM/DAOS lookup hit 비율이 있다.
staging의 파랑은 2초 구간 내 표본 최댓값, 초록은 표본 평균이다. 짧은 점유 급증과 구간 평균은 다르다.
hit의 파랑은 DRAM, 초록은 DAOS다. lookup hit는 최종 재사용 성공과 다를 수 있다.
각 그래프의 시간 0은 해당 단계의 시작이다. 서로 다른 실행의 같은 시각이 같은 요청을 뜻하지 않는다.

## warm

|동시 요청|DRAM GiB|staging GiB|OFF|ON/취소 OFF|ON/취소 ON|
|---:|---:|---:|---|---|---|
|8|8|8|[그래프](c8_d8_s8_off/warm/staging_hits.png)|[그래프](c8_d8_s8_wait/warm/staging_hits.png)|[그래프](c8_d8_s8_cancel/warm/staging_hits.png)|
|8|8|4|[그래프](c8_d8_s4_off/warm/staging_hits.png)|[그래프](c8_d8_s4_wait/warm/staging_hits.png)|[그래프](c8_d8_s4_cancel/warm/staging_hits.png)|
|8|4|8|[그래프](c8_d4_s8_off/warm/staging_hits.png)|[그래프](c8_d4_s8_wait/warm/staging_hits.png)|[그래프](c8_d4_s8_cancel/warm/staging_hits.png)|
|8|4|4|[그래프](c8_d4_s4_off/warm/staging_hits.png)|[그래프](c8_d4_s4_wait/warm/staging_hits.png)|[그래프](c8_d4_s4_cancel/warm/staging_hits.png)|
|8|2|8|[그래프](c8_d2_s8_off/warm/staging_hits.png)|[그래프](c8_d2_s8_wait/warm/staging_hits.png)|[그래프](c8_d2_s8_cancel/warm/staging_hits.png)|
|8|2|4|[그래프](c8_d2_s4_off/warm/staging_hits.png)|[그래프](c8_d2_s4_wait/warm/staging_hits.png)|[그래프](c8_d2_s4_cancel/warm/staging_hits.png)|
|16|8|8|[그래프](c16_d8_s8_off/warm/staging_hits.png)|[그래프](c16_d8_s8_wait/warm/staging_hits.png)|[그래프](c16_d8_s8_cancel/warm/staging_hits.png)|
|16|8|4|[그래프](c16_d8_s4_off/warm/staging_hits.png)|[그래프](c16_d8_s4_wait/warm/staging_hits.png)|[그래프](c16_d8_s4_cancel/warm/staging_hits.png)|
|16|4|8|[그래프](c16_d4_s8_off/warm/staging_hits.png)|[그래프](c16_d4_s8_wait/warm/staging_hits.png)|[그래프](c16_d4_s8_cancel/warm/staging_hits.png)|
|16|4|4|[그래프](c16_d4_s4_off/warm/staging_hits.png)|[그래프](c16_d4_s4_wait/warm/staging_hits.png)|[그래프](c16_d4_s4_cancel/warm/staging_hits.png)|
|16|2|8|[그래프](c16_d2_s8_off/warm/staging_hits.png)|[그래프](c16_d2_s8_wait/warm/staging_hits.png)|[그래프](c16_d2_s8_cancel/warm/staging_hits.png)|
|16|2|4|[그래프](c16_d2_s4_off/warm/staging_hits.png)|[그래프](c16_d2_s4_wait/warm/staging_hits.png)|[그래프](c16_d2_s4_cancel/warm/staging_hits.png)|
## cold

|동시 요청|DRAM GiB|staging GiB|OFF|ON/취소 OFF|ON/취소 ON|
|---:|---:|---:|---|---|---|
|8|8|8|[그래프](c8_d8_s8_off/cold/staging_hits.png)|[그래프](c8_d8_s8_wait/cold/staging_hits.png)|[그래프](c8_d8_s8_cancel/cold/staging_hits.png)|
|8|8|4|[그래프](c8_d8_s4_off/cold/staging_hits.png)|[그래프](c8_d8_s4_wait/cold/staging_hits.png)|[그래프](c8_d8_s4_cancel/cold/staging_hits.png)|
|8|4|8|[그래프](c8_d4_s8_off/cold/staging_hits.png)|[그래프](c8_d4_s8_wait/cold/staging_hits.png)|[그래프](c8_d4_s8_cancel/cold/staging_hits.png)|
|8|4|4|[그래프](c8_d4_s4_off/cold/staging_hits.png)|[그래프](c8_d4_s4_wait/cold/staging_hits.png)|[그래프](c8_d4_s4_cancel/cold/staging_hits.png)|
|8|2|8|[그래프](c8_d2_s8_off/cold/staging_hits.png)|[그래프](c8_d2_s8_wait/cold/staging_hits.png)|[그래프](c8_d2_s8_cancel/cold/staging_hits.png)|
|8|2|4|[그래프](c8_d2_s4_off/cold/staging_hits.png)|[그래프](c8_d2_s4_wait/cold/staging_hits.png)|[그래프](c8_d2_s4_cancel/cold/staging_hits.png)|
|16|8|8|[그래프](c16_d8_s8_off/cold/staging_hits.png)|[그래프](c16_d8_s8_wait/cold/staging_hits.png)|[그래프](c16_d8_s8_cancel/cold/staging_hits.png)|
|16|8|4|[그래프](c16_d8_s4_off/cold/staging_hits.png)|[그래프](c16_d8_s4_wait/cold/staging_hits.png)|[그래프](c16_d8_s4_cancel/cold/staging_hits.png)|
|16|4|8|[그래프](c16_d4_s8_off/cold/staging_hits.png)|[그래프](c16_d4_s8_wait/cold/staging_hits.png)|[그래프](c16_d4_s8_cancel/cold/staging_hits.png)|
|16|4|4|[그래프](c16_d4_s4_off/cold/staging_hits.png)|[그래프](c16_d4_s4_wait/cold/staging_hits.png)|[그래프](c16_d4_s4_cancel/cold/staging_hits.png)|
|16|2|8|[그래프](c16_d2_s8_off/cold/staging_hits.png)|[그래프](c16_d2_s8_wait/cold/staging_hits.png)|[그래프](c16_d2_s8_cancel/cold/staging_hits.png)|
|16|2|4|[그래프](c16_d2_s4_off/cold/staging_hits.png)|[그래프](c16_d2_s4_wait/cold/staging_hits.png)|[그래프](c16_d2_s4_cancel/cold/staging_hits.png)|
