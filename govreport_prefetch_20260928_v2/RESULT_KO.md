# GovReport 프리페치 OFF/ON 비교

Qwen3-14B BF16 · DRAM8GiB / staging8GiB · 동시 요청16 / max-num-seqs16 · 청크128.
LongBench GovReport200개를 원래 순서로 cold200 → 같은 프로세스에서 warm200으로 실행했다.
OFF/ON은 DRAM 프리페치만 변경. DAOS 프리페치 ON, 복사 작업자1, 대기열 취소/조기 알림 OFF.
각 조건은 새 프로세스·빈 DRAM·새 DAOS namespace. 요청이 끝날 때 다음 요청을 투입하는 rolling 방식.
한 페이지 요약, 출력 최대512토큰, 정상 EOS. 원문이 긴 일부 입력은 32K 한도에 맞춰 앞/뒤를 보존했다.
원본 전체 GovReport 학습/검증셋이 아니라 LongBench의 GovReport200이며, 요약 품질 점수는 평가하지 않았다.

|단계|프리페치|평균 TTFT ms|p95 ms|전체 s|DRAM hit %|DAOS hit %|staging 평균/최대 GiB|생성 토큰|
|---|---|---:|---:|---:|---:|---:|---|---:|
|cold|OFF|1859.17|7632.55|302.25|0.00|0.00|0.051/1.270|91630|
|warm|OFF|355.88|1301.72|147.83|0.00|100.00|0.220/7.988|91508|
|cold|ON|1837.26|7544.49|304.69|0.00|0.00|0.050/1.289|91780|
|warm|ON|479.50|2476.64|151.44|0.00|100.00|0.306/7.988|91571|

## 해석 시 주의

- 조건당 1회다. 처리 속도에 따라 실제 도착 시각·완료 순서·DRAM 보관 상태·생성량은 달라질 수 있다.
- hit 비율은 최초 CPU-tier 조회 후보 청크 기준이다. 재조회가 있으면 조회 횟수 기준이며 실제 재사용 토큰 비율과 다르다.
- 서로 다른 긴 문서200개의 KV는 DRAM8GiB보다 훨씬 크다. warm도 DRAM all-hit가 아니며 DAOS 중심일 수 있다.
- staging 그래프는 2초 구간의 샘플 평균/최대다. 첫 점은 0초 순간이 아니라 첫2초 구간이다.
- 재계산 원인 검증에 문제가 있는 요청은 summary.json에 표시하며 해당 단계의 원인별 재계산율은 단정하지 않는다.

## 그래프

- [OFF cold](off/cold/staging_hits.png) · [OFF warm](off/warm/staging_hits.png)
- [ON cold](on/cold/staging_hits.png) · [ON warm](on/warm/staging_hits.png)

[전체 수치](summary.json) · [입력/절단 정보](dataset.json) · [조건 검증](validation.json)
