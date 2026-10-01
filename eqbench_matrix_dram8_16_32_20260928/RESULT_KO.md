# EQ-Bench Longform 36조건 비교

진행 중인 조건은 표에서 제외한다. 조건마다 빈 DRAM·새 DAOS namespace로 시작한다.
첫4개 주제를 각각4번: 총16대화×13단계=208호출/조건. 실제 답변 누적, 반복 주제 간 재사용 포함.
전체 warm 반복이 아니라 cold 시작 후 자연스러운 재사용이다. 출력·계산량 차이도 확인해야 한다.
wait/cancel은 모두 early_ready ON: 취소 외 설정 동일. 이전 wait(early_ready OFF) 실험과 다르다.
취소는 retrieve 시 아직 시작하지 않은 DRAM 복사에만 적용. DAOS 프리페치는 모든 조건에서 ON.
모델 Qwen3-14B BF16, 청크128, 작업자1, max-num-seqs16. 조건별1회, 공식 품질 채점 아님.

|조건|시간(s)|평균 TTFT(ms)|DRAM hit %|DAOS hit %|staging 최대 GiB|용량 부족 재계산 토큰|대기 취소|생성 토큰|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|[c8_d8_s4_off](c8_d8_s4_off/staging_hits.png)|576.42|244.34|9.45|76.67|2.461|0|0|263279|
|[c8_d8_s4_wait](c8_d8_s4_wait/staging_hits.png)|582.89|248.25|9.97|76.12|3.438|0|0|262727|
|[c8_d8_s4_cancel](c8_d8_s4_cancel/staging_hits.png)|567.88|248.76|8.92|77.39|3.984|6656|0|257099|
|[c8_d8_s8_wait](c8_d8_s8_wait/staging_hits.png)|568.03|234.88|10.83|75.41|3.105|0|0|259391|
