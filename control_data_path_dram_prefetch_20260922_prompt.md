Generation method: built-in image_gen tool (imagegen skill). Conceptual diagram, not a measured trace.

Use case: scientific-educational / infographic-diagram.
Create one crisp Korean technical explainer image, landscape 16:9 high resolution, suitable for a presentation. White background, large legible Korean sans-serif type, generous spacing, no photos or decorative hardware. Title exactly: "Control path와 Data path는 다르다". Subtitle: "DRAM 캐시 hit · 현재 비레이어별 LMCache 경로".
The purpose is to distinguish control/API events from actual KV byte movement, comparing CPU staging prefetch OFF versus ON. This is a conceptual timeline, NOT measured timings or evidence of actual overlap.
Use two horizontal panels stacked top and bottom (not left/right panels). Each panel has left-to-right progression and two clearly labeled swimlanes: blue "Control path · 호출과 완료 통보" above orange/purple "Data path · 실제 KV 이동". Top panel title "① DRAM 프리페치 OFF". Bottom panel title "② DRAM 프리페치 ON".
TOP PANEL:
Control event order: "lookup 시작" → "CPU 객체 확보·준비 통보" → "배치 선택·KV 페이지 확보" → "retrieve() 시작" → "retrieve() 반환" → "모델 계산".
Data lane BEFORE retrieve marker: a pale gray bar labeled "KV는 DRAM에 유지".
Exactly under the retrieve-start-to-return interval: one long orange bar labeled "DRAM → vLLM KV 페이지" and a smaller sublabel "전송 + 페이지별 배치". An orange-to-blue completion dependency arrow goes UP from this bar's right end to "retrieve() 반환". Data transfer must NOT start during lookup in OFF. Model compute starts only after this transfer completes.
BOTTOM PANEL:
Control event order: "lookup 시작" → "캐시 확인·프리페치 요청" → "GPU 복사 완료·준비 통보" → "배치 선택·KV 페이지 확보" → "retrieve() 시작" → "retrieve() 반환" → "모델 계산".
Data lane has an orange bar AFTER the prefetch request and BEFORE the readiness notification, labeled "DRAM → GPU staging". A thin orange-to-blue UP dependency arrow from the copy end to "GPU 복사 완료·준비 통보" is important. Between this transfer and retrieve show a pale gray bar labeled "staging에 보관". Within retrieve interval show a short purple bar labeled "GPU staging → vLLM KV 페이지", with completion dependency arrow up to retrieve return. No data transfer arrow into the GPU model before data preparation finishes.
Within EACH panel draw separate fine vertical dotted guides at retrieve() start and retrieve() return through both lanes, making the API duration unambiguous. Control path arrows BLUE; CPU-to-GPU payload movement ORANGE; GPU-internal payload movement PURPLE; compute GREEN. Do not portray these lanes as independent: show the completion arrows connecting data and control.
Bottom small legend: "파랑: 제어 흐름   주황: DRAM→GPU   보라: GPU 내부 이동". Bottom takeaway in large clear type: "프리페치 = 데이터 이동을 retrieve() 이전으로 옮기기".
Small caveat exactly: "개념도 · 시간축 비례 아님 · ON/OFF의 실제 retrieve 호출 시각은 달라질 수 있음".
Avoid durations, latency measurements, speedup percentages, claims that lookup automatically starts DMA instantly, or claims that a lookup-to-retrieve gap itself proves overlap. All text readable, no duplicate arrows, no DAOS lane since this figure compares DRAM-hit paths only.

