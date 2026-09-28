# 통신 스택 선택 이유 그림 — 생성 프롬프트

도구: 내장 `image_gen` (CLI/API fallback 미사용).
용도: UCX 대신 CUDA 지원 libfabric을 선택한 근거를 설명하는 PPT용 PNG.
근거: 사용자가 제공한 글루시스 문서 §4, §6.1, §8, §9. 수치는 문서 인용이며 우리 재현 측정값이 아니다.

```text
Use case: infographic-diagram.
Asset type: Korean technical PowerPoint slide, single high-resolution landscape 16:9 PNG. Completely opaque solid white background across the entire canvas (not transparent). Clean scientific infographic, dark navy readable Korean sans-serif type, muted gray-blue for the UCX comparison and teal for the selected libfabric option, generous whitespace, no photo, no watermark, no logos.
Primary request: explain WHY a DAOS GPU-direct project selected CUDA-enabled libfabric verbs instead of the previously tested UCX transport. The central evidence is a published-in-the-supplied-Gluesys-document raw GPU-direct READ bandwidth comparison, NOT the user's new benchmark and NOT end-to-end LLM performance.
Composition: title top, two equally sized comparison cards in the middle with a left-to-right decision arrow between them, then one shared implementation strip, then a conclusion and evidence footnotes. Keep all wording below exact, legible and concise. Do not add any extra performance claims.
Top title exactly: "왜 UCX 대신 CUDA 지원 libfabric인가?"
Subtitle exactly: "같은 GPU-direct 읽기 조건에서 확인한 전송 성능 차이"
Left comparison card:
small label "글루시스 초기 검증"
large heading "UCX"
technical sublabel "ucx+rc_v"
simple small GPU-chip and network-link icons with the label "GPU 직접 읽기"
large number "16.0 GB/s"
small neutral text "해당 구성에서 낮은 대역폭 관측"
Right comparison card with subtle teal border and small selection check:
small label "선택한 전송 경로"
large heading "CUDA 지원 libfabric"
technical sublabel "verbs;ofi_rxm"
the SAME simple GPU-chip and network-link icons and label "GPU 직접 읽기"
large number "35.3 GB/s"
small text "같은 조건에서 더 높은 대역폭 확인"
Between cards, small rightward arrow, short label "전송 경로 교체".
Below both cards, one wide pale-teal strip with three clearly separated short phrases, connected in a logical sequence:
"CUDA 메모리 지원 빌드" → "GPU 메모리 등록 경로 보완" → "DFS · Object 공통 적용"
Conclusion below strip in bold navy:
"실측으로 검증한 GPU-direct 통신 스택을 통합본에 재사용"
Two readable footer lines:
"출처: 글루시스 문서 §6.1 · S16 · 16 workers · 8 GiB 파일 · 32 MiB 요청"
"문서의 raw 읽기 실측값이며 우리 재현값·E2E 결과가 아님 · UCX도 GPU-direct 지원"
Scientific accuracy constraints: Both options support GPU-direct; never draw UCX as necessarily bouncing through CPU DRAM, never label UCX inherently slow, never imply libfabric bypasses UCX as a software layer, these are alternative transport paths. Do not include speedup ratio or any claim of inference speedup. Do not depict the original discos as originally using UCX; the left card is the Gluesys initial validation only. No red error crosses. Entire canvas opaque white, including margins.
```
