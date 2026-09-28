# 공통 GPU-direct 통신 스택 그림 생성 기록

생성 방식: 내장 image_gen 도구. 최초 이미지 생성 후 같은 도구로 배경을 불투명 흰색으로 수정함.
최종 이미지: common_gpu_direct_stack_20260918_v1.png

## 생성 프롬프트

```text
Use case: infographic-diagram.
Asset type: ONE Korean research PowerPoint architecture figure, landscape 16:9, high-resolution bitmap, preferably 2048x1152.
Primary request: Explain that discos_minji compares two different DAOS storage APIs using the SAME GPU-direct DAOS and CUDA-enabled libfabric installations and network configuration. Main visual message: two alternative upper storage paths converge into ONE shared lower communication stack. This is a logical software architecture/comparison schematic, NOT a directional read-payload flow chart and NOT simultaneous execution of two workloads.

Style: clean white academic slide, flat vector-like rounded boxes, straight disciplined connectors, navy typography, muted blue for DFS and muted amber for object, teal for the common communication stack. Sharp large Korean sans-serif labels, ample whitespace, strong visual hierarchy, no photos, no 3D, no vendor logos, no watermark.

Layout top-to-bottom:
1. Top title and subtitle.
2. One modest-width centered neutral common box "vLLM + LMCache" and smaller line "공통 GPU staging · 비동기 프리페치".
3. Branch down into TWO equally sized side-by-side alternative API boxes, not nested in each other:
LEFT blue box:
"DFS 사용"
"파일 기반 접근"
"libdfs.so"
"dfs_read_gpu / dfs_write_gpu"
RIGHT amber box:
"DFS 우회 (Object)"
"오브젝트 키 기반 접근"
"libdaosgdr.so"
"daos_obj_fetch_gpu / daos_obj_update_gpu"
The API function lines should be monospaced or especially readable, not tiny.
4. Draw a downward connector from EACH alternative API box, merging into ONE large clearly highlighted teal rounded rectangle. Its heading is "공통 GPU-direct 통신 스택". Inside it, show two clean stacked rows:
Upper row: "DAOS · CaRT · Mercury", secondary text "/opt/daos-gds-gpu"
Lower row: "CUDA 지원 libfabric", secondary text "/opt/ofi-cuda/lib64", and smaller text "provider: verbs;ofi_rxm".
These are the SAME builds/installation paths reused by either mode, not two different shared processes.
5. Below the stack, one compact shared destination bar connected downward:
"동일 클라이언트 NIC · 동일 DAOS pool / container"
Secondary line: "ConnectX-7 · discospool / kvcache"
This is a grouping of common experimental conditions, not a claim NIC is software or that the NIC resides in the storage server.
6. Bottom concise takeaway and small scope caveat.

Exact title: "통신 스택 통일"
Exact subtitle: "DFS / Object 비교를 위한 공통 GPU-direct 환경"
Exact takeaway: "저장 API는 다르게, 통신 환경은 동일하게"
Exact small caveat: "파일·키·메타데이터 구조 차이는 비교 대상에 포함 · 두 모드는 각각 실행"

Accuracy constraints:
- Both paths are GPU-direct. Do not portray DFS as CPU staging or as dfuse/FUSE.
- Object path still uses DAOS. It bypasses DFS only; never bypass the DAOS/common stack.
- Do not put UCX on either branch or imply only one mode uses libfabric.
- The common libfabric provider is exactly "verbs;ofi_rxm"; this label is part of libfabric, not a third physical layer.
- Do not claim identical storage formats, identical dkey mappings, identical RPC counts, guaranteed fairness, or performance superiority.
- Do not claim GPU staging was invented in this integration.
- Text must be accurate verbatim, especially "CaRT", "vLLM", paths, underscores, function names, and provider semicolon.
- Keep the diagram readable at slide presentation size with balanced spacing. It should be usable as a nearly full-slide illustration, with no surrounding editor interface or fake slide frame.
```

## 최종 수정 프롬프트

```text
Edit the provided architecture diagram. Keep all boxes, Korean/English wording, code names, paths, two-path branching and shared communication stack EXACTLY the same. Change ONLY its background/transparency treatment: place the complete diagram on a solid, fully OPAQUE pure WHITE (#FFFFFF) rectangular slide canvas, edge to edge. The present transparent/dark background makes the navy title and arrows unreadable. No transparency anywhere in the final image; fill all transparent and black background regions white while preserving the navy text, colored boxes, thin connector lines, and all labels. Use clean opaque antialiasing with no blue/cyan/yellow transparency fringes around letters or box edges. This must be a conventional white-background PowerPoint diagram, not a cutout or transparent asset. Keep landscape 16:9, preserve all scientific content, do not add or remove text, do not crop labels.
```

