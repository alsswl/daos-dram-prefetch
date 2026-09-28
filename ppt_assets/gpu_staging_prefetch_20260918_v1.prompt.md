# GPU staging 그림 생성 프롬프트

생성 방식: 내장 image_gen 도구. 생성 후 PPT용 이미지를 검토하고 프로젝트에 복사함.
이미지: gpu_staging_prefetch_20260918_v1.png

```text
Use case: infographic-diagram / scientific-educational.
Create ONE polished Korean educational diagram as a high-resolution 16:9 landscape bitmap for a research PowerPoint, ideally 2048x1152. Topic: GPU staging in the discos_minji async KV-cache READ path.
Style: clean academic slide graphic, pure white background, crisp flat vector-like shapes, restrained navy/blue/teal palette, abundant whitespace, readable large Korean sans-serif typography, no photography, no 3D perspective, no logos, no watermark.

Scientific lesson: GPU staging is a temporary buffer INSIDE GPU HBM, distinct from the model's paged KV cache, which is also inside the SAME GPU HBM. DAOS KV payload is prefetched directly into staging after a cache hit is found during lookup; retrieve later copies/scatters it within GPU memory into vLLM paged KV cache for model computation. BOTH DFS and direct-object transports share this staging path. Metadata uses host memory. Do NOT depict staging as CPU RAM or as another GPU; do NOT depict the NIC directly filling vLLM KV cache; do NOT imply all host memory is eliminated; do NOT imply staging was newly invented here.

Composition: one left-to-right flow across the center. Left, outside the GPU boundary: a compact DAOS storage/server box with simple storage icon and a few KV chunk tiles. Center and right: ONE large light-tinted rounded rectangle, conspicuously labeled "GPU 메모리 (HBM)", enclosing TWO distinct memory compartments and their connecting arrow. The first inner compartment is blue "GPU staging", with three ordered chunk tiles A, B, C and the caption "미리 받아두는 임시 버퍼". The second inner compartment is teal "vLLM paged KV cache", with a small organized grid of page blocks and the caption "모델 계산에 사용하는 공간". The grid is conceptual, not to scale. Clearly keep both compartments INSIDE the large GPU boundary.

Two thick right-pointing arrows only:
1. From DAOS storage directly into GPU staging (crossing the outer GPU boundary), labeled above "① 프리페치", with two short readable lines below: "lookup 중 hit 확인 후" and "GPU 직접 읽기".
2. From staging to vLLM KV cache (entirely within the GPU boundary), labeled above "② retrieve", and below "GPU 내부 복사·배치".
Give the arrows enough horizontal space that labels do not overlap box boundaries or compartment text.

Exact Korean/English text:
Top title: "GPU staging: KV를 미리 받아두는 공간"
Smaller subtitle: "discos_minji 비동기 로딩 경로 · enable_async_loading=true"
Left storage title: "DAOS 저장소"
Left storage caption: "저장된 KV 캐시"
Outer GPU title: "GPU 메모리 (HBM)"
Blue compartment title: "GPU staging"
Blue compartment caption: "미리 받아두는 임시 버퍼"
Teal compartment title: "vLLM paged KV cache"
Teal compartment caption: "모델 계산에 사용하는 공간"
Arrow 1 labels: "① 프리페치", "lookup 중 hit 확인 후", "GPU 직접 읽기"
Arrow 2 labels: "② retrieve", "GPU 내부 복사·배치"
Footer, readable and unobtrusive: "DFS 사용·우회 모두 같은 GPU staging 경로 사용"
Small bottom note: "KV 읽기 경로 기준 · 키와 메타데이터는 호스트 메모리에서 처리"

Render text accurately and verbatim, maintain case of vLLM and GPU, no extraneous labels or paragraphs. Avoid dense decoration; visually emphasize the staging buffer with a strong blue border. Flat, publication-quality diagram suitable to insert directly in a PPT.
```

