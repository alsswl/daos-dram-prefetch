# 왜 UCX 대신 CUDA 지원 libfabric인가?

작성: 2026-09-18. 용도: 통신 스택 선택 근거를 설명하는 PPT 1장.
이번 작업에서는 벤치마크를 새로 실행하지 않았다. 아래 성능값은 사용자가 제공한 글루시스 문서의 결과다.

## 핵심 메시지

글루시스의 선행 검증에서 UCX보다 높은 GPU 직접 읽기 대역폭을 보인 CUDA 지원 libfabric(verbs) 스택을 선택하고, 이를 discos_minji의 DFS와 직접 오브젝트 구현에 공통 적용했다.

## 권장 배치

제목 → 두 전송 경로의 GPU 직접 읽기 실측값 → 필요한 GPU 지원 구현 → 통합본에 재사용한 이유.

그림: [why_libfabric_over_ucx_20260918_v1.png](why_libfabric_over_ucx_20260918_v1.png).

## 슬라이드에 넣을 내용

| 전송 경로 | GPU 직접 읽기 대역폭 |
|---|---:|
| UCX (`ucx+rc_v`) | 16.0 GB/s |
| CUDA 지원 libfabric (`verbs;ofi_rxm`) | 35.3 GB/s |

조건: 글루시스 문서 §6.1, S16, 16 workers, 같은 8 GiB 파일, 32 MiB 읽기 요청. 문서는 같은 GDS 클라이언트·서버·드라이브에서 전송 경로만 바꿨다고 기술한다. verbs의 3회 결과는 35.1~35.4 GB/s다. 이는 raw 데이터 읽기이며 모델 추론 E2E 지표가 아니다.

- 선택 이유: 해당 구성의 직접 읽기 성능 검증에서 더 높은 대역폭을 확인했다.
- 적용 전제: CUDA 지원 libfabric 빌드, DAOS device-memory 옵션, CUDA DMA-BUF 등록 및 파일 디스크립터 누수 수정이 필요했다.
- 우리 작업: 위 조건을 갖춘 `/opt/daos-gds-gpu`와 `/opt/ofi-cuda/lib64` 스택을 DFS·Object 모드에 공통 적용했다.

## 발표 예시

“GPU 메모리로 직접 데이터를 읽는 기능이 있어도, 실제 속도는 사용하는 통신 경로에 따라 달라집니다. 글루시스의 동일 조건 시험에서는 UCX가 16.0GB/s, CUDA 지원 libfabric이 35.3GB/s였습니다. 그래서 저희는 선행 검증에서 더 높은 대역폭을 보인 스택을 재사용했고, DFS와 직접 오브젝트 구현 모두에 같은 스택을 적용했습니다. 여기의 수치는 글루시스의 데이터 읽기 시험 결과로, 저희 E2E 실험 결과와는 구분합니다.”

## 질의응답 시 주의

1. UCX도 GPU-direct를 지원한다. UCX=CPU DRAM 경유, libfabric=GPU 직접 전송으로 나누면 틀리다. 두 전송 경로 모두 GPU 메모리 직접 읽기를 수행한 비교다.
2. libfabric이면 무조건 빠른 것이 아니다. 위 구현·버전·구성·워크로드에서 관측한 결과다. 원문은 UCX 경로의 낮은 대역폭에 대한 구체적 내부 원인을 미규명으로 남겼다. 불필요한 복사나 특정 UCX 내부 동작을 원인으로 단정하지 않는다.
3. 처음부터 UCX를 사용하던 원래 discos를 우리가 libfabric으로 교체했다는 뜻이 아니다. UCX→verbs는 글루시스 선행 검증의 이력이다. 원래 discos도 OFI 관련 라이브러리 경로를 사용했다.
4. 원문 §8.2에는 UCX 첫 연결 시 약 15초 스톨과 연결 사전 준비로 해결한 기록이 있다. 이를 UCX의 보편적 결함이나 이번 모든 성능 차이의 원인으로 설명하지 않는다. libfabric 연결에서 영향이 없다는 내부 설명도 원문에서는 추정으로 한정되어 있다.
5. 원문 §9에는 RP_2/RP_3 복제 구성의 64KiB 이상 GPU 소스 쓰기가 verbs에서 실패하고 UCX에서 정상인 제한도 있다. 따라서 모든 구성에서 verbs가 더 빠르고 더 안정적이라고 일반화하면 안 된다. 검증된 비복제 KV 조건의 선택이다.
6. 과거 libfabric의 데이터 손상으로 지목된 문제는 원문 §8.1에서 스토리지 서버들의 공유 NVMe 오구성으로 정정되었다.
7. GPU memory 지원 설정은 호스트 메모리 사용을 모두 없앤다는 뜻이 아니다. 메타데이터와 제어에는 호스트 메모리를 사용할 수 있다.

## 근거

- [사용자가 제공한 글루시스 문서: 전송 변경과 필수 수정](/root/.codex/attachments/c4072db7-0d74-4d30-bc7b-04c9c71cae8f/pasted-text.txt:318), §4.
- [같은 조건 대역폭 비교 및 원인 미규명](/root/.codex/attachments/c4072db7-0d74-4d30-bc7b-04c9c71cae8f/pasted-text.txt:447), §6.1.
- [스토리지 오구성 정정 및 UCX 연결 스톨](/root/.codex/attachments/c4072db7-0d74-4d30-bc7b-04c9c71cae8f/pasted-text.txt:633), §8.
- [복제 GPU 쓰기 제한](/root/.codex/attachments/c4072db7-0d74-4d30-bc7b-04c9c71cae8f/pasted-text.txt:702), §9.
- [현재 두 모드의 공통 환경 설정](/root/discos_minji/run_vllm.sh:5).
- [UCX 공식 FAQ: GPU 메모리 zero-copy RDMA 지원](https://openucx.readthedocs.io/en/master/faq.html#does-ucx-support-zero-copy-for-gpu-memory-over-rdma).

그림은 내장 image_gen으로 생성했다. [정확한 생성 프롬프트](why_libfabric_over_ucx_20260918_v1.prompt.md).
