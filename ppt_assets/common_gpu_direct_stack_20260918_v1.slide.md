# 통신 스택 통일

## 슬라이드 제목

DFS / Object 비교를 위한 공통 GPU-direct 환경

## 핵심 메시지

저장 API는 다르게, 통신 환경은 동일하게.

## 배치

흰 배경의 16:9 슬라이드. 그림을 본문 중앙에 크게 배치한다. 그림 자체에 제목과 요약이 포함돼 있으므로 전체 슬라이드로 사용할 수도 있다. 발표 메모의 상세 설명을 모두 슬라이드에 옮기지는 않는다.

이미지: [common_gpu_direct_stack_20260918_v1.png](common_gpu_direct_stack_20260918_v1.png)

## 슬라이드에 넣을 세 문장

- 문제: 저장 API뿐 아니라 DAOS·통신 라이브러리 빌드까지 다르면 성능 차이의 원인을 구분하기 어렵다.
- 조치: DFS와 직접 Object 경로를 동일한 GPU-direct DAOS 및 CUDA 지원 libfabric 설치본으로 통일했다.
- 확인: 실행 프로세스에 로딩된 라이브러리 경로와 통신 provider가 두 모드에서 동일한지 검증했다.

## 발표 메모

이 그림은 위쪽의 저장 API와 아래쪽의 통신 환경을 구분해서 보여줍니다. DFS 모드는 파일 API로, Object 모드는 오브젝트 API로 접근합니다. 하지만 두 방식 모두 같은 GPU-direct DAOS 설치본과 CUDA 지원 libfabric을 사용합니다. Object용 C 연결 라이브러리도 같은 DAOS 설치본에 맞춰 재빌드했습니다. 따라서 서로 다른 통신 라이브러리 빌드가 비교 결과에 섞이는 요인을 줄였습니다. 다만 파일·키·메타데이터와 데이터 배치 차이는 남아 있으므로, 두 저장 구현 전체를 비교하는 실험입니다.

## 근거

- 공통 실행 환경: ../run_vllm.sh
- C shim 공통 링크 대상: ../Makefile
- DFS 실제 로딩 경로: ../full_agentic_restart10_20260917_v1/dfs_run1_native_maps.json
- Object 실제 로딩 경로: ../full_agentic_restart10_20260917_v1/object_run1_native_maps.json
- 공통 DAOS/Mercury: /opt/daos-gds-gpu
- 공통 libfabric: /opt/ofi-cuda/lib64
- libfabric provider: verbs;ofi_rxm
- 당시 DAOS agent의 전체 통신 설정 문자열: ofi+verbs;ofi_rxm

## 주의할 표현

이번 슬라이드는 UCX 대비 libfabric 성능 우위를 측정했다는 뜻이 아니다. 기존 discos 런처에도 OFI 경로가 있었다. 동일한 스택은 동일 설치본·설정을 뜻하며, 두 모드는 각각 별도 프로세스로 실행했다. 파일 시스템 경로와 직접 오브젝트 경로의 저장 형식이 같다는 뜻도 아니다.
