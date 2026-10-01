# 512GiB pinned memory 준비 및 변경 기록

Qwen3-14B BF16, DRAM512GiB, staging8GiB, 동시요청16, 청크128토큰(20MiB),
OFF/ON 각각 cold1 + warm4 조건을 유지했다. 캐시 유지 반복 사이 삭제하지 않는다.

## 기존 실패

`../sharegpt_cold_warm4_d512_s8_20260929/c16_off/server.log`에
`cudaHostAlloc failed: 2`가 기록됐다. LMCache CPU allocator 초기화 실패로 본 요청은0개다.
상위 스크립트의 `Unexpected loaded stack libdaos.so: []`는 LMCache 초기화가 실패해
DAOS 백엔드 로딩까지 도달하지 못한 결과였다.

일반 메모리 여유만으로 큰 CUDA pinned allocation 성공을 보장할 수 없다.
호스트는 NUMA 노드0/1 각각 약503GiB다. 프로세스에 interleave를 적용해도 단일512GiB
cudaHostAlloc 검사 중 드라이버 `failed to allocate page table`가 재현돼 해당 검사 프로세스를 종료했다.
따라서 NUMA 배치만이 원인이라고 결론 내리지 않는다. OS/드라이버 설정은 변경하지 않았다.

## 적용 방식

- 익명 mmap으로 연속 가상주소의512GiB CPU 버퍼를 확보한다.
- 7.8125GiB(20MiB KV청크400개)씩 CUDA에 등록한다. 마지막은 남은 크기로 등록한다.
- 별도 등록 구간을 가로지르는 CUDA memcpy는 invalid argument가 확인됐다.
  따라서 청크 경계에 등록 구간을 맞추고, allocator가 반환하는 모든 대상청크가
  20MiB 정렬이며 단일 등록 구간 안에 있는지 검사한다. 다른 모델/청크 크기에 일반화하지 않는다.
- LMCache에는 하나의512GiB 텐서와 기존 allocator/LRU를 제공한다. pageable fallback은 없다.
- OFF/ON 모두 프로세스 수준 NUMA interleave(0,1)와 같은 등록 방식을 사용한다.
- 로컬 vLLM general plugin이며 `DAOS_SEGMENTED_PINNED=1`과 명시적 PYTHONPATH에서만 활성화한다.
  설치된 LMCache/vLLM 파일과 `/root/discos`는 수정하지 않았다.

## 검증

2026-09-29 도구 실행 출력 기준:

- 512GiB 전체 등록74.7357초, `is_pinned=True`.
- NUMA 페이지 수 N0=67,108,864 / N1=67,108,864, 각4KiB.
- 서로 떨어진16개 위치에서1MiB GPU 왕복 바이트 비교 통과.
- 65개 등록 경계의 양쪽130개20MiB 청크 GPU 왕복 바이트 비교 통과.
- 검사 후512GiB 고정 메모리 등록 해제, 검사 프로세스 정상 종료.
- 단위 테스트222개 통과; vLLM plugin 발견·설치 확인.

검증 원본 출력은 도구 실행에 있고 별도 전체 stdout 파일은 저장하지 않았다.
재현 스크립트: `tests/segmented_pinned_probe.py`, 실행 환경은 `with_numa_interleave.py` 사용.

## 비교 제한

512GiB OFF/ON은 같은 새 할당 방식이지만, 이전256GiB 기본 cudaHostAlloc 결과와 비교하면
용량뿐 아니라 NUMA 배치와 메모리 등록 방식도 다르다. 용량만의 효과라고 단정하지 않는다.
warm4회는 같은 프로세스와 캐시를 이어 쓰므로 독립 반복이 아니다.
staging은 단계 사이 drain하며 DRAM/DAOS는 유지한다.
