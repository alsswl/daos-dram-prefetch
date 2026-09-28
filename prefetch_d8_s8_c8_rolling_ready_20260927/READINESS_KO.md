# 동시 요청 유지 방식 준비 — 성능 측정 미실행

DRAM 8GiB, GPU staging 8GiB, 동시 요청 최대 8개. 기존 256개 입력을 사용하며
하나의 응답이 끝나면 다음 입력을 투입한다. OFF/ON은 DRAM 프리페치만 바꾼다.
각 실행은 새 프로세스·빈 DRAM·새 DAOS namespace로 시작한다.

이 디렉터리는 `--arrival-mode rolling --dry-run`으로 생성한 계획/소스 사본이다.
실제 모델 서버를 시작하거나 성능을 측정한 결과가 아니다.

## 실험을 시작하지 못한 이유

2026-09-27 실제 GPU 데이터 20MiB 사전 저장에서 다음 오류가 발생했다.

```text
rank 1 tag 6: DER_NOSPACE(-1007): 'No space on storage target'
daos_obj_update_gpu rc=-1007
```

사용한 진단 명령:

```bash
env DAOSGDS_TRANSPORT=object ./run_vllm.sh ./venv/bin/python3 \
  tests/object_gpu_roundtrip.py --size-mib 20
```

테스트 스크립트의 finally에서 이번 UUID 키만 정리했다:
`minji-v2:gpu-roundtrip:d3eaf42d0fdb41ae9bb4f134cdae36f9`.
기존 실험 namespace, 공유 컨테이너, 다른 사용자 데이터는 삭제하지 않았다.

풀 조회에는 NVMe 총 2.0TB, free132GB가 표시되지만 실제 저장 호출은 공간 부족으로
거부됐다. 표시 free만으로 저장 가능하다고 판단하지 않는다. 하위 저장장치·예약 공간 등
세부 원인은 미확인이다. 공유 데이터 정리나 서버 조정은 별도 사용자 지시가 필요하다.

## 공간 확보 후 실행

```bash
cd /root/discos_minji
./venv/bin/python3 repeat_prefetch_c8.py --arrival-mode rolling \
  --output /root/discos_minji/prefetch_d8_s8_c8_ROLLING_NEW
./venv/bin/python3 report_prefetch_c8.py \
  /root/discos_minji/prefetch_d8_s8_c8_ROLLING_NEW
```

출력 디렉터리는 새 이름을 사용한다. 실행기는 사전 저장 검증에 실패하면 서버를 시작하지
않고, 실험 중 저장 오류가 나도 새 요청 투입을 중단한다. 이전 묶음 실험 결과는 보존했다.
