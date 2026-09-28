# DRAM 프리페치 지연 진단 방법

대상은 `/root/discos_minji`의 실험용 `CapacityProbeBackend`다. 원본 `/root/discos`나 설치된 LMCache/vLLM 파일은 변경하지 않는다.

## 무엇을 재는가

`DAOS_GDS_PREFETCH_TIMING=1`일 때만 기존 CPU 프리페치 executor에 계측을 붙인다.

- `queued_ns → worker_start_ns`: executor 제출/dispatch/대기 시간.
- `reserve_start_ns → reserve_end_ns`: 기존 shared staging 할당 시도.
- `copy_start_ns → copy_end_ns`: 기존 H2D 함수의 호스트 경과 시간. 복사 명령 제출과 기존 stream 동기화가 포함된다. 순수 DMA 시간은 아니다.
- 기존 `cpu_get_ready`, `retrieve_start`, `retrieve_return`을 HTTP 요청 ID에 연결해 준비 이후 보유 시간과 retrieve API 시간을 비교한다.

정책은 변경하지 않는다. 작업 스레드 1개, 복사 stream 1개, 실제 용량 한계에서의 fallback, 취소 시 안전한 참조 정리를 유지한다. 추가 CUDA 동기화나 복사를 수행하지 않는다. 완료 시 JSON 이벤트 1개를 추가하므로 계측 오버헤드는 존재한다.

## 실행

```bash
cd /root/discos_minji
./venv/bin/python3 -u diagnose_prefetch_timing.py --output /root/discos_minji/새_결과_디렉터리
./venv/bin/python3 report_prefetch_timing.py /root/discos_minji/새_결과_디렉터리
```

러너는 Qwen3-14B, DRAM/staging 각 8GiB, 청크 128, 동시 요청 8/16, 프리페치 OFF/ON을 조건별 2회 수행한다. 입력 256개는 기존 DiscoveryBench 기록을 재생한다. 새 입력은 요청 하나가 끝날 때마다 투입하며 Python 도구는 재실행하지 않는다. 실행 순서는 두 번째 반복에서 반대로 바꾼다.

각 실행은 새 프로세스와 새 DAOS namespace로 시작한다. 모델 로딩 시간은 제외하고 첫 요청부터 마지막 요청 완료까지 잰다. 모델의 EOS를 허용하므로 생성량, 정확한 도착 시간과 실행 중 캐시 상태는 달라질 수 있다. DAOS 자체 서버 캐시는 지우지 않는다.

실행 전 UUID 테스트 키 하나로 20MiB GPU 왕복을 검증하고 해당 테스트 키만 정리한다. 실험 namespace는 보존하고 공용 풀/컨테이너/오브젝트를 삭제하지 않는다.

`--dry-run`은 계획과 소스 사본만 남긴다. 분석기의 `--partial`은 완료된 실행만 임시 보고하며 최종 결과와 구분한다.

## 중단/원복

계측은 실험 러너의 자식 프로세스에만 환경변수로 적용된다. 일반 실행에 이 변수를 지정하지 않으면 동작하지 않으므로 성능 정책을 되돌릴 필요가 없다. 용량 probe에 추가한 분기도 계측 모듈을 불러오지 않는다.

## 해석 주의

복사 대기가 늘었다는 것만으로 TTFT 차이를 전부 설명할 수 없다. ON/OFF의 실제 캐시 hit와 미재사용 입력량도 확인한다. `ready → retrieve`가 양수여도 동일 시간만큼 계산과 겹쳤다고 단정할 수 없다. 프리페치 준비 완료가 요청의 스케줄 가능 시점 자체를 바꿀 수 있기 때문이다.

서버 `request_queue_time`은 executor 큐가 아니라 vLLM 스케줄 대기 지표다. `prefill_time` 역시 스케줄부터 첫 토큰까지의 구간이며 순수 GPU 연산 시간으로 해석하지 않는다. 첫 16개/나머지 240개 분리는 보조 분석일 뿐 엄밀히 통제된 cold/warm 그룹 비교가 아니다.
