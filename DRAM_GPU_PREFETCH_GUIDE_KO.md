# DRAM hit → GPU staging 프리페치

2026-09-22. 기존 DRAM 캐시 보관 기능에 **선택형 GPU 프리페치**를 추가했다. 기본은 OFF다. DRAM 보관·교체 정책이나 DAOS 저장 형식은 바꾸지 않았다.

## 무엇이 달라졌나

| DRAM 캐시 ON 조건 | GPU 프리페치 OFF | GPU 프리페치 ON |
|---|---|---|
| lookup 뒤 비동기 읽기 | CPU 캐시 객체의 참조를 반환 | CPU 캐시 데이터를 공통 GPU staging으로 복사한 뒤 반환 |
| retrieve | CPU의 KV를 vLLM GPU KV로 이동 | 이미 staging에 있는 KV를 vLLM GPU KV로 이동 |
| DRAM miss | 기존 DAOS GPU-direct 경로 | 동일 |
| staging 여유 부족 | 원래 CPU 경로 | 기다리지 않고 원래 CPU 경로로 복귀 |

즉, **DRAM→GPU 전송을 없앤 것이 아니라, 모델의 retrieve보다 앞선 비동기 구간으로 옮겼다.** DAOS에만 있는 데이터를 DRAM에 자동으로 채우는 기능은 아니다. 새로 저장되는 KV의 CPU 복사본을 보관하는 기존 기능과 조합한다.

새 `DaosDramPrefetchBackend`는 기존 `DaosGdsBackend`를 상속한다. 이 실행에 사용되는 LocalCPUBackend 인스턴스의 비동기 get·close 참조만 교체한다. 설치된 LMCache/vLLM 파일이나 전역 CPU 클래스는 수정하지 않는다. OFF는 원래 `DaosGdsBackend`를 선택한다.

## 복사와 메모리 수명

- 전용 작업 스레드 1개와 CUDA stream 1개가 CPU→GPU 복사를 수행한다. 복사 완료 대기는 작업 스레드에서 하며, 모델 스레드나 asyncio 이벤트 루프에서 하지 않는다. 여러 H2D batch를 동시에 복사하는 구현은 아니다.
- 복사가 끝나야 GPU 객체를 반환한다. retrieve 또는 요청 정리 시점까지 GPU 버퍼와 원래 CPU 객체의 참조를 유지한다.
- 복사 도중 요청이 취소되면 전송 완료 후 버퍼를 반환한다. 전송 중인 메모리를 다른 요청이 재사용하지 않도록 한다.
- GPU staging은 DAOS와 **같은 allocator/pool**을 사용한다. 새 GPU pool을 별도로 만들지 않는다.
- 기본 허용 기준은 전체 GPU staging 용량의 절반이다. `현재 공유 pool 할당량 + 이번 CPU hit batch 필요량`이 이 기준 이하여야 미리 복사한다. 이미 읽기가 끝나 retrieve를 기다리는 버퍼도 실제 해제 전까지 포함한다.
- 이는 CPU 프리페치의 **입장 기준**이지, 전체 pool 사용량을 항상 절반 이하로 제한하는 설정은 아니다. DAOS 작업은 그 이후에도 남은 공간을 사용할 수 있다.
- batch 전체 공간을 확보하지 못하면 부분 복사를 하지 않고 CPU 객체를 그대로 반환한다. 캐시 miss로 처리하거나 DAOS를 다시 읽지 않는다.

## 실행·롤백

기존 `run_dram_cache.py --dram on ... -- 실행명령`에 다음 옵션만 추가한다. 모드 변경에는 프로세스 재시작이 필요하며 C 재빌드는 필요 없다.

```bash
# DRAM 보관 + GPU 프리페치
--dram on --gpu-prefetch on

# DRAM 보관은 유지하고 새 프리페치만 롤백
--dram on --gpu-prefetch off

# DRAM 보관까지 끄기
--dram off --gpu-prefetch off
```

`--prefetch-gb 5`처럼 허용 기준을 GiB로 지정할 수 있다. 지정값은 GPU staging 전체 용량 이하여야 한다. 생략 시 전체 용량의 절반이다. CLI의 기본 GPU 프리페치는 OFF이므로 기존 실행 명령의 동작은 유지된다. ON은 DRAM ON·async loading·비레이어별 경로만 지원한다.

기존 기본 YAML, `gds_backend.py`, `run_vllm.sh`, C shim, `/root/discos`는 이번 작업에서 수정하지 않았다. 실행기·벤치·기존 테스트 파일의 변경 전 사본은 `dram_prefetch_backup_20260922.nUF03m/`에 있다. 이후 사용자 수정이 생겼다면 사본으로 무조건 덮어쓰지 않는다.

## 비교 실험 재실행

다음은 Qwen3-14B, DRAM 4GiB, staging 10GiB, 허용 기준 5GiB, 청크 128의 비교다. 각 결과 폴더는 새 이름을 사용해야 한다.

```bash
cd /root/discos_minji
./venv/bin/python3 dram_cache_bench.py --profile --dram on \
  --gpu-prefetch off --output /root/discos_minji/cpu_prefetch_off_next

./venv/bin/python3 dram_cache_bench.py --profile --dram on \
  --gpu-prefetch on --output /root/discos_minji/cpu_prefetch_on_next
```

기본 transport는 이 벤치에서 object다. DFS는 `--transport dfs`를 추가한다. 실제 성능 비교를 완료한 경로는 object이며, DFS는 별도 바이트 일치 기능 검증까지 수행했다. 실행 시 공용 컨테이너를 지우지 않고 새 namespace를 사용한다. 벤치 캐시는 재검증용으로 남는다.

이 워크로드는 서로 다른 4096토큰 입력 4개에 대해 64토큰씩 생성하고, 동시성 1과 4에서 재사용한다. **full agentic DiscoveryBench가 아니다.** 재사용 KV가 DRAM에 모두 들어가므로 캐시 보관 정책의 우열을 검증하지 않는다.

로그의 `CPU staging prefetch`는 요청 batch의 복사 완료, `CPU staging fallback`은 기존 CPU 경로 복귀다. 개별 DAOS API/RPC 호출 수와 다르다. 복사 로그 시간은 작업 스레드 내 할당·복사·완료 대기이며 작업 큐에서 기다린 시간은 포함하지 않는다. 전체 지연은 HTTP TTFT/E2E로 확인한다.

현재 결과는 [검증·성능 결과](DRAM_GPU_PREFETCH_RESULT_20260922_KO.md)에 정리한다. 장시간 부하, 실제 GPU 용량 포화, 혼합 CPU/DAOS hit의 경합, 모든 취소·종료 조합은 추가 검증 대상이다.
