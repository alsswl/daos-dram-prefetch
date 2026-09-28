# 선택형 DRAM 캐시 + DAOS 저장

2026-09-26 추가: 아래 기존 CPU 경유 저장과 달리, GPU-direct 주 경로를 유지하고
별도 비동기 복사로 DRAM을 채우는 [새 구현 안내](ASYNC_DRAM_STORE_GUIDE_KO.md)가 있다.

추가 기능: DRAM hit를 GPU staging으로 미리 복사하는 선택형 기능은 [GPU 프리페치 안내](DRAM_GPU_PREFETCH_GUIDE_KO.md)를 참고한다. 기본 OFF이며, 아래 기존 DRAM 보관 기능과 별도로 켜고 끈다.

2026-09-22. 변경 범위는 `/root/discos_minji` 안의 새 실행기·검증 스크립트·문서뿐이다. 기존 `run_vllm.sh`, YAML, GDS 백엔드, C shim, 설치된 LMCache/vLLM, `/root/discos`는 수정하지 않는다.

## 구현한 동작

`run_dram_cache.py`가 기본 YAML을 읽어 별도 실행 디렉터리에 유효 설정을 만든다. 기본은 `--dram off`다. `--dram on`은 LMCache의 기존 `LocalCPUBackend` 캐시 보관 기능을 켠다. CPU 캐시 용량은 `--cpu-gb`로 명시할 수 있으며 GiB 단위다. 생략하면 기본 YAML/LMCache 기본값을 유지한다.

일반 저장 경로는 이미 `vLLM GPU KV → CPU 임시 MemoryObj → GPU staging → DAOS`이다. ON은 이 CPU MemoryObj의 참조를 캐시에 남긴다. **CPU 복사본을 만들기 위해 새 GPU→CPU 복사를 추가하는 방식이 아니다.** CPU 캐시 등록은 짧은 동기 참조 등록이며, 기존 백엔드의 DAOS 쓰기는 작업 스레드에서 비동기로 계속 수행한다. GPU에서 DAOS와 CPU로 독립된 두 DMA를 동시에 발행하는 새 구현은 아니다.

재사용할 때는 CPU 캐시의 연속 prefix를 먼저 확인하고, 없는 suffix는 기존 DAOS GPU-direct 경로에서 읽는다. 실제 저장 형식과 namespace는 바꾸지 않으며 DFS/object 양쪽에 같은 설정 방식을 적용한다. CPU 캐시 보관 정책은 기본 설정의 LRU를 사용한다.

주의할 범위:

- 이번 ON 경로는 비동기 로딩·비레이어별 모드만 허용한다. 다른 경로는 검증되지 않아 실행기가 거절한다.
- DAOS 저장이 꺼져 있거나 특정 한 계층에만 저장하도록 설정된 ON 구성은 거절한다.
- CPU hit가 있다고 DAOS 비동기 저장 완료까지 보장되는 것은 아니다. 이번 검증에서는 프로세스 재시작 후 읽기로 DAOS 복사본을 별도 확인한다.
- **기존 DAOS 데이터를 읽을 때 DRAM으로 자동 승격하는 기능은 추가하지 않았다.** 설치된 async prefetch 완료 callback에는 CPU write-back이 구현돼 있지 않다. ON의 DRAM은 해당 프로세스에서 새로 저장하는 KV로 채워진다.
- 프로세스가 재시작하면 CPU 캐시는 사라진다. 다른 vLLM 프로세스와 공유하는 MP 캐시도 아니다.
- CPU 전체 용량을 넘는 eviction, 취소·실패·고부하 전체 조합은 별도 검증 대상이다.

## 실행과 롤백

기존 실행 명령을 `--` 뒤에 넣는다. 예시는 object 모드 API 서버다. 기존 vLLM 서버가 있으면 해당 서버를 정상 종료한 뒤 새로 시작해야 한다.

```bash
cd /root/discos_minji
env DAOSGDS_TRANSPORT=object ./venv/bin/python3 run_dram_cache.py \
  --dram on --cpu-gb 4 \
  --config /root/discos_minji/lmcache_config_daosgds_unified.yaml \
  -- ./venv/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-14B --host 127.0.0.1 --port 8017 \
  --max-model-len 8192 --gpu-memory-utilization 0.75 --enforce-eager \
  --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

실행기가 `dram_profile_*` 새 폴더에 `lmcache_effective.yaml`, `launch.json`을 저장하고 원래 `run_vllm.sh`로 실행한다. `--state-dir`로 새 폴더를 지정할 수도 있다. 기존 폴더를 덮어쓰지 않는다. `--dry-run`은 설정만 생성한다.

**같은 자원 조건에서 OFF로 복귀:** 프로세스를 종료하고 같은 명령의 `--dram on`만 `--dram off`로 바꿔 재시작한다. `--cpu-gb 4` 등 다른 조건은 유지한다. 단순히 실행 중인 프로세스의 환경변수만 바꾸면 안 된다.

**기능을 완전히 우회:** 새 실행기를 쓰지 않고 원래 `run_vllm.sh` 명령으로 실행한다. 파일 복원·재빌드·DAOS 데이터 삭제가 필요 없다. `LMCACHE_LOCAL_CPU` 같은 환경변수를 셸에서 별도로 설정했다면 함께 확인한다. 새 실행기는 자식 프로세스의 환경만 변경하므로 부모 셸을 수정하지 않는다.

**파일 롤백이 필요할 때:** 기준 실행 폴더 `dram_baseline_20260922_v1/executed_sources/`에 기존 코드·설정·C 라이브러리를 복사하고 SHA-256을 남겼다. 이번 작업은 해당 기존 파일을 수정하지 않았으므로 원칙적으로 되덮어쓸 필요가 없다. 이후 사용자 수정이 생겼다면 이 스냅샷으로 무조건 덮어쓰면 안 된다.

## 비교 실험 방법

`dram_cache_bench.py`는 별도 namespace와 설정을 만들며 공용 컨테이너나 이전 캐시를 지우지 않는다. 결과 폴더는 없는 이름을 지정한다.

```bash
cd /root/discos_minji
# 기존 실행 경로 기준값
./venv/bin/python3 dram_cache_bench.py \
  --output /root/discos_minji/dram_baseline_next --skip-restart

# 새 실행기 OFF
./venv/bin/python3 dram_cache_bench.py --profile --dram off \
  --output /root/discos_minji/dram_off_next

# 새 실행기 ON
./venv/bin/python3 dram_cache_bench.py --profile --dram on \
  --output /root/discos_minji/dram_on_next
```

DFS는 `--transport dfs`를 추가한다. `--dry-run`은 설정·소스 스냅샷만 만들고 모델/DAOS I/O는 실행하지 않는다.

조건은 Qwen3-14B, BF16, 청크 128, CPU pool 4GiB, GPU staging 10GiB, I/O/meta workers 각각 16이다. 서로 다른 입력 4개 × 4,096토큰, 요청당 64토큰을 생성한다. vLLM prefix caching은 끄고, 별도 warmup 입력을 사용한다. 실제 생성은 하지만 full agentic DiscoveryBench는 아니다.

각 서버에서 첫 저장 후 같은 입력을 동시성 1과 4로 각각 3회 재사용한다. ON/OFF 모두 CPU allocator 용량이 같고, 보관 여부만 다르다. 5개 입력(warmup 포함)의 기본 KV payload 3.125GiB가 CPU pool 4GiB에 들어가므로 이번 조건은 **DRAM에 재사용 대상이 모두 들어가는 유리한 조건**이다.

HTTP `cached_tokens`만으로 DRAM hit와 DAOS hit를 구분할 수 없다. 스크립트는 해당 구간의 DAOS prefetch 호출 로그를 남기고, ON 재사용 때 DAOS prefetch 0회, OFF 때 요청당 1회인지 검사한다. vLLM prefix caching을 끄고 CPU/DAOS 두 계층만 사용하는 설정과 함께 읽기 출처를 확인한다. 이 호출 수는 청크별 DAOS API/RPC 개수가 아니라 **요청 단위 prefetch batch 수**다. CPU hot-cache metric도 수집을 시도하지만 현재 vLLM `/metrics`에는 해당 LMCache worker 항목이 노출되지 않는다. 미노출은 0개가 아니라 관측 불가다. 초반 실행의 raw JSON에 있는 `cpu_hot_chunks: 0`은 실제 측정값이 아니므로 분석에서 제외한다. 이후 스크립트는 미노출을 `null`로 기록하도록 수정했다.

재시작 검증은 저장하지 않는 `kv_consumer`로 실행한다. CPU 캐시가 비어 있는 새 프로세스에서 같은 입력이 DAOS hit로 읽히는지 검사한다. 재시작 검증 시간에는 새 프로세스 초기 실행 효과가 포함되므로 warm ON/OFF 성능 표와 섞지 않는다.

TTFT는 HTTP 요청 시작부터 첫 비어 있지 않은 텍스트까지, E2E는 해당 요청의 64토큰 응답 종료까지다. fill 응답 시간은 비동기 DAOS 저장 완료 시간과 다르다. 서버 시작/모델 로딩은 요청 시간에 포함하지 않는다. 기본 INFO 로깅을 사용하고 staging 상세 계측은 끈다.

실험 결과·한계는 [2026-09-22 검증 결과](DRAM_CACHE_RESULT_20260922_KO.md)에 정리했다. object의 모델 추론 비교와 object/DFS의 실제 DRAM·DAOS 바이트 일치 검증을 완료했다.
