# 새 KV의 DRAM 우회 저장

실제 3회 비교 결과는 [검증·성능 결과](gpu_store_ab_20260926_v3/RESULT_KO.md)에 정리했다.

## 바뀌는 경로

```text
host_staged (기존 기본값)
vLLM GPU paged KV → CPU 임시 KV → GPU staging → DAOS

gpu_direct (추가 옵션)
vLLM GPU paged KV → GPU staging → DAOS
```

GPU connector의 KV gather 커널이 데이터를 모으는 목적지를 CPU 객체에서
GPU staging 객체로 변경한다. GPU 객체를 그대로 DAOS 백엔드에 전달한다.
파일/오브젝트 저장 형식과 읽기 경로는 바뀌지 않는다.

`local_cpu: false`만으로는 기존 CPU 임시 버퍼를 없애지 못한다.
새 옵션은 엔진의 할당 대상과 StorageManager의 저장 라우팅을 함께 바꾼다.
프로세스 안에서만 적용하는 패치이며 설치된 LMCache 파일을 수정하지 않는다.
옵션을 켠 DAOS backend가 있는 manager에만 적용한다.

## 사용과 롤백

기존 YAML의 `extra_config` 아래에 다음 옵션을 추가하고 프로세스를 재시작한다.

```yaml
local_cpu: false
use_layerwise: false
extra_config:
  daosgds.store_path: gpu_direct
```

위는 기존 설정에 **추가할 항목**이다. 풀/컨테이너와 플러그인 설정도 필요하다.
완성 예시는 `lmcache_config_daosgds_gpu_store.yaml`을 사용한다.

```bash
cd /root/discos_minji
env DAOSGDS_TRANSPORT=object \
  LMCACHE_CONFIG_FILE=/root/discos_minji/lmcache_config_daosgds_gpu_store.yaml \
  ./run_vllm.sh ./venv/bin/python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-14B --max-model-len 16384 \
  --gpu-memory-utilization 0.75 --enforce-eager --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

롤백은 `daosgds.store_path: host_staged`로 변경하거나 해당 항목을 삭제한 뒤
프로세스를 재시작한다. 기존 `lmcache_config_daosgds_unified.yaml`의 기본 경로는
변경하지 않았다. `/root/discos`와 설치된 패키지도 수정하지 않는다.

## 범위와 주의점

- KV payload가 CPU 임시 메모리를 경유하지 않는다는 의미다. 키, JSON 메타데이터,
  포인터 목록, 제어 정보는 계속 CPU를 사용한다. 시스템 전체 DRAM 사용량이 0이 되지는 않는다.
- 기본 GPU-only 플러그인은 DRAM 캐시 보관과 동시에 켤 수 없다. `local_cpu: true`, 다른 저장 계층,
  layerwise/PD/blending 조합은 명시적으로 거부한다. DRAM 비동기 보관은 아래 별도 플러그인 안내를 참고한다.
- 비활성 LocalCPUBackend의 호스트 allocator 예약 자체는 남아 있다. 비교 실험에서는
  양쪽 모두 8GiB로 동일하게 두고 실제 KV 목적지와 복사 호출을 관측한다.
- 기존 GPU staging 풀을 읽기와 쓰기가 공유한다. 풀이 부족해도 CPU로 fallback하지 않는다.
  할당 실패 시 기존 엔진의 저장 조기 종료 동작이 적용되어 일부 KV가 저장되지 않을 수 있다.
  `alloc_fail`과 실제 저장 토큰 수를 확인해야 한다. VRAM이 자동으로 늘어나지는 않는다.
- GPU gather가 끝나기 전에 버퍼가 DAOS에 전달되거나 반환되지 않도록 제출 전 CUDA 동기화를 한다.
  취소/중복/저장 비활성/제출 실패에서도 참조를 정리한다. 이는 안전 우선 구현이며
  스트림 단위 이벤트 최적화까지 완료한 것은 아니다.
- LMCache 내부 인터페이스에 의존한다. 현재 설치 버전과 Qwen3 V2 GPU connector를 대상으로 검증하며,
  MP, 다중 GPU, 다른 모델/connector, 장애 복구 전체를 검증한 것은 아니다.

## 검증과 비교 실행

```bash
cd /root/discos_minji
PYTHONPATH=/root/discos_minji ./venv/bin/python3 -m pytest -q tests
env DAOSGDS_TRANSPORT=object ./run_vllm.sh ./venv/bin/python3 tests/gpu_store_roundtrip.py
./venv/bin/python3 compare_gpu_store.py \
  --output /root/discos_minji/gpu_store_ab_NEW --repeats 3
```

실제 GPU 검증은 40층, 128토큰, BF16 KV 20MiB를 사용한다. CPU 목적지와 GPU 목적지의
gather 결과, DAOS 저장/읽기, paged KV로 scatter한 결과를 바이트 단위 비교한다.
검증용 UUID dkey는 성공·실패와 관계없이 해당 키만 정리한다.

비교는 Qwen3-14B, 8192토큰 입력 8개, 생성 64토큰, 청크 128, staging 10GiB,
object 모드, 양쪽 DRAM 보관 OFF다. 입력 4개를 순차 처리하고 별도 입력 4개를
동시에 처리한다. 각 cold 저장 후 동일 입력을 warm 읽기한다. warm은 skip-save로
읽기만 한다. 매 실험 프로세스마다 새 UUID namespace를 사용하고 경로 순서를 교대한다.
워밍업 입력은 측정 입력과 첫 청크가 다르다. 모델 로딩 시간은 요청 시간에서 제외한다.

`save_decode_cache: true`를 양쪽에 적용하여 chunked prefill의 경계에 걸린 마지막
프롬프트 청크도 이후 decode 단계에서 저장하게 한다. 출력 64토큰은 추가 128토큰
청크를 만들지 않는다. 실제 저장량과 cold/warm hit 수가 기대치와 다르면 실험을 중단한다.

HTTP 지연과 함께 저장의 host→GPU 복사량, GPU gather 목적지, staging 점유,
DAOS 완료 개수와 LMCache store 로그를 남긴다. LMCache `Stored ... cost`는 DAOS
비동기 작업 완료 시간이 아니므로 이를 네트워크 저장 완료 시간으로 해석하면 안 된다.
이번 비교는 쓰기 경로를 분리하는 고정 입력 실험이지 full agentic DiscoveryBench가 아니다.
실험 캐시는 새 namespace에 남는다. 공용 컨테이너는 삭제하지 않는다.

## 추가 옵션: DAOS 우선 비동기 DRAM 보관

기본 GPU-only 플러그인과 별개로, GPU staging→DAOS 경로를 유지하면서
DAOS 쓰기 성공 후 DRAM에 별도 비동기 복제하는 플러그인을 추가했다.
[비동기 DRAM 보관 안내](ASYNC_DRAM_STORE_GUIDE_KO.md)를 참고한다.
이 문서의 GPU-only 비교 결과와 새 옵션의 성능은 서로 구분해야 한다.
