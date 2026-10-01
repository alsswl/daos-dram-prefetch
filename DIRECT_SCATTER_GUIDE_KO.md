# DAOS → vLLM KV page 직접 배치 실험

`DirectScatterBackend`는 GPU payload staging pool을 생성하지 않고,
최종 vLLM KV page 주소를 SGL로 전달한다. retrieve의 GPU scatter kernel과
store의 GPU gather kernel도 호출하지 않는다. CPU에는 key, 메타데이터,
slot mapping, IOV/extent descriptor만 둔다. 서버 내부 buffer 제거를 뜻하지 않는다.

## 지원 범위와 비교 시 주의할 차이

- 단일 GPU/rank, FP16 또는 BF16, 비-MLA, non-layerwise V2 connector.
- 연속적인 `NL × [NB, 2, BS, NH, HS]` 또는 `NL × [2, NB, BS, NH, HS]` layout.
- payload prefetch, DRAM cache/promotion, 추가 storage tier는 비활성화한다.
- **저장과 복원 모두 동기식이다.** 특히 store는 NIC가 최종 KV page를 읽는 동안
  vLLM이 page를 재사용하지 못하도록 호출 안에서 완료를 기다린다. 기존의 비동기
  staging store와 처리량을 단순 비교하면 이 차이가 섞인다.
- batch의 모든 I/O가 끝나기 전에는 반환하지 않는다. 이전 GPU 작업 완료 후
  I/O를 시작하고, 완료 뒤 CPU에서 CUDA synchronize를 호출한다. compute/I/O overlap은 없다.
- 모델 지원 범위를 벗어나면 오류로 거부한다. staging fallback은 없다.
- 현재 shim은 19KiB 미만 payload I/O를 거부한다. 이 DAOS build의 작은 inline
  I/O 경로는 GPU pointer에 안전하지 않기 때문이다. Qwen3-4B의 한 token 전체
  layer KV는 144KiB여서 부분 prefix의 마지막 한 token도 이 제한을 넘는다.

## 데이터 형식

기존 baseline의 `libdaosgdr.so` 및 `kv` SINGLE akey를 변경하지 않는다.
새 공유 라이브러리 `libdaosgdr_scatter.so`는 다음 형식을 사용한다.

```text
같은 OC_SX object
  dkey = 설정 namespace + "scatter-v1:" + CacheEngineKey.to_string()
    meta          = schema / layers / tokens / hidden / dtype / length
    kv_scatter_v1 = byte ARRAY, 논리적 순서 [K/V, layer, token, hidden]
```

저장은 page별 source IOV를 전체 array에 대응시키고 metadata와 같은 update에
기록한다. 읽기는 cached prefix를 제외한 byte extents만 선택해 destination IOV로
받는다. DAOS I/O map으로 요청 구간의 실제 coverage도 확인해 array hole을 hit로
인정하지 않는다. 입력 slot 중복, prefix alias, layer allocation 겹침도 거부한다.

기존 namespace를 설정해도 자동 `scatter-v1:` 접미사가 붙어서 baseline payload와
섞이지 않는다. 따라서 **새 형식으로 warm-up/store를 먼저 해야 한다.**

## 빌드와 하드웨어 gate

이 checkout의 기존 GDR 실행에서 사용한 DAOS/OFI/CUDA 환경을 동일하게 적용한다.
새로운 DAOS 서버나 pool을 만드는 절차는 포함하지 않는다.

```bash
cd /root/discos_minji
make libdaosgdr_scatter.so DAOS_PREFIX=/opt/daos-gds-gpu
venv/bin/python -m pytest -q tests/test_scatter_plan.py tests/test_scatter_binding.py tests/test_direct_scatter_backend.py
venv/bin/python tests/direct_scatter_roundtrip.py --pool discospool --container kvcache
```

hardware gate는 36 layers, 128 tokens, 576 IOV의 전체 fetch와 prefix 5/127-token
skip을 검증한다. 다른 physical blocks로 복원하고 모든 untouched 영역까지 비교한다.
고유 테스트 dkey만 생성하며 종료 시 그 key들만 제거한다. 기준값 생성과 CPU 복사는
검증 코드에만 있으며 실제 I/O에는 없다. 이것이 통과하기 전 서빙 성능을 결론내리지 않는다.

## vLLM 시작

설정 예시는 `lmcache_config_daosgds_direct_scatter.yaml`이다. 실험마다
`object_namespace`를 새 값으로 복사해 사용한다. 별도 launcher는 부모와 spawn된
worker 모두에서 connector의 intermediate GPU buffer 및 LMCache CPU payload
allocator 생성을 차단한다. `gpu_buffer_gb: 0`만 바꾸는 것으로는 충분하지 않다.

```bash
LMCACHE_CONFIG_FILE=/root/discos_minji/lmcache_config_daosgds_direct_scatter.yaml \
venv/bin/python -m lmcache_daos.direct_scatter_launch serve MODEL_PATH \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

MODEL_PATH와 기타 모델·scheduler 옵션은 기존 실험과 맞춘다. 이 명령에는 GPU/DAOS
환경 변수를 다시 지정하지 않았다. 기존 GDR client 환경을 먼저 적용해야 한다.
로그에서 `DirectScatterBackend enabled: GPU staging=0`과
`Direct scatter ... staging_bytes=0`을 확인한다. 후자는 backend의 설계상 payload
할당량이며 GPU 전체 메모리 실측값은 아니다. 모델 KV cache와 CUDA allocator reserve는
여전히 존재하므로 NVML/PyTorch peak 메모리도 따로 측정한다.

## 검증 및 성능 비교

1. CPU 주소 planner와 fake DAOS로 C IOV/extent·coverage 계약을 확인한다.
2. 실제 GPU/DAOS roundtrip gate를 통과시킨다.
3. 새 namespace에서 cold→warm 요청으로 KV hit 및 생성 결과를 확인한다.
4. vLLM KV 용량을 고정해 전송 자체의 비용을 비교한다.
5. 총 GPU 메모리 예산을 고정하고 절약한 공간을 KV cache로 돌려 TTFT, throughput,
   scheduler 대기, preemption을 비교한다.

새 모드는 DRAM tier를 지원하지 않으므로 baseline도 DRAM cache/promotion을 끄고
비교해야 한다. 비동기 저장 여부, schema, metadata/IOM 비용의 차이도 결과에 명시한다.
실패한 읽기는 해당 지점 이후를 유효한 prefix로 표시하지 않는다. 반환 mask는 실제
fetch한 token만 센다. 기존 adapter의 실패 block 처리로 재계산되며, chunk 내부의
이미 캐시된 prefix block까지 보수적으로 재계산 대상으로 포함될 수 있다.

## 이 환경에서 확인한 범위

Python CPU 검증과 C mock harness는 실제 GPU RDMA 정합성을 대신하지 않는다.
2026-10-01에는 sandbox 밖의 H100 NVL 및 RDMA 장치에서 위 roundtrip gate를
5회 반복 통과했다. 전체 chunk 및 prefix 5/127-token skip의 복원값과
미사용 영역 보존을 검증했다. sandbox 안에서는 GPU 장치가 보이지 않으므로
그 결과만으로 호스트의 GPU 사용 가능 여부를 판단하면 안 된다.

vLLM 초기화에서는 CPU tier가 없을 때도 `LocalCPUBackend`를 찾는 설치 버전의
allocator 선택을 direct backend로 연결한다. 실제 allocator는
`NoPayloadAllocator`여서 payload 할당 시도를 오류로 거부한다.
end-to-end 성능은 `realqa_direct_scatter_experiment.py`의 C8×S10 multi-round
실험으로 별도 검증한다. roundtrip 성공 자체는 서빙 성능 개선의 근거가 아니다.
