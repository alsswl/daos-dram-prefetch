# 이전 36조건 실험 KV 삭제 기록

사용자의 삭제 승인에 따라 `prefetch_capacity_sweep_20260927`의 완료된 36조건 설정을 확인하고 해당 namespace의 Qwen3-14B KV 키만 삭제했다.

- 풀: `discospool` (`f973c142-2353-41da-b154-5079ba6969f2`)
- 컨테이너: `kvcache` (`a2d875e3-195b-4598-98b4-b381ef49867a`)
- 대상 OID: `281543696187392.1000` — 오브젝트 전체를 삭제하지 않고 명시된 dkey만 삭제.
- 삭제: 36개 namespace, 59,638개 키.
- 보존: 나머지 18,431개 키. 삭제 전후의 전체 목록을 대조해 정확히 일치함을 확인.
- 실험 로그·그래프·코드·공용 설정 및 최근 worker 비교 실험은 보존.
- KV payload 백업 없음. 삭제한 캐시를 재사용하려면 재계산이 필요.
- 공간 회수 후 풀 여유: 1,528,361,607,168바이트 (약 1.53TB). 삭제 전 277,662,097,408바이트.

검증 근거: `manifest.json`, `result.json`, `before_keys.json`, `target_keys.json`, `after_keys.json`, `deletion.jsonl`, `pool_reclaimed.json`.

실행 스크립트: `../cleanup_capacity_sweep_cache.py`. 대상은 정확한 완료 실험 설정으로 제한되며 기본 namespace나 컨테이너 전체 삭제를 수행하지 않는다.
