# 완료된 실험 KV 캐시 정리 결과

- 사용자 승인: 이전 실험의 KV 캐시 정리. 기본 캐시·다른 사용자의 데이터·결과 파일 보존.
- 대상: discospool / kvcache, OID `281543696187392.1000`의 지정 dkey만.
- 로컬 설정·완료 상태·로그와 대조한 최근 프리페치 및 DiscoveryBench 실험 namespace 30개.
- 삭제 전 총 키 83,897개 중 지정 namespace의 Qwen3-14B KV 키 **68,814개 삭제**.
- 삭제 후 대상 키 **0개**. 나머지 **15,083개 키는 삭제 전후 집합이 정확히 일치**.
- pool/container/전체 OID/서버 NVMe 파일은 삭제하지 않음.
- 로그·CSV·그래프·소스·설정은 삭제하지 않음.
- 키 목록은 백업했지만 KV payload 백업은 없음. 삭제된 캐시는 직접 복구할 수 없고 다시 계산/저장해야 함.

삭제 직후에는 공간 회수가 늦게 반영됐다. 이후 pool query에서 NVMe 여유가
약 155GB에서 **약 1.6TB**, target별 최소 여유가 약 6.7GB에서 **약 99GB**로 늘어난 것을 확인했다.

`manifest.json`에 정확한 namespace/설정 근거 및 대상 키 해시를 기록했다.
`before_keys.json`, `target_keys.json`, `after_keys.json`, `deletion.jsonl`, `result.json`으로 삭제 범위와 보존 여부를 확인할 수 있다.
