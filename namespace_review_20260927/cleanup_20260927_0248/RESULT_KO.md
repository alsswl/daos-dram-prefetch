# DAOS namespace 정리 결과

- 대상: discospool/kvcache, OID 281543696187392.1000
- 승인 근거: 사용자의 데이터 삭제 요청과 client-5의 우선 후보 26개 및 OID 지정.
- 원본 priority_review.json의 deletion_authorized=false는 검토 목록 생성 당시 값이며 원본은 수정하지 않았음.
- 26개 namespace를 첨부 목록 및 client-5의 각 실행 설정과 대조함.
- client-5에서 실행 중인 해당 실험 프로세스 없음 확인.
- 삭제 전 전체 dkey: 85,034개
- 지정 namespace에서 삭제한 dkey: 38,425개 (kv 및 meta 포함)
- 삭제 후 지정 namespace dkey: 0개
- 보존한 dkey: 46,609개. 삭제 전후 집합의 정확한 일치 확인.
- 로그·CSV·그래프·소스·설정은 삭제하지 않았음.
- 풀·컨테이너·전체 OID 삭제나 NVMe 파일 직접 삭제는 하지 않았음.
- 삭제 직후 풀의 NVMe 여유 공간은 131,595,460,608 bytes로 변동 없음. 물리 공간 회수 완료를 의미하지 않음.

result.json에 검증 결과와 namespace별 삭제 수, target_dkeys.json에 삭제한 정확한 키 목록을 기록함.
before.keys 및 after.keys는 키 목록이며 데이터 값의 백업이 아님.
