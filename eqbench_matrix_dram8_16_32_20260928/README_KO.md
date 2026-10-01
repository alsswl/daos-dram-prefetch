# DRAM 8/16/32GiB × staging 4/8GiB 비교

사용자 요청으로 기존 eqbench_matrix_20260928_v2는 중지했다. 이전 로그·완료 결과는 보존하고 해당 실험의 KV는 정리했다. 이 실험은 이전 완료 조건을 재사용하지 않는 새 36조건 측정이다.

|항목|조건|
|---|---|
|동시 대화|8 / 16|
|DRAM|8 / 16 / 32GiB|
|GPU staging|4 / 8GiB|
|DRAM 프리페치|OFF / ON·대기 / ON·대기 중 취소|
|모델|Qwen3-14B BF16|
|워크로드|EQ-Bench Longform 원본 첫4주제 × 4대화씩, 실제 생성 답변 누적|
|모델 호출|16대화 × 13단계 = 208회/조건, 총7,488회|
|기타|청크128, vLLM max-num-seqs16, DRAM 복사 작업자1, DAOS 프리페치 항상ON|

ON 두 방식은 모두 early_ready=True로 고정하고 대기열 취소 여부만 다르다. 취소는 retrieve 시 아직 시작하지 않은 복사만 취소해 DRAM 경로로 소비하는 기능이다. 실행 중인 복사나 이미 준비된 staging 데이터를 삭제하는 기능은 아니다.

조건마다 새 프로세스·빈 DRAM·새 DAOS namespace에서 시작한다. 중간에는 캐시를 유지하고 실제 대화의 재사용을 관측한다. 전체 warm 반복이나 외부 품질 채점은 없다. 반복 주제 간 캐시 공유도 포함한다. 생성 답변이 다음 입력을 바꾸므로 출력량·입력량·실제 재사용량을 함께 해석해야 한다.

## 진행 및 결과

- [현재 조건](status.json)
- [5초 간격 진행 상태](supervision/heartbeat.json)
- [감독 상태](supervision/status.json)
- [완료 조건별 결과표](RESULT_KO.md), [상세 수치](summary.json), [CSV](summary.csv)
- 각 조건의 staging_hits.png: 2초 구간별 staging 평균/최대, DRAM/DAOS 조회 hit 비율
- 각 조건의 requests_summary.json: TTFT·실제 hit·용량 부족 재계산 귀속

완료 시 해당 조건의 KV만 manifest 기준으로 삭제하고 다른 namespace 키가 보존됐는지 검증한다. 실패 기록과 중단 시도는 보존하며, 부분 측정은 완주 결과에 섞지 않는다. 삭제한 KV의 payload 백업은 없다.

## 서비스와 복구

서비스는 root 사용자 서비스 `discos-minji-eqbench-large.service`다. 대화 도구 실행 세션과 분리했다. 실행기 로그·PID·heartbeat·종료 코드·재시도 이유는 supervision에 저장한다. 실행기 종료·일시적 통신 중단·15분간 진행 없음은 조건별 최대3회 자동 재시도한다. 용량/동시성/출력 제한 등을 임의로 바꾸지 않는다. 코드·데이터 검증 오류, OOM, 저장 공간 부족이나 반복 실패는 조사 필요 상태로 남긴다.

```bash
XDG_RUNTIME_DIR=/run/user/0 systemctl --user status discos-minji-eqbench-large.service --no-pager
```

명시적 중지는 위 명령에서 status 대신 stop을 사용한다. root Linger=no이므로 서버 재부팅이나 마지막 root 로그인 종료까지 보장하는 영구 서비스는 아니다. 보안 정책과 사용자 로그인 유지 정책은 변경하지 않았다. 이번 변경과 결과는 기존 GitHub 백업에 자동 반영되지 않는다.
