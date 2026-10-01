# 2026-09-28 중단 조사

## 확인된 사실

- 완료된 조건은 c8_d8_s8_off, c8_d8_s8_wait, c8_d8_s8_cancel의 3개다.
- c8_d8_s4_wait에서 208회 중 98회 응답이 저장됐다. 마지막 완료 시각은 18:23:22.755 KST다.
- 18:23:23.076~077에 서버 로그에 진행 중 요청 8개의 abort-time cleanup 경고가 함께 나타났다. 18:23:30부터 Running/Waiting 모두 0이었다.
- 조사 시 요청 실행기 PID 848632는 사라졌지만 vLLM PID 856015, 엔진 PID 856224는 살아 있었다. 서버 PPID는 1, /health는 HTTP 200이었다.
- 실행기 상태 파일은 running으로 남았으며, phase.json에 end_ns가 없었다. Python 오류 처리에서 기록하는 failed 상태도 없었다.
- DAOS 요청 67배치, 요청/반환 5,424/5,424청크. capacity_failed_chunks 및 other_failed_chunks 모두 0이다.
- 마지막 관측 시 DRAM 프리페치 72배치, capacity_rejections/copy_errors 0. staging used_bytes 0, 비동기 DRAM 저장 pending_bytes 0, errors 0이었다.
- DAOS 풀 여유 약736.9GB, 타깃 최소43.3GB였다. 서버 로그에 DER_NOSPACE, CUDA OOM, Python traceback은 발견하지 못했다.
- 해당 session-1406 cgroup의 oom/oom_kill/oom_group_kill은 모두0이고 MemoryMax는 무제한이었다.
- 완료 3조건의 KV는 각각 4,088 / 3,919 / 4,192개 삭제됐고, 다른 키 보존 검증을 통과했다.

## 해석과 한계

요청을 보내는 실행기가 정상 종료 절차 없이 사라져 클라이언트 연결이 끊긴 정황이다. 살아 있는 vLLM이 연결 해제된 요청들을 취소한 것으로 해석된다. DAOS 용량 부족, staging 부족 또는 확인된 OOM을 원인으로 볼 근거는 없다.

단, 실행기 자체의 표준출력·종료 코드가 독립 파일/서비스 관리자에 보존되지 않았고 기존 도구 세션도 더 이상 조회할 수 없다. 외부 종료 신호 또는 실행 세션 수명과의 연관성은 의심할 수 있지만, **정확한 종료 주체·신호는 확정할 수 없다**. 구현 버그를 완전히 배제했다는 뜻도 아니다.

커널에는 이전부터 반복된 NVRM RUSD_SEQ_DATA_VALID assertion 경고가 있었다. 서버는 이후에도 살아 있었으며 이 경고와 실행기 종료의 인과관계는 확인되지 않았다. NetworkManager의 eth0 상태 변경 기록도 있지만, 이것만으로 클라이언트 종료 원인이라고 결론내릴 수 없다.

## 정리와 후속 실행

중단 조건의 로그·98개 완료 응답·입력·trace는 보존한다. 실행기가 사라진 상태와 서버 identity를 확인한 뒤, 해당 서버 프로세스 그룹만 정상 종료한다. 중단 namespace의 KV만 별도 manifest로 삭제하고 다른 키 보존을 검증한다. KV payload 백업은 없으므로 삭제 후 복구할 수 없다.

이 중단 조건은 완주 결과와 합산하지 않는다. 재실행한다면 새 프로세스·빈 DRAM·새 namespace에서 조건 전체208회를 다시 측정해야 한다. 완료3조건은 그대로 보존한다. 장시간 실행은 도구 세션과 분리하고 실행기 PID·지속 로그·종료 코드·heartbeat를 기록하는 방식이 적절하다. 이 조사 자체에서는 실험 코드를 수정하거나 재실행하지 않았다.
