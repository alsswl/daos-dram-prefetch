# 재개와 자동 복구

서비스 이름: `discos-minji-eqbench.service` (root 사용자 서비스). 대화 도구의 실행 세션과 분리된 systemd 사용자 서비스로 운영한다. 서비스는 서버 재부팅 시 자동 부팅하도록 설치한 영구 서비스가 아니라 이번 실험용 transient 서비스다. SELinux가 시스템 서비스의 /root 로그 접근을 거부하여 사용자 서비스로 전환했으며 보안 정책을 변경하지 않았다. 이 초기 기동 실패 때는 모델 요청을 보내지 않았다.

확인 당시 root의 Linger=no였다. 따라서 서버 재부팅 또는 마지막 root 로그인 세션 종료까지 무조건 유지되는 서비스는 아니다. 사용자 전체 로그인 유지 정책은 변경하지 않았다.

완료3조건은 유지하고, 중단된 c8_d8_s4_wait는 로그를 `../aborted_attempts`에 보존한 뒤 새 프로세스·새 namespace에서 다시 시작한다. 해당 중단 캐시1,947개는 이미 삭제/기존 키 보존 검증이 끝났다. 다음 실패도 실패 기록과 정리 manifest를 남긴다. 성공한 조건은 성능 수치에 따라 다시 돌리지 않는다.

- 실행기의 stdout/stderr: `runner_<실행번호>.log`
- 감독 서비스 로그: `service.log`
- 감독 상태: `status.json`
- 실행기 PID와 명령: `runner.json`
- 5초마다 기록하는 진행 상태: `heartbeat.json`
- 중단 원인 분류와 종료 코드: `exit_<실행번호>.json`

실행기 종료 신호·통신 끊김은 해당 조건의 미완료 시도를 보존/정리한 뒤 재시도한다. 15분 동안 완료 요청/조건 전환이 전혀 없으면 정지로 간주하고 복구한다. 조건별 자동 재시도는 최대3회이며 최초 중단도 실패 시도 수에 포함된다. 재시도 전에 해당 실험 서버 프로세스 그룹의 설정 경로를 대조하고, 그 그룹만 종료한다. DAOS 삭제는 매번 정확한 namespace 키 목록을 먼저 만든 뒤 수행한다.

데이터 불일치, 코드 예외, CUDA OOM, 저장 공간 부족, 반복 실패는 성공 결과로 처리하거나 실험 조건을 바꿔 숨기지 않는다. `needs_investigation` 또는 `retry_limit`로 기록하며 원인을 조사한 뒤 수정·재개한다. 자동 실행기가 임의로 코드를 고치는 기능은 아니다. 감독 서비스 자체가 갑자기 종료되면 systemd가 재시작하며 서비스 소속 프로세스들도 정리한다.

새 실행은 모델·프롬프트·seed·동시성·캐시 용량·프리페치 방식·취소 여부를 기존 계획대로 유지한다. 원래 데이터/백엔드 구현은 변경하지 않고 실행 수명·로그·복구 관리만 추가했다. 중단 시도와 재시도는 과학적 해석을 위해 반드시 구분한다.

```bash
XDG_RUNTIME_DIR=/run/user/0 systemctl --user status discos-minji-eqbench.service --no-pager
cat /root/discos_minji/eqbench_matrix_20260928_v2/supervision/heartbeat.json
```

자동 실행을 명시적으로 멈출 때만 `XDG_RUNTIME_DIR=/run/user/0 systemctl --user stop discos-minji-eqbench.service`를 사용한다. GitHub 백업 저장소에는 이번 감독 코드 변경이 아직 반영되지 않았다.
