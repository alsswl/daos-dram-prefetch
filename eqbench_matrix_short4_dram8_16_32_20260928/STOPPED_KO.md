# 사용자 요청으로 중단

2026-09-28 L-Eval 문서 QA 실험으로 전환하기 위해 `discos-minji-eqbench-short.service`를 중단했다. 완료 10조건과 모든 로그·그래프를 보존한다. 진행 중이던 `c8_d16_s8_wait`의 불완전 측정은 완료 결과에 포함하지 않는다.

해당 중단 namespace의 KV 2865개를 삭제했으며 나머지 54225개 키를 보존했다. 삭제 KV는 백업되지 않아 복구되지 않지만 재실행으로 다시 생성할 수 있다. 근거: `c8_d16_s8_wait/recovery_cleanup_f705cf2412e74c5e84a3cd39ad82fb67/result.json`.

새 실험: `/root/discos_minji/leval_prefetch_20260928`. 워크로드가 다르므로 결과를 합산하지 않는다.
