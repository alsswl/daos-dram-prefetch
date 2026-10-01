# 축소본으로 새 실험 시작

사용자 요청에 따라 2026-09-28부터 `/root/discos_minji/eqbench_matrix_short4_dram8_16_32_20260928`에서 모든 36조건을 처음부터 실행한다.

이 폴더의 8챕터·출력 상한 4000토큰 결과는 보존하되 새 4챕터·2500토큰 결과와 합산하지 않는다. 중단 조건 `c8_d8_s8_cancel`의 KV 4302개는 해당 namespace에 한해 삭제했으며 다른 54225개 키와 이 폴더의 로그·그래프는 보존했다. 정리 증거는 해당 조건의 `recovery_cleanup_700e2e04dd5c44ea81d2fd31adc2c3fc/result.json`이다.
