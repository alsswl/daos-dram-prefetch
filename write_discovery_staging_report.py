#!/usr/bin/env python3
"""Render the completed, read-only analyzer output as a Korean report."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('folder',type=Path)
    a=p.parse_args(); folder=a.folder.resolve()
    status=json.loads((folder/'status.json').read_text())
    if status['status'] != 'completed':
        raise SystemExit('Refusing a final report for an unfinished experiment')
    results=json.loads((folder/'analysis.json').read_text())
    lines=['# DiscoveryBench 장시간 DRAM / DAOS / GPU staging 관측 결과', '',
        '실행일: 2026-09-26. 작업 위치: `/root/discos_minji`.', '',
        '## 무엇을 실행했나', '',
        '- Qwen3-14B BF16, H100 NVL 1개, vLLM 1개 프로세스.',
        '- DiscoveryBench real의 train/test 메타데이터 32개에서 각각 첫 query를 반복 투입했다. 모델이 Python 코드를 만들고 실행 결과를 받아 다음 추론을 하는 에이전트 방식이다. 전체 데이터셋 실행이나 과학적 정답 채점은 아니다.',
        '- DRAM 캐시 **8GiB**, GPU staging **10GiB**, 청크 **128토큰**. DAOS는 **직접 오브젝트/GPU-direct** 경로다.',
        '- 동시 에이전트 **4개 5분 → 8개 5분 → 16개 10분**. 각 단계 뒤에 최대 약 90초의 마무리 시간이 있다. 모델 시작 시간은 아래 관측 시간에서 제외했다.',
        '- DRAM→GPU staging 프리페치 OFF/ON을 각각 새 프로세스·새 DAOS namespace로 실행했다. DAOS 비동기 프리페치는 두 조건 모두 ON이다.',
        '- ON의 DRAM 프리페치는 기존 공유 staging 입장 기준 5GiB를 사용한다. 10GiB 풀을 두 개로 분할하거나 동적으로 늘린 것이 아니다.', '',
        '8GiB는 짧은 예비 실행에서 DRAM과 DAOS hit가 함께 관측된 실험값이다. 최적 운영 용량으로 검증된 값은 아니며, 아래 장시간 결과에서는 DRAM 재사용 한계가 드러났다.', '',
        '## 시간에 따른 변화', '',
        '![staging, tier hits, DRAM occupancy and LLM concurrency](timeline.png)', '',
        '[원본 SVG](timeline.svg). 왼쪽 OFF, 오른쪽 ON. 세로 점선은 동시 에이전트 수 변경 시점이다. GPU staging은 5초 구간 최대값과 시간 가중 평균을 함께 그렸다. 조회가 없는 구간의 hit 비율은 빈 구간이다.', '',
        'staging 그래프는 **실제 활성 할당 바이트**다. 예약된 10GiB 전체를 의미하지 않는다. DAOS 읽기·DRAM 프리페치뿐 아니라 신규 KV 저장용 GPU 버퍼도 포함한다.', '',
        '## 전체 요약', '',
        '| DRAM 프리페치 | 관측 시간(분) | 작업 수 | 완료 / 미완료 / 실패 | 모델 호출 | Python 실행 | staging 최대(GiB) | 평균(GiB) | 비어 있던 시간 | 할당 실패 |',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for s in results:
        counts=s['workflow_status']; mode=s['condition'].replace('prefetch_','').upper()
        lines.append(f"| {mode} | {s['elapsed_seconds']/60:.2f} | {s['workflows']} | {counts.get('complete',0)} / {counts.get('incomplete',0)} / {counts.get('failed',0)} | {s['llm_calls']} | {s['python_calls']} | {s['peak_staging_gib']:.3f} | {s['mean_staging_gib']:.3f} | {100*s['occupancy_time_fraction'].get('empty',0):.1f}% | {s['allocator_failures']} |")
    lines += ['', '완료는 Python 도구 사용 후 최종 답변에 도달했다는 뜻이다. 정답이라는 의미가 아니다. 모델 호출은 클라이언트 콜백 수로, 실패·취소된 호출도 포함할 수 있다.', '',
        '## DRAM / DAOS hit 비율', '',
        '**분모는 비동기 prefix lookup 대상 청크 수**다. DRAM과 DAOS 양쪽에 있는 청크는 먼저 선택된 DRAM에만 계상한다. miss는 새로 계산해야 하는 부분이다. 전체 입력 토큰 비율이나 DAOS에 실제 질의한 것 중 성공 비율과는 다르다.', '',
        '| 조건 | 구간 | 조회 청크 | DRAM hit | DAOS hit | miss |', '|---|---|---:|---:|---:|---:|']
    for s in results:
        mode=s['condition'].replace('prefetch_','').upper()
        lines.append(f"| {mode} | 전체 | {s['lookup_chunks']:,} | {100*s['dram_lookup_ratio']:.2f}% | {100*s['daos_lookup_ratio']:.2f}% | {100*s['miss_lookup_ratio']:.2f}% |")
        for w in s['lookup_time_windows']:
            label={'first_120s':'처음 2분','after_120s':'2분 이후'}[w['window']]
            lines.append(f"| {mode} | {label} | {w['chunks']:,} | {w['dram_pct']:.2f}% | {w['daos_pct']:.2f}% | {w['miss_pct']:.2f}% |")
    lines += ['', '| 조건 | 실제 DRAM 읽기 청크 | staging으로 미리 옮긴 DRAM 청크 | staging 프리페치 요청 수 | 마지막 DRAM hit 시점(시작 후 초) |',
        '|---|---:|---:|---:|---:|']
    for s in results:
        last=s['last_dram_hit_elapsed_seconds']
        last_text=f'{last:.1f}' if last is not None else '없음'
        lines.append(f"| {s['condition'].replace('prefetch_','').upper()} | {s['cpu_get_chunks']:,} | {s['cpu_staged_chunks']:,} | {s['cpu_staged_requests']} | {last_text} |")
    lines += ['', '위 청크 수는 반복 읽기를 포함한다. DRAM에 저장된 고유 청크 수가 아니다.', '', '## 동시 에이전트 단계별 결과', '',
        '| 조건 | 동시 에이전트 | 시간(분, 마무리 포함) | 모델 호출 | DRAM hit | DAOS hit | miss | staging 최대(GiB) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in results:
        for ph in s['phases']:
            lines.append(f"| {s['condition'].replace('prefetch_','').upper()} | {ph['concurrency']} | {ph['elapsed_seconds']/60:.2f} | {ph['llm_calls']} | {ph['dram_pct']:.2f}% | {ph['daos_pct']:.2f}% | {ph['miss_pct']:.2f}% | {ph['peak_staging_gib']:.3f} |")
    lines += ['', '단계가 진행될수록 캐시의 이력도 달라진다. 위 표만으로 동시성 증가의 인과 효과나 OFF/ON의 성능 우열을 확정하면 안 된다.', '',
        '## 핵심 해석', '',
        '1. **DRAM이 가득 차 있어도 DRAM hit가 계속 나오는 것은 아니었다.** 처음에는 DRAM을 재사용하지만 이후 DAOS에 의존하는 구간이 나타난다. 따라서 전체 평균 hit만 보면 중요한 시간 변화가 가려진다.',
        '2. **staging 최대 점유와 상시 점유를 구분해야 한다.** 요청의 읽기/저장 시점에는 점유가 증가하고 소비·저장 완료 후 감소한다. 순간 여유가 있다는 사실만으로 DRAM 프리페치가 안전하거나 빨라진다고 결론낼 수는 없다.',
        '3. **이 결과만으로 장기 DRAM 프리페치 이득을 평가할 수 없다.** DRAM hit가 사라진 구간에서는 ON이어도 DRAM에서 미리 가져올 데이터가 없다. OFF/ON 후반 차이를 DRAM 프리페치 효과로 해석하지 않는다.', '',
        '### DRAM 재사용이 사라지는 이유의 코드상 후보', '',
        '현재 설치된 LMCache의 CPU 비동기 조회는 앞 청크부터 찾다가 첫 누락에서 멈춘다. 앞부분이 퇴거되면 뒤 청크가 DRAM에 남아 있어도 그 뒤를 다시 CPU에서 찾지 않는다. DAOS 비동기 읽기 완료 뒤 CPU write-back 콜백은 TODO 상태이므로 빠진 앞부분도 자동 복구되지 않는다.', '',
        '`touch_cache()`가 최근 사용 순서를 갱신하지만, 확인한 비동기 조회 경로에서는 동기 lookup처럼 이를 호출하는 부분이 보이지 않는다. 위 조합은 이번 DRAM hit 소멸에 부합한다. **실제 퇴거 키 전체를 기록하지 않아 최초 원인을 확정한 것은 아니다.** 측정 중 캐시 정책을 바꾸지 않았다.', '',
        '따라서 후속 실험은 먼저 prefix 유지·비동기 경로의 최근 사용 순서 갱신·DAOS→DRAM 승격 중 필요한 동작을 검증하고, 혼합 hit가 지속되는 상태에서 staging의 동적 입장 정책을 비교하는 순서가 적절하다. 이번 실행에서 해당 정책을 새로 구현하거나 교체하지 않았다.', '',
        '## 실행 오류와 검증 한계', '',
        '| 조건 | 파싱 오류를 포함한 작업 | 도구 import 오류 포함 | SciPy/statsmodels 호환 오류 포함 | 문맥 길이 초과 포함 | 제한 시간 강제 종료 작업 |',
        '|---|---:|---:|---:|---:|---:|']
    for s in results:
        e=s['agent_issue_jobs']
        lines.append(f"| {s['condition'].replace('prefetch_','').upper()} | {e.get('output_parse_error_jobs',0)} | {e.get('tool_import_error_jobs',0)} | {e.get('scipy_statsmodels_compatibility_jobs',0)} | {e.get('context_overflow_jobs',0)} | {s['forced_workflows']} |")
    lines += ['', '오류 열은 해당 문자열이 기록된 작업 수이며 서로 중복될 수 있다. 도구 오류에서 회복해 완료된 작업도 포함한다. 파일명·컬럼명 오류처럼 모델이 생성한 코드의 오류도 원본 로그에 남겨두었다. 실패를 제거한 성공 작업만으로 성능을 계산하지 않았다. 도구 호환 오류는 추가 시도와 대화 길이를 늘려 캐시 점유에도 영향을 줄 수 있으므로, 정상적인 분석 작업의 대표 부하로 일반화하려면 환경을 정리한 뒤 다시 측정해야 한다.', '',
        '- 같은 문제 목록을 쓰지만 closed-loop 실행이므로 생성 내용·호출 순서·실제 요청 수가 조건 사이에 달라진다. 고정 trace를 재생한 paired benchmark가 아니다.',
        '- OFF→ON 고정 순서 한 번씩 실행했다. 순서 교대·여러 반복·신뢰 구간 측정은 하지 않았다.',
        '- DRAM/DAOS는 새 namespace의 논리적 cold 시작이다. 서버 전체 캐시나 장치 캐시는 비우지 않았다.',
        '- 기존 로컬 ReAct 프롬프트/실제 데이터와 Python 도구를 사용하지만 32문제 부분집합이다. 로컬 프롬프트에는 기존 `/no_think`·단계별 분석 지시가 있어 공식 upstream과 완전히 동일한 재현이라고 주장하지 않는다. 정답률 채점·전체 DiscoveryBench 평가가 아니다.',
        '- 계측 오버헤드가 있다. allocator 시계열이지 CUDA 연산/전송별 프로파일은 아니다.',
        '- 기존 SciPy/statsmodels 호환 문제와 일부 에이전트 형식 오류가 있다. 실행 중 환경을 바꾸지 않았으며 두 조건에 같은 환경을 사용했다.', '',
        '## 원본과 재현 방법', '',
        '- [실행 계획 및 소스 해시](plan.json), [32개 문제 목록 및 입력 해시](tasks.json)',
        '- [소스·입력 해시 및 로드한 라이브러리 검증](validation.json)',
        '- [집계 JSON](analysis.json), [OFF 시계열 CSV](prefetch_off/timeline.csv), [ON 시계열 CSV](prefetch_on/timeline.csv)',
        '- [OFF 서버 로그](prefetch_off/server.log), [ON 서버 로그](prefetch_on/server.log)',
        '- [OFF 작업 결과](prefetch_off/jobs.json), [ON 작업 결과](prefetch_on/jobs.json)',
        '- [상세 실행 방법](../DISCOVERY_STAGING_METHOD_KO.md)', '',
        '공용 DAOS 데이터를 삭제하지 않았고 `/root/discos` 원본 및 설치된 LMCache/vLLM은 수정하지 않았다. 이번 실험용 namespace 데이터와 결과를 보존한다.', '']
    path=folder/'RESULT_KO.md'; path.write_text('\n'.join(lines))
    print(path)


if __name__=='__main__':
    main()
