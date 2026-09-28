#!/usr/bin/env python3
"""Summarize the saved, instrumented mixed-tier pressure diagnostic."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics

from staging_mixed_pressure import read_events


def main():
    p = argparse.ArgumentParser()
    p.add_argument('folder', type=Path)
    a = p.parse_args()
    folder = a.folder.resolve()
    rows = json.loads((folder/'summary.json').read_text())
    groups = {}
    for row in rows:
        groups.setdefault((row['condition'], row['workload'], row['concurrency']), []).append(row)
    aggregate = []
    for (condition, workload, concurrency), samples in sorted(groups.items()):
        aggregate.append(dict(condition=condition, workload=workload, concurrency=concurrency,
            batches=len(samples),
            dram_hit_share=statistics.mean(s['dram_share_of_hit_chunks'] for s in samples),
            peak_gib=max(s['peak_used_gib'] for s in samples),
            mean_gib=sum(s['mean_used_gib']*s['wall_seconds'] for s in samples)/sum(s['wall_seconds'] for s in samples),
            ttft_ms=statistics.mean(s['mean_ttft_ms'] for s in samples),
            e2e_ms=statistics.mean(s['mean_e2e_ms'] for s in samples),
            cpu_staged_batches=sum(s['cpu_prefetch_batches'] for s in samples),
            cpu_unstaged_batches=sum(s['cpu_unstaged_batches'] for s in samples),
            failed_alloc_events=sum(s['failed_alloc_events'] for s in samples),
            partial_daos_reads=sum(s['partial_daos_reads'] for s in samples)))
    (folder/'aggregate.json').write_text(json.dumps(aggregate, indent=2)+'\n')
    with (folder/'aggregate.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    warnings = {}
    event_sets = {}
    for condition in ('prefetch_off', 'prefetch_on'):
        log = (folder/condition/'server.log').read_text()
        warnings[condition] = {
            'gpu_buffer_full': log.count('GPU buffer full'),
            'negative_refs': len(re.findall(r'negative: -|Double free|Double release', log)),
            'backend_failures': len(re.findall(r'DaosGdsBackend.*failed|Failed to create.*backend', log)),
            'semaphore_shutdown_warning': 'leaked semaphore' in log,
            'engine_dead': 'EngineDeadError' in log,
            'cpu_staging_fallback_all_phases': log.count('CPU staging fallback['),
            'stores_all_phases': len(re.findall(r'Stored \d+ out of total', log)),
        }
        event_sets[condition] = read_events(folder/condition)
    (folder/'log_checks.json').write_text(json.dumps(warnings, indent=2)+'\n')

    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="960" viewBox="0 0 1320 960">',
           '<rect width="1320" height="960" fill="white"/>',
           '<g font-family="sans-serif" font-size="13" fill="#222">',
           '<text x="40" y="28" font-size="20">Qwen3-14B: 8K inputs, 16 concurrent requests, staging occupancy (repeat 1)</text>']
    for i, workload in enumerate(('dram_heavy', 'mixed', 'daos_heavy')):
        for j, condition in enumerate(('prefetch_off', 'prefetch_on')):
            batch = next(r for r in rows if r['condition'] == condition and r['repeat'] == 1
                         and r['concurrency'] == 16 and r['workload'] == workload)
            events = [e for e in event_sets[condition] if batch['start_ns'] <= e['time_ns'] <= batch['end_ns']]
            left, top, width, height = 65+j*655, 85+i*275, 570, 195
            duration = (batch['end_ns']-batch['start_ns'])/1e9
            def x_at(ns):
                return left+(ns-batch['start_ns'])/1e9/duration*width
            def y_at(gib):
                return top+height-gib/10.5*height
            svg.append(f'<text x="{left}" y="{top-18}" font-size="16">{workload} / {condition} / C=16</text>')
            for level in (0,2,4,5,6,8,10):
                y = y_at(level)
                color = '#d55e00' if level == 10 else '#d69a00' if level == 5 else '#ddd'
                svg.append(f'<line x1="{left}" y1="{y}" x2="{left+width}" y2="{y}" stroke="{color}" stroke-dasharray="4 3"/>')
                svg.append(f'<text x="{left-32}" y="{y+4}">{level}</text>')
            for tick in range(5):
                x = left+tick*width/4
                svg.append(f'<text x="{x-10}" y="{top+height+20}">{duration*tick/4:.2f}</text>')
            svg.append(f'<text x="{left+width/3}" y="{top+height+42}">Seconds from batch submission</text>')
            svg.append(f'<text x="{left-40}" y="{top-3}">GiB</text>')
            for series, field, color in [(events, 'used_bytes', '#0072b2'),
                    (events, 'ready_bytes', '#009e73')]:
                points = []
                previous = None
                for e in series:
                    x, y = x_at(e['time_ns']), y_at(e[field]/2**30)
                    if previous is not None:
                        points.append(f'{x:.2f},{previous:.2f}')
                    points.append(f'{x:.2f},{y:.2f}')
                    previous = y
                svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="1.7"/>')
    for idx, (color, label) in enumerate([('#0072b2','Total active staging'),('#009e73','Ready, not yet freed'),
            ('#d69a00','CPU watermark: 5 GiB'),('#d55e00','Capacity: 10 GiB')]):
        x = 50+idx*315
        svg.append(f'<line x1="{x}" y1="925" x2="{x+25}" y2="925" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<text x="{x+32}" y="929">{label}</text>')
    svg.append('</g></svg>')
    (folder/'staging_timeline_c16.svg').write_text('\n'.join(svg))

    c16 = {(r['condition'], r['workload']): r for r in aggregate if r['concurrency'] == 16}
    hot = c16['prefetch_on', 'dram_heavy']
    mixed = c16['prefetch_on', 'mixed']
    lines = ['# DRAM/DAOS 혼합 부하의 GPU staging 점유 측정', '',
        '2026-09-23. 실제 object GPU-direct 경로. 합성 부하 진단이며 DiscoveryBench가 아니다.', '',
        '## 핵심 관측', '',
        '- 요청 수가 같아도 DRAM hit가 많으면 DAOS가 쓰는 staging이 줄어든다. OFF의 DRAM-only 재사용은 staging 점유가 0이었다.',
        f'- ON, 동시성 16, DRAM hit 100%: 최대 {hot["peak_gib"]:.2f}GiB. '
        f'두 묶음 32요청 중 {hot["cpu_staged_batches"]}개만 staging으로 미리 복사했고 '
        f'{hot["cpu_unstaged_batches"]}개는 CPU 직접 로드로 돌아갔다. 기존 5GiB admission 제한이 적용됐다.',
        f'- ON, 동시성 16, DRAM/DAOS 50:50: 순간 최대 {mixed["peak_gib"]:.2f}GiB/10GiB. '
        'CPU 입장 기준이 5GiB여도 DAOS가 이후 독립적으로 공간을 사용할 수 있어 전체 사용량은 이를 넘는다.',
        '- 따라서 DAOS 수요가 낮을 때 CPU 프리페치 한도를 빌려주는 정책의 실험 근거가 된다. '
        '다만 5GiB를 넘겨 허용하는 동적 정책 자체는 구현/비교하지 않았고, 이를 바꾸면 지연이 더 좋아진다는 증거는 아직 아니다.', '',
        '## 실행 조건', '',
        '- Qwen/Qwen3-14B BF16, H100 NVL 1개, vLLM 서버 1개, 자체 prefix caching OFF.',
        '- 입력 16종 × 8192토큰, 출력 64토큰. 청크 128, DRAM 4GiB, GPU staging 10GiB.',
        '- DRAM 프리페치 OFF/ON 별 새 프로세스·새 DAOS namespace. ON의 기존 입장 기준은 공유 풀 5GiB.',
        '- 순차 fill 후 최근 입력 2종을 DRAM-hot으로 선택, 앞 입력 8종은 DAOS-only로 확인.',
        '- 각 측정 묶음은 16요청. DRAM-heavy는 hot 2종을 반복, DAOS-heavy는 cold 8종을 반복, mixed는 둘을 반씩 사용.',
        '- 세 workload의 입력 다양성이 다르므로 workload 사이 시간 차이를 순수 hit 비율 효과로 단정하지 않는다.',
        '- 동시성 1/4/8/16, 프로세스 안에서 2회 반복. OFF→ON 실행 순서는 고정이며 독립 반복 성능 검증이 아니다.',
        '- 준비 이후 요청은 기존 lmcache.skip_save 설정으로 새 저장을 생략. fill 1024청크의 DAOS 저장 완료와 staging 0 복귀 확인.',
        '- 측정 중 생성 결과를 다음 입력에 추가하지 않으며, 캐시 보관/퇴거/프리페치 정책은 바꾸지 않았다.', '',
        '## staging 점유와 지연', '',
        '최대는 두 묶음의 event-wise 최대, 평균은 전체 HTTP 묶음 시간 가중 평균이다. '
        '평균에는 staging이 비어 있는 decode 시간도 포함하므로 모든 여유를 프리페치에 쓸 수 있다는 뜻이 아니다.', '',
        '| 프리페치 | 부하 | 동시성 | DRAM hit 비중 | 최대 GiB | 평균 GiB | 평균 TTFT ms | CPU staging/미사용 요청 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    names = dict(dram_heavy='DRAM 위주', mixed='혼합', daos_heavy='DAOS 위주')
    for r in aggregate:
        lines.append(f'| {r["condition"].replace("prefetch_", "").upper()} | {names[r["workload"]]} | '
                     f'{r["concurrency"]} | {r["dram_hit_share"]:.0%} | {r["peak_gib"]:.3f} | '
                     f'{r["mean_gib"]:.3f} | {r["ttft_ms"]:.2f} | '
                     f'{r["cpu_staged_batches"]}/{r["cpu_unstaged_batches"]} |')
    lines += ['', 'CPU staging/미사용은 CPU get batch 수이며 청크/API/RPC 수가 아니다. OFF의 미사용은 '
              '원래 CPU 경로이고, ON의 미사용은 staging 예산 부족 등의 fallback이다. '
              'hit 비중은 실제 lookup에서 선택한 DRAM·DAOS 청크 수 기준이며 API cached-token 비율과 다르다.', '',
              '## 시간에 따른 점유', '', '![동시성 16의 첫 반복](staging_timeline_c16.svg)', '',
              'Total은 전송 중인 버퍼를 포함한 실제 allocator 사용량이다. Ready는 복사 완료 후 '
              '아직 해제되지 않은 버퍼만 나타내므로 Total과 일치할 필요가 없다. '
              '주기적 관측은 명목 5ms이며 스케줄링 지연이 있을 수 있다. allocation/free 이벤트도 기록했다.', '',
              '## 검증과 한계', '',
              f'- 모든 측정 요청의 8191토큰 재사용: {all(r["all_hit"] for r in rows)}.',
              f'- 예상 DRAM/DAOS 청크 비중 일치: {all(r["tier_mix_valid"] for r in rows)}.',
              f'- 측정 중 할당 실패 이벤트: {sum(r["failed_alloc_events"] for r in rows)}.',
              f'- 측정 중 DAOS 부분 읽기: {sum(r["partial_daos_reads"] for r in rows)}.',
              '- allocator 점유는 nvidia-smi 메모리 예약량과 다르다. 풀 10GiB가 예약되어도 내부 사용량은 0일 수 있다.',
              '- 이번 혼합은 요청 사이 DRAM/DAOS hit 혼합이다. 한 요청 안의 부분 DRAM hit를 의도적으로 만든 실험은 아니다.',
              '- 전부 재사용 요청이며 실제 서비스의 신규 입력·쓰기·취소·불규칙 도착을 재현하지 않는다.',
              '- 계측 오버헤드가 있고 timeline 겹침은 CUDA profiler로 확인하지 않았다. 지연은 진단 참고값이다.',
              '- 전체 KV 바이트 비교는 수행하지 않았다. 종료 경고와 오류 횟수는 log_checks.json 참조.', '',
              '- 준비 실행 `staging_mixed_20260923_v1`은 skip_save를 bool로 전달해 async lookup의 문자열 스키마 검사에서 실패했다. '
              '본 결과에서 제외했고, 문자열로 수정한 v2를 새 프로세스·새 namespace에서 실행했다. 실패 로그도 보존했다.', '',
              '## 원본과 재실행', '',
              '- [요약 CSV](aggregate.csv), [모든 묶음 결과](summary.json), [검증 상태](status.json), [로그 검사](log_checks.json).',
              '- 각 조건 폴더에 원시 trace JSONL, 요청별 결과, 설정, 프로세스 명령, native library 경로를 보관했다.',
              '- 기존 캐시를 삭제하지 않았다. 이번 실험 전용 DAOS 키도 남아 있다.', '',
              '```bash', 'cd /root/discos_minji',
              './venv/bin/python3 staging_mixed_pressure.py --output /root/discos_minji/staging_mixed_next --repeats 2',
              './venv/bin/python3 report_staging_mixed.py /root/discos_minji/staging_mixed_next', '```', '']
    (folder/'RESULT_KO.md').write_text('\n'.join(lines))
    print(folder/'RESULT_KO.md')


if __name__ == '__main__':
    main()
