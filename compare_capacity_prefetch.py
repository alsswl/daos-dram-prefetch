#!/usr/bin/env python3
"""Compare completed capacity matrices without changing their raw results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import yaml

from report_capacity_matrix import save_chart


def read(path):
    return json.loads(path.read_text())


def validate_pair(on, off):
    plans = [read(p/'plan.json') for p in (on, off)]
    assert read(on/'requests.json') == read(off/'requests.json')
    for field in ('model', 'chunk_tokens', 'max_model_len', 'requests_per_case', 'concurrency', 'generation'):
        assert plans[0][field] == plans[1][field], field
    changed = [n for n, h in plans[0]['source_sha256'].items()
               if plans[1]['source_sha256'].get(n) != h]
    assert set(changed) <= {'discovery_capacity_matrix.py', 'report_capacity_matrix.py'}, changed
    namespaces = []
    audit = dict(changed_sources=changed, same_backend_and_replay_code=True, cases={})
    for root, plan in zip((on, off), plans):
        assert read(root/'status.json')['status'] == 'completed'
        for name, digest in plan['source_sha256'].items():
            assert hashlib.sha256((root/'executed_sources'/name).read_bytes()).hexdigest() == digest
    for spec in plans[0]['cases']:
        name = spec['name']
        configs = []
        for root, enabled in ((on, True), (off, False)):
            cfg = yaml.safe_load((root/name/'config.yaml').read_text())
            extra = cfg['extra_config']
            assert extra.pop('daosgds.dram_prefetch') is enabled
            namespaces.append(extra.pop('daosgds.object_namespace'))
            extra.pop('daosgds.root')
            configs.append(cfg)
            initial = read(root/name/'initial_sample.json')
            assert initial['used_bytes'] == initial['cpu_hot_bytes'] == initial['daos_puts'] == 0
        assert configs[0] == configs[1], name
        assert read(on/name/'native_maps.json') == read(off/name/'native_maps.json'), name
        # Ports/config paths may differ, but all model/scheduler flags must match.
        commands = []
        for root in (on, off):
            cmd = read(root/name/'command.json')
            commands.append(cmd)
        assert commands[0] == commands[1], name
        audit['cases'][name] = dict(same_config_except_toggle_namespace=True,
            same_native_libraries=True, same_server_command=True, both_started_empty=True)
    assert len(set(namespaces)) == len(namespaces)
    audit['all_namespaces_unique'] = True
    return audit


def bars(out, pairs):
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="820">',
           '<rect width="1440" height="820" fill="white"/><g font-family="sans-serif" fill="#222">',
           '<text x="35" y="30" font-size="22">DRAM prefetch OFF vs ON: fixed 256 inputs, concurrency 16</text>',
           '<text x="35" y="56" font-size="14">Blue=OFF, orange=ON. DAOS prefetch and DRAM caching remain enabled in both.</text>']
    metrics = [('mean_client_ttft_ms', 'Mean TTFT (ms)'),
               ('capacity_recompute_input_pct', 'Capacity-caused recompute / all input tokens (%)'),
               ('elapsed_seconds', 'Total replay time (s)'),
               ('actual_cached_input_pct', 'Actual reused input tokens (%)')]
    for panel, (key, title) in enumerate(metrics):
        left, top = 65+(panel % 2)*710, 112+(panel//2)*350
        height, width = 220, 620
        maximum = max(1, max(s[key] for pair in pairs for s in pair)*1.12)
        svg.append(f'<text x="{left}" y="{top-16}" font-size="16">{title}</text>')
        for tick in range(5):
            val=maximum*tick/4; y=top+height-height*tick/4
            svg.append(f'<line x1="{left}" x2="{left+width}" y1="{y}" y2="{y}" stroke="#ddd"/>')
            svg.append(f'<text x="{left-48}" y="{y+4}" font-size="11">{val:.1f}</text>')
        for i, pair in enumerate(pairs):
            x=left+i*103+12
            for j, s in enumerate(pair):
                val=s[key]; bar=height*val/maximum; bx=x+j*40
                color=('#0072b2','#d55e00')[j]
                svg.append(f'<rect x="{bx}" y="{top+height-bar}" width="34" height="{bar}" fill="{color}"/>')
                label=f'{val:.0f}' if key=='mean_client_ttft_ms' else f'{val:.1f}'
                svg.append(f'<text x="{bx}" y="{top+height-bar-5}" font-size="10">{label}</text>')
            s=pair[0]
            svg.append(f'<text x="{x}" y="{top+height+22}" font-size="12">D{s["cpu_gib"]:g}/S{s["staging_gib"]:g}</text>')
    svg += ['<text x="35" y="802" font-size="13">D=DRAM GiB, S=staging GiB. One run per condition; generated token counts and cache evolution can differ.</text>', '</g></svg>']
    save_chart(out,'comparison',svg)


def timelines(out, on, off, pairs):
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="1250">',
         '<rect width="1500" height="1250" fill="white"/><g font-family="sans-serif" font-size="12" fill="#222">',
         '<text x="35" y="27" font-size="22">DRAM prefetch OFF vs ON: occupancy and lookup hits over time</text>',
         '<text x="35" y="53">Occupancy: blue=OFF, orange=ON; solid=5s peak, dashed=time-weighted mean.</text>',
         '<text x="35" y="75">Hits: blue=DRAM, green=DAOS, orange=miss. Gaps=no lookups. X=seconds since first wave.</text>']
    for offstats,onstats in pairs:
        name=onstats['condition']; d=onstats['cpu_gib']; s=onstats['staging_gib']
        left,top=55+(8,4,2).index(d)*495, 135+(8,4).index(s)*535
        duration=max(offstats['elapsed_seconds'],onstats['elapsed_seconds'])
        bins=[]
        for root in (off,on):
            with (root/name/'timeline.csv').open() as stream:
                bins.append([{k:float(v) for k,v in r.items()} for r in csv.DictReader(stream)])
        svg.append(f'<text x="{left}" y="{top-15}" font-size="17">DRAM {d:g} GiB / staging {s:g} GiB</text>')
        for panel in range(3):
            y0=top+panel*162; height=112; width=415
            title=('Staging occupancy (%)','OFF: lookup fractions (%)','ON: lookup fractions (%)')[panel]
            svg.append(f'<text x="{left}" y="{y0}">{title}</text>')
            y0+=10
            for tick in (0,50,100):
                y=y0+height*(1-tick/100)
                svg.append(f'<line x1="{left}" x2="{left+width}" y1="{y}" y2="{y}" stroke="#ddd"/>')
                svg.append(f'<text x="{left-28}" y="{y+4}">{tick}</text>')
            for tick in range(5):
                svg.append(f'<text x="{left+width*tick/4}" y="{y0+height+16}">{duration*tick/4:.0f}</text>')
            series=[]
            if panel==0:
                for mode in range(2):
                    for key,dash in (('peak_staging_gib',''),('mean_staging_gib','4 3')):
                        series.append((bins[mode],key,('#0072b2','#d55e00')[mode],dash,100/s))
            else:
                for key,color in zip(('dram_chunks','daos_chunks','miss_chunks'),('#0072b2','#009e73','#d55e00')):
                    series.append((bins[panel-1],key,color,'',None))
            for rows,key,color,dash,scale in series:
                segments=[[]]
                for b in rows:
                    if scale is None and not b['lookup_chunks']:
                        if segments[-1]: segments.append([])
                        continue
                    value=b[key]*scale if scale is not None else b[key]/b['lookup_chunks']*100
                    segments[-1].append((left+width*b['t_seconds']/duration,y0+height*(1-value/100)))
                for segment in segments:
                    if not segment: continue
                    points=' '.join(f'{x:.2f},{y:.2f}' for x,y in segment)
                    svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="1.5" stroke-dasharray="{dash}"/>')
                    if len(segment)==1:
                        x,y=segment[0]; svg.append(f'<circle cx="{x}" cy="{y}" r="2" fill="{color}"/>')
    svg+=['<text x="35" y="1226">Aligned by elapsed time, not identical wall-clock arrivals: each wave starts after the preceding wave finishes.</text>','</g></svg>']
    save_chart(out,'timeline_comparison',svg)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--on',type=Path,required=True); p.add_argument('--off',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    audit=validate_pair(a.on,a.off)
    on={s['condition']:s for s in read(a.on/'analysis.json')}
    off={s['condition']:s for s in read(a.off/'analysis.json')}
    assert set(on)==set(off)
    pairs=[(off[k],on[k]) for k in on]
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'validation.json').write_text(json.dumps(audit,indent=2)+'\n')
    bars(a.output,pairs); timelines(a.output,a.on,a.off,pairs)
    rows=[]; detail=[]; data=[]
    for f,t in pairs:
        delta=100*(t['mean_client_ttft_ms']/f['mean_client_ttft_ms']-1)
        data.append(dict(condition=t['condition'],off=f,on=t,ttft_on_change_pct=delta))
        rows.append(f'| {t["cpu_gib"]:g} | {t["staging_gib"]:g} | {f["mean_client_ttft_ms"]:.0f} / {t["mean_client_ttft_ms"]:.0f} | {delta:+.1f}% | '
            f'{f["capacity_recompute_input_pct"]:.2f}% / {t["capacity_recompute_input_pct"]:.2f}% | '
            f'{f["peak_staging_gib"]:.3f} / {t["peak_staging_gib"]:.3f} | {f["elapsed_seconds"]:.2f} / {t["elapsed_seconds"]:.2f} |')
        for mode,s in (('OFF',f),('ON',t)):
            detail.append(f'| {s["cpu_gib"]:g} | {s["staging_gib"]:g} | {mode} | {s["dram_lookup_ratio"]*100:.2f}% | {s["daos_lookup_ratio"]*100:.2f}% | '
                f'{s["actual_cached_input_pct"]:.2f}% | {s["capacity_affected_requests"]} | {s["total_completion_tokens"]:,} |')
    report='''# DRAM 프리페치 OFF / ON 용량 비교

## 실험 조건

Qwen3-14B BF16, object 경로, 청크 128, 동시 요청 16. 동일한 DiscoveryBench 기록 입력 256개를 16개씩 묶어 재생한다. Python 도구는 재실행하지 않는다. 각 조건은 새 프로세스·빈 DRAM·새 DAOS namespace로 시작한다. 기존 DAOS 데이터를 삭제하거나 서버/OS 캐시를 flush하지 않았다.

OFF는 **DRAM→GPU staging 프리페치만 끈 것**이다. DRAM 캐시 보관, DAOS 비동기 프리페치, 비동기 DRAM 쓰기 미러·읽기 승격은 양쪽에서 유지한다. ON에는 soft watermark가 없고 실제 용량까지 할당을 시도한다. 물리적 용량 및 기존 DAOS serializer/mirror 예산은 유지한다.

기존 ON 6개 결과를 보존하고, 이후 OFF 6개를 같은 순서(D8S8→D8S4→D4S8→D4S4→D2S8→D2S4)로 추가 측정했다. 총 12조건, 3,072개 요청이다. 백엔드·재생 코드·native 라이브러리·입력은 동일하고, 실행 도구의 토글 선택 및 보고서 라벨만 확장했다. validation.json에 비교 검증을 기록했다.

## 주요 비교 — 각 칸은 OFF / ON

| DRAM GiB | staging GiB | 평균 TTFT ms | ON TTFT 변화 | 실패 재계산율 | staging 최대 GiB | 전체 시간 s |
|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

TTFT 변화의 양수는 ON에서 더 느렸다는 뜻이다. 실패 재계산율은 전체 입력 토큰 중 DAOS 읽기 할당 실패로 잃은 usable prefix 토큰 비율이다. Cold miss는 분자에서 제외하고 실제 usage.cached_tokens와 대조했다. 요청 실패 비율과는 다르다.

## hit와 실제 재사용

| DRAM GiB | staging GiB | 프리페치 | DRAM lookup hit | DAOS lookup hit | 실제 입력 재사용률 | 실패 영향 요청 /256 | 생성 토큰 |
|---:|---:|---|---:|---:|---:|---:|---:|
'''+ '\n'.join(detail)+'''

Lookup hit는 존재 확인 결과이며 성공적인 데이터 재사용과 다르다. 같은 입력이라도 실행 타이밍·읽기 성공·승격·퇴출에 따라 캐시 상태와 hit 비율이 달라질 수 있다.

## 그래프

- [핵심 지표 OFF/ON 비교](comparison.png)
- [시간별 staging 점유 및 hit 비교](timeline_comparison.png)

시간 그래프: 열=DRAM 8/4/2, 행=staging 8/4. 각 칸 위는 staging(파랑 OFF, 주황 ON; 실선 5초 최대, 점선 시간 가중 평균). 가운데는 OFF hit, 아래는 ON hit(파랑 DRAM, 초록 DAOS, 주황 miss). X축은 첫 묶음 이후 초다. 높은 최대치가 계속 유지된다는 뜻은 아니다.

## 해석 한계

조건별 1회 측정이며 ON/OFF를 교차 반복한 실험은 아니다. 입력 목록·묶음 구성은 같지만 이전 묶음 완료 후 다음 묶음을 보내므로 벽시계 도착 시각은 다르다. EOS를 허용해 생성량도 다르므로 전체 시간 차이를 순수 I/O 개선량으로 해석하지 않는다. OFF에서도 DAOS 읽기와 저장은 staging을 사용한다. ON/OFF 차이는 프리페치 토글이 캐시 상태 변화에 미친 영향까지 포함한 구현 전체의 관측 결과이며 모든 지연을 DRAM staging 점유 하나로 분해한 결과는 아니다.
'''
    report+=f'\n원본 결과: [ON 보고서]({a.on.resolve()}/RESULT_KO.md), [OFF 보고서]({a.off.resolve()}/RESULT_KO.md).\n'
    (a.output/'RESULT_KO.md').write_text(report)
    (a.output/'analysis.json').write_text(json.dumps(data,indent=2)+'\n')
    print('\n'.join(rows))


if __name__=='__main__':
    main()
