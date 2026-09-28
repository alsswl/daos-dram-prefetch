#!/usr/bin/env python3
"""Validate and report three fresh-process OFF/ON repetitions."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import yaml

from analyze_discovery_staging import summarize_case, plot
from report_capacity_matrix import attribute_recomputation, save_chart
from staging_mixed_pressure import read_events


def read(p): return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser(); p.add_argument('folder',type=Path)
    p.add_argument('--completed-only',action='store_true',help='Report completed pairs and disclose incomplete cases')
    a=p.parse_args()
    folder=a.folder.resolve(); plan=read(folder/'plan.json'); records=read(folder/'requests.json')
    rolling=plan.get('arrival_mode','waves')=='rolling'
    if rolling and a.completed_only:
        p.error('Rolling report requires all planned runs complete; do not mix partial runs into the summary')
    assert a.completed_only or read(folder/'status.json')['status']=='completed'
    stats_all=[]; configs=[]; native=[]; commands=[]; namespaces=[]; audit={}
    for spec in plan['cases']:
        case=folder/spec['name']
        if not (case/'status.json').exists() or read(case/'status.json')['status']!='completed':
            assert a.completed_only
            log=(case/'server.log').read_text(errors='replace')
            audit[spec['name']]=dict(excluded=True,reason='incomplete run',
                nospace_errors=log.count('daosgdr_put failed: rc=-1007'),
                saved_requests=len(read(case/'replay_calls.json')))
            continue
        calls=read(case/'replay_calls.json')
        assert 'LMCache ERROR' not in (case/'server.log').read_text(errors='replace'), 'Storage error: invalid benchmark'
        assert len(calls)==256 and not any('error' in c for c in calls)
        assert [(c['index'],c['prompt_sha256']) for c in calls]==[(r['index'],r['prompt_sha256']) for r in records]
        initial=read(case/'initial_sample.json'); final=read(case/'final_sample.json')
        assert initial['used_bytes']==initial['cpu_hot_bytes']==initial['daos_puts']==0
        assert final['used_bytes']==final['dram_mirror']['pending_bytes']==final['dram_mirror']['errors']==0
        cfg=yaml.safe_load((case/'config.yaml').read_text()); extra=cfg['extra_config']
        assert cfg['max_local_cpu_size']==extra['daosgds.gpu_buffer_gb']==8
        assert extra.pop('daosgds.dram_prefetch') is spec['prefetch']
        assert extra['daosgds.dram_prefetch_policy']=='capacity'
        namespaces.append(extra.pop('daosgds.object_namespace')); extra.pop('daosgds.root')
        configs.append(cfg); native.append(read(case/'native_maps.json')); commands.append(read(case/'command.json'))
        if rolling:
            assert not (case/'waves.json').exists()
            assert read(case/'phases.json')[0]['arrival_mode']=='rolling'
            boundaries=sorted([(c['start_ns'],1) for c in calls]+[(c['end_ns'],-1) for c in calls])
            active=peak=0
            for _,delta in boundaries:
                active+=delta; peak=max(peak,active)
                assert 0<=active<=8
            assert active==0 and peak==8
            assert max(c['start_ns'] for c in calls[:8]) < min(c['start_ns'] for c in calls[8:])
            assert min(c['start_ns'] for c in calls[8:]) < max(c['end_ns'] for c in calls[:8]), 'Unexpected initial wave barrier'
        else:
            waves=read(case/'waves.json'); assert len(waves)==32 and all(w['requests']==8 for w in waves)
        att=attribute_recomputation(calls,read_events(case),128)
        assert not any(r['attribution_issues'] or r['unattributed_shortfall_tokens'] or r['other_failed_chunks'] for r in att)
        (case/'recomputation_by_request.json').write_text(json.dumps(att,indent=2)+'\n')
        s,bins,phases=summarize_case(case,8)
        assert not s['negative_ref_log_count'] and not s['llm_call_errors']
        assert sum(r['capacity_failed_chunks'] for r in att)==s['gpu_full_log_count']
        if not spec['prefetch']: assert final['cpu_prefetch'] is None and s['cpu_staged_requests']==0
        else: assert final['cpu_prefetch']['watermark_rejections']==0
        total=sum(c['prompt_tokens'] for c in calls)
        lost=sum(r['capacity_recomputed_tokens'] for r in att)
        s.update(prefetch=spec['prefetch'],repeat=spec['repeat'],total_prompt_tokens=total,
                 total_completion_tokens=sum(c['completion_tokens'] for c in calls),
                 capacity_recomputed_tokens=lost,capacity_recompute_input_pct=100*lost/total,
                 capacity_affected_requests=sum(r['capacity_recomputed_tokens']>0 for r in att),
                 actual_cached_input_pct=100*sum(c['cached_tokens'] for c in calls)/total,
                 prefetch_counters=final['cpu_prefetch'],request_throughput=256/s['elapsed_seconds'])
        (case/'analysis.json').write_text(json.dumps(s,indent=2)+'\n')
        stats_all.append(s)
        plot(case,[(s,bins,phases)],title=f'D8/S8, concurrency8, DRAM prefetch {"ON" if spec["prefetch"] else "OFF"}',
             phase_label='requests',caption=f'256 fixed inputs; {"rolling concurrency8" if rolling else "wave size8"}; fresh process and namespace; 5s peak/mean occupancy.')
        audit[spec['name']]=dict(started_empty=True,requests=256,arrival_mode='rolling' if rolling else 'waves',
                               attribution_validated=True,final_drained=True)
    assert all(c==configs[0] for c in configs) and all(n==native[0] for n in native) and all(c==commands[0] for c in commands)
    assert len(set(namespaces))==len(stats_all)
    for location,root in [('archived',folder/'executed_sources'),('current',Path(__file__).resolve().parent)]:
        audit[location+'_source_mismatches']=[n for n,h in plan['source_sha256'].items() if hashlib.sha256((root/n).read_bytes()).hexdigest()!=h]
        assert not audit[location+'_source_mismatches'] or (location=='current' and a.completed_only and
            audit[location+'_source_mismatches']==['report_prefetch_c8.py'])
    audit['same_config_native_commands']=True
    (folder/'validation.json').write_text(json.dumps(audit,indent=2)+'\n')
    (folder/'analysis.json').write_text(json.dumps(stats_all,indent=2)+'\n')
    paired_repeats=[rep for rep in range(1,4) if {s['prefetch'] for s in stats_all if s['repeat']==rep}=={True,False}]
    aggregate={}
    for enabled in (False,True):
        mode='ON' if enabled else 'OFF'; group=[s for s in stats_all if s['prefetch']==enabled and s['repeat'] in paired_repeats]
        aggregate[mode]={k:dict(mean=statistics.mean(s[k] for s in group),
                               stdev=statistics.stdev(s[k] for s in group),
                               min=min(s[k] for s in group),max=max(s[k] for s in group))
                         for k in ('mean_client_ttft_ms','elapsed_seconds','request_throughput','peak_staging_gib',
                                   'dram_lookup_ratio','daos_lookup_ratio','capacity_recompute_input_pct','actual_cached_input_pct')}
    (folder/'aggregate.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="950" height="470"><rect width="950" height="470" fill="white"/>',
         '<g font-family="sans-serif" fill="#222"><text x="30" y="30" font-size="21">D8/S8, concurrency8: DRAM prefetch OFF vs ON</text>',
         '<text x="30" y="55" font-size="14">Mean TTFT (ms), each bar = 256 requests in one fresh process. Blue=OFF, orange=ON.</text>']
    maximum=max(s['mean_client_ttft_ms'] for s in stats_all)*1.15
    for tick in range(5):
        y=360-tick*260/4
        svg.extend([f'<line x1="70" x2="910" y1="{y}" y2="{y}" stroke="#ddd"/>',f'<text x="15" y="{y+4}">{maximum*tick/4:.0f}</text>'])
    for rep in paired_repeats:
        for j,enabled in enumerate((False,True)):
            s=next(s for s in stats_all if s['repeat']==rep and s['prefetch']==enabled)
            v=s['mean_client_ttft_ms']; x=120+(rep-1)*275+j*90; h=260*v/maximum
            color=('#0072b2','#d55e00')[j]
            svg.extend([f'<rect x="{x}" y="{360-h}" width="70" height="{h}" fill="{color}"/>',
                        f'<text x="{x}" y="{350-h}">{v:.1f}</text>'])
        svg.append(f'<text x="{145+(rep-1)*275}" y="390">Repeat {rep}</text>')
    caption=('Completed pairs only; repeat3 ON aborted due to DAOS storage NOSPACE (not staging exhaustion).'
             if a.completed_only else 'Order: OFF/ON, ON/OFF, OFF/ON. DAOS prefetch and DRAM storage enabled in all cases.')
    svg+=[f'<text x="30" y="440" font-size="13">{caption}</text></g></svg>']
    save_chart(folder,'comparison',svg)
    rows=[]
    for s in stats_all:
        rows.append(f'| {s["repeat"]} | {"ON" if s["prefetch"] else "OFF"} | {s["mean_client_ttft_ms"]:.2f} | {s["elapsed_seconds"]:.2f} | {s["request_throughput"]:.3f} | {s["peak_staging_gib"]:.3f} | {s["dram_lookup_ratio"]*100:.2f}% | {s["daos_lookup_ratio"]*100:.2f}% | {s["capacity_recompute_input_pct"]:.2f}% | {s["total_completion_tokens"]} |')
    aggrows=[]
    for mode,v in aggregate.items():
        aggrows.append(f'| {mode} | {v["mean_client_ttft_ms"]["mean"]:.2f} ± {v["mean_client_ttft_ms"]["stdev"]:.2f} | {v["elapsed_seconds"]["mean"]:.2f} ± {v["elapsed_seconds"]["stdev"]:.2f} | {v["request_throughput"]["mean"]:.3f} | {v["capacity_recompute_input_pct"]["mean"]:.2f}% |')
    delta=100*(aggregate['ON']['mean_client_ttft_ms']['mean']/aggregate['OFF']['mean_client_ttft_ms']['mean']-1)
    report='''# DRAM 8GiB / staging 8GiB / 동시 요청 8: 프리페치 비교

Qwen3-14B BF16, object 경로, 청크 128. 동일한 DiscoveryBench 기록 입력 256개를 8개씩 묶어 재생했다. Python 도구를 재실행한 full agentic 실험은 아니다. 매 실행은 새 vLLM 프로세스, 빈 DRAM, 새 DAOS namespace에서 시작한다. 서버/OS 캐시를 flush하지 않았다.

OFF/ON은 DRAM→GPU staging 프리페치만 바꾼다. DRAM 보관, DAOS 프리페치, 비동기 쓰기 미러·읽기 승격은 모두 유지한다. ON은 soft watermark 없이 실제 GPU 할당을 시도한다. vLLM max-num-seqs는 기존 16을 유지하고 클라이언트 동시 요청만 8로 제한했다.

각 조건 3회, 총 1,536요청. 실행 순서는 OFF→ON→ON→OFF→OFF→ON이다. 생성은 temperature0, seed0, 최대2048토큰, EOS 및 Observation 종료를 허용한다. 모델 로딩은 시간에서 제외하고 첫 cold 요청부터 마지막 요청 완료까지 측정했다. 다음 묶음은 이전 묶음 완료 후 투입하므로 벽시계 도착 시각은 고정하지 않았다.

## 반복별 결과

| 반복 | 프리페치 | 평균 TTFT ms | 전체 시간 s | 요청/s | staging 최대 GiB | DRAM hit | DAOS hit | 실패 재계산율 | 생성 토큰 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

## 3회 요약

±는 요청별 분산이 아니라 **새 프로세스 3개에서 얻은 평균값의 표본 표준편차**다. 신뢰구간은 아니다.

| 프리페치 | 평균 TTFT ms (평균 ± SD) | 전체 시간 s (평균 ± SD) | 평균 요청/s | 평균 실패 재계산율 |
|---|---:|---:|---:|---:|
'''+ '\n'.join(aggrows)+f'\n\nON의 평균 TTFT 변화: {delta:+.2f}% (음수면 개선).\n'+'''
## 지표와 한계

Hit 비율은 lookup 청크 기준이다. 실패 재계산율은 전체 입력 토큰 중 DAOS GPU 할당 실패로 잃은 usable prefix 토큰 비율이며 cold miss를 분자에서 제외했다. 실제 usage.cached_tokens로 검증했다. 프리페치가 캐시 승격·퇴출 시점에 미친 영향까지 포함하며 고정된 hit trace 실험은 아니다. 생성량 차이 때문에 전체 시간 차이를 순수 I/O 개선량으로 해석하지 않는다. 3회 반복은 변동 확인을 위한 작은 표본이며 보편적인 성능 우위를 증명하지 않는다.

- [반복별 TTFT 그래프](comparison.png)
'''+ '\n'.join(f'- [{s["condition"]} staging·hit 시간 그래프]({s["condition"]}/timeline.png)' for s in stats_all)
    if rolling:
        report=report.replace('동일한 DiscoveryBench 기록 입력 256개를 8개씩 묶어 재생했다.',
            '동일한 DiscoveryBench 기록 입력 256개를 재생했다. 처음 최대 8개를 보내고 하나가 완료되면 다음 입력을 즉시 투입한다. 묶음 전체 완료를 기다리지 않는다.')
        report=report.replace('다음 묶음은 이전 묶음 완료 후 투입하므로 벽시계 도착 시각은 고정하지 않았다.',
            '요청 완료에 맞춰 다음 입력을 투입하는 closed-loop 방식이다. 총 256개와 최대 동시성 8은 고정하지만 서버 도착 시각과 완료 순서는 OFF/ON에서 달라질 수 있다. 마지막 요청들이 끝나는 구간은 새 입력이 없어 동시성이 감소한다. 워커 인계에 짧은 공백이 있어 정확히 모든 순간에 8개인 것은 아니다.')
    if a.completed_only:
        report=report.replace('각 조건 3회, 총 1,536요청.',
            '각 조건 3회(1,536요청)를 계획했으나 마지막 ON에서 DAOS 저장 오류가 발생해 중단했다. 정상 완료는 5회(1,280요청), 비교 가능한 완전한 쌍은 2회다. 마지막 ON의 저장된 224요청은 통계에서 제외한다.')
        report=report.replace('## 3회 요약','## 정상 완료된 2쌍 요약 (r1, r2)')
        report=report.replace('새 프로세스 3개','새 프로세스 2개')
        report=report.replace('3회 반복은 변동 확인을 위한 작은 표본','완료된 2쌍은 변동 확인을 위한 작은 표본')
        report+='''

## 중단 사유와 미완료 범위

r3_on에서 `daos_obj_update_gpu rc=-1007` / `DER_NOSPACE` 저장 대상 공간 부족 오류가 발생했다. GPU staging 할당 실패와는 다르다. 요청을 추가 투입하지 않도록 실행을 중단하고 해당 실험 서버를 종료했다. 공용 DAOS 데이터는 삭제하지 않았다. r1/r2의 OFF/ON과 r3_off 로그에는 ERROR가 없었다.

r3_off는 반복별 표에 보존하지만 대응하는 r3_on이 비정상이므로 평균·표준편차·비교 그래프에서는 **r1/r2의 완전한 2쌍만** 사용했다. 실패한 실행을 정상 표본으로 섞거나 성능이 나쁘다는 이유로 제외한 것이 아니라, 명시적인 저장 오류로 실험 조건이 깨졌기 때문에 분리했다. 3쌍 비교는 미완료다.

중단 후 pool query에서 NVMe 총 2.0TB, 표시 free 132GB, target별 최소 6.2GB가 확인됐다. 논리적 free 합계가 남아 있어도 실제 저장 호출은 공간 부족으로 거부됐다. 오류를 발생시킨 target 및 예약 공간·하위 파일시스템 상태는 아직 확인하지 않았으므로 세부 원인을 단정하지 않는다. 서버 공간 확보 후 r3 OFF/ON 쌍을 다시 측정하는 것이 좋다. 삭제에는 별도 사용자 승인이 필요하다.

후처리 보고서만 미완료 실행을 구분하도록 수정했다. 실행 당시 소스 스냅샷은 보존했고 백엔드·재생 코드는 변경하지 않았다.
'''
    (folder/'RESULT_KO.md').write_text(report+'\n')
    print('\n'.join(rows+aggrows))


if __name__=='__main__': main()
