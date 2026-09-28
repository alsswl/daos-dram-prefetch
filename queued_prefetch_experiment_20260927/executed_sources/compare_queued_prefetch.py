#!/usr/bin/env python3
"""Cold/warm replay comparison: original wait vs cancel-queued-at-retrieve.

DRAM prefetch is ON in every arm. Optional early_wait isolates early scheduler
readiness from actual cancellation. No queue padding, delays or cache injection.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess

import yaml

import compare_e2e as common
from cold_warm_prefetch import run_case
from report_cold_warm_prefetch import select_events
from report_prefetch_timing import join, stats
from staging_mixed_pressure import read_events


MODES = {
    'wait': dict(cancel_queued=False, early_ready=False),
    'cancel': dict(cancel_queued=True, early_ready=True),
    'early_wait': dict(cancel_queued=False, early_ready=True),
}


def read(path):
    return json.loads(path.read_text())


def cases_for(concurrencies, repeats, control):
    modes = ['wait', 'cancel'] + (['early_wait'] if control else [])
    cases = []
    # Rotate/reverse condition order across repetitions; no result-based reruns.
    for rep in range(1, repeats + 1):
        order = modes[rep % len(modes):] + modes[:rep % len(modes)]
        if rep % 2 == 0: order = list(reversed(order))
        for concurrency in concurrencies:
            for mode in order:
                cases.append(dict(name=f'r{rep}_c{concurrency}_{mode}', repeat=rep,
                                  concurrency=concurrency, mode=mode))
    return cases


def report(root):
    plan, records = read(root/'plan.json'), read(root/'requests.json')
    assert read(root/'status.json')['status'] == 'completed', 'Incomplete experiment, no final report'
    expected = [(r['index'], r['prompt_sha256']) for r in records]
    summaries, details, configs, natives, namespaces = [], {}, [], [], []
    for spec in plan['cases']:
        case = root/spec['name']
        assert read(case/'status.json')['status'] == 'completed'
        cfg = yaml.safe_load((case/'config.yaml').read_text())
        ec = cfg['extra_config']
        assert ec['daosgds.dram_prefetch'] is True
        assert ec.pop('daosgds.dram_prefetch_cancel_queued') == MODES[spec['mode']]['cancel_queued']
        assert ec.pop('daosgds.dram_prefetch_early_ready') == MODES[spec['mode']]['early_ready']
        namespaces.append(ec.pop('daosgds.object_namespace')); ec.pop('daosgds.root')
        configs.append(cfg); natives.append(read(case/'native_maps.json'))
        events = read_events(case)
        assert len({e['pid'] for e in events}) == 1
        initial = read(case/'initial_sample.json')
        assert initial['used_bytes'] == initial['cpu_hot_bytes'] == initial['daos_puts'] == 0
        seen_ids = set()
        for phase_name in ('cold', 'warm'):
            folder = case/phase_name
            calls, phase = read(folder/'replay_calls.json'), read(folder/'phase.json')
            assert len(calls) == len(records) and not any('error' in c for c in calls)
            assert [(c['index'], c['prompt_sha256']) for c in calls] == expected
            ids = {c['server_request_id'] for c in calls}
            assert len(ids) == len(records) and not (ids & seen_ids)
            seen_ids.update(ids)
            before, after = read(folder/'initial_sample.json'), read(folder/'final_sample.json')
            assert after['used_bytes'] == after['dram_mirror']['pending_bytes'] == 0
            assert after['dram_mirror']['errors'] == 0
            assert after['cpu_prefetch']['copy_errors'] == after['cpu_prefetch']['deferred_pending_batches'] == 0
            assert before == (initial if phase_name == 'cold' else read(case/'cold/final_sample.json'))
            selected = select_events(events, calls)
            rows = join(calls, selected)
            assert not any(r['other_failed_chunks'] for r in rows)
            decisions = [e for e in selected if e['event'] == 'cpu_prefetch_retrieve_decision']
            pf = {k: v-before['cpu_prefetch'].get(k, 0) for k, v in after['cpu_prefetch'].items()}
            for decision in ('cancelled_queued', 'ready_gpu', 'waited_gpu', 'capacity_cpu'):
                assert sum(e['decision'] == decision for e in decisions) == pf['retrieve_'+decision]
            candidates = sum(e['queried_chunks'] for e in selected
                             if e['event'] == 'tier_lookup' and e['tier'] == 'dram')
            dram, daos = sum(r['dram_lookup_chunks'] for r in rows), sum(r['daos_lookup_chunks'] for r in rows)
            assert 0 <= dram+daos <= candidates
            scoped = [e for e in events if phase['start_ns'] <= e['time_ns'] <= phase['end_ns']]
            s = dict(**spec, phase=phase_name, requests=len(calls),
                     ttft=stats(r['ttft_ms'] for r in rows),
                     retrieve=stats(r.get('retrieve_ms') for r in rows),
                     worker_queue=stats(r.get('queue_ms') for r in rows),
                     resolve_wait=stats((e['resolve_end_ns']-e['resolve_start_ns'])/1e6 for e in decisions),
                     elapsed_seconds=(phase['end_ns']-phase['start_ns'])/1e9,
                     completion_tokens=sum(c['completion_tokens'] for c in calls),
                     computed_input_tokens=sum(r['computed_prompt_tokens'] for r in rows),
                     capacity_recomputed_tokens=sum(r['capacity_recomputed_tokens'] for r in rows),
                     dram_hit_pct=100*dram/candidates if candidates else 0,
                     daos_hit_pct=100*daos/candidates if candidates else 0,
                     peak_staging_gib=max(e['used_bytes'] for e in scoped)/2**30,
                     counters=pf)
            common.dump(folder/'timing_by_request.json', rows)
            common.dump(folder/'retrieve_decisions.json', decisions)
            summaries.append(s); details[(spec['repeat'], spec['concurrency'], spec['mode'], phase_name)] = rows
    assert len(namespaces) == len(set(namespaces))
    assert all(c == configs[0] for c in configs)
    assert all(n == natives[0] for n in natives)
    for name, digest in plan['source_sha256'].items():
        assert hashlib.sha256((root/'executed_sources'/name).read_bytes()).hexdigest() == digest
    paired = []
    for rep in range(1, plan['repeats']+1):
        for c in plan['concurrency']:
            for phase in ('cold', 'warm'):
                for reference in ('wait', 'early_wait'):
                    if (rep, c, reference, phase) not in details: continue
                    pairs = list(zip(details[(rep,c,reference,phase)], details[(rep,c,'cancel',phase)], strict=True))
                    same_cache = [(x,y) for x,y in pairs if x['cached_tokens'] == y['cached_tokens']]
                    same_tier = [(x,y) for x,y in same_cache if (x['dram_lookup_chunks'],x['daos_lookup_chunks']) ==
                                 (y['dram_lookup_chunks'],y['daos_lookup_chunks'])]
                    paired.append(dict(repeat=rep, concurrency=c, phase=phase, reference=reference,
                                       same_cached_requests=len(same_cache), same_tier_requests=len(same_tier),
                                       total_requests=len(pairs)))
    common.dump(root/'summary.json', summaries); common.dump(root/'paired_checks.json', paired)
    common.dump(root/'validation.json', dict(same_config_except_switches=True,
        same_native_libraries=True, unique_namespaces=True, same_inputs=True,
        same_worker_between_cold_warm=True, all_buffers_drained=True))
    lines = ['# 대기 중 프리페치 취소 비교', '',
        '모든 조건에서 DRAM 프리페치는 ON이다. wait=기존 완료 후 준비 알림, '
        'cancel=조기 준비 알림+retrieve 소비 시 대기 작업 취소, early_wait=조기 준비 알림만 적용.', '',
        '**wait vs cancel은 준비 알림 시점과 취소 정책을 함께 비교한다. 취소만의 효과는 early_wait vs cancel을 확인한다.**', '',
        '|반복|동시 요청|방식|단계|평균 TTFT ms|p95 ms|큐 취소|진행 중 대기|이미 GPU 준비|공간 실패 CPU|DRAM hit %|DAOS hit %|새 계산 토큰|',
        '|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in summaries:
        p = s['counters']
        lines.append(f"|{s['repeat']}|{s['concurrency']}|{s['mode']}|{s['phase']}|{s['ttft']['mean']:.2f}|"
                     f"{s['ttft']['p95']:.2f}|{p['retrieve_cancelled_queued']}|{p['retrieve_waited_gpu']}|"
                     f"{p['retrieve_ready_gpu']}|{p['retrieve_capacity_cpu']}|{s['dram_hit_pct']:.2f}|"
                     f"{s['daos_hit_pct']:.2f}|{s['computed_input_tokens']}|")
    lines += ['', 'wait에서는 retrieve 결정 이벤트를 만들지 않으므로 그 열의 0은 GPU 사용이 없다는 뜻이 아니다.',
              '작업 대기 시간은 실행된 작업만 포함한다. 취소된 작업은 별도 retrieve_decisions.json에 기록한다.',
              '취소 건수가 0이면 이 워크로드에서는 대기 취소 효과를 검증하지 못한 것이다. 인위적인 큐 지연은 넣지 않았다.',
              '완료 기반 rolling이므로 실제 도착 시각과 생성량은 달라질 수 있다. paired_checks.json의 실제 재사용량도 확인한다.', '',
              '## Warm 반복 평균 (각 실행의 평균 TTFT 기준)', '',
              '|동시 요청|방식|실행 수|평균 ms|실행 간 표준편차 ms|', '|---:|---|---:|---:|---:|']
    for c in plan['concurrency']:
        for mode in MODES:
            values = [s['ttft']['mean'] for s in summaries if s['phase']=='warm' and s['concurrency']==c and s['mode']==mode]
            if values:
                sd = f'{statistics.stdev(values):.2f}' if len(values)>1 else '미측정'
                lines.append(f'|{c}|{mode}|{len(values)}|{statistics.mean(values):.2f}|{sd}|')
    (root/'RESULT_KO.md').write_text('\n'.join(lines)+'\n')
    with (root/'summary.csv').open('w') as stream:
        fields = ['repeat','concurrency','mode','phase','requests','ttft_mean_ms','ttft_p95_ms',
                  'elapsed_seconds','computed_input_tokens','capacity_recomputed_tokens','completion_tokens',
                  'dram_hit_pct','daos_hit_pct','peak_staging_gib','cancelled_queued']
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for s in summaries:
            row={k:s[k] for k in fields if k in s}
            row.update(ttft_mean_ms=s['ttft']['mean'], ttft_p95_ms=s['ttft']['p95'],
                       cancelled_queued=s['counters']['retrieve_cancelled_queued'])
            writer.writerow(row)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--concurrency', type=int, nargs='+', choices=(1,4,8,16), default=[8,16])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--with-early-wait', action='store_true')
    p.add_argument('--port', type=int, default=8017)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--report-only', action='store_true')
    a=p.parse_args()
    if a.repeats < 1 or len(set(a.concurrency)) != len(a.concurrency): p.error('positive repeats and unique concurrency required')
    root=a.output.resolve()
    if a.report_only:
        report(root); return
    a.model, a.max_model_len='Qwen/Qwen3-14B',32768
    root.mkdir(parents=True,exist_ok=False)
    records=read(common.ROOT/'discovery_capacity_matrix_20260927/requests.json')
    assert len(records)==256
    assert all(r['index']==i and hashlib.sha256(r['prompt'].encode()).hexdigest()==r['prompt_sha256'] for i,r in enumerate(records))
    common.dump(root/'requests.json',records)
    cases=cases_for(a.concurrency,a.repeats,a.with_early_wait)
    names=['compare_queued_prefetch.py','cold_warm_prefetch.py','discovery_fixed_replay.py',
           'discovery_rolling_replay.py','staging_mixed_pressure.py','compare_e2e.py','run_vllm.sh',
           'libdaosgdr.so','lmcache_config_daosgds_async_dram.yaml','report_prefetch_timing.py',
           'report_cold_warm_prefetch.py','tests/test_queued_prefetch.py']
    names += [str(x.relative_to(common.ROOT)) for x in sorted((common.ROOT/'lmcache_daos').glob('*.py'))]
    hashes={}
    for name in names:
        dest=root/'executed_sources'/name; dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(common.ROOT/name,dest); hashes[name]=hashlib.sha256(dest.read_bytes()).hexdigest()
    common.dump(root/'plan.json',dict(cases=cases,repeats=a.repeats,concurrency=a.concurrency,
        model=a.model,cpu_gib=8,staging_gib=8,chunk_tokens=128,requests_per_phase=256,
        source_sha256=hashes,notes=[
            'DRAM prefetch ON in ALL arms; toggle is cancellation, not prefetch itself.',
            'wait vs cancel includes changed scheduler readiness. early_wait isolates cancellation.',
            'Physical 8GiB pool, no soft watermark, no artificial queue blocking or delays.',
            'Each case: fresh process/DRAM/DAOS namespace; cold256 then same-process warm256.',
            'DAOS namespace isolation, not destructive deletion or OS/server cache flush.',
            'Same recorded prompts, rolling arrivals; EOS/output and arrival timestamps can differ.',
            'This is recorded DiscoveryBench input replay, not full agent/tool execution.']))
    if a.dry_run:
        common.dump(root/'status.json',dict(status='dry_run')); print(root); return
    os.environ['DAOS_GDS_PREFETCH_TIMING']='1'
    try:
        cmd=[str(common.ROOT/'run_vllm.sh'),str(common.ROOT/'venv/bin/python3'),
             str(common.ROOT/'tests/object_gpu_roundtrip.py'),'--size-mib','20']
        result=subprocess.run(cmd,cwd=common.ROOT,env=dict(os.environ,DAOSGDS_TRANSPORT='object'),
                              capture_output=True,text=True,timeout=120)
        (root/'storage_preflight.log').write_text(result.stdout+result.stderr)
        if result.returncode: raise RuntimeError('DAOS preflight failed')
        for spec in cases:
            # Reject source changes during a long repeated experiment.
            assert all(hashlib.sha256((common.ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
            common.dump(root/'status.json',dict(status='running',case=spec['name']))
            a.cancel_queued=MODES[spec['mode']]['cancel_queued']
            a.early_ready=MODES[spec['mode']]['early_ready']
            folder=root/spec['name']; folder.mkdir()
            run_case(a,folder,records,spec['concurrency'],True)
        common.dump(root/'status.json',dict(status='completed'))
        report(root)
    except BaseException as exc:
        common.dump(root/'status.json',dict(status='failed',error=repr(exc))); raise


if __name__=='__main__':
    main()
