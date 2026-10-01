#!/usr/bin/env python3
"""DiscoveryBench recorded replay: OFF, ungated ON, 60%-occupancy gated ON."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import yaml

import compare_e2e as common
from cold_warm_prefetch import run_case
from report_cold_warm_prefetch import select_events
from report_prefetch_timing import join, stats
from report_prefetch_capacity_sweep import timeline
from staging_mixed_pressure import read_events

ROOT = common.ROOT
HISTORICAL = ROOT/'prefetch_cold_warm_d8_s8_20260927'
MODES = ('off', 'on', 'gate60')


def read(path):
    return json.loads(path.read_text())


def cases():
    return [dict(name=f'c{c}_{m}', concurrency=c, mode=m, prefetch=m != 'off',
                 occupancy_stop_ratio=.6 if m == 'gate60' else None)
            for c, order in [(16, tuple(reversed(MODES))), (8, MODES)] for m in order]


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    records = read(HISTORICAL/'requests.json')
    assert len(records) == 256
    assert all(r['index'] == i and hashlib.sha256(r['prompt'].encode()).hexdigest() == r['prompt_sha256']
               for i, r in enumerate(records))
    shutil.copy2(HISTORICAL/'requests.json', root/'requests.json')
    shutil.copy2(HISTORICAL/'summary.json', root/'historical_summary.json')
    names = ['discovery_occupancy_gate.py', 'cold_warm_prefetch.py', 'discovery_fixed_replay.py',
             'discovery_rolling_replay.py', 'staging_mixed_pressure.py', 'compare_e2e.py',
             'report_cold_warm_prefetch.py', 'report_prefetch_timing.py', 'report_capacity_matrix.py',
             'report_prefetch_capacity_sweep.py', 'run_vllm.sh', 'libdaosgdr.so', 'libdaosgdr.c',
             'cleanup_experiment_cache.py', 'list_experiment_dkeys', 'lmcache_config_daosgds_async_dram.yaml']
    names += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'lmcache_daos').glob('*.py'))]
    hashes = {}
    for name in names:
        dest = root/'executed_sources'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
        hashes[name] = hashlib.sha256(dest.read_bytes()).hexdigest()
    common.dump(root/'plan.json', dict(cases=cases(), model='Qwen/Qwen3-14B',
        cpu_gib=8, staging_gib=8, workers=1, cancel_queued=False, early_ready=False,
        requests_per_phase=256, phases=['cold', 'warm'], repetitions=1,
        source_sha256=hashes, request_sha256=hashlib.sha256((root/'requests.json').read_bytes()).hexdigest(),
        historical=str(HISTORICAL), notes=[
            'Same frozen DiscoveryBench prompt replay as the historical cold/warm experiment; not full agentic.',
            'Each case: fresh process, empty DRAM/staging and independent DAOS namespace.',
            'Warm follows cold in the same process after drain, reusing the same caches.',
            'One worker, no early readiness or queued cancellation: match historical settings explicitly.',
            'Gate tests current total occupancy >=60% atomically just before allocation, not projected occupancy.',
            'Rejected batches return original CPU objects. No eviction, miss, DAOS restriction, or copy cancellation.',
            'Threshold is not a hard cap. Admitted batches and DAOS/store allocations can exceed 60%.',
            'Default null disables gate. Existing physical capacity fallback is unchanged.',
            'Each completed case namespace is deleted after process exit, never the shared object/container.',
            'One trial: historical comparison is context, contemporaneous OFF/ON/gate isolates policy better.']))
    common.dump(root/'status.json', dict(status='prepared', completed=[]))


def cleanup(case):
    """Executed only in a DAOS-native child after the case server has exited."""
    import cleanup_experiment_cache as cleaner
    case = case.resolve()
    plan = read(case.parent/'plan.json')
    assert case.parent.parent == ROOT and case.name in {s['name'] for s in plan['cases']}
    assert read(case/'status.json')['status'] == 'completed'
    cfgpath = case/'config.yaml'
    ec = yaml.safe_load(cfgpath.read_text())['extra_config']
    ns = ec['daosgds.object_namespace']
    assert re.fullmatch(r'minji-cold-warm-[0-9a-f]{32}:', ns)
    assert ec['daosgds.pool'] == 'discospool' and ec['daosgds.container'] == 'kvcache'
    assert ec['daosgds.transport'] == 'object'
    assert Path(ec['daosgds.object_library']).resolve() == ROOT/'libdaosgdr.so'
    rows = [dict(namespace=ns, sources=[str(cfgpath.relative_to(ROOT))],
                 pool='discospool', container='kvcache', experiments=[case.parent.name])]
    cleaner.eligible = lambda: rows
    folder = case/'namespace_cleanup'
    if (folder/'result.json').exists():
        result = read(folder/'result.json')
        assert result['remaining_targets'] == 0 and result['preserved_set_unchanged']
        return
    assert not folder.exists(), 'Partial cleanup requires fresh audited manifest'
    sys.argv = ['cleanup', '--output', str(folder)]
    cleaner.main()
    sys.argv.append('--execute')
    cleaner.main()


def summarize_case(root, spec):
    case = root/spec['name']
    events = read_events(case)
    assert len({e['pid'] for e in events}) == 1
    expected = [(r['index'], r['prompt_sha256']) for r in read(root/'requests.json')]
    initial = read(case/'initial_sample.json')
    assert initial['used_bytes'] == initial['cpu_hot_bytes'] == initial['daos_puts'] == 0
    summaries = []
    for phase_name in ('cold', 'warm'):
        folder = case/phase_name
        calls, phase = read(folder/'replay_calls.json'), read(folder/'phase.json')
        assert len(calls) == 256 and not any('error' in c for c in calls)
        assert [(c['index'], c['prompt_sha256']) for c in calls] == expected
        before, after = read(folder/'initial_sample.json'), read(folder/'final_sample.json')
        assert before == (initial if phase_name == 'cold' else read(case/'cold/final_sample.json'))
        assert before['used_bytes'] == after['used_bytes'] == after['dram_mirror']['pending_bytes'] == 0
        assert after['dram_mirror']['errors'] == 0
        selected = select_events(events, calls)
        rows = join(calls, selected)
        assert not any(r['other_failed_chunks'] for r in rows)
        common.dump(folder/'timing_by_request.json', rows)
        q = sum(e['queried_chunks'] for e in selected if e['event'] == 'tier_lookup' and e['tier'] == 'dram')
        d, a = sum(r['dram_lookup_chunks'] for r in rows), sum(r['daos_lookup_chunks'] for r in rows)
        assert 0 <= d+a <= q
        pf = {k: v-(before['cpu_prefetch'] or {}).get(k, 0)
              for k, v in (after['cpu_prefetch'] or {}).items()}
        assert pf.get('copy_errors', 0) == 0
        start, end = phase['start_ns'], phase['end_ns']
        scoped = [e for e in events if start <= e['time_ns'] <= end]
        rejects = [e for e in scoped if e['event'] == 'cpu_prefetch_occupancy_reject']
        assert len(rejects) == pf.get('occupancy_rejections', 0)
        assert all(e['used_bytes'] >= e['threshold_bytes'] for e in rejects)
        inp = sum(c['prompt_tokens'] for c in calls)
        s = dict(**spec, phase=phase_name, requests=256,
            ttft=stats(r['ttft_ms'] for r in rows), elapsed_seconds=(end-start)/1e9,
            input_tokens=inp, computed_input_tokens=sum(r['computed_prompt_tokens'] for r in rows),
            cached_input_pct=100*sum(c['cached_tokens'] for c in calls)/inp,
            output_tokens=sum(c['completion_tokens'] for c in calls),
            dram_hit_pct=100*d/q, daos_hit_pct=100*a/q,
            capacity_recomputed_tokens=sum(r['capacity_recomputed_tokens'] for r in rows),
            peak_staging_gib=max(e['used_bytes'] for e in scoped)/2**30,
            prefetch_counters=pf,
            timings={k: stats(r.get(k) for r in rows) for k in
                     ['queue_ms', 'copy_ms', 'retrieve_ms', 'http_to_retrieve_start_ms']})
        timeline(folder, events, start, end, 8)
        common.dump(folder/'summary.json', s)
        summaries.append(s)
    common.dump(case/'summary.json', summaries)


def report(root):
    plan = read(root/'plan.json')
    rows = []
    configs, natives = [], []
    for spec in plan['cases']:
        case = root/spec['name']
        if not (case/'summary.json').exists():
            continue
        rows += read(case/'summary.json')
        cfg = yaml.safe_load((case/'config.yaml').read_text())
        ec = cfg['extra_config']
        assert ec.pop('daosgds.dram_prefetch') == spec['prefetch']
        assert ec.pop('daosgds.dram_prefetch_stop_occupancy_ratio') == spec['occupancy_stop_ratio']
        for k in ('daosgds.object_namespace', 'daosgds.root'):
            ec.pop(k)
        configs.append(cfg)
        natives.append(read(case/'native_maps.json'))
    assert all(c == configs[0] for c in configs)
    assert all(n == natives[0] for n in natives)
    paired = []
    for c in (8, 16):
        for phase in ('cold', 'warm'):
            reference = root/f'c{c}_on'/phase/'timing_by_request.json'
            if not reference.exists():
                continue
            for mode in ('off', 'gate60'):
                path = root/f'c{c}_{mode}'/phase/'timing_by_request.json'
                if path.exists():
                    pairs = list(zip(read(reference), read(path), strict=True))
                    assert all(a['index'] == b['index'] for a, b in pairs)
                    paired.append(dict(concurrency=c, phase=phase, reference='on', mode=mode,
                        same_cached_requests=sum(a['cached_tokens'] == b['cached_tokens'] for a,b in pairs),
                        same_tier_requests=sum((a['dram_lookup_chunks'],a['daos_lookup_chunks']) ==
                            (b['dram_lookup_chunks'],b['daos_lookup_chunks']) for a,b in pairs)))
    common.dump(root/'summary.json', rows)
    common.dump(root/'paired_checks.json', paired)
    lines = ['# DiscoveryBench: staging 60% DRAM 프리페치 차단 실험', '',
        'Qwen3-14B / DRAM 8GiB / staging 8GiB / 청크128 / 복사 작업자1 / 조기 준비·대기열 취소 OFF.',
        '각 조건: 빈 캐시 cold 256요청 → drain → 같은 프로세스·캐시로 warm 256요청. 조건별 1회.',
        'OFF=DRAM 프리페치 없음, ON=기존 용량 한계만 적용, gate60=현재 전체 점유율 60% 이상이면 DRAM 프리페치 생략.',
        'DAOS 프리페치는 모든 조건에서 ON. gate60은 단단한 점유율 상한이 아니며 이미 시작한 복사를 취소하지 않는다.', '',
        '|동시 요청|정책|단계|평균 TTFT ms|p95 ms|전체 s|DRAM hit %|DAOS hit %|새 계산 토큰|60% 차단 건수|최대 staging GiB|DAOS 실패 재계산 토큰|',
        '|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in rows:
        lines.append(f'|{s["concurrency"]}|{s["mode"]}|[{s["phase"]}]({s["name"]}/{s["phase"]}/staging_hits.png)|'
            f'{s["ttft"]["mean"]:.2f}|{s["ttft"]["p95"]:.2f}|{s["elapsed_seconds"]:.2f}|'
            f'{s["dram_hit_pct"]:.2f}|{s["daos_hit_pct"]:.2f}|{s["computed_input_tokens"]}|'
            f'{s["prefetch_counters"].get("occupancy_rejections",0)}|{s["peak_staging_gib"]:.3f}|'
            f'{s["capacity_recomputed_tokens"]}|')
    lines += ['', '## 현재 코드에서 ON 대비 gate60 변화', '',
              '|동시 요청|단계|ON TTFT ms|gate60 TTFT ms|TTFT 감소율|', '|---:|---|---:|---:|---:|']
    by = {(s['concurrency'],s['phase'],s['mode']):s for s in rows}
    for c in (8,16):
        for phase in ('cold','warm'):
            if (c,phase,'on') in by and (c,phase,'gate60') in by:
                a,b = [by[c,phase,m]['ttft']['mean'] for m in ('on','gate60')]
                lines.append(f'|{c}|{phase}|{a:.2f}|{b:.2f}|{100*(1-b/a):+.2f}%|')
    lines += ['', '## 이전 2026-09-27 관측값 (직접적인 정책 대조군이 아닌 참고)', '',
              '|동시 요청|정책|단계|평균 TTFT ms|', '|---:|---|---|---:|']
    for s in read(root/'historical_summary.json'):
        lines.append(f'|{s["concurrency"]}|{"ON" if s["prefetch"] else "OFF"}|{s["phase"]}|{s["timings"]["ttft_ms"]["mean"]:.2f}|')
    lines += ['', '프리페치 정책만의 효과를 판단하려면 현재 ON과 gate60의 hit/새 계산량·생성량도 확인해야 한다.',
              '한 번씩 실행한 예비 실험이므로 통계적 우열은 확정하지 않는다. cold/warm은 독립 반복이 아니다.',
              '[요청별 재사용 일치 검사](paired_checks.json) · [원시 집계](summary.json)', '']
    (root/'RESULT_KO.md').write_text('\n'.join(lines))


def run(root):
    import fcntl
    with (root/'runner.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read(root/'plan.json')
        assert plan['cases'] == cases()
        for name, digest in plan['source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
        assert hashlib.sha256((root/'requests.json').read_bytes()).hexdigest() == plan['request_sha256']
        records = read(root/'requests.json')
        os.environ['DAOS_GDS_PREFETCH_TIMING'] = '1'
        completed = []
        try:
            for spec in plan['cases']:
                case = root/spec['name']
                common.dump(root/'status.json', dict(status='running', current=spec['name'], completed=completed,
                                                     updated_ns=time.time_ns()))
                if not (case/'status.json').exists() or read(case/'status.json')['status'] != 'completed':
                    assert not case.exists(), 'Incomplete case requires diagnosis; do not mix partial runs'
                    case.mkdir()
                    a = SimpleNamespace(model=plan['model'], max_model_len=32768, port=8017,
                        cpu_gib=8, staging_gib=8, prefetch_workers=1, cancel_queued=False, early_ready=False,
                        occupancy_stop_ratio=spec['occupancy_stop_ratio'])
                    run_case(a, case, records, spec['concurrency'], spec['prefetch'])
                if not (case/'summary.json').exists():
                    summarize_case(root, spec)
                report(root)
                with (case/'cleanup.log').open('a') as stream:
                    subprocess.run([str(ROOT/'run_vllm.sh'), str(ROOT/'venv/bin/python3'),
                        str(Path(__file__).resolve()), '--cleanup-case', str(case)], cwd=ROOT,
                        env=dict(os.environ, DAOSGDS_TRANSPORT='object'), stdout=stream,
                        stderr=subprocess.STDOUT, check=True, timeout=300)
                completed.append(spec['name'])
            common.dump(root/'status.json', dict(status='completed', completed=completed, updated_ns=time.time_ns()))
        except BaseException as exc:
            common.dump(root/'status.json', dict(status='failed', completed=completed, error=repr(exc),
                current=spec['name'], updated_ns=time.time_ns()))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path)
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--report', action='store_true')
    p.add_argument('--cleanup-case', type=Path)
    a = p.parse_args()
    if a.cleanup_case:
        cleanup(a.cleanup_case)
        return
    assert a.output is not None
    root = a.output.resolve()
    assert root.parent == ROOT
    if a.prepare:
        prepare(root)
    elif a.report:
        report(root)
    else:
        run(root)


if __name__ == '__main__':
    main()
