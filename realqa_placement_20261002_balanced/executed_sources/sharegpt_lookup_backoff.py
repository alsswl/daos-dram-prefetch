#!/usr/bin/env python3
"""Isolated 10ms -> 1ms lookup-backoff ablation; no installed code edits.

Performance: fresh cold + warm1, paired with existing unprofiled 10ms warm1.
Diagnostics: separate fresh cold + original 256-request profile replay, paired
with the existing 10ms diagnostic. Never compare profiled TTFT as performance.
"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import subprocess
import time

import yaml
import sharegpt_early_lookup as early
from report_early_lookup import summarize_case

ROOT, base = early.ROOT, early.base
read, dump = early.read, early.dump
BASELINE = ROOT/'sharegpt_early_lookup_d256_s8_20260930'
PROFILE_BASELINE = ROOT/'sharegpt_gpu_overlap_d256_s8_20260930'


def config():
    cfg = original_config()
    cfg['extra_config']['lookup_backoff_time'] = 0.001
    return cfg


original_config = early.config


def comparable_config(cfg):
    cfg = deepcopy(cfg)
    ec = cfg['extra_config']
    for key in ('lookup_backoff_time', 'daosgds.object_namespace', 'daosgds.root'):
        ec.pop(key, None)
    return cfg


def check_config(old, new):
    assert old['extra_config'].get('lookup_backoff_time', 0.01) == 0.01
    assert new['extra_config']['lookup_backoff_time'] == 0.001
    assert comparable_config(old) == comparable_config(new), 'Unexpected config change'


def prepare(root):
    assert root.parent == ROOT and not root.exists(), 'Use a new direct child experiment directory'
    plan = read(BASELINE/'plan.json')
    # Verify unchanged backend, executor and workload implementations BEFORE launch.
    for name, digest in plan['source_sha256'].items():
        assert base.digest(ROOT/name) == digest, f'Baseline source changed: {name}'
    assert base.digest(BASELINE/'requests.json') == plan['request_sha256']
    check_config(yaml.safe_load((BASELINE/'c16_early/config.yaml').read_text()), config())
    root.mkdir()
    for name in ('requests.json', 'sessions.json', 'capacity_estimate.json', 'params.json', 'runtime_versions.json'):
        shutil.copy2(BASELINE/name, root/name)
    plan.update(baseline=str(BASELINE), after_unit=None, phases=['cold', 'warm1'], warm_repeats=1,
                requests_per_case=3368, lookup_backoff_seconds=0.001,
                cases=[dict(name='c16_backoff_1ms', prefetch=True, concurrency=16)],
                profile_output=str(root.with_name(root.name+'_profile')),
                profile_baseline=str(PROFILE_BASELINE),
                notes=['Only extra_config.lookup_backoff_time changes: default .01 -> .001 seconds.',
                       'No changes to locks, result handling, queue, memory or prefetch policy.',
                       'Fresh cold 1684 then warm1 1684; pair warm1, NOT old warm4 mean.',
                       'Same input list and rolling C16; actual arrivals, tier placement and generation can vary.',
                       'Separate fresh process/cache for profiling, with full cold fill then original 256 requests.',
                       'One new run, historical control: preliminary ablation, not statistical significance.',
                       'Cleanup only completed experiment UUID namespace; preserve logs and failed data.'])
    files = set(plan['source_sha256']) | {
        'sharegpt_lookup_backoff.py', 'profile_sharegpt_overlap.py', 'analyze_gpu_overlap.py'}
    plan['source_sha256'] = {}
    for name in sorted(files):
        dest = root/'executed_sources'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
        plan['source_sha256'][name] = base.digest(dest)
    dump(root/'plan.json', plan)
    dump(root/'status.json', dict(status='prepared', updated_ns=time.time_ns()))


def performance_report(root):
    oldcase, case = BASELINE/'c16_early', root/'c16_backoff_1ms'
    check_config(yaml.safe_load((oldcase/'config.yaml').read_text()),
                 yaml.safe_load((case/'config.yaml').read_text()))
    assert read(oldcase/'command.json') == read(case/'command.json')
    assert read(oldcase/'native_maps.json') == read(case/'native_maps.json')
    before = {s['phase']: s for s in read(oldcase/'summary.json')}
    rows = []
    for after in read(case/'summary.json'):
        phase = after['phase']
        a, b = read(oldcase/phase/'replay_calls.json'), read(case/phase/'replay_calls.json')
        keys = ('index', 'prompt_sha256', 'prompt_tokens', 'max_tokens')
        assert [tuple(c[k] for k in keys) for c in a] == [tuple(c[k] for k in keys) for c in b]
        same_cached = sum(x['cached_tokens'] == y['cached_tokens'] for x, y in zip(a, b))
        old = before[phase]
        rows.append(dict(phase=phase, before_10ms=old, after_1ms=after,
            ttft_change_pct=100*(after['ttft']['mean']/old['ttft']['mean']-1),
            elapsed_change_pct=100*(after['elapsed_seconds']/old['elapsed_seconds']-1),
            same_cached_requests=same_cached, requests=len(a),
            same_generated_output_requests=sum(x['output_sha256'] == y['output_sha256'] for x,y in zip(a,b))))
    dump(root/'comparison.json', dict(rows=rows, performance_only=True,
        limitations=['One new run versus historical control; not independent repeated trials.',
                     'Cold placement and generated outputs can vary; inspect paired cached token counts.',
                     'Profiled timing is excluded. Negative percent means faster.']))
    print('Performance comparison written:', root/'comparison.json', flush=True)


def run(root):
    plan = read(root/'plan.json')
    try:
        early.verify_sources(root, plan)
        assert early.idle_gpu(), 'GPU occupied; no unrelated job will be stopped'
        assert shutil.disk_usage(ROOT).free > 8*2**30
        mem = {s.split(':')[0]: int(s.split()[1])*1024
               for s in Path('/proc/meminfo').read_text().splitlines()}
        assert mem['MemAvailable'] >= 320*2**30
        os.environ['DAOS_GDS_PREFETCH_TIMING'] = '1'
        dump(root/'status.json', dict(status='performance', updated_ns=time.time_ns()))
        # Isolated runner-process factory override, restored before diagnostics.
        early.config = config
        try:
            early.run_case(root, plan['cases'][0])
        finally:
            early.config = original_config
        case = root/'c16_backoff_1ms'
        summarize_case(root, plan['cases'][0])
        performance_report(root)
        early.cleanup(case)
        dump(root/'status.json', dict(status='profiling', performance='completed', updated_ns=time.time_ns()))
        profile_root = Path(plan['profile_output'])
        import profile_sharegpt_overlap as profiler
        profiler.run(profile_root, lookup_backoff_seconds=0.001)
        check_config(yaml.safe_load((PROFILE_BASELINE/'c16_profile/config.yaml').read_text()),
                     yaml.safe_load((profile_root/'c16_profile/config.yaml').read_text()))
        assert base.digest(PROFILE_BASELINE/'requests.json') == base.digest(profile_root/'requests.json')
        assert read(PROFILE_BASELINE/'warm_requests.json') == read(profile_root/'warm_requests.json')
        # Analyze each trace in a separate process so multi-million-event JSON is released.
        for source, dest in ((PROFILE_BASELINE, root/'baseline_profile_analysis'),
                             (profile_root, root/'backoff_profile_analysis')):
            code = ('from pathlib import Path; from analyze_gpu_overlap import analyze; '
                    f'analyze(Path({str(source / "c16_profile")!r}), Path({str(dest)!r}))')
            with (root/(dest.name+'.log')).open('x') as log:
                subprocess.run([str(ROOT/'venv/bin/python3'), '-c', code], cwd=ROOT,
                               stdout=log, stderr=subprocess.STDOUT, timeout=600, check=True)
        dump(root/'profile_comparison.json', dict(
            before_10ms=read(root/'baseline_profile_analysis/summary.json'),
            after_1ms=read(root/'backoff_profile_analysis/summary.json'),
            note='Profiled diagnostic only. DAOS span is host I/O, not measured NIC DMA. '
                 'Kernel gaps exclude initial/final idle; kernel absence is not proof GPU DMA is idle.'))
        dump(profile_root/'status.json', dict(status='completed', analysis=str(root/'profile_comparison.json')))
        dump(root/'status.json', dict(status='completed', updated_ns=time.time_ns()))
    except BaseException as exc:
        dump(root/'status.json', dict(status='failed', error=repr(exc), updated_ns=time.time_ns()))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    if args.prepare:
        prepare(root)
    else:
        run(root)
