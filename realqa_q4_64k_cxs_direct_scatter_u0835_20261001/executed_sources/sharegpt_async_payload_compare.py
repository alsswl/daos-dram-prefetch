#!/usr/bin/env python3
"""Fresh paired async-control arms: payload at retrieve vs speculative payload.

No installed-code edits; no CUDA profiler in performance trials. Both use
metadata-first notification, 1ms backoff and identical cache/memory policies.
"""
import argparse
from copy import deepcopy
import fcntl
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
BASELINE = ROOT/'sharegpt_backoff_1ms_d256_s8_20260930'
original_config = early.config
NEW_FILES = ('lmcache_daos/async_demand_backend.py', 'sharegpt_async_payload_compare.py',
             'tests/test_async_demand.py', 'tests/async_demand_roundtrip.py')


def config(prefetch):
    cfg = original_config()
    cfg['extra_config'].update({
        'lookup_backoff_time': .001,
        'storage_plugin.daosgds.module_path': 'lmcache_daos.async_demand_backend',
        'storage_plugin.daosgds.class_name': 'AsyncDemandBackend',
        'daosgds.early_payload_prefetch': prefetch})
    return cfg


def comparable_config(cfg):
    result = deepcopy(cfg)
    for key in ('daosgds.object_namespace', 'daosgds.root', 'daosgds.early_payload_prefetch'):
        result['extra_config'].pop(key)
    return result


def prepare(root):
    assert root.parent == ROOT and not root.exists()
    plan = read(BASELINE/'plan.json')
    for name, digest in plan['source_sha256'].items():
        assert base.digest(ROOT/name) == digest, f'Baseline source changed: {name}'
    assert base.digest(BASELINE/'requests.json') == plan['request_sha256']
    # Explicitly limit divergence from reference E to new wrapper and its flag.
    old = yaml.safe_load((BASELINE/'c16_backoff_1ms/config.yaml').read_text())
    new = config(True)
    new['extra_config'].pop('daosgds.early_payload_prefetch')
    for c in (old, new):
        for key in ('storage_plugin.daosgds.module_path', 'storage_plugin.daosgds.class_name',
                    'daosgds.object_namespace', 'daosgds.root'):
            c['extra_config'].pop(key)
    assert old == new
    root.mkdir()
    for name in ('requests.json', 'sessions.json', 'capacity_estimate.json', 'params.json', 'runtime_versions.json'):
        shutil.copy2(BASELINE/name, root/name)
    plan.update(baseline=str(BASELINE), phases=['cold', 'warm1'], warm_repeats=1,
        requests_per_case=3368, profile_output=None, after_unit=None,
        cases=[dict(name='c16_async_demand', prefetch=False, concurrency=16),
               dict(name='c16_async_prefetch', prefetch=True, concurrency=16)],
        notes=['Both arms rerun now with same AsyncDemandBackend and async lookup 1ms.',
               'Only early_payload_prefetch differs, plus fresh UUID namespaces.',
               'OFF does metadata lookup and publishes a plan; reads payload only at retrieve.',
               'ON uses existing speculative DRAM/DAOS workers after metadata notification.',
               'Both retain same running-read wait and queued-to-demand ownership implementation.',
               'Fresh process, empty DRAM/staging and new namespace per arm; retain cache cold->warm1.',
               'Identical original 421 ShareGPT conversations x4 turns, rolling C16, output cap256.',
               'Actual arrivals, generated output and cache placement may differ; report paired checks.',
               'No ready-return patch, lock probe, profiler, artificial delays or forced cache placement.',
               'One trial per arm, OFF then ON: exploratory, not statistical significance.',
               'GPU byte tests and independent small cold/warm smoke precede performance.',
               'Clean only exact completed experimental namespace keys; retain failed evidence.'])
    for name in set(plan['source_sha256']) | set(NEW_FILES):
        dest = root/'executed_sources'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
        plan['source_sha256'][name] = base.digest(dest)
    dump(root/'plan.json', plan)
    dump(root/'status.json', dict(status='prepared', updated_ns=time.time_ns()))


def validate_policy(case, prefetch):
    events = base.read_events(case)
    policy = [e for e in events if e['event'] == 'early_payload_policy']
    assert len(policy) == 1 and policy[0]['speculative'] is prefetch
    if prefetch:
        assert any(e['event'] == 'early_payload_start' for e in events)
    else:
        assert not any(e['event'] in ('early_payload_start', 'early_payload_ready') for e in events)
        retrieves = {e['request_id']: e['monotonic_ns'] for e in events if e['event'] == 'retrieve_start'}
        demands = [e for e in events if e['event'] == 'early_daos_demand_start']
        for e in demands:
            assert retrieves[e['request_id']] <= e['monotonic_ns'], 'DAOS read before retrieve'
        decisions = [e for e in events if e['event'] == 'early_retrieve_decision']
        assert decisions and all(e['decision'] == 'queued_to_demand' for e in decisions)
    dump(case/'payload_policy_check.json', dict(passed=True, speculative=prefetch))


def run_one(root, spec):
    early.config = lambda: config(spec['prefetch'])
    try:
        early.run_case(root, spec)
    finally:
        early.config = original_config
    summarize_case(root, spec)
    validate_policy(root/spec['name'], spec['prefetch'])


def smoke(root, plan):
    folder = root.with_name(root.name+'_validation')
    folder.mkdir()
    records = read(root/'requests.json')[:8]
    dump(folder/'requests.json', records)
    pilot = dict(plan, phases=['cold', 'warm1'], requests_per_phase=8, requests_per_case=16)
    dump(folder/'plan.json', pilot)
    dump(root/'validation_location.json', dict(path=str(folder)))
    for spec in plan['cases']:
        dump(root/'status.json', dict(status='vllm_validation', current=spec['name'], updated_ns=time.time_ns()))
        run_one(folder, spec)
        assert read(folder/spec['name']/'summary.json')[1]['cached_tokens'] > 0
        early.cleanup(folder/spec['name'])
    dump(folder/'status.json', dict(status='completed'))


def report(root):
    off, on = root/'c16_async_demand', root/'c16_async_prefetch'
    a, b = [yaml.safe_load((c/'config.yaml').read_text()) for c in (off, on)]
    assert a['extra_config']['daosgds.early_payload_prefetch'] is False
    assert b['extra_config']['daosgds.early_payload_prefetch'] is True
    assert comparable_config(a) == comparable_config(b)
    assert read(off/'command.json') == read(on/'command.json')
    assert read(off/'native_maps.json') == read(on/'native_maps.json')
    rows = []
    for x, y in zip(read(off/'summary.json'), read(on/'summary.json'), strict=True):
        assert x['phase'] == y['phase']
        left, right = [read(c/x['phase']/'replay_calls.json') for c in (off, on)]
        keys = ('index', 'prompt_sha256', 'prompt_tokens', 'max_tokens')
        assert [tuple(r[k] for k in keys) for r in left] == [tuple(r[k] for k in keys) for r in right]
        rows.append(dict(phase=x['phase'], async_demand=x, async_prefetch=y,
            ttft_change_pct=100*(y['ttft']['mean']/x['ttft']['mean']-1),
            elapsed_change_pct=100*(y['elapsed_seconds']/x['elapsed_seconds']-1),
            requests=len(left), same_cached_requests=sum(a['cached_tokens']==b['cached_tokens'] for a,b in zip(left,right)),
            same_generated_output_requests=sum(a['output_sha256']==b['output_sha256'] for a,b in zip(left,right))))
    dump(root/'comparison.json', dict(rows=rows, performance_only=True,
        limitations=['One fresh trial per arm; fixed OFF->ON order, not statistical significance.',
                     'Same input workload and control protocol, not guaranteed identical runtime cache placement.',
                     'Negative change means faster. Read paired cached-token and generation checks.']))


def run(root):
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read(root/'plan.json')
        try:
            early.verify_sources(root, plan)
            assert early.idle_gpu(), 'GPU occupied; no unrelated job will be stopped'
            assert shutil.disk_usage(ROOT).free > 8*2**30
            mem = {s.split(':')[0]: int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()}
            assert mem['MemAvailable'] >= 320*2**30
            os.environ['DAOS_GDS_PREFETCH_TIMING'] = '1'
            os.environ['DAOS_LOOKUP_READY_RETURN'] = '0'
            os.environ['DAOS_LOOKUP_LOCK_PROBE'] = '0'
            dump(root/'status.json', dict(status='gpu_validation', updated_ns=time.time_ns()))
            with (root/'gpu_validation.log').open('x') as log:
                subprocess.run([str(ROOT/'run_vllm.sh'), str(ROOT/'venv/bin/python3'),
                    str(ROOT/'tests/async_demand_roundtrip.py'), '--output', str(root/'gpu_validation')],
                    cwd=ROOT, env=dict(os.environ, DAOSGDS_TRANSPORT='object'),
                    stdout=log, stderr=subprocess.STDOUT, timeout=240, check=True)
            assert read(root/'gpu_validation/result.json')['result'] == 'PASS'
            smoke(root, plan)
            for spec in plan['cases']:
                early.verify_sources(root, plan)
                dump(root/'status.json', dict(status='performance', current=spec['name'], updated_ns=time.time_ns()))
                run_one(root, spec)
                early.cleanup(root/spec['name'])
            report(root)
            dump(root/'status.json', dict(status='completed', updated_ns=time.time_ns()))
        except BaseException as exc:
            dump(root/'status.json', dict(status='failed', error=repr(exc), updated_ns=time.time_ns()))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    assert root.parent == ROOT
    if args.prepare:
        prepare(root)
    elif args.report:
        report(root)
    else:
        run(root)
