#!/usr/bin/env python3
"""Ready-result recheck ON vs completed 1ms baseline; cold1/warm1 + separate profile."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import yaml
import sharegpt_lookup_backoff as previous
import sharegpt_early_lookup as early
from report_early_lookup import summarize_case

ROOT, base = early.ROOT, early.base
read, dump = early.read, early.dump
BASELINE = ROOT/'sharegpt_backoff_1ms_d256_s8_20260930'
BASECASE = BASELINE/'c16_backoff_1ms'
PROFILE_BASELINE = ROOT/'sharegpt_backoff_1ms_d256_s8_20260930_profile'
NEW_FILES = ('lookup_ready_return.py', 'sharegpt_ready_return.py',
             'tests/test_lookup_ready_return.py',
             'experiment_plugins/lookup_ready_return_hook-0.1.dist-info/METADATA',
             'experiment_plugins/lookup_ready_return_hook-0.1.dist-info/entry_points.txt')


def check_config(a, b):
    assert a['extra_config']['lookup_backoff_time'] == b['extra_config']['lookup_backoff_time'] == 0.001
    assert previous.comparable_config(a) == previous.comparable_config(b)


def prepare(root):
    assert root.parent == ROOT and not root.exists()
    assert read(BASELINE/'status.json')['status'] == 'completed'
    plan = read(BASELINE/'plan.json')
    for name, digest in plan['source_sha256'].items():
        assert base.digest(ROOT/name) == digest, f'Baseline source changed: {name}'
    check_config(yaml.safe_load((BASECASE/'config.yaml').read_text()), previous.config())
    root.mkdir()
    for name in ('requests.json', 'sessions.json', 'capacity_estimate.json', 'params.json', 'runtime_versions.json'):
        shutil.copy2(BASELINE/name, root/name)
    plan.update(baseline=str(BASELINE), ready_return=True,
                cases=[dict(name='c16_ready_return', prefetch=True, concurrency=16)],
                profile_baseline=str(PROFILE_BASELINE),
                profile_output=str(root.with_name(root.name+'_profile')),
                notes=['Same 1ms backoff; only completed-result recheck after original initial lookup is enabled.',
                       'lookup_cache lock/sleep/timeout, prefetch queues and cache placement policy are unchanged.',
                       'Local opt-in vLLM general plugin, disabled by default; installed package is not edited.',
                       'Fresh cold1+warm1; same 1684 requests per phase and C16, DRAM256/staging8.',
                       'Small isolated cold/warm smoke precedes full run; separate profiling follows.',
                       'Compare with historical 1ms warm1, not warm4 averages. Single-run preliminary evidence.'])
    for name in set(plan['source_sha256']) | set(NEW_FILES):
        dest = root/'executed_sources'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
        plan['source_sha256'][name] = base.digest(dest)
    dump(root/'plan.json', plan)
    dump(root/'status.json', dict(status='prepared'))


def hook_stats(case):
    text = (case/'server.log').read_text(errors='replace')
    assert 'LOOKUP_READY_RETURN installed' in text, 'Plugin was not installed'
    rows = [json.loads(m.group(1)) for m in re.finditer(r'LOOKUP_READY_RETURN_(?:STATS|FINAL) (\{[^\n]*\})', text)]
    assert rows, 'No patched lookup calls observed'
    # First16 calls + periodic snapshots: lower bound if close() is not called.
    last = max(rows, key=lambda row: row['calls'])
    dump(case/'lookup_ready_return_stats.json', dict(last_observed=last,
        note='First16 then every128 initial lookups plus close, may exclude the final partial interval.'))
    return last


def report(root):
    case = root/'c16_ready_return'
    check_config(yaml.safe_load((BASECASE/'config.yaml').read_text()), yaml.safe_load((case/'config.yaml').read_text()))
    assert read(BASECASE/'command.json') == read(case/'command.json')
    assert read(BASECASE/'native_maps.json') == read(case/'native_maps.json')
    before = {s['phase']: s for s in read(BASECASE/'summary.json')}
    rows = []
    for after in read(case/'summary.json'):
        phase = after['phase']
        a, b = read(BASECASE/phase/'replay_calls.json'), read(case/phase/'replay_calls.json')
        keys = ('index', 'prompt_sha256', 'prompt_tokens', 'max_tokens')
        assert [tuple(c[k] for k in keys) for c in a] == [tuple(c[k] for k in keys) for c in b]
        old = before[phase]
        rows.append(dict(phase=phase, before_1ms=old, after_ready_return=after,
            ttft_change_pct=100*(after['ttft']['mean']/old['ttft']['mean']-1),
            elapsed_change_pct=100*(after['elapsed_seconds']/old['elapsed_seconds']-1),
            requests=len(a), same_cached_requests=sum(x['cached_tokens']==y['cached_tokens'] for x,y in zip(a,b)),
            same_generated_output_requests=sum(x['output_sha256']==y['output_sha256'] for x,y in zip(a,b))))
    dump(root/'comparison.json', dict(rows=rows, hook=hook_stats(case),
        note='Unprofiled performance only; historical single-run control, tier placement and generated output may vary.'))


def run(root):
    plan = read(root/'plan.json')
    try:
        early.verify_sources(root, plan)
        assert early.idle_gpu(), 'GPU occupied; no unrelated process will be stopped'
        assert shutil.disk_usage(ROOT).free > 8*2**30
        mem = {s.split(':')[0]: int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()}
        assert mem['MemAvailable'] >= 320*2**30
        os.environ.update(DAOS_GDS_PREFETCH_TIMING='1', DAOS_LOOKUP_READY_RETURN='1')
        os.environ['PYTHONPATH'] = str(ROOT/'experiment_plugins')+os.pathsep+str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH','')
        assert not os.environ.get('VLLM_PLUGINS'), 'Plugin allow-list requires review'
        dump(root/'runtime_overrides.json', dict(DAOS_LOOKUP_READY_RETURN='1',
            extra_pythonpath=str(ROOT/'experiment_plugins'), lookup_backoff_time=0.001))
        early.config = previous.config
        early.smoke(root, plan)
        assert hook_stats(root.with_name(root.name+'_validation')/'pilot')['ready'] > 0, 'Smoke did not exercise ready return'
        dump(root/'status.json', dict(status='performance', updated_ns=time.time_ns()))
        spec = plan['cases'][0]
        early.run_case(root, spec)
        summarize_case(root, spec)
        report(root)
        early.cleanup(root/spec['name'])
        dump(root/'status.json', dict(status='profiling', performance='completed', updated_ns=time.time_ns()))
        import profile_sharegpt_overlap as profiler
        profile_root = Path(plan['profile_output'])
        profiler.run(profile_root, lookup_backoff_seconds=0.001)
        hook_stats(profile_root/'c16_profile')
        check_config(yaml.safe_load((PROFILE_BASELINE/'c16_profile/config.yaml').read_text()),
                     yaml.safe_load((profile_root/'c16_profile/config.yaml').read_text()))
        assert base.digest(PROFILE_BASELINE/'requests.json') == base.digest(profile_root/'requests.json')
        assert read(PROFILE_BASELINE/'warm_requests.json') == read(profile_root/'warm_requests.json')
        dest = root/'ready_return_profile_analysis'
        code = ('from pathlib import Path; from analyze_gpu_overlap import analyze; '
                f'analyze(Path({str(profile_root / "c16_profile")!r}), Path({str(dest)!r}))')
        with (root/'profile_analysis.log').open('x') as log:
            subprocess.run([str(ROOT/'venv/bin/python3'), '-c', code], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        dump(root/'profile_comparison.json', dict(
            before_1ms=read(BASELINE/'backoff_profile_analysis/summary.json'),
            after_ready_return=read(dest/'summary.json'),
            note='Separate profiled diagnostic, not TTFT measurement or actual NIC DMA time.'))
        dump(profile_root/'status.json', dict(status='completed', analysis=str(root/'profile_comparison.json')))
        dump(root/'status.json', dict(status='completed', updated_ns=time.time_ns()))
    except BaseException as exc:
        dump(root/'status.json', dict(status='failed', error=repr(exc), updated_ns=time.time_ns()))
        raise
    finally:
        early.config = previous.original_config


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    (prepare if args.prepare else run)(args.output.resolve())
