#!/usr/bin/env python3
"""Fresh full cold fill, then bounded GPU profiling of original warm conversations.

Not a performance trial: Kineto overhead and trace flush affect request timings.
No installed package edits, fabricated sleeps, placement, or background workloads.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace

import yaml
import sharegpt_early_lookup as early
import staging_mixed_pressure as runtime

ROOT, base, cw = early.ROOT, early.base, early.cw
read, dump = early.read, early.dump
BASELINE = ROOT/'sharegpt_early_lookup_d256_s8_20260930'


def run(root, lookup_backoff_seconds=None):
    if lookup_backoff_seconds is not None:
        assert 0 < lookup_backoff_seconds <= 0.01
    assert root.parent == ROOT and not root.exists()
    assert early.idle_gpu(), 'GPU occupied; unrelated jobs will not be stopped'
    assert shutil.disk_usage(ROOT).free > 8*2**30, 'Insufficient local trace disk headroom'
    root.mkdir()
    case = root/'c16_profile'
    case.mkdir()
    plan = read(BASELINE/'plan.json')
    records = read(BASELINE/'requests.json')
    sessions = sorted({r['session'] for r in records})[:64]
    warm = [r for r in records if r['session'] in sessions]
    assert len(warm) == 256 and len(records) == 1684
    plan.update(cases=[dict(name=case.name, prefetch=True, concurrency=16)],
                phases=['cold', 'warm_profile'], diagnostic=True,
                full_cold_requests=len(records), warm_requests=len(warm),
                profiler_max_iterations=512,
                note='Profiling changes runtime; not a TTFT benchmark. Full cold fill retained. '
                     'Warm replays first64 original conversations, 4turns each. GPU recording '
                     'covers first512 engine steps only; no synthetic prompts or cache placement.')
    plan['lookup_backoff_seconds'] = lookup_backoff_seconds if lookup_backoff_seconds is not None else 0.01
    plan['notes'] = [plan['note'], 'Only the explicitly recorded lookup backoff may differ from the prior diagnostic.']
    plan['source_sha256'] = {}
    for path in sorted((ROOT/'lmcache_daos').glob('*.py')) + [
            ROOT/'profile_sharegpt_overlap.py', ROOT/'sharegpt_early_lookup.py',
            ROOT/'sharegpt_cold_warm.py', ROOT/'sharegpt_scale_experiment.py',
            ROOT/'staging_mixed_pressure.py']:
        name = str(path.relative_to(ROOT))
        dest = root/'executed_sources'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        plan['source_sha256'][name] = base.digest(dest)
    dump(root/'plan.json', plan)
    shutil.copy2(BASELINE/'requests.json', root/'requests.json')
    dump(root/'warm_requests.json', warm)
    dump(root/'status.json', dict(status='preflight', updated_ns=time.time_ns()))
    os.environ['DAOS_GDS_PREFETCH_TIMING'] = '1'
    cfg = early.config()
    if lookup_backoff_seconds is not None:
        cfg['extra_config']['lookup_backoff_time'] = lookup_backoff_seconds
    (case/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    cap = plan['capacity']
    required = 2*cap['conservative_stored_kv_gib']-cap['unique_input_kv_gib']
    cw.wait_space(case, required)
    mem = {line.split(':')[0]: int(line.split()[1])*1024
           for line in Path('/proc/meminfo').read_text().splitlines()}
    assert mem['MemAvailable'] >= 320*2**30
    profiler = dict(profiler='torch', torch_profiler_dir=str(case/'gpu_trace'),
        torch_profiler_with_stack=False, torch_profiler_record_shapes=False,
        torch_profiler_with_memory=False, torch_profiler_with_flops=False,
        torch_profiler_use_gzip=True, torch_profiler_dump_cuda_time_total=False,
        ignore_frontend=True, max_iterations=512)
    original_command = runtime.inference_command
    runtime.inference_command = lambda a: original_command(a)+['--profiler-config', json.dumps(profiler)]
    args = SimpleNamespace(model=plan['model'], max_model_len=16384, max_num_seqs=16, port=8017)
    try:
        dump(root/'status.json', dict(status='starting_server', updated_ns=time.time_ns()))
        with base.server(args, case/'config.yaml', case) as client:
            initial = base.await_empty(case)
            assert initial['used_bytes'] == initial['cpu_hot_bytes'] == initial['daos_puts'] == 0
            dump(case/'initial_sample.json', initial)
            health = base.LogHealth(case/'server.log')
            dump(root/'status.json', dict(status='cold', updated_ns=time.time_ns()))
            cold = cw.run_phases(case, records, 16, client, health, initial, phases=['cold'])
            dump(root/'status.json', dict(status='profiling_warm', updated_ns=time.time_ns()))
            client.post('/start_profile', timeout=60).raise_for_status()
            dump(case/'profile_start.json', dict(time_ns=time.time_ns(), config=profiler))
            final = cw.run_phases(case, warm, 16, client, health, cold, phases=['warm_profile'],
                warm_extra_gib=cap['conservative_stored_kv_gib']-cap['unique_input_kv_gib'])
            client.post('/stop_profile', timeout=180).raise_for_status()
            dump(case/'profile_stop.json', dict(time_ns=time.time_ns()))
            dump(case/'final_sample.json', final)
            assert list((case/'gpu_trace').glob('*.pt.trace.json.gz')), 'No CUDA profiler trace exported'
            assert final['daos_alloc_fail'] == 0 and final['dram_mirror']['errors'] == 0
        dump(case/'status.json', dict(status='completed', requests=len(records)+len(warm)))
        early.cleanup(case)
        dump(root/'status.json', dict(status='completed', analysis='pending', updated_ns=time.time_ns()))
    except BaseException as exc:
        dump(root/'status.json', dict(status='failed', error=repr(exc), updated_ns=time.time_ns()))
        raise
    finally:
        runtime.inference_command = original_command


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output.resolve())
