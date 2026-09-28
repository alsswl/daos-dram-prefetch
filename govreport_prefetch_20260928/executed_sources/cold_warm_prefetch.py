#!/usr/bin/env python3
"""One cold replay then one warm replay in the SAME process and namespace.

No backend policy changes: retain stores, read promotion, EOS and rolling input
admission. A fresh process and namespace isolate each OFF/ON/concurrency case.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import uuid

import yaml

import compare_e2e as common
from discovery_fixed_replay import await_empty, make_config, replay_one
from discovery_rolling_replay import LogHealth, rolling_requests
from staging_mixed_pressure import latest_sample, server


def drain(case, health, timeout=90):
    """GPU objects retain store/read/mirror lifetimes; require a stable drain."""
    deadline = time.monotonic() + timeout
    stable = 0
    while time.monotonic() < deadline:
        health()
        sample = latest_sample(case) or {}
        mirror = sample.get('dram_mirror') or {}
        prefetch = sample.get('cpu_prefetch') or {}
        if prefetch.get('copy_errors', 0):
            raise RuntimeError('CPU prefetch error; comparison invalid')
        if (sample and sample['used_bytes'] == 0 and mirror.get('pending_bytes') == 0
                and prefetch.get('deferred_pending_batches', 0) == 0):
            stable += 1
        else:
            stable = 0
        if stable >= 10:
            if mirror.get('errors'):
                raise RuntimeError('DRAM mirror error; comparison invalid')
            return sample
        time.sleep(.2)
    raise RuntimeError('GPU staging / asynchronous mirror failed to drain')


def run_case(a, case, records, concurrency, enabled, *, request_fn=None):
    request_fn = request_fn or replay_one
    namespace = 'minji-cold-warm-' + uuid.uuid4().hex
    cfg = make_config(enabled, namespace, cpu_gib=getattr(a, 'cpu_gib', 8),
                      staging_gib=getattr(a, 'staging_gib', 8))
    if hasattr(a, 'prefetch_workers'):
        cfg['extra_config']['daosgds.dram_prefetch_workers'] = a.prefetch_workers
    # Defaults preserve the original benchmark. Optional switches are used by
    # compare_queued_prefetch.py; both arms keep DRAM prefetch enabled.
    if hasattr(a, 'cancel_queued'):
        cfg['extra_config']['daosgds.dram_prefetch_cancel_queued'] = a.cancel_queued
        cfg['extra_config']['daosgds.dram_prefetch_early_ready'] = a.early_ready
    cfg['extra_config'].update({
        'storage_plugin.daosgds.module_path': 'lmcache_daos.capacity_probe_backend',
        'storage_plugin.daosgds.class_name': 'CapacityProbeBackend'})
    (case/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    active = None
    try:
        with server(a, case/'config.yaml', case) as client:
            initial = await_empty(case)
            common.dump(case/'initial_sample.json', initial)
            health = LogHealth(case/'server.log')
            health()
            previous = initial
            for name in ('cold', 'warm'):
                active = case/name
                active.mkdir()
                # warm inherits the exact drained cold process/cache, no reset.
                common.dump(active/'initial_sample.json', previous)
                (active/'metrics_before.txt').write_text(client.get('/metrics').text)
                phase = dict(name=name, concurrency=concurrency, arrival_mode='rolling',
                             start_ns=time.time_ns())
                common.dump(active/'phase.json', phase)
                calls = []
                def save(row):
                    calls.append(row)
                    common.dump(active/'replay_calls.json', sorted(calls, key=lambda r: r['index']))
                    if len(calls) % 32 == 0 or len(calls) == len(records):
                        print(f'{case.name}/{name} {len(calls)}/{len(records)} completed', flush=True)
                rolling_requests(records, concurrency, lambda r, b: request_fn(client, r, b), save, health)
                phase['end_ns'] = time.time_ns()
                common.dump(active/'phase.json', phase)
                (active/'metrics_after.txt').write_text(client.get('/metrics').text)
                previous = drain(case, health)
                common.dump(active/'final_sample.json', previous)
                common.dump(active/'status.json', dict(status='completed', requests=len(calls)))
                print(f'{case.name}/{name} drained: DRAM={previous["cpu_hot_bytes"]/2**30:.3f}GiB, '
                      f'DAOS successful puts={previous["daos_puts"]}', flush=True)
            common.dump(case/'final_sample.json', previous)
        common.dump(case/'status.json', dict(status='completed', requests=2*len(records)))
    except BaseException as exc:
        status = dict(status='failed', error=repr(exc))
        if active is not None and not (active/'status.json').exists():
            common.dump(active/'status.json', status)
        common.dump(case/'status.json', status)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--port', type=int, default=8017)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    a.model, a.max_model_len = 'Qwen/Qwen3-14B', 32768
    root = a.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    records = json.loads((common.ROOT/'discovery_capacity_matrix_20260927/requests.json').read_text())
    assert len(records) == 256
    assert all(r['index'] == i and hashlib.sha256(r['prompt'].encode()).hexdigest() == r['prompt_sha256']
               for i, r in enumerate(records))
    common.dump(root/'requests.json', records)
    cases = [dict(name=f'c{c}_{"on" if on else "off"}', concurrency=c, prefetch=on)
             for c, on in [(8, False), (8, True), (16, True), (16, False)]]
    names = ['cold_warm_prefetch.py', 'discovery_fixed_replay.py', 'discovery_rolling_replay.py',
             'staging_mixed_pressure.py', 'compare_e2e.py', 'run_vllm.sh', 'libdaosgdr.so',
             'lmcache_config_daosgds_async_dram.yaml', 'tests/test_cold_warm_prefetch.py']
    names += [str(p.relative_to(common.ROOT)) for p in sorted((common.ROOT/'lmcache_daos').glob('*.py'))]
    hashes = {}
    for name in names:
        dest = root/'executed_sources'/name; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(common.ROOT/name, dest)
        hashes[name] = hashlib.sha256(dest.read_bytes()).hexdigest()
    common.dump(root/'plan.json', dict(cases=cases, model=a.model, cpu_gib=8, staging_gib=8,
        chunk_tokens=128, requests_per_phase=256, arrival_mode='rolling', source_sha256=hashes,
        notes=['One cold + one warm per case; same live vLLM process and namespace between phases.',
               'Fresh process/empty DRAM/new DAOS namespace across cases, not shared-container deletion.',
               'Wait for shared GPU buffers and asynchronous DRAM mirror to drain before warm.',
               'Backend policies unchanged: stores, read promotion, physical capacity limits retained.',
               'DRAM prefetch only OFF/ON. DAOS prefetch remains enabled.',
               'Same 256 recorded prompts in each phase; no Python agent-tool execution.',
               'Keep EOS/Observation stop and max_tokens2048; output counts and wall-clock arrivals can differ.',
               'One trial per condition; cold and warm are distinct phases, not independent repetitions.',
               'No server/OS cache flush. Warm means reused working set, not guaranteed all-hit or identical tier placement.']))
    if a.dry_run:
        common.dump(root/'status.json', dict(status='dry_run')); return
    os.environ['DAOS_GDS_PREFETCH_TIMING'] = '1'
    try:
        cmd = [str(common.ROOT/'run_vllm.sh'), str(common.ROOT/'venv/bin/python3'),
               str(common.ROOT/'tests/object_gpu_roundtrip.py'), '--size-mib', '20']
        r = subprocess.run(cmd, cwd=common.ROOT, env=dict(os.environ, DAOSGDS_TRANSPORT='object'),
                           capture_output=True, text=True, timeout=120)
        (root/'storage_preflight.log').write_text(r.stdout+r.stderr)
        if r.returncode:
            raise RuntimeError('DAOS preflight failed; no inference started')
        for spec in cases:
            common.dump(root/'status.json', dict(status='running', case=spec['name']))
            case = root/spec['name']; case.mkdir()
            run_case(a, case, records, spec['concurrency'], spec['prefetch'])
        common.dump(root/'status.json', dict(status='completed'))
    except BaseException as exc:
        common.dump(root/'status.json', dict(status='failed', error=repr(exc)))
        raise


if __name__ == '__main__':
    main()
