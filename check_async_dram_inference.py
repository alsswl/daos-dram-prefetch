#!/usr/bin/env python3
"""Qwen3-14B functional test, not a repeated performance comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import uuid

import yaml

import compare_e2e as common
from staging_mixed_pressure import server, wait_drained, read_events, ReadOnlyRequests


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--port', type=int, default=8017)
    p.add_argument('--cpu-gb', type=float, default=8)
    p.add_argument('--require-mixed', action='store_true')
    a = p.parse_args()
    a.model, a.max_tokens, a.max_model_len = 'Qwen/Qwen3-14B', 64, 16384
    a.context_tokens, a.chunk_size = [8192] * 4, 128
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load((common.ROOT/'lmcache_config_daosgds_async_dram.yaml').read_text())
    cfg['max_local_cpu_size'] = a.cpu_gb
    cfg['extra_config'].update({'storage_plugin.daosgds.module_path': 'lmcache_daos.store_probe_backend',
        'storage_plugin.daosgds.class_name': 'StoreProbeAsyncDramBackend',
        'daosgds.object_namespace': 'minji-async-dram-inference-' + uuid.uuid4().hex + ':'})
    config = a.output/'config.yaml'
    config.write_text(yaml.safe_dump(cfg, sort_keys=False))
    prompts, _, description = common.workload(a)
    common.dump(a.output/'workload.json', prompts)
    results = []
    with server(a, config, a.output) as client:
        for phase in ['cold', 'warm1', 'warm2']:
            start_ns = time.time_ns()
            target = client if phase == 'cold' else ReadOnlyRequests(client)
            with ThreadPoolExecutor(max_workers=4) as executor:
                rows = list(executor.map(lambda x: common.request(target, x, a), prompts))
            assert all(r['cached_tokens'] == (0 if phase == 'cold' else 8191) for r in rows)
            sample = wait_drained(a.output, 256)
            assert sample['daos_alloc_fail'] == 0
            assert sample['dram_mirror']['errors'] == 0
            assert sample['dram_mirror']['pending_bytes'] == 0
            assert sample['cpu_hot_chunks'] > 0
            events = [e for e in read_events(a.output) if e['time_ns'] >= start_ns]
            hits = {tier: sum(e['hit_chunks'] for e in events if e['event'] == 'tier_lookup' and e['tier'] == tier)
                    for tier in ['dram', 'daos']}
            if phase != 'cold':
                assert hits['dram'] > 0
                if a.require_mixed:
                    assert hits['daos'] > 0
                assert all(r['output_token_sha256'] == c['output_token_sha256']
                           for r, c in zip(rows, results[0]['responses']))
            gathers = [e for e in events if e['event'] == 'store_gather_start']
            if phase == 'cold':
                assert sum(e['bytes'] for e in gathers) == 5 * 2**30
                assert all(d.startswith('cuda') for e in gathers for d in e['devices'])
                assert not any(e['event'] == 'store_manager_copy' for e in events)
            results.append(dict(phase=phase, responses=rows, sample=sample, tier_hit_chunks=hits))
            common.dump(a.output/'results.json', results)
            print(phase, 'tier hits:', hits, 'mirror:', sample['dram_mirror'], flush=True)
    text = (a.output/'server.log').read_text()
    assert not any(x in text for x in ['negative ref', 'Traceback (most recent', 'CUDA error'])
    print('PASS', a.output, flush=True)


if __name__ == '__main__':
    main()
