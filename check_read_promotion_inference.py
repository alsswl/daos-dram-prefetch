#!/usr/bin/env python3
"""Prove DAOS-only cache becomes CPU hits after a fresh-process read, no new writes."""
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
    a = p.parse_args()
    a.model, a.max_tokens, a.max_model_len = 'Qwen/Qwen3-14B', 64, 16384
    a.context_tokens, a.chunk_size = [8192]*4, 128
    a.output.mkdir(parents=True, exist_ok=False)
    prompts, _, _ = common.workload(a)
    common.dump(a.output/'workload.json', prompts)
    ns = 'minji-read-promotion-inference-' + uuid.uuid4().hex + ':'
    rows_out = []
    for role in ['seed', 'consumer']:
        folder = a.output/role
        folder.mkdir()
        name = 'lmcache_config_daosgds_gpu_store.yaml' if role == 'seed' else 'lmcache_config_daosgds_async_dram.yaml'
        cfg = yaml.safe_load((common.ROOT/name).read_text())
        cfg['extra_config'].update({'daosgds.object_namespace': ns,
            'storage_plugin.daosgds.module_path': 'lmcache_daos.store_probe_backend',
            'storage_plugin.daosgds.class_name': 'StoreProbeBackend' if role == 'seed' else 'StoreProbeAsyncDramBackend'})
        if role == 'consumer':
            cfg['extra_config'].update({'daosgds.store': False, 'daosgds.dram_promote_on_read': True})
        config = folder/'config.yaml'
        config.write_text(yaml.safe_dump(cfg, sort_keys=False))
        with server(a, config, folder) as client:
            for phase in (['fill'] if role == 'seed' else ['daos_read', 'cpu_reuse']):
                start_ns = time.time_ns()
                target = client if role == 'seed' else ReadOnlyRequests(client)
                with ThreadPoolExecutor(max_workers=4) as executor:
                    rows = list(executor.map(lambda x: common.request(target, x, a), prompts))
                assert all(r['cached_tokens'] == (0 if role == 'seed' else 8191) for r in rows)
                sample = wait_drained(folder, 256 if role == 'seed' else 0)
                events = [e for e in read_events(folder) if e['time_ns'] >= start_ns]
                hits = {tier: sum(e['hit_chunks'] for e in events if e['event']=='tier_lookup' and e['tier']==tier)
                        for tier in ['dram', 'daos']}
                if role == 'seed':
                    assert sample['cpu_hot_chunks'] == 0
                    assert hits == {'dram': 0, 'daos': 0}
                else:
                    assert hits == ({'dram': 0, 'daos': 256} if phase=='daos_read' else {'dram': 256, 'daos': 0})
                    assert sample['daos_puts'] == 0
                    assert sample['dram_mirror']['write_copied'] == 0
                    assert sample['dram_mirror']['read_copied'] == 256
                    assert sample['dram_mirror']['errors'] == 0
                    assert sample['dram_mirror']['read_submit_errors'] == 0
                    assert sample['dram_mirror']['pending_bytes'] == 0
                    assert all(r['output_token_sha256'] == c['output_token_sha256']
                               for r,c in zip(rows, rows_out[0]['responses']))
                assert sample['daos_alloc_fail'] == 0
                rows_out.append(dict(phase=phase, responses=rows, tier_hit_chunks=hits, sample=sample))
                common.dump(a.output/'results.json', rows_out)
                print(phase, hits, 'CPU entries', sample['cpu_hot_chunks'], flush=True)
        text = (folder/'server.log').read_text()
        assert not any(x in text for x in ['negative ref', 'Traceback (most recent', 'CUDA error'])
    print('PASS', a.output, flush=True)


if __name__ == '__main__': main()
