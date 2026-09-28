#!/usr/bin/env python3
"""Paired cold-write / warm-read comparison; no DRAM retention in either arm."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time
import uuid

import yaml

import compare_e2e as common
from staging_mixed_pressure import server, wait_drained, read_events, ReadOnlyRequests

ROOT = common.ROOT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--port', type=int, default=8017)
    a = p.parse_args()
    if a.repeats < 1:
        p.error('repeats must be positive')
    a.model, a.max_tokens, a.max_model_len = 'Qwen/Qwen3-14B', 64, 16384
    a.context_tokens, a.chunk_size = [8192] * 8, 128
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    prompts, warmup, desc = common.workload(a)
    common.dump(root/'workload.json', dict(prompts=prompts, warmup=warmup, description=desc))
    base = yaml.safe_load((ROOT/'lmcache_config_daosgds_unified.yaml').read_text())
    base.update(chunk_size=128, local_cpu=False, max_local_cpu_size=8,
                enable_async_loading=True, use_layerwise=False, store_location=None)
    base['extra_config'].update({'daosgds.transport': 'object', 'daosgds.gpu_buffer_gb': 10,
        'daosgds.io_workers': 16, 'daosgds.meta_workers': 16, 'daosgds.store': True,
        'storage_plugin.daosgds.module_path': 'lmcache_daos.store_probe_backend',
        'storage_plugin.daosgds.class_name': 'StoreProbeBackend', 'daosgds.probe_interval_ms': 5})
    hashes = {}
    for name in ['compare_gpu_store.py', 'compare_e2e.py', 'staging_mixed_pressure.py',
                 'lmcache_daos/gpu_store.py', 'lmcache_daos/gds_backend.py',
                 'lmcache_daos/store_probe_backend.py', 'lmcache_daos/staging_probe_backend.py',
                 'lmcache_daos/staging_trace.py', 'run_vllm.sh', 'libdaosgdr.so']:
        dest = root/'executed_sources'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
        hashes[name] = hashlib.sha256(dest.read_bytes()).hexdigest()
    common.dump(root/'plan.json', dict(model=a.model, tokens=8192, generation=64,
        chunk=128, staging_gib=10, cpu_retention=False, cpu_allocator_gib=8,
        repeats=a.repeats, transport='object', source_hashes=hashes,
        note='Fixed-token micro-workload, not agentic DiscoveryBench. Each arm has its own new namespace.'))
    all_rows = []
    for repeat in range(1, a.repeats + 1):
        order = ['host_staged', 'gpu_direct'] if repeat % 2 else ['gpu_direct', 'host_staged']
        for mode in order:
            folder = root/f'r{repeat}_{mode}'
            folder.mkdir()
            cfg = json.loads(json.dumps(base))
            cfg['extra_config']['daosgds.store_path'] = mode
            cfg['extra_config']['daosgds.object_namespace'] = 'minji-store-ab-' + uuid.uuid4().hex + ':'
            config = folder/'config.yaml'
            config.write_text(yaml.safe_dump(cfg, sort_keys=False))
            with server(a, config, folder) as client:
                warm = common.request(client, warmup, a, max_tokens=1)
                assert warm['cached_tokens'] == 0
                puts = len(warmup['tokens']) // a.chunk_size
                wait_drained(folder, puts)
                warm = common.request(ReadOnlyRequests(client), warmup, a, max_tokens=1)
                assert warm['cached_tokens'] == common.expected_hit(warmup, a.chunk_size)
                wait_drained(folder, puts)
                for concurrency, batch in [(1, prompts[:4]), (4, prompts[4:])]:
                    for phase in ['cold', 'warm']:
                        start_ns, start = time.time_ns(), time.perf_counter()
                        target = client if phase == 'cold' else ReadOnlyRequests(client)
                        with ThreadPoolExecutor(max_workers=concurrency) as executor:
                            rows = list(executor.map(lambda prompt: common.request(target, prompt, a), batch))
                        http_ms = (time.perf_counter() - start) * 1000
                        http_end_ns = time.time_ns()
                        expected = 0 if phase == 'cold' else common.expected_hit(batch[0], a.chunk_size)
                        assert all(r['cached_tokens'] == expected for r in rows), rows
                        if phase == 'cold':
                            puts += sum(len(prompt['tokens']) // a.chunk_size for prompt in batch)
                        sample = wait_drained(folder, puts)
                        assert sample['cpu_hot_chunks'] == 0
                        assert sample['daos_alloc_fail'] == 0
                        end_ns = time.time_ns()
                        events = [e for e in read_events(folder) if start_ns <= e['time_ns'] <= end_ns]
                        gathers = [e for e in events if e['event'] == 'store_gather_start']
                        copies = [e for e in events if e['event'] == 'store_manager_copy']
                        if phase == 'cold':
                            assert sum(e['bytes'] for e in gathers) == 4 * 8192 * 163840
                            assert all(all(d.startswith('cuda') if mode == 'gpu_direct' else d == 'cpu'
                                           for d in e['devices']) for e in gathers)
                            if mode == 'gpu_direct':
                                assert not copies, 'Unexpected manager copy in direct path'
                        completed = next((e['time_ns'] for e in events if e['event'] == 'occupancy_sample'
                            and e['daos_puts'] >= puts and e['used_bytes'] == 0), end_ns)
                        record = dict(repeat=repeat, mode=mode, concurrency=concurrency, phase=phase,
                            http_batch_ms=http_ms, http_plus_drain_ms=(max(http_end_ns, completed)-start_ns)/1e6,
                            responses=rows, gather_chunks=sum(e['chunks'] for e in gathers),
                            gather_bytes=sum(e['bytes'] for e in gathers),
                            manager_copy_bytes=sum(e['bytes'] for e in copies),
                            manager_copy_ms=sum(e['ms'] for e in copies),
                            peak_staging_gib=max((e['used_bytes'] for e in events), default=0)/2**30,
                            daos_puts=sample['daos_puts'], allocation_failures=sample['daos_alloc_fail'])
                        all_rows.append(record)
                        common.dump(folder/f'c{concurrency}_{phase}.json', record)
                        common.dump(root/'results.json', all_rows)
                        print(f'r{repeat} {mode} c{concurrency} {phase}: {http_ms:.1f}ms; '
                              f'copy={record["manager_copy_bytes"]/2**30:.2f}GiB', flush=True)
            text = (folder/'server.log').read_text()
            forbidden = ['negative ref', 'DaosGdsBackend put ', 'CUDA error', 'Traceback (most recent']
            assert not any(s in text for s in forbidden), 'Inspect server.log for errors'
    print('COMPLETE', root, flush=True)


if __name__ == '__main__':
    main()
