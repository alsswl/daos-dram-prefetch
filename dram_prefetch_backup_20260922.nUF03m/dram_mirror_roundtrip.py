#!/usr/bin/env python3
"""Live CPU retention + async DAOS mirror byte check through StorageManager.

Run with run_vllm.sh on an idle GPU. Uses one UUID key and removes only that
test entry after a successful verification; failure data is retained.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time
import uuid

import torch
import yaml
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager
from lmcache.v1.memory_management import MemoryFormat
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.storage_manager import StorageManager
from run_dram_cache import build_profile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dram', choices=['off', 'on'], required=True)
    p.add_argument('--transport', choices=['object', 'dfs'], required=True)
    p.add_argument('--result', type=Path)
    a = p.parse_args()
    if a.result is not None and a.result.exists():
        p.error('result path already exists')
    os.environ['DAOSGDS_TRANSPORT'] = a.transport
    os.environ['DAOS_PROBE_CHUNKS'] = '0'
    root = Path(__file__).resolve().parents[1]
    cfg = build_profile(yaml.safe_load((root/'lmcache_config_daosgds_unified.yaml').read_text()),
                        a.dram, .125)
    ns = 'minji-dram-bytes-' + uuid.uuid4().hex
    cfg.update(chunk_size=128)
    cfg['extra_config'].update({'daosgds.transport': a.transport, 'daosgds.root': '/'+ns,
        'daosgds.object_namespace': ns+':', 'daosgds.dfs_oclass': (1 << 24) | 0xffff,
        'daosgds.gpu_buffer_gb': .125, 'daosgds.io_workers': 2, 'daosgds.meta_workers': 2})
    config = LMCacheEngineConfig.from_dict(cfg)
    md = LMCacheMetadata(model_name='dram-byte-check', world_size=1, local_world_size=1,
                         worker_id=0, local_worker_id=0, kv_dtype=torch.float16,
                         kv_shape=(1, 2, 128, 1, 2048), role='worker', chunk_size=128)
    manager = StorageManager(config, md, EventManager())
    cpu = manager.local_cpu_backend
    daos = next(b for b in manager.storage_backends.values() if str(b) == 'DaosGdsBackend')
    key = CacheEngineKey(md.model_name, 1, 0, uuid.uuid4().int, torch.float16)
    succeeded = False
    fetched = []
    try:
        torch.manual_seed(20260922)
        source = torch.randint(0, 256, tuple(md.get_shapes()[0]), device='cuda').half()
        expected = source.cpu().view(torch.uint8)
        obj = manager.allocate(md.get_shapes(), md.get_dtypes(), fmt=MemoryFormat.KV_2LTD)
        assert obj is not None and obj.tensor.device.type == 'cpu'
        obj.tensor.copy_(source, non_blocking=True)
        torch.cuda.synchronize()
        manager.batched_put([key], [obj])  # Real fan-out, H2D copy and async DAOS put.
        deadline = time.monotonic() + 30
        while True:
            with daos._put_lock:
                pending = bool(daos._put_tasks)
            if not pending:
                break
            if time.monotonic() > deadline:
                raise TimeoutError('DAOS async put did not drain')
            time.sleep(.01)
        assert daos.stats['put'] == 1, daos.stats
        assert cpu.contains(key) == (a.dram == 'on')
        cpu_entries_after_put = len(cpu.hot_cache)
        if a.dram == 'on':
            cpu_objs = asyncio.run(cpu.batched_get_non_blocking('cpu-check', [key]))
            try:
                assert cpu_objs[0] is obj  # Original host buffer, no duplicate D2H.
                assert torch.equal(cpu_objs[0].tensor.view(torch.uint8), expected)
            finally:
                for cpu_obj in cpu_objs:
                    cpu_obj.ref_count_down()
            assert cpu.remove(key)  # Only drop local copy; DAOS must survive.
        assert not cpu.contains(key)
        fetched = asyncio.run(daos.batched_get_non_blocking('fallback-check', [key]))
        assert len(fetched) == 1 and fetched[0].tensor.device.type == 'cuda'
        assert torch.equal(fetched[0].tensor.cpu().view(torch.uint8), expected)
        assert daos.remove(key)
        succeeded = True
        result = dict(result='PASS', dram=a.dram, transport=a.transport,
            bytes=expected.numel(), cpu_entries_after_put=cpu_entries_after_put,
            cpu_payload='match' if a.dram == 'on' else 'not retained',
            daos_payload='match', daos_survives_cpu_removal=True,
            namespace=ns, cleanup='only test DAOS entry removed; DFS directory retained')
        print(json.dumps(result))
        if a.result is not None:
            a.result.write_text(json.dumps(result, indent=2) + '\n')
    finally:
        for obj in fetched:
            daos._release_memory_obj(obj)
        if not succeeded:
            print(f'Failure: test data retained for inspection: {ns}')
        manager.close()


if __name__ == '__main__':
    main()
