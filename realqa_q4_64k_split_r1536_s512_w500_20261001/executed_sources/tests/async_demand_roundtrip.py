"""Live 20MiB payload equality for actual new OFF/ON configurations.

Only a UUID test key is deleted on success. No speculative worker overrides.
"""
import argparse
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
import uuid

import torch
import yaml
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager, EventType
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.storage_backend.storage_manager import StorageManager
from async_dram_promotion_roundtrip import wait_for
from sharegpt_async_payload_compare import config, dump


def check(folder, prefetch):
    folder.mkdir()
    os.environ['DAOS_GDS_STAGING_TRACE'] = str(folder/'trace')
    cfg = config(prefetch)
    cfg['max_local_cpu_size'] = .125
    cfg['extra_config'].update({'daosgds.gpu_buffer_gb': .125,
        'daosgds.dram_mirror_max_pending_gb': .0625, 'daosgds.dram_mirror_max_age_ms': 10000})
    (folder/'config.yaml').write_text(yaml.safe_dump(cfg))
    md = LMCacheMetadata(model_name='async-demand-byte-test', world_size=1, local_world_size=1,
        worker_id=0, local_worker_id=0, kv_dtype=torch.bfloat16,
        kv_shape=(40, 2, 128, 8, 128), role='worker', chunk_size=128)
    engine_cfg = LMCacheEngineConfig.from_dict(cfg)
    PinMonitor.GetOrCreate(engine_cfg)
    notices = {}
    manager = StorageManager(engine_cfg, md, EventManager(), async_lookup_server=SimpleNamespace(
        send_response_to_scheduler=lambda rid,n: notices.__setitem__(rid,n)))
    be, cpu = manager.allocator_backend, manager.local_cpu_backend
    key = CacheEngineKey(md.model_name, 1, 0, uuid.uuid4().int, torch.bfloat16)
    live = []
    try:
        obj = manager.allocate(md.get_shapes(), md.get_dtypes(), fmt=None)
        assert obj is not None and obj.tensor.is_cuda
        obj.tensor.copy_(torch.randn_like(obj.tensor))
        torch.cuda.synchronize()
        expected = obj.tensor.cpu().view(torch.uint8)
        manager.batched_put([key], [obj])
        wait_for(lambda: cpu.contains(key) and not be.exists_in_put_tasks(key)
                 and be.dram_mirror.snapshot()['pending_bytes'] == 0
                 and be.memory_allocator.allocator.total_allocated_size == 0)
        for tier in ('daos', 'dram'):
            if tier == 'daos':
                assert cpu.remove(key)
            else:
                assert cpu.contains(key)
            rid = f'{tier}-{prefetch}'
            asyncio.run_coroutine_threadsafe(manager.async_lookup_and_prefetch(
                rid, [key], [0,128], pin=True), manager.loop).result(timeout=20)
            assert notices[rid] == 128
            view = manager.event_manager.get_event_future(EventType.LOADING,rid).result()[0][0][1]
            live.append(view)
            assert view.batch.tier == tier
            if prefetch:
                view.batch.future.result(timeout=20)
            else:
                assert not view.batch.future.done()
                assert be.memory_allocator.allocator.total_allocated_size == 0
            view.batch.resolve()
            assert torch.equal(view.tensor.cpu().view(torch.uint8), expected)
            assert view.tensor.is_cuda == (prefetch or tier == 'daos')
            view.ref_count_down()
            live.remove(view)
            manager.event_manager.pop_event(EventType.LOADING,rid)
            wait_for(lambda: be.dram_mirror.snapshot()['pending_bytes'] == 0
                     and be.memory_allocator.allocator.total_allocated_size == 0)
            assert view.batch.is_retired and cpu.contains(key)
        assert cpu.remove(key) and be.remove(key)
        dump(folder/'result.json', dict(result='PASS', prefetch=prefetch, bytes=20*2**20,
            paths=['daos','dram'], staging_bytes_after=0, own_key_removed=True))
    finally:
        for view in live:
            view.batch.abort()
        manager.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--arm', choices=['off','on'])
    args = parser.parse_args()
    if args.arm:
        check(args.output, args.arm == 'on')
    else:
        import subprocess
        import sys
        args.output.mkdir()
        for arm in ('off','on'):
            subprocess.run([sys.executable, __file__, '--output', str(args.output/arm), '--arm', arm], check=True)
        dump(args.output/'result.json', dict(result='PASS', arms=['off','on']))
