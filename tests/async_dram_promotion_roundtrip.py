"""Live read promotion: delayed D2H must not block GET or publish partial CPU data."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import threading
import time
import uuid

import torch
import yaml
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.storage_backend.storage_manager import StorageManager


def wait_for(predicate):
    deadline = time.monotonic() + 30
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError('copy/store did not complete')
        time.sleep(.005)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--transport', choices=['object', 'dfs'], default='object')
    p.add_argument('--prefetch', action='store_true')
    p.add_argument('--capacity', action='store_true', help='Disable watermark; test actual pool exhaustion')
    a = p.parse_args()
    if a.capacity and not a.prefetch: p.error('--capacity requires --prefetch')
    if a.result.exists(): p.error('result file exists')
    os.environ['DAOSGDS_TRANSPORT'], os.environ['DAOS_PROBE_CHUNKS'] = a.transport, '0'
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root/'lmcache_config_daosgds_async_dram.yaml').read_text())
    ns = 'minji-read-promote-' + uuid.uuid4().hex
    cfg['max_local_cpu_size'] = .125
    cfg['extra_config'].update({'daosgds.transport': a.transport, 'daosgds.root': '/'+ns,
        'daosgds.object_namespace': ns+':', 'daosgds.gpu_buffer_gb': .125,
        'daosgds.dram_mirror_max_pending_gb': .0625, 'daosgds.dram_mirror_max_age_ms': 10000,
        'daosgds.dram_promote_on_read': True, 'daosgds.dram_prefetch': a.prefetch,
        'daosgds.cpu_prefetch_gpu_gb': .0625,
        'daosgds.dram_prefetch_policy': 'capacity' if a.capacity else 'watermark'})
    md = LMCacheMetadata(model_name='read-promotion-bytes', world_size=1, local_world_size=1,
        worker_id=0, local_worker_id=0, kv_dtype=torch.bfloat16,
        kv_shape=(40, 2, 128, 8, 128), role='worker', chunk_size=128)
    config = LMCacheEngineConfig.from_dict(cfg)
    PinMonitor.GetOrCreate(config)
    manager = StorageManager(config, md, EventManager())
    be, cpu = manager.allocator_backend, manager.local_cpu_backend
    key = CacheEngineKey(md.model_name, 1, 0, uuid.uuid4().int, torch.bfloat16)
    release, entered = threading.Event(), threading.Event()
    fetched = None
    success = False
    try:
        obj = manager.allocate(md.get_shapes(), md.get_dtypes(), fmt=None)
        assert obj.tensor.is_cuda
        source = torch.randn_like(obj.tensor)
        obj.tensor.copy_(source)
        torch.cuda.synchronize()
        manager.batched_put([key], [obj])
        wait_for(lambda: cpu.contains(key) and be.dram_mirror.snapshot()['pending_bytes'] == 0
                 and not be.exists_in_put_tasks(key))
        assert cpu.remove(key)
        original_copy = be.dram_mirror.copy_payload
        def delayed(*args):
            entered.set()
            assert release.wait(10), 'GET incorrectly waited for D2H'
            return original_copy(*args)
        be.dram_mirror.copy_payload = delayed
        fetched = asyncio.run(be.batched_get_non_blocking('test-read', [key]))[0]
        assert entered.wait(3)
        assert not cpu.contains(key), 'CPU data published before copy finished'
        assert torch.equal(fetched.tensor.view(torch.uint8), source.view(torch.uint8))
        be._release_memory_obj(fetched)
        fetched = None
        assert be.memory_allocator.allocator.total_allocated_size > 0, 'source freed during DMA'
        release.set()
        wait_for(lambda: cpu.contains(key) and be.dram_mirror.snapshot()['pending_bytes'] == 0)
        assert be.memory_allocator.allocator.total_allocated_size == 0
        assert asyncio.run(cpu.batched_async_contains('next-request', [key])) == 1
        cpu_obj = cpu.get_blocking(key)
        try:
            assert torch.equal(cpu_obj.tensor.view(torch.uint8), source.cpu().view(torch.uint8))
        finally:
            cpu_obj.ref_count_down()
        assert be.dram_mirror.snapshot()['read_copied'] == 1
        assert be.dram_mirror.snapshot()['write_copied'] == 1
        if a.prefetch:
            from lmcache_daos.dram_prefetch_backend import StagedCPUObject, discard
            assert asyncio.run(cpu.batched_async_contains('prefetch-next', [key], pin=True)) == 1
            staged = asyncio.run(cpu.batched_get_non_blocking('prefetch-next', [key]))
            try:
                assert isinstance(staged[0], StagedCPUObject)
                assert torch.equal(staged[0].tensor.view(torch.uint8), source.view(torch.uint8))
            finally:
                discard(staged)
            assert be.memory_allocator.allocator.total_allocated_size == 0
            # An exhausted admission budget must retain a valid CPU hit.
            held = []
            if a.capacity:
                assert be.cpu_prefetch.limit_bytes is None
                # 6 x 20MiB in a 128MiB pool leaves less than one chunk free.
                for _ in range(6):
                    held.append(be.memory_allocator.allocate(md.get_shapes(), md.get_dtypes(),
                                                            cpu_obj.metadata.fmt))
                    assert held[-1] is not None
            else:
                be.cpu_prefetch.limit_bytes = 1
            assert asyncio.run(cpu.batched_async_contains('fallback', [key], pin=True)) == 1
            fallback = asyncio.run(cpu.batched_get_non_blocking('fallback', [key]))
            try:
                assert not isinstance(fallback[0], StagedCPUObject)
                assert torch.equal(fallback[0].tensor.view(torch.uint8), source.cpu().view(torch.uint8))
            finally:
                discard(fallback)
                for held_obj in held:
                    be._release_memory_obj(held_obj)
            assert be.memory_allocator.allocator.total_allocated_size == 0
        # Read promotion OFF must leave a DAOS-only read and no CPU entry.
        be.promote_on_read = False
        assert cpu.remove(key)
        fetched = be.get_blocking(key)
        assert fetched is not None and not cpu.contains(key)
        assert torch.equal(fetched.tensor.view(torch.uint8), source.view(torch.uint8))
        be._release_memory_obj(fetched)
        fetched = None
        assert be.remove(key)
        result = dict(result='PASS', transport=a.transport, bytes=source.numel()*source.element_size(),
            async_get_returned_while_d2h_blocked=True, cpu_invisible_before_copy=True,
            source_retained_until_copy=True, promoted_cpu_payload='byte_equal',
            next_cpu_lookup_hit=True, promotion_off_verified=True,
            dram_prefetch_enabled=a.prefetch,
            physical_capacity_fallback_tested=a.capacity,
            prefetch_stats=be.cpu_prefetch.stats if a.prefetch else None,
            stats=be.dram_mirror.snapshot(), cleanup='own UUID key removed; DFS directory retained')
        a.result.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result))
        success = True
    finally:
        release.set()
        if fetched is not None: be._release_memory_obj(fetched)
        if not success: print('Failure: own test data retained:', ns)
        manager.close()


if __name__ == '__main__': main()
