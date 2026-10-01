"""Live 20MiB GPU test: metadata-first ready/queued paths for DAOS and DRAM.

Deliberate job deferral ONLY in this byte-validation test, never the benchmark.
Uses one UUID key, deletes only that key on success; preserves evidence on failure.
"""
import argparse
import asyncio
import json
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    folder = parser.parse_args().output.resolve()
    folder.mkdir(exist_ok=False)
    os.environ['DAOS_GDS_STAGING_TRACE'] = str(folder/'trace')
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root/'lmcache_config_daosgds_async_dram.yaml').read_text())
    ns = 'minji-early-test-'+uuid.uuid4().hex
    cfg.update(enable_async_loading=True, max_local_cpu_size=.125)
    cfg['extra_config'].update({
        'storage_plugin.daosgds.module_path': 'lmcache_daos.early_lookup_backend',
        'storage_plugin.daosgds.class_name': 'EarlyLookupBackend',
        'daosgds.early_lookup': True, 'daosgds.dram_prefetch': True,
        'daosgds.dram_prefetch_early_ready': False, 'daosgds.dram_prefetch_cancel_queued': False,
        'daosgds.dram_prefetch_policy': 'capacity',
        'daosgds.dram_promote_on_read': True, 'daosgds.transport': 'object',
        'daosgds.object_namespace': ns+':', 'daosgds.root': '/'+ns,
        'daosgds.gpu_buffer_gb': .125, 'daosgds.dram_mirror_max_pending_gb': .0625,
        'daosgds.dram_mirror_max_age_ms': 10000})
    (folder/'config.yaml').write_text(yaml.safe_dump(cfg))
    md = LMCacheMetadata(model_name='early-lookup-bytes', world_size=1, local_world_size=1,
        worker_id=0, local_worker_id=0, kv_dtype=torch.bfloat16,
        kv_shape=(40, 2, 128, 8, 128), role='worker', chunk_size=128)
    config = LMCacheEngineConfig.from_dict(cfg)
    PinMonitor.GetOrCreate(config)
    notices = {}
    manager = StorageManager(config, md, EventManager(), async_lookup_server=SimpleNamespace(
        send_response_to_scheduler=lambda rid,n: notices.__setitem__(rid,n)))
    be, cpu = manager.allocator_backend, manager.local_cpu_backend
    coordinator = be.early_lookup
    original_start = coordinator.start
    key = CacheEngineKey(md.model_name, 1, 0, uuid.uuid4().int, torch.bfloat16)
    live_views = []
    passed = []
    try:
        obj = manager.allocate(md.get_shapes(), md.get_dtypes(), fmt=None)
        assert obj is not None and obj.tensor.is_cuda
        source = torch.randn_like(obj.tensor)
        obj.tensor.copy_(source)
        torch.cuda.synchronize()
        expected = source.cpu().view(torch.uint8)
        manager.batched_put([key], [obj])
        wait_for(lambda: cpu.contains(key) and not be.exists_in_put_tasks(key)
                 and be.dram_mirror.snapshot()['pending_bytes'] == 0)
        wait_for(lambda: be.memory_allocator.allocator.total_allocated_size == 0)

        for tier, queued in (('daos', False), ('dram', False), ('daos', True), ('dram', True)):
            if tier == 'daos':
                assert cpu.remove(key)
            else:
                assert cpu.contains(key)
            rid = tier+('-queued' if queued else '-ready')
            # Hold submission only to deterministically validate queued takeover.
            coordinator.start = (lambda *args: None) if queued else original_start
            asyncio.run_coroutine_threadsafe(manager.async_lookup_and_prefetch(
                rid, [key], [0, 128], pin=True), manager.loop).result(timeout=20)
            assert notices[rid] == 128
            tiers = manager.event_manager.get_event_future(EventType.LOADING, rid).result()
            view = tiers[0][0][1]
            live_views.append(view)
            assert view.batch.tier == tier
            if queued:
                assert not view.batch.future.running() and not view.batch.future.done()
            else:
                view.batch.future.result(timeout=20)
            view.batch.resolve()
            assert torch.equal(view.tensor.cpu().view(torch.uint8), expected)
            assert view.tensor.is_cuda == (tier == 'daos' or not queued)
            if queued:
                assert view.batch.future.cancelled()
                assert not view.batch.claim(), 'Cancelled speculative job restarted'
            view.ref_count_down()
            live_views.remove(view)
            manager.event_manager.pop_event(EventType.LOADING, rid)
            coordinator.start = original_start
            wait_for(lambda: be.dram_mirror.snapshot()['pending_bytes'] == 0
                     and be.memory_allocator.allocator.total_allocated_size == 0)
            assert cpu.contains(key)
            assert view.batch.is_retired
            passed.append(rid)
        assert cpu.remove(key), 'CPU pin/ref leak'
        assert be.remove(key)
        (folder/'result.json').write_text(json.dumps(dict(result='PASS', bytes=20*2**20,
            byte_equal_paths=passed, staging_bytes_after=0, own_key_removed=True), indent=2)+'\n')
        print('EARLY_LOOKUP_ROUNDTRIP_PASS', folder)
    finally:
        coordinator.start = original_start
        for view in live_views:
            view.batch.abort()
        manager.close()


if __name__ == '__main__':
    main()
