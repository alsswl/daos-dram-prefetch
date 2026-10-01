"""Live GPU/DAOS demand read, CPU promotion and no-lookup-allocation check."""
import argparse
import json
import os
from pathlib import Path
import uuid

import torch
import yaml
from lmcache.utils import CacheEngineKey
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.event_manager import EventManager
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.storage_backend.storage_manager import StorageManager

from lmcache_daos.demand_read_backend import _operation
from async_dram_promotion_roundtrip import wait_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    folder = parser.parse_args().output.resolve()
    folder.mkdir(exist_ok=False)
    os.environ['DAOS_GDS_STAGING_TRACE'] = str(folder/'trace')
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root/'lmcache_config_daosgds_async_dram.yaml').read_text())
    ns = 'minji-demand-test-' + uuid.uuid4().hex
    cfg.update(enable_async_loading=False, max_local_cpu_size=.125)
    cfg['extra_config'].update({
        'storage_plugin.daosgds.module_path': 'lmcache_daos.demand_read_backend',
        'storage_plugin.daosgds.class_name': 'DemandReadBackend',
        'daosgds.demand_read_only': True, 'daosgds.dram_prefetch': False,
        'daosgds.dram_promote_on_read': True, 'daosgds.transport': 'object',
        'daosgds.object_namespace': ns+':', 'daosgds.root': '/'+ns,
        'daosgds.gpu_buffer_gb': .125, 'daosgds.dram_mirror_max_pending_gb': .0625,
        'daosgds.dram_mirror_max_age_ms': 10000})
    (folder/'config.yaml').write_text(yaml.safe_dump(cfg))
    md = LMCacheMetadata(model_name='demand-read-bytes', world_size=1, local_world_size=1,
        worker_id=0, local_worker_id=0, kv_dtype=torch.bfloat16,
        kv_shape=(40, 2, 128, 8, 128), role='worker', chunk_size=128)
    config = LMCacheEngineConfig.from_dict(cfg)
    PinMonitor.GetOrCreate(config)
    manager = StorageManager(config, md, EventManager())
    be, cpu = manager.allocator_backend, manager.local_cpu_backend
    key = CacheEngineKey(md.model_name, 1, 0, uuid.uuid4().int, torch.bfloat16)
    objects = []
    try:
        obj = manager.allocate(md.get_shapes(), md.get_dtypes(), fmt=None)
        source = torch.randn_like(obj.tensor)
        obj.tensor.copy_(source)
        torch.cuda.synchronize()
        manager.batched_put([key], [obj])
        wait_for(lambda: cpu.contains(key) and not be.exists_in_put_tasks(key)
                 and be.dram_mirror.snapshot()['pending_bytes'] == 0)
        assert cpu.remove(key)
        wait_for(lambda: be.memory_allocator.allocator.total_allocated_size == 0)
        token = _operation.set(('lookup', 'live-test'))
        try:
            hit, mapping = manager.batched_contains([key])
            assert hit == 1
            assert not cpu.contains(key)
            assert be.stats['get'] == 0
            assert be.memory_allocator.allocator.total_allocated_size == 0
        finally:
            _operation.reset(token)
        trace = be._staging_trace
        trace.emit('retrieve_start', request_id='live-test')
        token = _operation.set(('retrieve', 'live-test'))
        try:
            objects = manager.batched_get([key], location=next(iter(mapping)))
            assert len(objects) == 1 and objects[0].tensor.is_cuda
            assert torch.equal(objects[0].tensor.view(torch.uint8), source.view(torch.uint8))
        finally:
            _operation.reset(token)
            trace.emit('retrieve_return', request_id='live-test')
        be._release_memory_obj(objects.pop())
        wait_for(lambda: cpu.contains(key) and be.dram_mirror.snapshot()['pending_bytes'] == 0)
        objects = manager.batched_get([key], location='LocalCPUBackend')
        assert not objects[0].tensor.is_cuda
        assert torch.equal(objects[0].tensor.view(torch.uint8), source.cpu().view(torch.uint8))
        objects.pop().ref_count_down()
        assert be.memory_allocator.allocator.total_allocated_size == 0
        assert be.dram_mirror.snapshot()['read_copied'] == 1
        assert cpu.remove(key)
        assert be.remove(key)
        (folder/'result.json').write_text(json.dumps(dict(result='PASS', bytes=20*2**20,
            lookup_payload_reads=0, lookup_staging_bytes=0, demand_gpu_bytes_equal=True,
            promoted_cpu_bytes_equal=True, gpu_objects_retained_in_cpu=False,
            cleanup='own UUID key removed'), indent=2)+'\n')
        print('DEMAND_READ_ROUNDTRIP_PASS', folder)
    finally:
        for obj in objects:
            if obj is not None: obj.ref_count_down()
        manager.close()


if __name__ == '__main__':
    main()
