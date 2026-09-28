"""Live manager -> DAOS -> asynchronous CPU side-cache validation."""
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
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.storage_manager import StorageManager


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result', type=Path, required=True)
    p.add_argument('--pending-mib', type=float, default=64)
    p.add_argument('--expect-mirror', choices=['yes', 'no'], default='yes')
    a = p.parse_args()
    if a.result.exists():
        p.error('result file already exists')
    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root/'lmcache_config_daosgds_async_dram.yaml').read_text())
    ns = 'minji-async-dram-test-' + uuid.uuid4().hex + ':'
    cfg['max_local_cpu_size'] = .125
    cfg['extra_config'].update({'daosgds.object_namespace': ns, 'daosgds.gpu_buffer_gb': .125,
        'daosgds.dram_mirror_max_pending_gb': a.pending_mib/1024, 'daosgds.dram_mirror_max_age_ms': 1000})
    os.environ['DAOS_PROBE_CHUNKS'] = '0'
    md = LMCacheMetadata(model_name='async-dram-byte-check', world_size=1, local_world_size=1,
        worker_id=0, local_worker_id=0, kv_dtype=torch.bfloat16,
        kv_shape=(40, 2, 128, 8, 128), role='worker', chunk_size=128)
    manager = StorageManager(LMCacheEngineConfig.from_dict(cfg), md, EventManager())
    be = manager.allocator_backend
    cpu = manager.local_cpu_backend
    key = CacheEngineKey(md.model_name, 1, 0, uuid.uuid4().int, torch.bfloat16)
    fetched = None
    success = False
    try:
        obj = manager.allocate(md.get_shapes(), md.get_dtypes(), fmt=None)
        assert obj.tensor.is_cuda
        source = torch.randn_like(obj.tensor)
        obj.tensor.copy_(source)
        torch.cuda.synchronize()
        manager.batched_put([key], [obj])
        deadline = time.monotonic() + 30
        while True:
            with be._put_lock:
                pending = bool(be._put_tasks)
            if not pending and (a.expect_mirror == 'no' or cpu.contains(key)):
                break
            assert time.monotonic() < deadline, be.dram_mirror.snapshot()
            time.sleep(.01)
        with be._put_lock:
            assert not be._put_tasks
        assert be.stats['put'] == 1
        if a.expect_mirror == 'yes':
            hit = asyncio.run(cpu.batched_async_contains('test', [key], pin=False))
            assert hit == 1
            cpu_obj = cpu.get_blocking(key)
            try:
                assert cpu_obj.tensor.device.type == 'cpu'
                assert torch.equal(cpu_obj.tensor.view(torch.uint8), source.cpu().view(torch.uint8))
            finally:
                cpu_obj.ref_count_down()
            assert cpu.remove(key)
        else:
            assert not cpu.contains(key)
            assert be.dram_mirror.snapshot()['skipped_budget'] == 1
        fetched = be.get_blocking(key)
        assert fetched is not None
        assert torch.equal(fetched.tensor.view(torch.uint8), source.view(torch.uint8))
        be._release_memory_obj(fetched)
        fetched = None
        assert be.remove(key)
        be.dram_mirror.close()
        assert be.dram_mirror.snapshot()['pending_bytes'] == 0
        assert be.memory_allocator.allocator.total_allocated_size == 0
        result = dict(result='PASS', bytes=source.numel()*source.element_size(),
            allocation_device='cuda', dram_payload='byte_equal' if a.expect_mirror == 'yes' else 'skipped_budget',
            daos_payload='byte_equal',
            cpu_removal_does_not_remove_daos=True if a.expect_mirror == 'yes' else None,
            daos_valid_without_cpu_copy=True, stats=be.dram_mirror.snapshot(),
            cleanup='only own UUID test key removed')
        a.result.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result))
        success = True
    finally:
        if fetched is not None:
            be._release_memory_obj(fetched)
        if not success:
            print('Failed; own test namespace retained for inspection:', ns)
        manager.close()


if __name__ == '__main__':
    main()
