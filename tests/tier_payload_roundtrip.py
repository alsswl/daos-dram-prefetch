"""Live mixed CPU-prefix + DAOS-suffix read for each prefetch policy.

Only two private UUID test keys are removed after successful byte validation.
CPU placement is forced ONLY in this correctness test, never the workload.
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
from lmcache.v1.event_manager import EventManager,EventType
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.pin_monitor import PinMonitor
from lmcache.v1.storage_backend.storage_manager import StorageManager
from async_dram_promotion_roundtrip import wait_for
from sharegpt_async_threeway import config,POLICIES,dump


def check(folder,policy):
    folder.mkdir()
    os.environ['DAOS_GDS_STAGING_TRACE'] = str(folder/'trace')
    cfg = config(policy)
    cfg['max_local_cpu_size'] = .125
    cfg['extra_config'].update({'daosgds.gpu_buffer_gb':.125,
        'daosgds.dram_mirror_max_pending_gb':.0625,'daosgds.dram_mirror_max_age_ms':10000})
    (folder/'config.yaml').write_text(yaml.safe_dump(cfg))
    md = LMCacheMetadata(model_name='tier-payload-byte-test',world_size=1,local_world_size=1,
        worker_id=0,local_worker_id=0,kv_dtype=torch.bfloat16,
        kv_shape=(40,2,128,8,128),role='worker',chunk_size=128)
    engine_cfg = LMCacheEngineConfig.from_dict(cfg)
    PinMonitor.GetOrCreate(engine_cfg)
    notices = {}
    manager = StorageManager(engine_cfg,md,EventManager(),async_lookup_server=SimpleNamespace(
        send_response_to_scheduler=lambda rid,n:notices.__setitem__(rid,n)))
    be,cpu = manager.allocator_backend,manager.local_cpu_backend
    keys = [CacheEngineKey(md.model_name,1,0,uuid.uuid4().int,torch.bfloat16) for _ in range(2)]
    live,expected = [],[]
    try:
        for key in keys:
            obj = manager.allocate(md.get_shapes(),md.get_dtypes(),fmt=None)
            assert obj is not None and obj.tensor.is_cuda
            obj.tensor.copy_(torch.randn_like(obj.tensor))
            torch.cuda.synchronize()
            expected.append(obj.tensor.cpu().view(torch.uint8))
            manager.batched_put([key],[obj])
            wait_for(lambda:cpu.contains(key) and not be.exists_in_put_tasks(key)
                     and be.dram_mirror.snapshot()['pending_bytes']==0
                     and be.memory_allocator.allocator.total_allocated_size==0)
        assert cpu.remove(keys[1])
        rid = 'mixed-'+policy
        asyncio.run_coroutine_threadsafe(manager.async_lookup_and_prefetch(
            rid,keys,[0,128,256],pin=True),manager.loop).result(timeout=20)
        assert notices[rid]==256
        groups = manager.event_manager.get_event_future(EventType.LOADING,rid).result()
        live = [group[0][1] for group in groups]
        assert [v.batch.tier for v in live]==['dram','daos']
        for view in live:
            if POLICIES[policy][view.batch.tier]:
                view.batch.future.result(timeout=20)
            else:
                assert not view.batch.future.done(), 'Disabled prefetch tier started work'
        for view,reference in zip(live,expected,strict=True):
            view.batch.resolve()
            assert torch.equal(view.tensor.cpu().view(torch.uint8),reference)
            assert view.tensor.is_cuda==(view.batch.tier=='daos' or POLICIES[policy]['dram'])
        for view in live: view.ref_count_down()
        retired = [v.batch.is_retired for v in live]
        live = []
        manager.event_manager.pop_event(EventType.LOADING,rid)
        wait_for(lambda:be.dram_mirror.snapshot()['pending_bytes']==0
                 and be.memory_allocator.allocator.total_allocated_size==0)
        assert all(retired)
        for key in keys:
            assert cpu.contains(key) and cpu.remove(key)
            assert be.remove(key)
        dump(folder/'result.json',dict(result='PASS',policy=policy,bytes=40*2**20,
            paths=['dram-prefix','daos-suffix'],staging_bytes_after=0,own_keys_removed=2))
    finally:
        for view in live: view.batch.abort()
        manager.close()


if __name__=='__main__':
    import subprocess
    import sys
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--policy',choices=list(POLICIES))
    args = parser.parse_args()
    if args.policy:
        check(args.output,args.policy)
    else:
        args.output.mkdir()
        for policy in POLICIES:
            subprocess.run([sys.executable,__file__,'--output',str(args.output/policy),'--policy',policy],check=True)
        dump(args.output/'result.json',dict(result='PASS',policies=list(POLICIES)))
