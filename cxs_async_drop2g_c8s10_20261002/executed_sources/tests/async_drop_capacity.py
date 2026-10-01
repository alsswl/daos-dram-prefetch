"""Real DAOS/GDR gate: native store truncation, async prefix miss and paged scatter."""
import argparse
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def run(root):
    import torch
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.metadata import LMCacheMetadata
    from lmcache.v1.token_database import ChunkedTokenDatabase
    from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2
    from realqa_async_drop import config
    import realqa_q4_cxs as qa
    from async_dram_promotion_roundtrip import wait_for

    folder=root/'gate';folder.mkdir()
    os.environ['DAOS_GDS_STAGING_TRACE']=str(folder/'trace')
    cfg=config(qa.read(root/'plan.json'))
    cfg['extra_config']['daosgds.object_namespace']='async-drop-gate-'+uuid.uuid4().hex+':'
    c=LMCacheEngineConfig.from_dict(cfg)
    md=LMCacheMetadata(model_name='async-drop-byte-gate',world_size=1,local_world_size=1,
        worker_id=0,local_worker_id=0,kv_dtype=torch.bfloat16,
        kv_shape=(36,2,128,8,128),role='worker',chunk_size=128)
    db=ChunkedTokenDatabase(c,md);connector=VLLMPagedMemGPUConnectorV2(1024,36)
    engine=LMCacheEngine(c,md,db,connector,lambda tensor,src:None,lambda obj,src:obj)
    notices={}
    engine.post_init(async_lookup_server=SimpleNamespace(send_response_to_scheduler=lambda rid,n:notices.__setitem__(rid,n)))
    manager=engine.storage_manager;be=manager.allocator_backend
    assert manager.local_cpu_backend is None
    assert type(manager.async_serializer).__name__=='AsyncSingleSerializer'
    chunks=128;tokens=list(range(chunks*128));keys=[k for _,_,k in db.process_tokens(tokens)]
    source=[torch.empty((2,chunks*8,16,8,128),dtype=torch.bfloat16,device='cuda') for _ in range(36)]
    values=(torch.arange(len(tokens),device='cuda')//128%127+1).to(torch.bfloat16)
    for layer in source:layer.view(2,len(tokens),8,128).copy_(values.view(1,-1,1,1))
    slots=torch.arange(len(tokens),device='cuda',dtype=torch.long)
    torch.cuda.synchronize()
    engine.store(tokens,kvcaches=source,slot_mapping=slots,req_id='gate-store')
    wait_for(lambda:not be._put_tasks and be.memory_allocator.allocator.total_allocated_size==0)
    present=[be.contains(k) for k in keys]
    assert sum(present)==113 and present==[True]*113+[False]*15
    # Populate the omitted tail only for this correctness gate, so read pressure
    # can be checked independently of original store truncation.
    for i in range(113,chunks):
        obj=manager.allocate(md.get_shapes(),md.get_dtypes(),fmt=None)
        obj.tensor.fill_(i%127+1);torch.cuda.synchronize()
        manager.batched_put([keys[i]],[obj]);wait_for(lambda:not be._put_tasks)
    wait_for(lambda:be.memory_allocator.allocator.total_allocated_size==0)
    engine.async_lookup_and_prefetch('gate-first',tokens=tokens,pin=True)
    wait_for(lambda:'gate-first' in notices)
    assert 0<notices['gate-first']<=113*128
    # Retained first-request prefix fills the shared pool. Another request must
    # complete its lookup with a shorter/zero prefix, without waiting for space.
    started=time.monotonic()
    engine.async_lookup_and_prefetch('gate-second',tokens=tokens,pin=True)
    wait_for(lambda:'gate-second' in notices)
    assert notices['gate-second']<=113*128-notices['gate-first']
    engine.cleanup_memory_objs('gate-second')
    destination=[torch.full_like(layer,-1) for layer in source]
    engine.gpu_connector=VLLMPagedMemGPUConnectorV2(1024,36)
    mask=engine.retrieve(tokens,kvcaches=destination,slot_mapping=slots,req_id='gate-first')
    torch.cuda.synchronize();loaded=int(mask.sum())
    assert loaded==notices['gate-first']
    assert torch.equal(mask,torch.arange(len(tokens))<loaded)
    expected=values.clone();expected[loaded:]=-1
    for layer in destination:
        assert bool((layer.view(2,len(tokens),8,128)==expected.view(1,-1,1,1)).all())
    engine.lookup_unpin('gate-first');engine.lookup_unpin('gate-second')
    wait_for(lambda:be.memory_allocator.allocator.total_allocated_size==0)
    events=qa.base.read_events(folder)
    outcomes=[e for e in events if e['event']=='daos_prefetch_outcome']
    assert len(outcomes)==2 and all(e['other_failed_chunks']==0 for e in outcomes)
    assert sum(e['capacity_failed_chunks'] for e in outcomes)>0
    for key in keys:assert be.remove(key)
    assert not any(be.contains(k) for k in keys)
    qa.dump(folder/'result.json',dict(passed=True,store_chunks=113,store_dropped_chunks=15,
        notices=notices,paged_scatter_verified=True,untouched_tail_verified=True,
        capacity_outcomes=outcomes,staging_after_bytes=0,own_keys_removed=True,
        second_lookup_and_verify_seconds=time.monotonic()-started))
    manager.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
