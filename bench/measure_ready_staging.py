"""Observe existing retrieve policy without changing scheduling or storage layout.

Uses the existing 9 GiB native LMCache/GPU-scatter correctness check. All extra
events are buffered in CPU memory until the run ends; no payload references are
retained. Wrappers are process-local and installed only in this executable.
"""
import argparse
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tests'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    # run_vllm.sh otherwise defaults to dfs and overrides the YAML transport.
    os.environ['DAOSGDS_TRANSPORT']='object'
    from lmcache_daos.capacity_pipeline_backend import CapacityPipelineBackend
    from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2
    from sync_retrieve_capacity import check
    rows=[];local=threading.local();by_id={}
    active=False
    def event(kind,**fields):
        rows.append(dict(event=kind,t=time.monotonic_ns(),**fields))

    original_init=CapacityPipelineBackend.__init__
    @wraps(original_init)
    def init(be,*args,**kwargs):
        original_init(be,*args,**kwargs)
        original_get=be._observed_get
        def observed_get(key):
            local.key=be._object_key(key)
            if active:event('worker_start',key=local.key)
            try:return original_get(key)
            finally:
                if active:event('worker_done',key=local.key)
                del local.key
        be._observed_get=observed_get
        native_get=be._object.get
        def get(key,ptr,n,device_id=0):
            if active:event('fetch_start',key=key,bytes=n)
            result=native_get(key,ptr,n,device_id)
            if active:event('fetch_done',key=key,bytes=n)
            return result
        be._object.get=get
        arena=be.budget
        allocate,free=arena.allocate,arena.free
        def alloc(*args,**kwargs):
            obj=allocate(*args,**kwargs)
            key=getattr(local,'key',None)
            if active and obj is not None and key is not None:
                by_id[id(obj)]=key
                event('allocated',key=key,bytes=obj.get_size())
            return obj
        def release(obj,*args,**kwargs):
            key=by_id.pop(id(obj),None)
            result=free(obj,*args,**kwargs)
            if active and key is not None:event('freed',key=key)
            return result
        arena.allocate,arena.free=alloc,release
        emit=be._staging_trace.emit
        def traced(kind,**fields):
            if active and kind in {'pipeline_submitted','pipeline_load_start','pipeline_load_done',
                                   'window_copy_start','window_copy_done','window_released'}:
                event(kind,**fields)
            return emit(kind,**fields)
        be._staging_trace.emit=traced
    CapacityPipelineBackend.__init__=init
    transfer=CapacityPipelineBackend.transfer_retrieve
    @wraps(transfer)
    def observed_transfer(be,spans,*args,**kwargs):
        nonlocal active
        assert not active
        active=True
        event('retrieve_start',chunks=len(spans),window_chunks=be.window_chunk_count,
              depth=be.pipeline_depth,capacity=be.gpu_buffer_bytes,chunk_bytes=be.store_chunk_bytes)
        try:return transfer(be,spans,*args,**kwargs)
        finally:
            event('retrieve_end')
            active=False
    CapacityPipelineBackend.transfer_retrieve=observed_transfer
    scatter=VLLMPagedMemGPUConnectorV2.batched_to_gpu
    @wraps(scatter)
    def observed_scatter(connector,objects,starts,ends,**kwargs):
        keys=[by_id[id(obj)] for obj in objects] if active else []
        if active:event('scatter_start',keys=keys,starts=list(starts),ends=list(ends))
        result=scatter(connector,objects,starts,ends,**kwargs)
        if active:event('scatter_done',keys=keys)
        return result
    VLLMPagedMemGPUConnectorV2.batched_to_gpu=observed_scatter
    sources=['bench/measure_ready_staging.py','tests/sync_retrieve_capacity.py',
             'lmcache_daos/capacity_pipeline.py','lmcache_daos/capacity_pipeline_backend.py',
             'lmcache_daos/demand_read_backend.py','lmcache_daos/windowed_demand_backend.py',
             'lmcache_daos/gds_backend.py','libdaosgdr.c','libdaosgdr.so']
    hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in sources}
    try:
        check(a.output.resolve(),2,512,window_mib=500,store_window_mib=500,
              capacity_pipeline=True,no_dram=True)
    finally:
        if a.output.exists():
            (a.output/'chunk_lifecycle.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (a.output/'measurement_plan.json').write_text(json.dumps(dict(
                source_sha256=hashes,clock='time.monotonic_ns',
                scheduling_policy_changed=False,payload_reference_retained_by_probe=False,
                notes=['Native synchronous DAOS fetch completion, real paged GPU scatter.',
                       'Extra events buffered in memory, written after engine shutdown.',
                       'No model compute, concurrent requests, or simultaneous store during retrieve.',
                       'Admission/depth saturation does not imply NIC idle time.',
                       'Fresh 9 GiB patterned dataset; no server cache flush.']),indent=2)+'\n')


if __name__=='__main__':main()
