from contextlib import nullcontext
from types import SimpleNamespace as NS
import threading

import pytest
import torch

from lmcache_daos import windowed_store_backend as ws


def fixture(monkeypatch, *, fail_alloc_at=None, fail_gather=False, fail_put=False):
    from lmcache.v1.gpu_connector import gpu_connectors
    used = NS(total_allocated_size=0)
    events, batches, pending, gathers, finished, waits = [], [], [], [], [], []
    chunk = 18*2**20

    class Obj:
        released = False
        def get_size(self): return chunk
        def release(self):
            assert not self.released
            self.released = True
            used.total_allocated_size -= chunk

    class Connector:
        store_stream = NS(synchronize=lambda: events.append('sync'))
        def batched_from_gpu(self, objects, starts, ends, **kwargs):
            if fail_gather: raise RuntimeError('gather error')
            gathers.append((starts, ends, kwargs['slot_mapping']))

    monkeypatch.setattr(gpu_connectors, 'VLLMPagedMemGPUConnectorV2', Connector)
    count = [0]
    def allocate(*a, **kw):
        count[0] += 1
        if count[0] == fail_alloc_at: return None
        assert used.total_allocated_size + chunk <= 2**30
        used.total_allocated_size += chunk
        return Obj()
    def put(keys, objects, **kw):
        assert ws._store_context.get()[0] is manager
        if fail_put:
            for o in objects: o.release()
            raise RuntimeError('put error')
        batches.append(keys); pending.append(objects)
    manager = NS(allocate=allocate, batched_put=put)
    stats = NS(profile_process_tokens=nullcontext, profile_from_gpu=nullcontext, profile_put=nullcontext)
    def process(tokens, mask, **kw):
        return [(s,s+128,tuple(tokens[:s+128])) for s in range(0,len(tokens)//128*128,128)
                if mask is None or mask[s]]
    engine = NS(is_healthy=lambda:True, _is_passive=lambda:False, is_frozen=lambda:False,
                gpu_connector=Connector(), _get_req_id=lambda kw:'test',
                token_database=NS(process_tokens=process), config=NS(chunk_size=128),
                metadata=NS(get_shapes=lambda n:[(n,)], get_dtypes=lambda:[torch.bfloat16]),
                fmt=None, store_location=None, storage_manager=manager,
                stats_monitor=NS(on_store_request=lambda n:stats,
                                 on_store_finished=lambda s,n:finished.append(n)))
    backend = NS(store_enabled=True, window_lock=threading.RLock(), gpu_buffer_bytes=2**30,
                 store_window_chunks=27, store_window_mib=500, store_chunk_bytes=chunk,
                 store_window_timeout_s=.1,
                 memory_allocator=NS(device_mem_lock=threading.Lock(), allocator=used),
                 _staging_trace=NS(emit=lambda *a,**kw:events.append((a,kw))),
                 _release_memory_obj=lambda o:o.release())
    original = ws.wait_for_space
    def sleep(t):
        waits.append(t)
        # Model delayed asynchronous ownership, including DRAM readers.
        for obj in pending.pop(0): obj.release()
    monkeypatch.setattr(ws, 'wait_for_space', lambda b,n,e: original(b,n,e,sleep=sleep))
    return engine,backend,NS(used=used,batches=batches,pending=pending,gathers=gathers,
                            finished=finished,waits=waits,events=events)


def test_bounded_store_keeps_absolute_offsets_and_async_tail(monkeypatch):
    engine,backend,state=fixture(monkeypatch)
    tokens=list(range(128*80+7))
    mask=torch.ones(len(tokens),dtype=torch.bool);mask[:128]=False
    slots=list(range(len(tokens)))
    ws.store_windows(engine,backend,tokens,mask,{'slot_mapping':slots})
    assert [len(x) for x in state.batches]==[27,27,25]
    keys=[k for batch in state.batches for k in batch]
    assert keys==[tuple(tokens[:e]) for e in range(256,128*80+1,128)]
    assert [s for starts,_,_ in state.gathers for s in starts]==list(range(128,128*80,128))
    assert all(sm is slots for _,_,sm in state.gathers)
    assert state.finished==[79*128] and state.waits
    assert state.used.total_allocated_size==52*18*2**20  # two asynchronous windows retained
    assert ws._store_context.get() is None


@pytest.mark.parametrize('failure', ['allocation','gather','put'])
def test_failure_does_not_leak_or_silently_skip_tail(monkeypatch,failure):
    engine,backend,state=fixture(monkeypatch,fail_alloc_at=3 if failure=='allocation' else None,
                                 fail_gather=failure=='gather',fail_put=failure=='put')
    with pytest.raises(RuntimeError):
        ws.store_windows(engine,backend,list(range(128*30)),None,{'slot_mapping':[]})
    assert state.used.total_allocated_size==0 and state.finished==[0]
    assert ws._store_context.get() is None


def test_backpressure_timeout_cannot_report_success():
    now=[0.]
    be=NS(gpu_buffer_bytes=1024,store_window_timeout_s=.01,
          memory_allocator=NS(device_mem_lock=threading.Lock(),allocator=NS(total_allocated_size=900)))
    def sleep(t):now[0]+=t
    with pytest.raises(TimeoutError,match='no tail may be skipped'):
        ws.wait_for_space(be,500,lambda *a,**kw:None,clock=lambda:now[0],sleep=sleep)


def test_empty_mask_never_allocates(monkeypatch):
    engine,backend,state=fixture(monkeypatch)
    ws.store_windows(engine,backend,list(range(128)),torch.zeros(128,dtype=torch.bool),{})
    assert not state.batches and state.used.total_allocated_size==0 and state.finished==[0]
