from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
import threading
from types import SimpleNamespace as NS

import pytest
import torch

from lmcache_daos.direct_scatter_backend import (
    DirectScatterBackend, NoPayloadAllocator, validate_config,
)


def backend(monkeypatch):
    be=DirectScatterBackend.__new__(DirectScatterBackend)
    be._operation_lock=threading.RLock()
    be.io_workers=2;be.device_id=0
    be.metadata=NS(kv_shape=(2,2,4,1,4),kv_dtype=torch.bfloat16)
    be.stats=dict(put=0,get=0,put_bytes=0,get_bytes=0,miss=0,errors=0,iov_count=0)
    be._pool=ThreadPoolExecutor(2)
    calls=[]
    be._prepare=lambda engine,kwargs: ([],[100000,200000],(16,4,4,2),kwargs['slot_mapping'],'NB_TWO')
    def transfer(key,segments,meta,write):
        calls.append((key,segments,write))
        return sum(s.length for s in segments)
    be._transfer=transfer
    barriers=[]
    monkeypatch.setattr(torch.cuda,'synchronize',lambda d:barriers.append(d))
    return be,calls,barriers


def test_direct_retrieve_protects_prefix_and_never_returns_memoryobjs(monkeypatch):
    be,calls,barriers=backend(monkeypatch)
    infos=[(0,4,'a'),(4,8,'b'),(8,12,'c')]
    engine=NS(token_database=NS(process_tokens=lambda **kw:infos))
    ret=torch.zeros(12,dtype=torch.bool)
    try:
        rows,total=be.retrieve_into(engine,list(range(12)),None,ret,
            slot_mapping=[-1,-1,10,11,20,21,22,23,40,41,42,43],vllm_cached_tokens=2)
        assert rows==[] and not ret[:2].any() and ret[2:].all() and total==10*2*2*4*2
        assert len(calls)==3 and barriers==[0]
        first=next(c for c in calls if c[0]=='a')
        assert min(s.offset for s in first[1])==2*4*2
    finally:be._pool.shutdown()


def test_failed_read_drains_workers_and_only_publishes_valid_prefix(monkeypatch):
    be,calls,barriers=backend(monkeypatch)
    done=threading.Event()
    def transfer(key,segments,meta,write):
        if key=='bad': raise RuntimeError('read failed')
        done.set()
        return sum(s.length for s in segments)
    be._transfer=transfer
    infos=[(0,4,'bad'),(4,8,'good'),(8,12,'never')]
    ret=torch.zeros(12,dtype=torch.bool)
    try:
        total,n=be._operate(None,infos,ret,{'slot_mapping':list(range(12))},False)
        assert total==n==0 and not ret.any() and done.is_set()
        assert barriers==[0] and be.stats['errors']==1
    finally:be._pool.shutdown()


def test_cached_prefix_cannot_hide_a_later_read_failure(monkeypatch):
    be,calls,_=backend(monkeypatch)
    original=be._transfer
    def transfer(key,*args):
        if key=='bad':raise RuntimeError('failed tail')
        return original(key,*args)
    be._transfer=transfer
    ret=torch.zeros(8,dtype=torch.bool)
    try:
        _,n=be._operate(None,[(0,4,'ok'),(4,8,'bad')],ret,
                       {'slot_mapping':list(range(8)),'vllm_cached_tokens':3},False)
        assert n==1 and ret.sum()==1 and ret[3]
        assert ret.sum() < 8-3  # adapter must detect the failure
    finally:be._pool.shutdown()


def test_submit_failure_still_drains_already_submitted_dma(monkeypatch):
    be,_,_=backend(monkeypatch)
    real=be._pool
    done=threading.Event()
    def transfer(*args):
        done.set()
        return 1
    be._transfer=transfer
    count=[0]
    def submit(*args):
        count[0]+=1
        if count[0]==2:raise RuntimeError('executor failure')
        return real.submit(*args)
    be._pool=NS(submit=submit)
    try:
        with pytest.raises(RuntimeError,match='executor failure'):
            be._batch([('a',[],{}),('b',[],{})],False)
        assert done.is_set()
    finally:real.shutdown()


def test_invalid_cross_chunk_alias_is_rejected_before_io(monkeypatch):
    be,calls,_=backend(monkeypatch)
    try:
        with pytest.raises(ValueError,match='aliased'):
            be._operate(None,[(0,4,'a'),(4,8,'b')],None,
                        {'slot_mapping':[0,1,2,3,0,1,2,3]},True)
        assert not calls
    finally:be._pool.shutdown()


def test_store_is_complete_before_return(monkeypatch):
    be,calls,barriers=backend(monkeypatch)
    try:
        size,n=be._operate(None,[(0,4,'a'),(4,8,'b')],None,
                           {'slot_mapping':list(range(8))},True)
        assert n==8 and size==8*2*2*4*2 and len(calls)==2 and barriers==[0]
        assert all(write for _,_,write in calls)
    finally:be._pool.shutdown()


def test_metadata_mismatch_does_not_issue_fetch():
    be=DirectScatterBackend.__new__(DirectScatterBackend)
    be.device_id=0;be.object_namespace='test:';be._ensure_cuda_ctx=lambda:None
    be._object=NS(stat=lambda key:b'{"schema":"old"}',getv=lambda *a:pytest.fail('unexpected fetch'))
    with pytest.raises(ValueError,match='metadata'):
        be._transfer(NS(to_string=lambda:'k'),[],{'schema':'new'},False)


def test_no_staging_allocator():
    a=NoPayloadAllocator()
    for f in (a.allocate,a.batched_allocate):
        with pytest.raises(RuntimeError,match='forbidden'):f()
    assert a.memcheck()


def test_failed_plugin_never_falls_back_to_baseline():
    from lmcache_daos.direct_scatter_backend import find_backend
    engine=NS(config=NS(extra_config={'daosgds.direct_scatter':True}),
              storage_manager=NS(storage_backends={}))
    with pytest.raises(RuntimeError,match='refusing fallback'):
        find_backend(engine)


def test_config_rejects_prefetch_and_staging():
    config=NS(extra_config={'daosgds.direct_scatter':True,'daosgds.object_namespace':'test:'},
              enable_async_loading=False)
    md=NS(world_size=1,use_mla=False,kv_dtype=torch.bfloat16)
    validate_config(config,md)
    config.enable_async_loading=True
    with pytest.raises(ValueError,match='enable_async_loading'):validate_config(config,md)
    config.enable_async_loading=False
    config.extra_config['daosgds.gpu_buffer_gb']=2
    with pytest.raises(ValueError,match='gpu_buffer_gb'):validate_config(config,md)


def test_engine_hooks_bypass_both_copy_paths(monkeypatch):
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache_daos.direct_scatter_backend import install_engine_hooks
    monkeypatch.setattr(LMCacheEngine,'_daos_direct_scatter_installed',False,raising=False)
    monkeypatch.setattr(LMCacheEngine,'_process_tokens_internal',lambda *a,**kw:pytest.fail('native read'))
    monkeypatch.setattr(LMCacheEngine,'store',lambda *a,**kw:pytest.fail('native store'))
    be=DirectScatterBackend.__new__(DirectScatterBackend)
    be.store_enabled=True
    calls=[]
    be.retrieve_into=lambda *a,**kw:([],42)
    be.store_from=lambda *a,**kw:calls.append('store')
    engine=NS(storage_manager=NS(storage_backends={'DirectScatterBackend':be}),
              is_healthy=lambda:True,_is_passive=lambda:False,is_frozen=lambda:False)
    install_engine_hooks()
    assert LMCacheEngine._process_tokens_internal(engine,[1],None,None)==([],42)
    LMCacheEngine.store(engine,[1])
    assert calls==['store']


def test_launch_selects_no_payload_backend_without_cpu_tier():
    # Isolate process-global startup patches; use the installed manager's real
    # selector so absence of LocalCPUBackend reproduces the startup regression.
    import subprocess
    import sys
    subprocess.run([sys.executable, '-c', '''
from types import SimpleNamespace as NS
from lmcache_daos.direct_scatter_backend import DirectScatterBackend, NoPayloadAllocator
from lmcache_daos.direct_scatter_launch import install
from lmcache.v1.storage_backend.storage_manager import StorageManager
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
install()
be = DirectScatterBackend.__new__(DirectScatterBackend)
be.memory_allocator = NoPayloadAllocator()
config = NS(extra_config={'daosgds.direct_scatter': True})
manager = NS(storage_backends={'daosgds': be})
selected = StorageManager._get_allocator_backend(manager, config)
assert selected is be and isinstance(selected, AllocatorBackendInterface)
try:
    selected.memory_allocator.allocate()
except RuntimeError as e:
    assert 'forbidden' in str(e)
else:
    raise AssertionError('payload allocation allowed')
try:
    StorageManager._get_allocator_backend(NS(storage_backends={}), config)
except RuntimeError as e:
    assert 'refusing fallback' in str(e)
else:
    raise AssertionError('missing plugin accepted')
normal = NS(storage_backends={'LocalCPUBackend': be}, enable_pd=False)
assert StorageManager._get_allocator_backend(normal, NS(extra_config={})) is be
'''], check=True)
