import threading
import contextvars
from types import SimpleNamespace as NS
import pytest
from lmcache_daos.split_pipeline import pipeline_windows


@pytest.mark.parametrize('fail_copy', [False,True])
def test_overlap_order_capacity_and_cleanup(fail_copy):
    barrier=threading.Barrier(3);lock=threading.Lock()
    alive=set();copied=[];peak=0
    context=contextvars.ContextVar('test',default=None);context.set('retrieve')
    def load(start,end):
        nonlocal peak
        assert context.get()=='retrieve'
        if start<6:barrier.wait(timeout=2)
        with lock:
            ids=list(range(start,end));alive.update(ids);peak=max(peak,len(alive))
        return [(i,NS(get_size=lambda:1),i,i+1) for i in ids]
    def copy(items):
        if fail_copy:raise RuntimeError('copy failed')
        copied.extend(x[0] for x in items)
    def release(items):
        with lock:
            for x in items:alive.remove(x[0])
    args=([(i,i+1) for i in range(10)],2,3,load,copy,release,lambda *a,**k:None)
    if fail_copy:
        with pytest.raises(RuntimeError,match='copy failed'):pipeline_windows(*args)
    else:
        assert pipeline_windows(*args)==10
        assert copied==list(range(10))
    assert not alive and peak<=6


def test_load_failure_drains_other_futures():
    alive=set();lock=threading.Lock()
    def load(start,end):
        if start==2:raise ValueError('load failed')
        with lock:alive.add(start)
        return [(start,NS(get_size=lambda:1),start,end)]
    def release(items):
        with lock:
            for x in items:alive.remove(x[0])
    with pytest.raises(ValueError,match='load failed'):
        pipeline_windows([(i,i+1) for i in range(6)],1,3,load,
                         lambda x:None,release,lambda *a,**k:None)
    assert not alive


def test_physical_split_no_borrow_and_free_reuse():
    import torch
    from lmcache_daos.split_staging_backend import SplitGPUMemoryAllocator,_allocation_pool
    a=SplitGPUMemoryAllocator(16384,4096,'cpu')
    t=_allocation_pool.set('store')
    store=a.allocate(torch.Size([4096]),torch.uint8)
    assert a.allocate(torch.Size([1]),torch.uint8) is None
    _allocation_pool.reset(t)
    reads=[a.allocate(torch.Size([4096]),torch.uint8) for _ in range(3)]
    assert all(x is not None for x in reads)
    assert a.allocate(torch.Size([1]),torch.uint8) is None
    assert all(x.tensor.data_ptr()>=a.allocator.boundary for x in reads)
    assert store.tensor.data_ptr()<a.allocator.boundary
    store.ref_count_down()
    assert a.allocate(torch.Size([1]),torch.uint8) is None, 'Retrieve borrowed store arena'
    for x in reads:x.ref_count_down()
    assert a.allocator.total_allocated_size==0 and a.memcheck()
