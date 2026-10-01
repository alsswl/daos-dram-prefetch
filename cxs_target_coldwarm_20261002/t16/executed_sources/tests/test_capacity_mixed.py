import threading
import pytest
import torch
from lmcache_daos.capacity_pipeline_backend import BudgetGPUAllocator
from lmcache_daos.capacity_pipeline import capacity_pipeline
from mixed_qa_scheduler import make_schedule,START_POSITIONS

@pytest.mark.parametrize('split',[False,True])
def test_reservation_isolation_and_reuse(split):
    a=BudgetGPUAllocator(16384,4096 if split else 0,'cpu',4096);b=a.allocator
    s=b.try_reserve('store',4096)
    r=b.try_reserve('retrieve',12288)
    assert s and r and b.try_reserve('retrieve',1) is None
    with s.activate():store=a.allocate(torch.Size([4096]),torch.uint8)
    with r.activate():reads=[a.allocate(torch.Size([4096]),torch.uint8) for _ in range(3)]
    s.close();r.close()
    assert sum(b.pending.values())==0 and b.total_allocated_size==16384
    store.ref_count_down()
    spare=b.try_reserve('retrieve',4096)
    assert (spare is None)==split
    if spare:spare.close()
    for obj in reads:obj.ref_count_down()
    assert b.total_allocated_size==0 and b.memcheck()

@pytest.mark.parametrize('copy_fail',[False,True])
def test_pipeline_dynamic_admission_and_cleanup(copy_fail):
    a=BudgetGPUAllocator(16384,0,'cpu',4096);b=a.allocator
    # Occupy half of shared pool as an earlier asynchronous store.
    ticket=b.try_reserve('store',8192)
    with ticket.activate():held=a.allocate(torch.Size([8192]),torch.uint8)
    ticket.close();started=threading.Event();copied=[];peak=0
    def load(start,end):
        started.set()
        return [(start,a.allocate(torch.Size([4096]),torch.uint8),start,end)]
    def copy(items):
        if copy_fail:raise RuntimeError('copy failed')
        copied.extend(x[0] for x in items)
    def release(items):
        for x in items:x[1].ref_count_down()
    def free_store():
        assert started.wait(2);held.ref_count_down()
    worker=threading.Thread(target=free_store);worker.start()
    args=([(i,i+1) for i in range(12)],1,4,load,copy,release,lambda *a,**k:None,b)
    if copy_fail:
        with pytest.raises(RuntimeError,match='copy failed'):capacity_pipeline(*args)
    else:
        assert capacity_pipeline(*args)==12*4096
        assert copied==list(range(12))
    worker.join();assert b.total_allocated_size==0 and sum(b.pending.values())==0

def test_schedule_reproducible_mixed_and_complete():
    s=make_schedule();assert s==make_schedule()
    pairs=[]
    for lane in s['lanes']:
        assert [r['position'] for r in lane if r['new_session']]==list(START_POSITIONS)
        turns={}
        for r in lane:
            assert r['turn']==turns.get(r['session_index'],0)
            turns[r['session_index']]=r['turn']+1
            pairs.append((r['session_index'],r['turn']))
        assert len(turns)==10 and set(turns.values())=={6}
    assert len(set(pairs))==480

@pytest.mark.parametrize('copy_fail',[False,True])
def test_near_capacity_single_window_release(copy_fail):
    a=BudgetGPUAllocator(16384,0,'cpu',4096);b=a.allocator
    copied=[];events=[]
    def load(start,end):
        return [(i,a.allocate(torch.Size([4096]),torch.uint8),i,i+1)
                for i in range(start,end)]
    def copy(items):
        if copy_fail:raise RuntimeError('copy failed')
        copied.extend(x[0] for x in items)
    def release(items):
        for x in items:x[1].ref_count_down()
    def emit(event,**fields):
        events.append(dict(event=event,used_bytes=b.total_allocated_size,
                           **b.pool_usage(),**fields))
    args=([(i,i+1) for i in range(8)],3,1,load,copy,release,emit,b)
    if copy_fail:
        with pytest.raises(RuntimeError,match='copy failed'):capacity_pipeline(*args)
    else:
        assert capacity_pipeline(*args)==8*4096 and copied==list(range(8))
        from lmcache_daos.capacity_validation import validate_capacity_events
        cfg=dict(event='capacity_pipeline_enabled',used_bytes=0,store_used_bytes=0,
                 retrieve_used_bytes=0,capacity_bytes=16384,shared=True,pipeline_depth=1)
        result=validate_capacity_events([cfg]+events)
        assert result['max_concurrent_loads']==1 and result['load_copy_overlap_events']==0
    assert b.total_allocated_size==0 and sum(b.pending.values())==0
