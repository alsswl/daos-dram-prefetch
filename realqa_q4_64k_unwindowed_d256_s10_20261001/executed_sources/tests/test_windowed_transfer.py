from types import SimpleNamespace as NS
import pytest

from lmcache_daos.windowed_transfer import window_chunks, transfer_windows


@pytest.mark.parametrize('mib,chunks',[(0,0),(18,1),(128,7),(512,28),(1024,56)])
def test_window_size(mib,chunks):
    assert window_chunks(mib,18*2**20,8*2**30)==chunks


@pytest.mark.parametrize('mib',[True,-1,float('nan'),float('inf'),1,8193])
def test_invalid_window(mib):
    with pytest.raises(ValueError): window_chunks(mib,18*2**20,8*2**30)


def simulate(limit=3, short=False, fail_copy=False):
    spans=[(i*128,(i+1)*128) for i in range(10)]
    log=[]; alive=set(); copied=[]
    def load(start,end):
        assert not alive, 'Previous buffers must be released before next load'
        ids=list(range(start//128,end//128))
        if short:ids=ids[:1]
        assert len(ids)<=limit
        alive.update(ids);log.append(('load',ids))
        return [(i,NS(get_size=lambda:18),(i*128),(i+1)*128) for i in ids]
    def copy(items):
        if fail_copy:raise RuntimeError('copy failed')
        copied.extend(i[0] for i in items);log.append(('copy',list(alive)))
    def release(items):
        log.append(('release',list(alive)));alive.clear()
    return spans,load,copy,release,log,alive,copied


@pytest.mark.parametrize('short',[False,True])
def test_order_and_short_prefix_progress(short):
    spans,load,copy,release,log,alive,copied=simulate(short=short)
    assert transfer_windows(spans,3,load,copy,release,lambda *a,**k:None)==180
    assert copied==list(range(10)) and not alive
    assert [e[0] for e in log]==['load','copy','release']*(10 if short else 4)


def test_copy_failure_releases_and_does_not_read_ahead():
    spans,load,copy,release,log,alive,_=simulate(fail_copy=True)
    with pytest.raises(RuntimeError,match='copy failed'):
        transfer_windows(spans,3,load,copy,release,lambda *a,**k:None)
    assert not alive and [x[0] for x in log]==['load','release']


def test_no_progress_is_bounded():
    now=[0.];events=[]
    def sleep(t):now[0]+=t
    with pytest.raises(TimeoutError,match='no progress'):
        transfer_windows([(0,128)],1,lambda *a:[],lambda *a:None,lambda *a:None,
            lambda *a,**k:events.append(a),timeout=.03,clock=lambda:now[0],sleep=sleep)
    assert 2<=len(events)<=4


def test_noncontiguous_result_released_without_copy():
    released=[];copied=[]
    with pytest.raises(RuntimeError,match='non-contiguous'):
        transfer_windows([(0,128)],1,lambda *a:[('bad',object(),128,256)],
            copied.append,released.extend,lambda *a,**k:None)
    assert len(released)==1 and not copied


def test_native_process_hook_window_scatter_and_mask(monkeypatch):
    import torch
    import threading
    import lmcache_daos.windowed_demand_backend as wd
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.gpu_connector import gpu_connectors
    copied=[];released=[];loaded=[]
    class Obj:
        tensor=NS(is_cuda=True)
        is_pinned=False
        def __init__(self,k):self.key=k;self.refs=1
        def get_size(self):return 18
        def get_ref_count(self):return self.refs
    class Connector:
        load_stream=NS(synchronize=lambda:None)
        def batched_to_gpu(self,objects,starts,ends,**kw):
            assert len(objects)<=2
            copied.extend(zip(starts,ends))
    def native(engine,tokens,mask,ret,**kw):
        rows=[]
        for i in range(0,len(tokens),128):
            if mask[i]:rows.append((i,Obj(i),i,i+128));ret[i:i+128]=True
        loaded.append((len(tokens),[(r[2],r[3]) for r in rows]))
        return rows,len(rows)*18
    monkeypatch.setattr(LMCacheEngine,'_process_tokens_internal',native)
    monkeypatch.setattr(LMCacheEngine,'_daos_windowed_installed',False,raising=False)
    monkeypatch.setattr(gpu_connectors,'VLLMPagedMemGPUConnectorV2',Connector)
    be=wd.WindowedDemandBackend.__new__(wd.WindowedDemandBackend)
    be.window_chunk_count=2;be.window_mib=1;be.window_timeout_s=1;be.window_lock=threading.RLock()
    be._staging_trace=NS(emit=lambda *a,**k:None)
    def release(o):o.refs-=1;released.append(o.key)
    be._release_memory_obj=release
    def spans(tokens,mask,**kwargs):
        return [(i,i+128,i) for i in range(0,len(tokens),128) if mask is None or mask[i]]
    engine=NS(storage_manager=NS(_daos_gpu_store=('daos',be)),async_loading=False,
        save_only_first_rank=False,remove_after_retrieve=False,gpu_connector=Connector(),
        _get_req_id=lambda kw:kw['req_id'],token_database=NS(process_tokens=spans))
    wd.install_window_hook()
    mask=torch.ones(640,dtype=torch.bool);mask[:128]=False
    ret=torch.zeros(640,dtype=torch.bool)
    rows,total=LMCacheEngine._process_tokens_internal(engine,list(range(640)),mask,ret,req_id='r')
    assert rows==[] and total==72
    assert torch.equal(ret,mask)
    assert copied==[(128,256),(256,384),(384,512),(512,640)]
    assert released==[128,256,384,512] and len(loaded)==2
    # Zero switch really calls the previous implementation without window copies.
    be.window_chunk_count=0
    rows,_=LMCacheEngine._process_tokens_internal(engine,list(range(640)),mask,ret,req_id='r')
    assert len(rows)==4 and len(copied)==4
