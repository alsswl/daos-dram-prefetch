"""Fixed-dataset concurrency sweep: native payload vs real LMCache retrieve.

No production policy changes. Native path reuses the same empty 2 GiB staging
slab, one 18 MiB scratch slice per worker; the engine path uses its usual allocator,
486 MiB windows, metadata reads, and paged GPU scatter. These are diagnostic
paths with different scheduling, not an isolated estimate of any one overhead.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import sys
import threading
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tests')]
os.environ['DAOSGDS_TRANSPORT']='object'


def dump(path,obj):path.write_text(json.dumps(obj,indent=2)+'\n')


def snapshot():
    root=Path('/sys/class/infiniband/mlx5_0/ports/1')
    counters={}
    for directory,names in [('counters',['port_rcv_data','port_xmit_data','port_rcv_errors','port_xmit_discards']),
                             ('hw_counters',['rp_cnp_handled','np_ecn_marked_roce_packets','out_of_buffer','packet_seq_err','roce_adp_retrans'])]:
        for name in names:
            try:counters[name]=int((root/directory/name).read_text())
            except OSError:pass
    threads={}
    for p in Path('/proc/self/task').glob('*/stat'):
        try:
            data=p.read_text();name=data[data.index('(')+1:data.rindex(')')]
            fields=data[data.rindex(')')+2:].split()
            threads[p.parent.name]=dict(name=name,ticks=int(fields[11])+int(fields[12]))
        except (OSError,ValueError):pass
    usage=resource.getrusage(resource.RUSAGE_SELF)
    return dict(t=time.monotonic_ns(),cpu_s=usage.ru_utime+usage.ru_stime,counters=counters,threads=threads)


def main():
    import torch,yaml,pynvml
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.metadata import LMCacheMetadata
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.token_database import ChunkedTokenDatabase
    from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2
    from lmcache_daos.no_dram import configure_no_dram
    from async_dram_promotion_roundtrip import wait_for
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,nargs='+',default=[1,2,4,8,16,32])
    p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
    os.environ['DAOS_GDS_STAGING_TRACE']=str(a.output/'trace')
    cfg=yaml.safe_load((ROOT/'lmcache_config_daosgds_async_dram.yaml').read_text())
    namespace='minji-path-profile-'+uuid.uuid4().hex+':'
    cfg.update(chunk_size=128,enable_async_loading=False,use_layerwise=False)
    cfg['extra_config'].update({
        'storage_plugin.daosgds.module_path':'lmcache_daos.capacity_pipeline_backend',
        'storage_plugin.daosgds.class_name':'CapacityPipelineBackend',
        'daosgds.demand_read_only':True,'daosgds.object_namespace':namespace,
        'daosgds.root':'/'+namespace[:-1],'daosgds.gpu_buffer_gb':2,
        'daosgds.retrieve_window_mib':500,'daosgds.store_window_mib':500,
        'daosgds.store_staging_gib':0,'daosgds.probe_interval_ms':10})
    configure_no_dram(cfg)
    (a.output/'config.yaml').write_text(yaml.safe_dump(cfg))
    config=LMCacheEngineConfig.from_dict(cfg)
    md=LMCacheMetadata(model_name='qwen4b-shaped-path-profile',world_size=1,local_world_size=1,
        worker_id=0,local_worker_id=0,kv_dtype=torch.bfloat16,
        kv_shape=(36,2,128,8,128),role='worker',chunk_size=128)
    db=ChunkedTokenDatabase(config,md)
    connector=VLLMPagedMemGPUConnectorV2(1024,36)
    engine=LMCacheEngine(config,md,db,connector,lambda tensor,src:None,lambda obj,src:obj)
    engine.post_init();be=engine.storage_manager.allocator_backend
    assert engine.storage_manager.local_cpu_backend is None and be.transport=='object'
    n=512;chunk_bytes=18*2**20;total=n*chunk_bytes
    tokens=[100+i%128 for i in range(n*128)]
    keys=[key for _,_,key in db.process_tokens(tokens)]
    object_keys=[be._object_key(k) for k in keys]
    files=['bench/profile_retrieve_path.py','libdaosgdr.so','lmcache_daos/capacity_pipeline.py',
           'lmcache_daos/capacity_pipeline_backend.py','lmcache_daos/gds_backend.py']
    dump(a.output/'plan.json',dict(namespace=namespace,keys=object_keys,workers=a.workers,repeats=a.repeats,
        chunks=n,chunk_bytes=chunk_bytes,staging_bytes=2**31,window_chunks=27,pipeline_depth=4,
        source_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
        rdma_counter_bytes_multiplier=4,link_mbps=int(Path('/sys/class/net/ens255np0/speed').read_text()),
        notes=['Fresh patterned data, then repeated reads without server cache flush.',
               'Native path skips metadata, allocator and scatter, and uses rolling worker scratch slices.',
               'Both paths issue exactly 512 native payload fetches and read 9 GiB.',
               'Lookup is measured with known-key set cleared and again warmed; metadata workers remain 16.',
               'CPU reference work, destination poisoning and validation excluded from timing.',
               'Stage summed durations overlap; use interval union/overlap for wall attribution.',
               'NVML utilization samples have vendor averaging windows and are only supporting evidence.']))
    import ctypes
    observer=ctypes.CDLL(os.environ['DAOS_COST_LIB']) if os.environ.get('DAOS_COST_LIB') else None
    if observer:
        observer.cost_phase_begin.argtypes=[ctypes.c_uint]
        observer.cost_phase_begin.restype=None
        observer.cost_phase_end.argtypes=[]
        observer.cost_phase_end.restype=None
    events=[];records=[];phase=None;trial=None
    def log(kind,start,end,**fields):
        if phase is not None:events.append(dict(trial=trial,phase=phase,kind=kind,start=start,end=end,**fields))
    stat=be._object.stat;get=be._object.get
    def observed_stat(key):
        t=time.monotonic_ns();r=stat(key);log('metadata',t,time.monotonic_ns());return r
    def observed_get(key,ptr,size,device_id=0):
        t=time.monotonic_ns();r=get(key,ptr,size,device_id);log('payload',t,time.monotonic_ns(),bytes=size);return r
    be._object.stat=observed_stat;be._object.get=observed_get
    original_alloc=be.budget.allocate
    def alloc(*args,**kwargs):
        t=time.monotonic_ns();r=original_alloc(*args,**kwargs);log('allocate',t,time.monotonic_ns());return r
    be.budget.allocate=alloc
    pynvml.nvmlInit();gpu=pynvml.nvmlDeviceGetHandleByIndex(0)
    samples=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():
            util=pynvml.nvmlDeviceGetUtilizationRates(gpu)
            samples.append(dict(t=time.monotonic_ns(),trial=trial,phase=phase,gpu_pct=util.gpu,memory_pct=util.memory,
                                power_mw=pynvml.nvmlDeviceGetPowerUsage(gpu)))
            stop.wait(.05)
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    def measure(kind,fn):
        nonlocal phase
        before=snapshot();phase=kind
        phase_id={'cold_lookup':1,'warm_lookup':2,'engine':3,'native':4}[kind]
        rep=int(trial.split('_')[0][1:]);workers=int(trial.split('_w')[1])
        phase_tag=(rep+2)*10000+workers*10+phase_id
        if observer:observer.cost_phase_begin(phase_tag)
        t=time.monotonic_ns()
        out=fn();torch.cuda.synchronize()
        end=time.monotonic_ns();phase=None
        if observer:observer.cost_phase_end()
        after=snapshot()
        record=dict(trial=trial,phase=kind,phase_tag=phase_tag,start=t,end=end,ms=(end-t)/1e6,
                    cpu_s=after['cpu_s']-before['cpu_s'],counter_interval_ms=(after['t']-before['t'])/1e6,
                    counter_delta={k:after['counters'][k]-v for k,v in before['counters'].items()},
                    thread_cpu=[dict(tid=k,name=v['name'],cpu_s=(v['ticks']-before['threads'].get(k,{'ticks':v['ticks']})['ticks'])/os.sysconf('SC_CLK_TCK'))
                                for k,v in after['threads'].items()])
        records.append(record)
        with (a.output/'trials.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        return out
    success=False
    try:
        source=[torch.empty((2,n*8,16,8,128),dtype=torch.bfloat16,device='cuda') for _ in range(36)]
        expected=(torch.arange(n*128,device='cuda')//128%127+1).to(torch.bfloat16)
        for layer in source:layer.view(2,n*128,8,128).copy_(expected.view(1,-1,1,1))
        slots=torch.arange(n*128,device='cuda',dtype=torch.long);torch.cuda.synchronize()
        engine.store(tokens,kvcaches=source,slot_mapping=slots,req_id='path-profile-store')
        wait_for(lambda:not be._put_tasks and be.budget.total_allocated_size==0)
        assert all(be.contains(k) for k in keys)
        del source;torch.cuda.empty_cache()
        destination=[torch.empty((2,n*8,16,8,128),dtype=torch.bfloat16,device='cuda') for _ in range(36)]
        engine.gpu_connector=VLLMPagedMemGPUConnectorV2(1024,36)
        copy=engine.gpu_connector.batched_to_gpu
        def scatter(*args,**kwargs):
            t=time.monotonic_ns();r=copy(*args,**kwargs);log('scatter',t,time.monotonic_ns());return r
        engine.gpu_connector.batched_to_gpu=scatter
        rng=random.Random(20261002)
        for rep in range(-1,a.repeats):
            order=list(a.workers);rng.shuffle(order)
            for w in order:
                assert be.budget.total_allocated_size==0
                be._pool.shutdown(wait=True)
                be._pool=ThreadPoolExecutor(max_workers=w,initializer=be._ensure_cuda_ctx,thread_name_prefix='profile-io')
                # Start all workers and bind CUDA outside measured intervals.
                barrier=threading.Barrier(w)
                list(be._pool.map(lambda _:barrier.wait(),range(w)))
                trial=f'r{rep}_w{w}'
                with be._known_lock:be._known.clear()
                rid=trial+'-lookup'
                assert measure('cold_lookup',lambda:engine.lookup(tokens,lookup_id=rid,pin=False))==len(tokens)
                assert measure('warm_lookup',lambda:engine.lookup(tokens,lookup_id=rid,pin=True))==len(tokens)
                paths=['engine','native'];rng.shuffle(paths)
                for path in paths:
                    if path=='engine':
                        for layer in destination:layer.fill_(-1)
                        torch.cuda.synchronize()
                        mask=measure('engine',lambda:engine.retrieve(tokens,kvcaches=destination,slot_mapping=slots,req_id=rid))
                        assert bool(mask.all())
                        for layer in destination:
                            assert bool((layer.view(2,n*128,8,128)==expected.view(1,-1,1,1)).all())
                        assert be.budget.total_allocated_size==0
                    else:
                        slab=be.memory_allocator.tensor
                        slab.zero_();torch.cuda.synchronize()
                        def loop(slot):
                            last=None
                            for idx in range(slot,n,w):
                                be._object.get(object_keys[idx],slab.data_ptr()+slot*chunk_bytes,chunk_bytes,0)
                                last=idx
                            return slot,last
                        last=measure('native',lambda:list(be._pool.map(loop,range(w))))
                        for slot,idx in last:
                            view=slab[slot*chunk_bytes:(slot+1)*chunk_bytes].view(torch.bfloat16)
                            assert bool((view==idx%127+1).all())
                engine.lookup_unpin(rid)
                assert be.stats['alloc_fail']==0
                print(json.dumps(dict(trial=trial,verified=True,phases=[dict(phase=r['phase'],ms=r['ms']) for r in records if r['trial']==trial])),flush=True)
        success=True
    finally:
        stop.set();thread.join()
        (a.output/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
        (a.output/'gpu_samples.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in samples))
        removed=0
        try:
            for key in keys:
                assert be._object_key(key).startswith(namespace)
                if be.remove(key):removed+=1
            assert all(be._object.stat(k) is None for k in object_keys)
            dump(a.output/'cleanup.json',dict(namespace=namespace,removed=removed,remaining=0))
        finally:engine.close()
        dump(a.output/'status.json',dict(status='completed' if success else 'failed',trials=len(records),all_verified=success))


if __name__=='__main__':main()
