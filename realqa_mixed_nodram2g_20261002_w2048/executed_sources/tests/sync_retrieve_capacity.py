"""Actual DAOS + native synchronous LMCache retrieve capacity validation.

Uses patterned KV tensors with Qwen3-4B dimensions and the real paged GPU
connector. No model forward pass or vLLM scheduler/HTTP failure-policy test.
CPU cache is emptied before lookup; read promotion and all prefetch are OFF.
"""
import argparse
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def dump(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def check(folder,capacity,chunks,window_mib=None,store_window_mib=None,split_store_gib=None,capacity_pipeline=False,no_dram=False):
    import torch
    import yaml
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.metadata import LMCacheMetadata
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.token_database import ChunkedTokenDatabase
    from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2
    from async_dram_promotion_roundtrip import wait_for
    from staging_mixed_pressure import read_events

    folder.mkdir(exist_ok=False)
    os.environ['DAOS_GDS_STAGING_TRACE']=str(folder/'trace')
    cfg=yaml.safe_load((ROOT/'lmcache_config_daosgds_async_dram.yaml').read_text())
    namespace='minji-sync-capacity-'+uuid.uuid4().hex+':'
    cfg.update(chunk_size=128,enable_async_loading=False,use_layerwise=False,max_local_cpu_size=.125)
    cfg['extra_config'].update({
        'storage_plugin.daosgds.module_path':'lmcache_daos.demand_read_backend',
        'storage_plugin.daosgds.class_name':'DemandReadBackend',
        'daosgds.demand_read_only':True,'daosgds.dram_prefetch':False,
        'daosgds.dram_promote_on_read':False,'daosgds.object_namespace':namespace,
        'daosgds.root':'/'+namespace[:-1],'daosgds.gpu_buffer_gb':capacity,
        'daosgds.dram_mirror_max_pending_gb':.0625,
        'daosgds.dram_mirror_max_age_ms':10000,'daosgds.probe_interval_ms':5})
    if window_mib is not None:
        cfg['extra_config'].update({
            'storage_plugin.daosgds.module_path':'lmcache_daos.windowed_demand_backend',
            'storage_plugin.daosgds.class_name':'WindowedDemandBackend',
            'daosgds.retrieve_window_mib':window_mib,
            'daosgds.retrieve_window_timeout_s':5})
    if store_window_mib is not None:
        cfg['extra_config'].update({
            'storage_plugin.daosgds.module_path':'lmcache_daos.windowed_store_backend',
            'storage_plugin.daosgds.class_name':'WindowedStoreBackend',
            'daosgds.store_window_mib':store_window_mib})
    if split_store_gib is not None:
        cfg['extra_config'].update({
            'storage_plugin.daosgds.module_path':'lmcache_daos.split_staging_backend',
            'storage_plugin.daosgds.class_name':'SplitStagingBackend',
            'daosgds.store_staging_gib':split_store_gib})
    if capacity_pipeline:
        cfg['extra_config'].update({
            'storage_plugin.daosgds.module_path':'lmcache_daos.capacity_pipeline_backend',
            'storage_plugin.daosgds.class_name':'CapacityPipelineBackend',
            'daosgds.store_staging_gib':split_store_gib or 0})
    if no_dram:
        from lmcache_daos.no_dram import configure_no_dram
        configure_no_dram(cfg)
    (folder/'config.yaml').write_text(yaml.safe_dump(cfg))
    config=LMCacheEngineConfig.from_dict(cfg)
    md=LMCacheMetadata(model_name='qwen4b-shaped-capacity-test',world_size=1,local_world_size=1,
        worker_id=0,local_worker_id=0,kv_dtype=torch.bfloat16,
        kv_shape=(36,2,128,8,128),role='worker',chunk_size=128)
    db=ChunkedTokenDatabase(config,md)
    connector=VLLMPagedMemGPUConnectorV2(1024,36)
    engine=LMCacheEngine(config,md,db,connector,lambda tensor,src:None,lambda obj,src:obj)
    engine.post_init()
    manager=engine.storage_manager
    be,cpu=manager.allocator_backend,manager.local_cpu_backend
    assert not engine.async_loading and manager.async_serializer is None
    native_path=Path(inspect.unwrap(LMCacheEngine._process_tokens_internal).__code__.co_filename)
    assert 'site-packages/lmcache/v1/cache_engine.py' in str(native_path)
    assert not hasattr(manager,'_daos_early_lookup'), 'Do not use early async demand path'
    tokens=[100+i%128 for i in range(chunks*128)]
    keys=[key for _,_,key in db.process_tokens(tokens)]
    assert len(keys)==chunks
    dump(folder/'manifest.json',dict(namespace=namespace,keys=[be._object_key(k) for k in keys],
        chunks=chunks,payload_bytes=chunks*18*2**20,native_process_tokens_source=str(native_path),
        window_mib=window_mib,store_window_mib=store_window_mib,
        notes=['Patterned KV, not model-generated KV; real DAOS writes/reads and real paged GPU scatter.',
               'Native synchronous engine retrieve; custom DAOS transport and passive tracing remain.',
               'No vLLM scheduler: returned mask is tested, not HTTP failure/recompute behavior.']))
    success=False
    try:
        if store_window_mib is None:
            for i,key in enumerate(keys):
                obj=manager.allocate(md.get_shapes(),md.get_dtypes(),fmt=None)
                assert obj is not None and obj.get_size()==18*2**20
                obj.tensor.fill_(i%127+1)
                torch.cuda.synchronize()
                manager.batched_put([key],[obj])
                wait_for(lambda:not be.exists_in_put_tasks(key))
                if (i+1)%64==0:
                    print(f'{capacity}GiB: stored {i+1}/{chunks} chunks',flush=True)
        else:
            # Exercise the actual engine gather/store path on a request larger
            # than the staging pool, including a resumed nonzero prefix mask.
            source=[torch.empty((2,chunks*8,16,8,128),dtype=torch.bfloat16,device='cuda')
                    for _ in range(36)]
            values=(torch.arange(len(tokens),device='cuda')//128%127+1).to(torch.bfloat16)
            for layer in source:
                layer.view(2,len(tokens),8,128).copy_(values.view(1,-1,1,1))
            slots=torch.arange(len(tokens),device='cuda',dtype=torch.long)
            torch.cuda.synchronize()
            prefix=min(3,chunks)*128
            engine.store(tokens[:prefix],kvcaches=source,slot_mapping=slots,req_id='window-store-prefix')
            mask=torch.ones(len(tokens),dtype=torch.bool);mask[:prefix]=False
            engine.store(tokens,mask=mask,kvcaches=source,slot_mapping=slots,req_id='window-store-tail')
            wait_for(lambda:not be._put_tasks)
            assert all(be.contains(k) for k in keys), 'Windowed store omitted keys'
            assert be.stats['alloc_fail']==0, 'Store staging allocation failed'
            print(f'{capacity}GiB: windowed engine store retained all {chunks} chunks',flush=True)
        wait_for(lambda:be.dram_mirror.snapshot()['pending_bytes']==0
                 and be.memory_allocator.allocator.total_allocated_size==0)
        if no_dram:
            assert cpu is None, 'DRAM-off run must have no LocalCPUBackend'
        else:
            for key in keys:
                if cpu.contains(key):
                    assert cpu.remove(key)
            assert not any(cpu.contains(key) for key in keys)
        if store_window_mib is not None:
            # Native V2 caches physical page pointers per device. Use a fresh
            # connector for a physically separate destination in this test.
            connector=VLLMPagedMemGPUConnectorV2(1024,36)
            engine.gpu_connector=connector
        # Real vLLM paged layout, physically separate from the staging arena.
        destination=[torch.full((2,chunks*8,16,8,128),-1,dtype=torch.bfloat16,device='cuda')
                     for _ in range(36)]
        slots=torch.arange(len(tokens),device='cuda',dtype=torch.long)
        torch.cuda.synchronize()
        rid=f'sync-capacity-{capacity}'
        look_start=time.time_ns()
        hits=engine.lookup(tokens,lookup_id=rid,pin=True)
        assert hits==len(tokens)
        assert be.stats['get']==0 and be.memory_allocator.allocator.total_allocated_size==0
        scatter=[]
        original=connector.batched_to_gpu
        def observed(objects,starts,ends,**kwargs):
            scatter.append(dict(chunks=len(objects),bytes=sum(o.get_size() for o in objects),
                staging_bytes_at_entry=be.memory_allocator.allocator.total_allocated_size,
                time_ns=time.time_ns()))
            return original(objects,starts,ends,**kwargs)
        connector.batched_to_gpu=observed
        started=time.time_ns()
        mask=engine.retrieve(tokens,kvcaches=destination,slot_mapping=slots,req_id=rid)
        torch.cuda.synchronize()
        ended=time.time_ns()
        loaded=int(mask.sum())
        assert torch.equal(mask,torch.arange(len(tokens))<loaded)
        assert loaded%128==0
        expected=(torch.arange(len(tokens),device='cuda')//128%127+1).to(torch.bfloat16)
        expected[loaded:]=-1
        for layer in destination:
            assert bool((layer.view(2,len(tokens),8,128)==expected.view(1,-1,1,1)).all()), 'GPU scatter data mismatch'
        engine.lookup_unpin(rid)
        wait_for(lambda:be.memory_allocator.allocator.total_allocated_size==0)
        events=read_events(folder)
        outcome=[e for e in events if e['event']=='daos_demand_outcome']
        assert outcome and all(o['other_failed_chunks']==0 for o in outcome)
        failures=sum(o['capacity_failed_chunks'] for o in outcome)
        assert not any(e['event'] in ('early_payload_start','prefetch_ready','cpu_prefetch_ready') for e in events)
        selected=[e for e in events if started<=e['time_ns']<=ended]
        peak=max(e['used_bytes'] for e in selected)
        theoretical=min(chunks,int(capacity*2**30)//(18*2**20))
        assert sum(s['chunks'] for s in scatter)==loaded//128
        if window_mib:
            effective_chunks=int(window_mib*2**20)//(18*2**20)
            assert loaded==len(tokens) and failures==0
            assert all(s['chunks']<=effective_chunks for s in scatter)
            assert len(scatter)==(chunks+effective_chunks-1)//effective_chunks
            peak_limit=((capacity-(split_store_gib or 0))*2**30 if capacity_pipeline or split_store_gib is not None
                        else effective_chunks*18*2**20)
            assert peak<=peak_limit
        elif chunks*18*2**20>capacity*2**30:
            assert 0<loaded//128<=theoretical and len(scatter)==1
            assert loaded<len(tokens) and outcome[0]['capacity_failed_chunks']>0
        else:
            assert len(scatter)==1
            assert loaded==len(tokens) and outcome[0]['capacity_failed_chunks']==0
        result=dict(status='passed',staging_gib=capacity,requested_tokens=len(tokens),
            window_mib=window_mib,store_window_mib=store_window_mib,
            requested_chunks=chunks,requested_gib=chunks*18/1024,
            lookup_hit_tokens=hits,returned_tokens=loaded,returned_chunks=loaded//128,
            returned_gib=loaded//128*18/1024,capacity_failed_chunks=failures,
            successful_tail_discarded_chunks=sum(o['successful_tail_discarded_chunks'] for o in outcome),
            peak_staging_gib=peak/2**30,staging_bytes_after=0,scatter_calls=scatter,
            all_returned_gpu_values_match=True,unreturned_gpu_region_unchanged=True,
            retrieve_ms=(ended-started)/1e6,lookup_start_ns=look_start,
            start_ns=started,end_ns=ended)
        if store_window_mib is not None:
            starts=[e for e in events if e['event']=='store_window_copy_start']
            finishes=[e for e in events if e['event']=='store_window_submitted']
            assert len(starts)==len(finishes) and sum(e['chunks'] for e in finishes)==chunks
            assert all(e['bytes']<=store_window_mib*2**20 for e in starts)
            assert not any(e.get('failed') for e in events if e['event'] in ('allocate','batched_allocate'))
            result.update(store_windows=len(finishes),store_allocation_failures=0,all_store_keys_present=True)
        if capacity_pipeline:
            from lmcache_daos.capacity_validation import validate_capacity_events
            result['capacity_validation']=validate_capacity_events(events)
        elif split_store_gib is not None:
            from lmcache_daos.split_validation import validate_split_events
            result['split_validation']=validate_split_events(events)
        if no_dram:
            from lmcache_daos.no_dram import validate_no_dram_events
            result['no_dram_validation']=validate_no_dram_events(events)
        dump(folder/'result.json',result)
        print(json.dumps(result),flush=True)
        success=True
    finally:
        if success:
            removed=0
            for key in keys:
                assert be._object_key(key).startswith(namespace)
                assert be.remove(key)
                removed+=1
            assert all(be._object.stat(be._object_key(k)) is None for k in keys)
            dump(folder/'cleanup.json',dict(removed=removed,namespace=namespace,remaining=0,logs_preserved=True))
        engine.close()


def run(folder,chunks):
    from report_capacity_matrix import save_chart
    from staging_mixed_pressure import read_events
    folder.mkdir(exist_ok=False)
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    results=[]
    for cap in (4,8,12):
        case=folder/f's{cap}'
        with (folder/f's{cap}.log').open('x') as log:
            subprocess.run([sys.executable,__file__,'--output',str(case),'--capacity',str(cap),
                '--chunks',str(chunks)],cwd=ROOT,env=dict(os.environ,DAOS_GDS_MULTI_PREFETCH='0'),
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
        r=json.loads((case/'result.json').read_text()); results.append(r)
        print(f's{cap}: {r["returned_chunks"]}/{chunks} chunks, capacity failures={r["capacity_failed_chunks"]}',flush=True)
    dump(folder/'summary.json',results)
    svg=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="760">',
        '<rect width="1000" height="760" fill="white"/>',
        '<g font-family="sans-serif" font-size="14">',
        '<text x="60" y="30" font-size="20">9GiB DAOS KV: native synchronous retrieve, prefetch OFF</text>',
        '<text x="60" y="55">Patterned Qwen3-4B-shaped KV; one real paged-GPU scatter call per case. Not inference.</text>']
    for idx,r in enumerate(results):
        top=100+idx*210
        events=[e for e in read_events(folder/f's{r["staging_gib"]}') if r['start_ns']<=e['time_ns']<=r['end_ns']]
        duration=(r['end_ns']-r['start_ns'])/1e6
        svg.append(f'<text x="60" y="{top-10}">Staging {r["staging_gib"]}GiB: returned {r["returned_chunks"]}/{chunks} chunks, peak {r["peak_staging_gib"]:.3f}GiB</text>')
        for tick in (0,25,50,75,100):
            y=top+130*(1-tick/100)
            svg.extend([f'<line x1="70" x2="960" y1="{y}" y2="{y}" stroke="#ddd"/>',f'<text x="25" y="{y+4}">{tick}%</text>'])
        points=' '.join(f'{70+890*(e["time_ns"]-r["start_ns"])/(r["end_ns"]-r["start_ns"]):.2f},{top+130*(1-e["used_bytes"]/(r["staging_gib"]*2**30)):.2f}' for e in events)
        svg.append(f'<polyline points="{points}" fill="none" stroke="#0072b2" stroke-width="1.5"/>')
        svg.extend([f'<text x="70" y="{top+155}">0ms</text>',f'<text x="860" y="{top+155}">{duration:.1f}ms</text>'])
    svg.extend(['<text x="60" y="745">Occupancy includes allocation/free events; failed allocation is not a whole-GPU CUDA OOM.</text>','</g></svg>'])
    save_chart(folder,'occupancy',svg)
    dump(folder/'status.json',dict(status='completed',cases=3))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--capacity',type=int,choices=[1,2,4,8,10,12])
    p.add_argument('--chunks',type=int,default=512)
    p.add_argument('--window-mib',type=int,help='Enable windowed backend; 0 is legacy baseline, e.g. 128/512/1024')
    p.add_argument('--store-window-mib',type=int)
    p.add_argument('--split-store-gib',type=float)
    p.add_argument('--capacity-pipeline',action='store_true')
    p.add_argument('--no-dram',action='store_true')
    a=p.parse_args()
    assert 1<=a.chunks<=512
    if a.window_mib is not None:
        assert a.capacity is not None, '--window-mib requires --capacity'
        assert a.window_mib==0 or 18<=a.window_mib<=a.capacity*1024
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(), 'GPU occupied; do not disturb other jobs'
    if a.capacity:
        check(a.output.resolve(),a.capacity,a.chunks,a.window_mib,a.store_window_mib,a.split_store_gib,a.capacity_pipeline,a.no_dram)
    else:
        run(a.output.resolve(),a.chunks)
