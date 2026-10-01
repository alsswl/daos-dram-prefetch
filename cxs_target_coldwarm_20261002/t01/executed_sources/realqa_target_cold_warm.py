"""Small CXS target-spread comparison: 8 sessions, 6 turns, cold + fixed warm replay."""
import argparse
import asyncio
import ctypes as C
import fcntl
import json
import os
from pathlib import Path
import random
import shutil
import sqlite3
import subprocess
import sys
import time
import uuid

import yaml
import realqa_q4_cxs as qa
from realqa_placement_comparison import environment

LIGHT=qa.ROOT/'cxs_light_workload_20261002'
COUNTS=[1,2,4,8,16]


def prepare(folder):
    folder.mkdir(parents=True,exist_ok=False)
    original=qa.read(LIGHT/'sessions.json')
    sessions=[dict(s,index=s['group'],slot=0) for s in original if s['slot']==0]
    assert len(sessions)==8 and sorted(s['index'] for s in sessions)==list(range(8))
    files={s['file'] for s in sessions}
    docs=[d for d in qa.read(LIGHT/'documents.json') if d['file'] in files]
    schedule=dict(concurrency=8,sessions=8,turns_per_session=6,requests=48,
                  lanes=[[dict(lane=lane,position=t,session_index=lane,turn=t,new_session=t==0)
                          for t in range(6)] for lane in range(8)])
    profile=qa.read(LIGHT/'profile.json')
    initial=sum(d['initial_prompt_tokens'] for d in docs)*qa.KV_BYTES/2**30
    profile.update(session_depth=1,sessions=8,expected_requests=48,
                   capacity=dict(unique_selected_books=len(docs),initial_unique_kv_upper_gib=initial,
                                 stored_kv_upper_gib=initial+2*8*6*(256+128)*qa.KV_BYTES/2**30),
                   phases=['cold','warm'],resume_token_fix=True,kv_load_failure_policy='fail')
    execution=COUNTS.copy();random.Random(20261002).shuffle(execution)
    reference=folder/f't{execution[0]:02d}'/'serving'/'cold'/'histories'
    # Reuse audited sources but snapshot their CURRENT contents for this new run.
    names=set(qa.read(qa.ROOT/'realqa_placement_20261002_baseline'/'plan.json')['source_sha256'])
    names.update(['realqa_target_cold_warm.py','prepare_cxs_light.py','tests/test_realqa_q4_cxs.py'])
    for count in COUNTS:
        root=folder/f't{count:02d}';root.mkdir();(root/'documents').mkdir()
        for doc in docs:shutil.copy2(LIGHT/'documents'/doc['file'],root/'documents'/doc['file'])
        for name,data in [('sessions.json',sessions),('documents.json',docs),('mixed_schedule.json',schedule)]:qa.dump(root/name,data)
        plan=dict(profile,target_count=count,experiment_namespace='minji-cxs-target-'+uuid.uuid4().hex+':',
                  placement_manifest=str(root/'placement.json'),cases=[dict(name='serving',policy='none')],
                  reference_histories=str(reference),reference_condition=execution[0],
                  notes=['Qwen3-4B, real Gutenberg 61440-token documents, C8 x S1 x 6 turns = 48 requests per phase.',
                         'Reduced from C8xS2 because one target cannot contain its full cold/warm KV working set.',
                         'All conditions use 16 I/O workers, 2GiB shared staging, 500MiB windows, DRAM/prefix cache/prefetch OFF.',
                         'Every chunk has its own dkey and kv/meta akeys; target count only changes salted dkey routing.',
                         'First executed cold phase generates reference conversation histories. All later phases replay those exact inputs.',
                         'Cold starts with an empty namespace but naturally reuses prefixes during its follow-up turns.',
                         'Warm retains the same process and DAOS namespace after all cold stores drain.',
                         'Warm outputs do not alter later prompts; frozen cold-generated histories preserve exact input equality.',
                         'No OS/DAOS server cache flush. One cold/warm pair per target count; no confidence interval.',
                         '1-to-2 targets also changes engine count; 2/4/8/16 split evenly across both engines.',
                         'Only isolated run-owned objects are cleaned after both phases.'])
        plan['input_sha256']={name:qa.base.digest(root/name) for name in ['sessions.json','documents.json','mixed_schedule.json']}
        plan['source_sha256']={}
        for name in sorted(names):
            dest=root/'executed_sources'/name;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(qa.ROOT/name,dest);plan['source_sha256'][name]=qa.base.digest(dest)
        qa.dump(root/'plan.json',plan);qa.dump(root/'status.json',dict(status='prepared'))
    qa.dump(folder/'comparison_plan.json',dict(target_counts=COUNTS,execution_order=execution,
        requests_per_phase=48,phases=['cold','warm'],total_requests=480,reference_histories=str(reference)))
    qa.dump(folder/'status.json',dict(status='prepared'))


def manifest(root):
    from experiments.chunk_placement.prepare_manifest import prepare as make
    plan=qa.read(root/'plan.json');path=Path(plan['placement_manifest'])
    m=make(path,'baseline')
    ranks=sorted({p[0] for p in m['layout'].values()});assert len(ranks)==2
    by_rank={rank:sorted([int(s) for s,p in m['layout'].items() if p[0]==rank],key=lambda s:m['layout'][s][1]) for rank in ranks}
    order=[by_rank[rank][i] for i in range(8) for rank in ranks]
    m['target_spread_shards']=order[:plan['target_count']]
    m['selected_physical_targets']=[m['layout'][s] for s in m['target_spread_shards']]
    qa.dump(path,m)


def make_config(plan):
    from prepare_windowed_config import make_config as window_config
    from lmcache_daos.no_dram import configure_no_dram
    from lmcache_daos.placement_backend import configure
    cfg=window_config(500,2,1);configure_no_dram(cfg)
    cfg['extra_config'].update({'daosgds.store_window_mib':500,'daosgds.store_staging_gib':0,
        'daosgds.object_namespace':plan['experiment_namespace'],'daosgds.root':'/'+plan['experiment_namespace'][:-1]})
    configure(cfg,plan['placement_manifest'])
    return cfg


def capacity_guard(root):
    plan=qa.read(root/'plan.json');pool=qa.base.storage_guard(root,plan['capacity']['stored_kv_upper_gib'])
    nvme=next(t for t in pool['response']['tier_stats'] if t['media_type']=='nvme')
    # Global min is conservative for every selected target, including the one-target arm.
    need=(plan['capacity']['stored_kv_upper_gib']/plan['target_count']*1.10+4)*2**30
    assert nvme['min']>need, f"Selected-target capacity headroom insufficient: minimum {nvme['min']/2**30:.1f}GiB, need {need/2**30:.1f}GiB"
    return pool


def execute(root):
    plan=qa.read(root/'plan.json');case=root/'serving'
    for name,digest in plan['source_sha256'].items():assert qa.base.digest(qa.ROOT/name)==digest,name
    for name,digest in plan['input_sha256'].items():assert qa.base.digest(root/name)==digest,name
    for doc in qa.read(root/'documents.json'):assert qa.base.digest(root/'documents'/doc['file'])==doc['sha256']
    assert qa.early.idle_gpu()
    capacity_guard(root);case.mkdir()
    os.environ.update(DAOS_GDS_PREFETCH_TIMING='1',DAOS_LOOKUP_READY_RETURN='0',DAOS_LOOKUP_LOCK_PROBE='0',DAOS_VLLM_RESUME_TOKEN_FIX='1')
    cfg=make_config(plan);(case/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    try:
        with qa.base.server(qa.server_args(plan),case/'config.yaml',case):
            initial=qa.base.await_empty(case)
            assert initial['used_bytes']==initial['cpu_hot_bytes']==initial['daos_puts']==0
            qa.dump(case/'initial_sample.json',initial)
            health=qa.base.LogHealth(case/'server.log')
            previous=initial
            for phase in ['cold','warm']:
                dest=case/phase;dest.mkdir()
                qa.dump(dest/'initial_sample.json',previous)
                reference=None if phase=='cold' and plan['target_count']==plan['reference_condition'] else Path(plan['reference_histories'])
                if reference is not None:assert len(list(reference.glob('*.json')))==8
                qa.dump(root/'phase.json',dict(phase=phase,target_count=plan['target_count']))
                window=dict(start_ns=time.time_ns(),phase=phase,replay_histories=str(reference) if reference else None)
                qa.dump(dest/'workload.json',window)
                asyncio.run(qa.workload(root,dest,health,replay_histories=reference))
                window['end_ns']=time.time_ns();qa.dump(dest/'workload.json',window)
                previous=qa.base.drain(case,health,timeout=180)
                qa.dump(dest/'final_sample.json',previous)
                assert previous['used_bytes']==0
                print(f"targets={plan['target_count']} phase={phase}: 48 requests completed and stores drained",flush=True)
            qa.dump(case/'final_sample.json',previous)
        qa.validate_windowed_run(case)
        from lmcache_daos.no_dram import validate_no_dram_events
        events=qa.base.read_events(case)
        qa.dump(case/'no_dram_check.json',validate_no_dram_events(events))
        enabled=[e for e in events if e['event']=='target_spread_enabled']
        assert len(enabled)==1 and enabled[0]['target_count']==plan['target_count']
        # All phases must use the exact same request payloads as reference cold.
        ref_calls=qa.read(Path(plan['reference_histories']).parent/'calls.json')
        hashes={(r['session_index'],r['turn']):r['messages_sha256'] for r in ref_calls}
        for phase in ['cold','warm']:
            calls=qa.read(case/phase/'calls.json')
            assert len(calls)==48 and all(c['status']=='success' for c in calls)
            assert all(c['messages_sha256']==hashes[c['session_index'],c['turn']] for c in calls)
        qa.dump(root/'status.json',dict(status='completed',requests=96,identical_reference_prompts=True))
    except BaseException as exc:
        qa.dump(root/'status.json',dict(status='failed',error=repr(exc)));raise


def cleanup(root):
    from lmcache_daos.placement_backend import PlacementObjectStore,checked
    plan=qa.read(root/'plan.json');assert qa.read(root/'status.json')['status']=='completed'
    m=qa.read(Path(plan['placement_manifest']));assert m['isolated_experiment_object']
    store=PlacementObjectStore(plan['placement_manifest'],m['pool'],m['container'])
    try:
        db=sqlite3.connect(f"file:{Path(plan['placement_manifest']).with_suffix('.positions.sqlite')}?mode=ro",uri=True)
        rows=db.execute('SELECT key,idx FROM positions WHERE key LIKE ?', (plan['experiment_namespace']+'%',)).fetchall();db.close()
        assert rows
        counts={}
        for key,index in rows:
            dk,_,_=store.address(key+'\x1f'+str(index))
            actual=C.c_uint();checked(store._lib.placement_shard(store._ctx,dk,C.byref(actual)))
            expected=m['target_spread_shards'][index%plan['target_count']]
            assert actual.value==expected
            counts[actual.value]=counts.get(actual.value,0)+1
        qa.dump(root/'placement_validation.json',dict(passed=True,validated_keys=len(rows),counts_by_shard=counts,
            selected_physical_targets=m['selected_physical_targets'],distinct_dkey_per_chunk=True))
        ctx=store._ctx;store._ctx=None
        rc=store._lib.placement_close(ctx,1)
        qa.dump(root/'owned_object_cleanup.json',dict(oid=m['oid'],rc=rc,production_object_touched=False));checked(rc)
    finally:store.close()


def run(folder):
    p=qa.read(folder/'comparison_plan.json')
    with (folder/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for count in p['execution_order']:
            root=folder/f't{count:02d}'
            if qa.read(root/'status.json')['status']=='completed':
                assert (root/'owned_object_cleanup.json').exists();continue
            assert qa.early.idle_gpu()
            for attempt in range(60):
                try:
                    capacity_guard(root);break
                except (AssertionError,RuntimeError) as exc:
                    if attempt==59:raise
                    print(f'Waiting for deleted DAOS space to be reclaimed: {exc}',flush=True)
                    time.sleep(10)
            qa.dump(folder/'status.json',dict(status='running',target_count=count))
            env=environment()
            if not (root/'placement.json').exists():
                subprocess.run([sys.executable,__file__,'manifest','--output',str(root)],env=env,check=True)
            gate=root/'gpu_gate'
            if not gate.exists():
                command=[str(qa.ROOT/'run_vllm.sh'),sys.executable,str(qa.ROOT/'tests/sync_retrieve_capacity.py'),
                    '--capacity','2','--chunks','128','--window-mib','500','--store-window-mib','500',
                    '--capacity-pipeline','--split-store-gib','0','--no-dram',
                    '--placement-manifest',str(root/'placement.json'),'--output',str(gate)]
                with (root/'gpu_gate.log').open('x') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
            result=qa.read(gate/'result.json')
            assert result['status']=='passed' and result['returned_tokens']==16384 and result['placement_validation']['passed']
            assert qa.read(gate/'cleanup.json')['remaining']==0
            with (root/'benchmark.log').open('x') as log:
                subprocess.run([sys.executable,__file__,'execute','--output',str(root)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            subprocess.run([sys.executable,__file__,'cleanup','--output',str(root)],env=env,check=True)
            print(f'Target count {count}: cold/warm complete; owned object removed',flush=True)
        qa.dump(folder/'status.json',dict(status='completed',requests=480))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','manifest','execute','cleanup','run'])
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    globals()[args.action](args.output.resolve())
