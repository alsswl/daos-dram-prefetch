"""Full C8 x S10 CXS, native async loading, shared 2GiB, immediate misses."""
import argparse
import asyncio
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import yaml
import realqa_q4_cxs as qa
from realqa_placement_comparison import environment
from recompute_experiment_support import RecomputeHealth

SOURCE = qa.ROOT/'realqa_placement_20261002_baseline'
REFERENCE = SOURCE/'c8_s10_placement_baseline'
CONFIG_SOURCE = qa.ROOT/'cxs_target_coldwarm_20261002/t16/serving/config.yaml'


def config(plan):
    cfg = yaml.safe_load(CONFIG_SOURCE.read_text())
    cfg['enable_async_loading'] = True
    ec = cfg['extra_config']
    for k in list(ec):
        if any(s in k for s in ('window', 'demand_read', 'store_staging', 'dram_mirror')):
            ec.pop(k)
    ec.update({'storage_plugin.daosgds.module_path':'lmcache_daos.async_drop_backend',
        'storage_plugin.daosgds.class_name':'AsyncDropBackend',
        'daosgds.object_namespace':plan['experiment_namespace'],
        'daosgds.root':'/'+plan['experiment_namespace'][:-1],
        'daosgds.placement_manifest':plan['placement_manifest']})
    return cfg


def prepare(root):
    root.mkdir(exist_ok=False)
    shutil.copytree(SOURCE/'documents', root/'documents')
    for name in ('documents.json','sessions.json','mixed_schedule.json'):
        shutil.copy2(SOURCE/name,root/name)
    plan=qa.read(SOURCE/'plan.json')
    for k in ('target_count','reference_condition','source_sha256'):
        plan.pop(k,None)
    plan.update(experiment_namespace='minji-async-drop-'+uuid.uuid4().hex+':',
        placement_manifest=str(root/'placement.json'),reference_histories=str(REFERENCE/'histories'),
        kv_load_failure_policy='recompute',store_window_mib=0,retrieve_window_mib=0,
        capacity_pipeline=False,enable_async_loading=True,phases=['cold'],
        cases=[dict(name='serving',policy='async_drop')],
        notes=['Same full C8 x S10 x 6 turns (480 requests), documents, mixed schedule and frozen histories as prior baseline CXS.',
               'Original full-key hash placement, 16 targets, 16 IO workers, 2GiB shared GPU staging, no DRAM.',
               'Native async loading; original AsyncSingleSerializer, not eight simultaneous request-level fetch batches.',
               'Both store and read allocations fail immediately without retry. Native store truncates its tail.',
               'Async lookup responds with actual contiguous loaded prefix, remaining input gets normal prefill.',
               'One 480-request run from empty namespace, natural multi-turn reuse. No complete-cache preload.',
               'No server cache flush; only run-owned isolated object is removed.'])
    names=set(qa.read(SOURCE/'plan.json')['source_sha256'])
    names.update(['realqa_async_drop.py','lmcache_daos/async_drop_backend.py','tests/async_drop_capacity.py'])
    plan['source_sha256']={}
    for name in sorted(names):
        dest=root/'executed_sources'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(qa.ROOT/name,dest);plan['source_sha256'][name]=qa.base.digest(dest)
    qa.dump(root/'plan.json',plan)
    qa.dump(root/'status.json',dict(status='prepared'))


def manifest(root):
    from experiments.chunk_placement.prepare_manifest import prepare as make
    make(root/'placement.json','baseline')


def execute(root):
    plan=qa.read(root/'plan.json');case=root/'serving';case.mkdir()
    for name,digest in plan['source_sha256'].items():assert qa.base.digest(qa.ROOT/name)==digest,name
    assert qa.early.idle_gpu()
    qa.base.storage_guard(root,plan['capacity']['stored_kv_upper_gib'])
    os.environ.update(DAOS_VLLM_RESUME_TOKEN_FIX='1',DAOS_LOOKUP_READY_RETURN='0',
        DAOS_LOOKUP_LOCK_PROBE='0',DAOS_GDS_PREFETCH_TIMING='0')
    (case/'config.yaml').write_text(yaml.safe_dump(config(plan),sort_keys=False))
    try:
        with qa.base.server(qa.server_args(plan),case/'config.yaml',case) as client:
            previous=qa.base.await_empty(case)
            assert previous['used_bytes']==previous['cpu_hot_bytes']==previous['daos_puts']==0
            qa.dump(case/'initial_sample.json',previous)
            health=RecomputeHealth(case/'server.log')
            for phase in plan['phases']:
                dest=case/phase;dest.mkdir()
                qa.dump(root/'phase.json',dict(phase=phase))
                qa.dump(dest/'initial_sample.json',previous)
                (dest/'metrics_before.txt').write_text(client.get('/metrics').text)
                window=dict(start_ns=time.time_ns(),phase=phase)
                qa.dump(dest/'workload.json',window)
                asyncio.run(qa.workload(root,dest,health,replay_histories=Path(plan['reference_histories'])))
                window['end_ns']=time.time_ns();qa.dump(dest/'workload.json',window)
                previous=qa.base.drain(case,health,timeout=180)
                qa.dump(dest/'final_sample.json',previous)
                (dest/'metrics_after.txt').write_text(client.get('/metrics').text)
                assert previous['used_bytes']==0
                print(f'{phase}: {plan["expected_requests"]} requests completed, staging drained',flush=True)
            qa.dump(case/'final_sample.json',previous)
        qa.dump(root/'status.json',dict(status='completed',requests=plan['expected_requests']*len(plan['phases'])))
    except BaseException as exc:
        qa.dump(root/'status.json',dict(status='failed',error=repr(exc)));raise


def cleanup(root):
    from lmcache_daos.async_drop_backend import OriginalKeyStore
    from lmcache_daos.placement_backend import checked
    assert qa.read(root/'status.json')['status']=='completed'
    m=qa.read(root/'placement.json');assert m['isolated_experiment_object'] and m['nonce']>=1000000
    store=OriginalKeyStore(root/'placement.json',m['pool'],m['container'])
    ctx=store._ctx;store._ctx=None;rc=store._lib.placement_close(ctx,1)
    qa.dump(root/'owned_object_cleanup.json',dict(oid=m['oid'],rc=rc,production_object_touched=False));checked(rc)


def run(root):
    env=environment()
    for action in ('manifest','gate','execute','cleanup'):
        command=([str(qa.ROOT/'run_vllm.sh'),sys.executable,str(qa.ROOT/'tests/async_drop_capacity.py'),'--output',str(root)]
                 if action=='gate' else [sys.executable,__file__,action,'--output',str(root)])
        with (root/(action+'.log')).open('x') as log:
            subprocess.run(command,cwd=qa.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        print(action+' complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','manifest','execute','cleanup','run'])
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    globals()[args.action](args.output.resolve())
