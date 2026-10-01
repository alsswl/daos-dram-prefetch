"""Mixed CXS 480 requests/condition, DRAM off, 2 GiB staging, 500 MiB windows."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

import realqa_q4_cxs as qa
from realqa_mixed_comparison import check_sources,wait_space

SOURCE=qa.ROOT/'realqa_mixed_nodram2g_20261002_w500'


def prepare(folder):
    assert folder.parent==qa.ROOT and not folder.exists()
    folder.mkdir();cases=[]
    from experiments.chunk_placement.prepare_manifest import prepare as manifest
    for mode in ('baseline','balanced'):
        root=qa.ROOT/f'{folder.name}_{mode}';root.mkdir()
        plan=qa.read(SOURCE/'plan.json')
        shutil.copytree(SOURCE/'documents',root/'documents')
        for name in ('documents.json','sessions.json','mixed_schedule.json'):shutil.copy2(SOURCE/name,root/name)
        m=root/'placement.json';manifest(m,mode)
        plan.update(placement_manifest=str(m),experiment_namespace='minji-placement-'+uuid.uuid4().hex+':',
                    cases=[dict(name=f'c8_s10_placement_{mode}',policy='none')],source_dataset=str(SOURCE),
                    store_window_mib=500,retrieve_window_mib=500)
        plan['notes']=[
            'Same documents and mixed 80-session x 6-turn schedule, 480 requests per condition.',
            'DRAM off, shared 2GiB staging, 500MiB store/read windows (486MiB effective), 16 I/O workers.',
            'Both use PlacementBackend, identical position tracking, metadata transport and per-chunk fetches.',
            'baseline: full key dkey + kv/meta akeys; balanced: index-derived placement dkey + full key akeys.',
            'Frozen 16-shard descriptor; absolute token chunk index, never window-local index.',
            'Local SQLite positions support key-only deletion/restart; this file must accompany the namespace.',
            'Fresh isolated OID per condition; generated histories and wall-clock arrivals may differ.',
            'After each completed run, its experiment OID is punched; results and local index retained.',
            'No server-side software changes or server cache flush. One serving trial per condition.']
        files=set(plan['source_sha256'])|{'realqa_placement_comparison.py','lmcache_daos/placement_backend.py',
            'experiments/chunk_placement/addressing.py','experiments/chunk_placement/placement.c',
            'experiments/chunk_placement/lmcache_transport.c','experiments/chunk_placement/lmcache_transport.so',
            'experiments/chunk_placement/prepare_manifest.py','tests/sync_retrieve_capacity.py'}
        plan['source_sha256']={}
        for name in sorted(files):
            dst=root/'executed_sources'/name;dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(qa.ROOT/name,dst);plan['source_sha256'][name]=qa.base.digest(dst)
        qa.dump(root/'plan.json',plan);qa.dump(root/'status.json',dict(status='prepared'))
        cases.append(dict(label=mode,root=str(root),description='hash placement' if mode=='baseline' else 'balanced absolute-index placement'))
    qa.dump(folder/'comparison_plan.json',dict(cases=cases,delta_key='balanced_vs_baseline_change_pct',
        chart_title='Mixed CXS | DRAM OFF | shared 2GiB | existing vs balanced placement'))
    qa.dump(folder/'status.json',dict(status='prepared',expected_requests=960))


def environment():
    return dict(os.environ,DAOSGDS_TRANSPORT='object',DAOS_GDS_MULTI_PREFETCH='0',DAOSGDR_TIMING='0',
        HF_HOME='/home/hf/hf_cache',VLLM_PLUGINS='daos_resume_tokens',
        PYTHONPATH=str(qa.ROOT/'experiment_plugins')+os.pathsep+str(qa.ROOT))


def gates(folder):
    for case in qa.read(folder/'comparison_plan.json')['cases']:
        root=Path(case['root']);plan=check_sources(root);gate=root/'gpu_9gib_check'
        assert qa.early.idle_gpu()
        # Full-serving guard has a fixed high per-target floor. This isolated
        # gate writes only 9 GiB total and deletes it before the next condition.
        pool=qa.base.pool_query()
        nvme=next(t for t in pool['response']['tier_stats'] if t['media_type']=='nvme')
        assert pool['status']==0 and not pool['response']['disabled_targets']
        assert nvme['free']>30*2**30 and nvme['min']>2*2**30
        qa.dump(root/'gate_pool_before.json',pool)
        if not (gate/'result.json').exists():
            command=[str(qa.ROOT/'run_vllm.sh'),sys.executable,str(qa.ROOT/'tests/sync_retrieve_capacity.py'),
                '--capacity','2','--chunks','512','--window-mib','500','--store-window-mib','500',
                '--capacity-pipeline','--split-store-gib','0','--no-dram',
                '--placement-manifest',plan['placement_manifest'],'--output',str(gate)]
            qa.dump(root/'gate_command.json',command)
            with (root/'gpu_9gib_check.log').open('x') as log:
                subprocess.run(command,cwd=qa.ROOT,env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
        result=qa.read(gate/'result.json')
        assert result['status']=='passed' and result['returned_tokens']==65536
        assert result['all_returned_gpu_values_match'] and result['no_dram_validation']['passed']
        assert result['placement_validation']['passed'] and qa.read(gate/'cleanup.json')['remaining']==0
        print(case['label']+' 9GiB placement/store/lookup/scatter/removal gate passed',flush=True)
    qa.dump(folder/'gates.json',dict(passed=True,conditions=2))


def cleanup_owned(root):
    from lmcache_daos.placement_backend import PlacementObjectStore,checked
    plan=qa.read(root/'plan.json');m=qa.read(Path(plan['placement_manifest']))
    assert m['isolated_experiment_object'] and m['nonce']>=1000000
    assert qa.read(root/'status.json')['status']=='completed'
    assert qa.early.idle_gpu()
    store=PlacementObjectStore(plan['placement_manifest'],m['pool'],m['container'])
    ctx=store._ctx;store._ctx=None
    rc=store._lib.placement_close(ctx,1)
    qa.dump(root/'owned_object_cleanup.json',dict(oid=m['oid'],rc=rc,production_object_touched=False))
    checked(rc)


def run(folder):
    with (folder/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert qa.read(folder/'gates.json')['passed']
        for case in qa.read(folder/'comparison_plan.json')['cases']:
            root=Path(case['root']);plan=check_sources(root)
            if qa.read(root/'status.json')['status']=='completed':
                if not (root/'owned_object_cleanup.json').exists():cleanup_process(root)
                continue
            assert qa.early.idle_gpu()
            wait_space(root,plan['capacity']['stored_kv_upper_gib'])
            qa.dump(folder/'status.json',dict(status='running',condition=case['label']))
            command=[sys.executable,str(qa.ROOT/'realqa_q4_cxs.py'),'run','--output',str(root)]
            qa.dump(root/'benchmark_command.json',command)
            with (root/'benchmark.log').open('x') as log:
                subprocess.run(command,cwd=qa.ROOT,env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
            assert qa.read(root/'status.json')['status']=='completed'
            events=qa.base.read_events(root/plan['cases'][0]['name'])
            enabled=[e for e in events if e['event']=='placement_enabled']
            assert len(enabled)==1 and enabled[0]['mode']==case['label']
            cleanup_process(root)
            print(case['label']+' completed 480 CXS requests; isolated object cleaned',flush=True)
        from report_mixed_comparison import report
        report(folder)
        qa.dump(folder/'status.json',dict(status='completed',requests=960))


def cleanup_process(root):
    # DAOS device initialization can retain a CUDA context until process exit.
    # Keep cleanup out of the orchestration process before the next idle check.
    subprocess.run([sys.executable,__file__,'cleanup','--output',str(root)],
                   cwd=qa.ROOT,env=environment(),check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','gates','run','cleanup'])
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    (cleanup_owned if a.action=='cleanup' else globals()[a.action])(a.output.resolve())
