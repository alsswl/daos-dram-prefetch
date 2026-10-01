"""Same deterministic mixed-session workload, 2GiB shared vs 1:3 split."""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid
import realqa_q4_cxs as qa
from mixed_qa_scheduler import make_schedule

SOURCE=qa.ROOT/'realqa_q4_64k_split_r1536_s512_w500_20261001'

def prepare(folder):
    assert folder.parent==qa.ROOT and not folder.exists()
    folder.mkdir();cases=[];schedule=make_schedule()
    for label,split in [('shared',0),('split',.5)]:
        root=qa.ROOT/f'{folder.name}_{label}'
        assert not root.exists();root.mkdir()
        plan=qa.read(SOURCE/'plan.json')
        shutil.copytree(SOURCE/'documents',root/'documents')
        for name in ('documents.json','sessions.json'):shutil.copy2(SOURCE/name,root/name)
        qa.dump(root/'mixed_schedule.json',schedule)
        plan.update(staging_gib=2,split_store_gib=split,store_window_mib=500,retrieve_window_mib=500,
                    gpu_memory_utilization=.835,capacity_pipeline=True,mixed_schedule=True,
                    experiment_namespace='minji-mixed-'+uuid.uuid4().hex+':',
                    source_dataset=str(SOURCE),cases=[dict(name='c8_s10_mixed_'+label,policy='none')])
        plan['notes']=[
            'Same 80 seeded document assignments, six real generated turns/session, max input+output 65536.',
            'Eight independent lanes. Each follows an immutable mixed schedule of 60 requests.',
            'New sessions start at lane positions 0,1,2,9,16,23,30,37,44,51; other positions are seeded follow-ups.',
            'No global turn barrier. One outstanding request/lane and one/session.',
            'Same 2GiB staging, .835 GPU utilization, DRAM256GiB and 500MiB read/store windows in both cases.',
            'Both use capacity-based admission and concurrent read-ahead; physical arena split is the condition difference.',
            'Lookup-time prefetch OFF. Native vLLM prefix cache OFF. Existing DRAM mirror/read promotion ON.',
            'Store .5GiB/read1.5GiB physical split.' if split else 'Shared2GiB arena, no store/read partition.',
            'Actual generated histories and wall-clock arrival times can differ between runs.',
            'One trial per condition. First generated comparison cache removed only after successful report and server shutdown.']
        files=set(plan['source_sha256'])|{'mixed_qa_scheduler.py','realqa_mixed_comparison.py',
            'lmcache_daos/capacity_pipeline.py','lmcache_daos/capacity_pipeline_backend.py',
            'lmcache_daos/capacity_validation.py','tests/test_capacity_mixed.py'}
        plan['source_sha256']={}
        for name in sorted(files):
            dst=root/'executed_sources'/name;dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(qa.ROOT/name,dst);plan['source_sha256'][name]=qa.base.digest(dst)
        plan['input_sha256']={name:qa.base.digest(root/name) for name in ('documents.json','sessions.json','mixed_schedule.json')}
        qa.dump(root/'plan.json',plan);qa.dump(root/'status.json',dict(status='prepared',expected_requests=480))
        cases.append(dict(label=label,root=str(root),namespace=plan['experiment_namespace']))
    qa.dump(folder/'comparison_plan.json',dict(cases=cases,expected_requests=960,schedule=schedule))
    qa.dump(folder/'status.json',dict(status='prepared',cases=cases))

def check_sources(root):
    plan=qa.read(root/'plan.json')
    for name,digest in plan['source_sha256'].items():
        assert qa.base.digest(qa.ROOT/name)==digest,f'Implementation changed: {name}'
    return plan

def wait_space(root,required):
    for attempt in range(60):
        try:return qa.base.storage_guard(root,required)
        except RuntimeError as exc:
            print(f'Waiting for reclaimed DAOS space ({attempt+1}/60): {exc}',flush=True)
            time.sleep(10)
    raise RuntimeError('Insufficient space after reclamation wait')

def gates(folder):
    for c in qa.read(folder/'comparison_plan.json')['cases']:
        root=Path(c['root']);plan=check_sources(root);gate=root/'gpu_9gib_check'
        assert qa.early.idle_gpu()
        if (gate/'result.json').exists():
            result=qa.read(gate/'result.json')
            assert result['status']=='passed' and qa.read(gate/'cleanup.json')['remaining']==0
            continue
        wait_space(root,10)
        command=[str(qa.ROOT/'run_vllm.sh'),sys.executable,str(qa.ROOT/'tests/sync_retrieve_capacity.py'),
            '--capacity','2','--chunks','512','--window-mib','500','--store-window-mib','500',
            '--capacity-pipeline','--split-store-gib',str(plan['split_store_gib']),'--output',str(gate)]
        qa.dump(root/'gate_command.json',command)
        with (root/'gpu_9gib_check.log').open('x') as log:
            subprocess.run(command,cwd=qa.ROOT,env=dict(os.environ,DAOSGDS_TRANSPORT='object',DAOS_GDS_MULTI_PREFETCH='0'),
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
        result=qa.read(gate/'result.json')
        assert result['status']=='passed' and result['returned_tokens']==65536
        assert result['all_returned_gpu_values_match'] and result['capacity_validation']['passed']
        assert qa.read(gate/'cleanup.json')['remaining']==0
        print(c['label']+' GPU gate passed',flush=True)

def cleanup_generated(root):
    import cleanup_experiment_cache as cleaner
    plan=qa.read(root/'plan.json');case=root/plan['cases'][0]['name'];ns=plan['experiment_namespace']
    assert root.parent==qa.ROOT and ns.startswith('minji-mixed-') and ns.endswith(':')
    assert qa.read(root/'status.json')['status']=='completed'
    assert qa.read(case/'capacity_policy_check.json')['passed']
    rows=[dict(namespace=ns,sources=[str((case/'config.yaml').relative_to(qa.ROOT))],
        pool='discospool',container='kvcache',experiments=[root.name])]
    def classify(keys,selected):
        assert selected==rows
        targets=[k for k in keys if k.startswith(ns)]
        assert all(k[len(ns):].startswith('Qwen/Qwen3-4B-Instruct-2507@') for k in targets)
        return targets,dict(Counter({ns:len(targets)}))
    cleaner.eligible=lambda:rows;cleaner.classify=classify
    for execute in (False,True):
        sys.argv=['cleanup','--output',str(root/'comparison_cache_cleanup')]+(['--execute'] if execute else [])
        cleaner.main()

def run(folder):
    with (folder/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        gates(folder)
        cases=qa.read(folder/'comparison_plan.json')['cases']
        for i,c in enumerate(cases):
            root=Path(c['root']);plan=check_sources(root)
            wait_space(root,plan['capacity']['stored_kv_upper_gib'])
            qa.dump(folder/'status.json',dict(status='running',condition=c['label'],index=i+1,total=2))
            # Separate runner processes isolate the monkey-patched Python classes.
            command=[sys.executable,str(qa.ROOT/'realqa_q4_cxs.py'),'run','--output',str(root)]
            with (root/'benchmark.log').open('x') as log:
                subprocess.run(command,cwd=qa.ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            assert qa.read(root/'status.json')['status']=='completed'
            if i==0:
                # DAOS cleanup needs its CLI environment and can create a CUDA
                # context. Exit that process before checking the next GPU run.
                command=[str(qa.ROOT/'run_vllm.sh'),sys.executable,
                         str(qa.ROOT/'realqa_mixed_comparison.py'),
                         'cleanup','--output',str(root)]
                with (root/'cleanup_process.log').open('x') as log:
                    subprocess.run(command,cwd=qa.ROOT,stdout=log,
                                   stderr=subprocess.STDOUT,check=True)
        qa.dump(folder/'status.json',dict(status='completed',conditions=2,requests=960))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','gates','run','cleanup']);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:
        action=cleanup_generated if a.action=='cleanup' else globals()[a.action]
        action(a.output.resolve())
    except BaseException as exc:
        if a.output.exists() and a.action!='cleanup':qa.dump(a.output/'status.json',dict(status='failed',error=repr(exc)))
        raise
