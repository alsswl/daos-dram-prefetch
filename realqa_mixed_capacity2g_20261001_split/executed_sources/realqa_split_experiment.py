"""Dedicated split-arena experiment; immutable inputs and executed source snapshots."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import realqa_q4_cxs as qa

SOURCE=qa.ROOT/'realqa_q4_64k_read500_store500_d256_s1_20261001'

def prepare(root):
    assert root.parent==qa.ROOT and not root.exists()
    plan=qa.read(SOURCE/'plan.json')
    assert qa.read(SOURCE/'status.json')['status']=='completed'
    root.mkdir()
    shutil.copytree(SOURCE/'documents',root/'documents')
    for name in ('documents.json','sessions.json'):
        shutil.copy2(SOURCE/name,root/name)
    plan.update(staging_gib=2,split_store_gib=.5,store_window_mib=500,
                retrieve_window_mib=500,gpu_memory_utilization=.835,
                source_dataset=str(SOURCE),cases=[dict(name='c8_s10_split_r1536_s512_w500',policy='none')])
    plan['notes']=[n for n in plan['notes'] if not any(x in n for x in
        ('gpu_memory_utilization=', 'staging=1GiB', 'SYNCHRONOUS metadata lookup', 'Stores also use'))]
    plan['notes'] += [
        'Physical 2GiB arena: store 0.5GiB, retrieve 1.5GiB; neither may borrow.',
        'Store and retrieve admission locks separated. Store DAOS puts remain asynchronous.',
        '500MiB windows = 27 x 18MiB =486MiB. Up to three retrieve windows, one full store window.',
        'Demand-time intra-request parallel read-ahead; lookup-time prefetch remains OFF.',
        'vLLM gpu_memory_utilization=0.835 (same model KV budget as prior 2GiB shared baseline).']
    files=set(plan['source_sha256']) | {'realqa_split_experiment.py',
        'lmcache_daos/split_pipeline.py','lmcache_daos/split_staging_backend.py',
        'lmcache_daos/split_validation.py','tests/test_split_pipeline.py'}
    plan['source_sha256']={}
    for name in sorted(files):
        dst=root/'executed_sources'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(qa.ROOT/name,dst)
        plan['source_sha256'][name]=qa.base.digest(dst)
    plan['input_sha256']={name:qa.base.digest(root/name) for name in ('documents.json','sessions.json')}
    qa.dump(root/'plan.json',plan)
    qa.dump(root/'status.json',dict(status='prepared',expected_requests=480))

def run(root):
    with (root/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=qa.read(root/'plan.json')
        for name,digest in plan['source_sha256'].items():
            assert qa.base.digest(qa.ROOT/name)==digest, f'Implementation changed: {name}'
        assert qa.early.idle_gpu(), 'GPU occupied; no other jobs stopped'
        gate=root/'gpu_9gib_check'
        qa.base.storage_guard(root,10)
        qa.dump(root/'status.json',dict(status='gpu_correctness_check',updated_ns=time.time_ns()))
        command=[str(qa.ROOT/'run_vllm.sh'),sys.executable,str(qa.ROOT/'tests/sync_retrieve_capacity.py'),
                 '--capacity','2','--chunks','512','--window-mib','500','--store-window-mib','500',
                 '--split-store-gib','0.5','--output',str(gate)]
        qa.dump(root/'gate_command.json',command)
        with (root/'gpu_9gib_check.log').open('x') as log:
            subprocess.run(command,cwd=qa.ROOT,env=dict(os.environ,DAOSGDS_TRANSPORT='object',DAOS_GDS_MULTI_PREFETCH='0'),
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
        result=qa.read(gate/'result.json')
        assert result['status']=='passed' and result['all_returned_gpu_values_match']
        assert result['returned_tokens']==65536 and result['capacity_failed_chunks']==0
        assert result['split_validation']['passed']
        assert qa.read(gate/'cleanup.json')['remaining']==0
        qa.base.storage_guard(root,plan['capacity']['stored_kv_upper_gib'])
        qa.run(root)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run']);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.output.resolve())
    else:
        try:run(a.output.resolve())
        except BaseException as exc:
            qa.dump(a.output/'status.json',dict(status='failed',error=repr(exc),updated_ns=time.time_ns()))
            raise
