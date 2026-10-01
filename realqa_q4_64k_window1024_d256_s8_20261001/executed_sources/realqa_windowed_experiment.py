#!/usr/bin/env python3
"""64K CxS OFF, adjustable window: 9GiB GPU correctness test before full QA.

No waiting/kill of other GPU jobs; run fails safely if the GPU is occupied.
Prepared inputs: 60K document + conversation <=64K, 80 sessions x 6 turns, C8.
Example: ... prepare --window-mib 1024 --output /root/discos_minji/EXPERIMENT
         ... run --output /root/discos_minji/EXPERIMENT
"""
import argparse
import fcntl
from pathlib import Path
import os
import shutil
import subprocess
import sys
import time

import realqa_q4_cxs as qa

SOURCE=qa.ROOT/'realqa_q4_64k_cxs_d256_s8_none_20260930'


def prepare(root,window_mib):
    assert root.parent==qa.ROOT and not root.exists()
    assert 18<=window_mib<=8192
    plan=qa.read(SOURCE/'plan.json')
    assert plan['max_model_len']==65536 and plan['document_tokens']==61440
    # Backend/transport must still match the prepared data's implementation.
    for name,digest in plan['source_sha256'].items():
        if name not in ('realqa_q4_cxs.py','tests/test_realqa_q4_cxs.py','tests/sync_retrieve_capacity.py'):
            assert qa.base.digest(qa.ROOT/name)==digest, f'Unexpected source change: {name}'
    root.mkdir()
    shutil.copytree(SOURCE/'documents',root/'documents')
    for name in ('documents.json','sessions.json'):
        shutil.copy2(SOURCE/name,root/name)
    plan.update(retrieve_window_mib=window_mib,source_dataset=str(SOURCE),
        cases=[dict(name='c8_s10_window'+str(window_mib),policy='none')])
    plan['notes']=[n for n in plan['notes'] if 'async metadata-first lookup' not in n]
    plan['notes'] += ['SYNCHRONOUS metadata lookup and bounded sequential read/copy/release; both prefetches OFF.',
        'Model KV capacity may limit admitted 60K requests below C8. C8 is client concurrency, not guaranteed GPU residency.',
        '1GiB window rounds down to 1008MiB (56 full Qwen4B BF16 chunks).',
        '9GiB patterned-GPU correctness gate is not model inference. Full QA runs only after it passes.',
        'Read promotion enabled in inference. Window buffers retained by D2H mirror are drained before the next window.',
        'Lookup control path differs from earlier async-OFF run: not a one-variable performance comparison.']
    files=set(plan['source_sha256']) | {'realqa_windowed_experiment.py','prepare_windowed_config.py',
        'lmcache_config_daosgds_windowed.yaml','lmcache_daos/windowed_transfer.py',
        'lmcache_daos/windowed_demand_backend.py','tests/test_windowed_transfer.py',
        'tests/sync_retrieve_capacity.py'}
    plan['source_sha256']={}
    for name in sorted(files):
        dst=root/'executed_sources'/name
        dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(qa.ROOT/name,dst)
        plan['source_sha256'][name]=qa.base.digest(dst)
    plan['input_sha256']={name:qa.base.digest(root/name) for name in ('documents.json','sessions.json')}
    qa.dump(root/'plan.json',plan)
    qa.dump(root/'status.json',dict(status='prepared',window_mib=window_mib,expected_requests=480))


def run(root):
    with (root/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=qa.read(root/'plan.json')
        for name,digest in plan['source_sha256'].items():
            assert qa.base.digest(qa.ROOT/name)==digest,f'Implementation changed: {name}'
        if not qa.early.idle_gpu():
            qa.dump(root/'status.json',dict(status='blocked_gpu',reason='Another GPU process is active; no workload started',updated_ns=time.time_ns()))
            return
        gate=root/'gpu_9gib_check'
        try:
            if not gate.exists():
                qa.base.storage_guard(root,10)
                qa.dump(root/'status.json',dict(status='gpu_correctness_check',updated_ns=time.time_ns()))
                command=[str(qa.ROOT/'run_vllm.sh'),sys.executable,
                    str(qa.ROOT/'tests/sync_retrieve_capacity.py'),'--capacity','8','--chunks','512',
                    '--window-mib',str(plan['retrieve_window_mib']),'--output',str(gate)]
                qa.dump(root/'gate_command.json',command)
                with (root/'gpu_9gib_check.log').open('x') as log:
                    subprocess.run(command,cwd=qa.ROOT,env=dict(os.environ,DAOSGDS_TRANSPORT='object',DAOS_GDS_MULTI_PREFETCH='0'),
                        stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
            result=qa.read(gate/'result.json')
            assert result['status']=='passed' and result['returned_tokens']==65536
            assert result['all_returned_gpu_values_match'] and result['capacity_failed_chunks']==0
            assert qa.read(gate/'cleanup.json')['remaining']==0
            # Fail before creating a benchmark case if a complete run cannot fit.
            qa.base.storage_guard(root,plan['capacity']['stored_kv_upper_gib'])
            qa.run(root)
        except BaseException as exc:
            qa.dump(root/'status.json',dict(status='failed',error=repr(exc),updated_ns=time.time_ns()))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--window-mib',type=int,default=1024)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.output.resolve(),a.window_mib)
    else:run(a.output.resolve())
