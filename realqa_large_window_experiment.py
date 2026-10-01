"""Same mixed QA inputs with shared 2GiB staging and near-capacity windows."""
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

BASELINE=qa.ROOT/'realqa_mixed_capacity2g_20261001_shared'

def prepare(root):
    assert root.parent==qa.ROOT and not root.exists()
    root.mkdir()
    plan=qa.read(BASELINE/'plan.json')
    shutil.copytree(BASELINE/'documents',root/'documents')
    for name in ('documents.json','sessions.json','mixed_schedule.json'):
        shutil.copy2(BASELINE/name,root/name)
    plan.update(retrieve_window_mib=2048,store_window_mib=2048,
                experiment_namespace='minji-mixed-'+uuid.uuid4().hex+':',
                cases=[dict(name='c8_s10_mixed_window2048',policy='none')],
                source_dataset=str(BASELINE))
    plan['notes']=[
        'Same documents, session assignments, and fixed mixed lane schedule as shared 500MiB baseline.',
        'Shared staging2GiB, store/retrieve window2048MiB -> 113 full chunks = 2034MiB.',
        'One full retrieve window fits; depth1 is allowed with unchanged capacity reservation and release.',
        'Same GPU utilization .835, CPU cache256GiB, model, 480 requests and context cap65536.',
        'Lookup prefetch OFF, native prefix cache OFF, DRAM mirror/read promotion retained.',
        'Store calls are bounded by produced KV; window size is an upper limit, not a forced batch size.',
        'The DAOS chunk size remains18MiB: larger windows do not merge per-chunk DAOS fetch calls.',
        'One run per condition; generated answers and actual arrival times can differ.']
    files=set(plan['source_sha256'])|{'realqa_large_window_experiment.py','report_mixed_comparison.py'}
    plan['source_sha256']={}
    for name in sorted(files):
        dst=root/'executed_sources'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(qa.ROOT/name,dst);plan['source_sha256'][name]=qa.base.digest(dst)
    qa.dump(root/'plan.json',plan)
    qa.dump(root/'status.json',dict(status='prepared',expected_requests=480))
    comp=root/'comparison_500_vs_2048';comp.mkdir()
    qa.dump(comp/'comparison_plan.json',dict(cases=[
        dict(label='window500',root=str(BASELINE),description='shared2GiB, effective window486MiB, depth4'),
        dict(label='window2048',root=str(root),description='shared2GiB, effective window2034MiB, depth1')],
        delta_key='window2048_vs_500_change_pct',
        chart_title='Mixed 80-session QA | shared2GiB | effective windows486MiB vs2034MiB'))

def run(root):
    with (root/'large_window_runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=check_sources(root)
        assert qa.early.idle_gpu()
        wait_space(root,plan['capacity']['stored_kv_upper_gib'])
        gate=root/'gpu_9gib_check'
        qa.dump(root/'status.json',dict(status='validating_gpu'))
        command=[str(qa.ROOT/'run_vllm.sh'),sys.executable,str(qa.ROOT/'tests/sync_retrieve_capacity.py'),
                 '--capacity','2','--chunks','512','--window-mib','2048','--store-window-mib','2048',
                 '--capacity-pipeline','--split-store-gib','0','--output',str(gate)]
        env=dict(os.environ,DAOSGDS_TRANSPORT='object',DAOS_GDS_MULTI_PREFETCH='0',
                 DAOSGDR_TIMING='0',HF_HOME='/home/hf/hf_cache',VLLM_PLUGINS='daos_resume_tokens',
                 PYTHONPATH=str(qa.ROOT/'experiment_plugins')+os.pathsep+str(qa.ROOT))
        qa.dump(root/'gate_command.json',command)
        if not (gate/'result.json').exists():
            with (root/'gpu_9gib_check.log').open('x') as log:
                subprocess.run(command,cwd=qa.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
        result=qa.read(gate/'result.json')
        assert result['status']=='passed' and result['returned_tokens']==65536
        assert result['all_returned_gpu_values_match'] and result['capacity_validation']['passed']
        assert qa.read(gate/'cleanup.json')['remaining']==0
        print('9GiB GPU correctness gate passed',flush=True)
        # Logical deletion finishes before DAOS has reclaimed the test payload.
        wait_space(root,plan['capacity']['stored_kv_upper_gib'])
        command=[sys.executable,str(qa.ROOT/'realqa_q4_cxs.py'),'run','--output',str(root)]
        qa.dump(root/'benchmark_command.json',dict(command=command,environment={k:env[k] for k in
            ('DAOSGDS_TRANSPORT','DAOS_GDS_MULTI_PREFETCH','DAOSGDR_TIMING','HF_HOME','VLLM_PLUGINS','PYTHONPATH')}))
        with (root/'benchmark.log').open('x') as log:
            subprocess.run(command,cwd=qa.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert qa.read(root/'status.json')['status']=='completed'
        from report_mixed_comparison import report
        report(root/'comparison_500_vs_2048')
        check_sources(root)
        assert qa.early.idle_gpu()
        qa.dump(root/'execution_validation.json',dict(passed=True,requests=480,
            sources_match_snapshot=True,gpu_idle_after_run=True,gpu_correctness_gate=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run'])
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    globals()[a.action](a.output.resolve())
