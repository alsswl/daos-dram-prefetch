"""DRAM-off, shared2GiB: compare 500/2048MiB windows on the same mixed QA."""
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

SOURCE=qa.ROOT/'realqa_mixed_shared2g_window2048_20261002'

def prepare(folder):
    assert folder.parent==qa.ROOT and not folder.exists()
    folder.mkdir();cases=[]
    for window in (500,2048):
        root=qa.ROOT/f'{folder.name}_w{window}';root.mkdir()
        plan=qa.read(SOURCE/'plan.json')
        shutil.copytree(SOURCE/'documents',root/'documents')
        for name in ('documents.json','sessions.json','mixed_schedule.json'):
            shutil.copy2(SOURCE/name,root/name)
        plan.update(cpu_gib=0,dram_disabled=True,store_window_mib=window,retrieve_window_mib=window,
                    cases=[dict(name=f'c8_s10_mixed_nodram_w{window}',policy='none')],
                    experiment_namespace='minji-mixed-'+uuid.uuid4().hex+':',source_dataset=str(SOURCE))
        plan['notes']=[
            'Same documents and fixed mixed per-lane schedule, 80 sessions x6 turns, 480 requests.',
            'DRAM KV capacity0: local_cpu false, no store mirror, no read promotion, no DRAM prefetch.',
            'All restored chunks must equal DAOS-returned chunks; CPU hot/ready and copied bytes must stay0.',
            'Shared GPU staging2GiB, utilization .835, context65536, output limit256, concurrency8.',
            f'Store/read window{window}MiB; effective {(window//18)*18}MiB; capacity-derived pipeline depth.',
            'The 18MiB DAOS chunk I/O unit and 16-worker pool remain unchanged.',
            'One run per condition, generated answers and wall-clock arrivals may differ.',
            'Fresh namespaces; after first run ends, only its KV is removed for second-run space.']
        files=set(plan['source_sha256'])|{'realqa_nodram_comparison.py','cleanup_nodram_comparison.py',
            'lmcache_daos/no_dram.py','tests/test_no_dram.py'}
        plan['source_sha256']={}
        for name in sorted(files):
            dst=root/'executed_sources'/name;dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(qa.ROOT/name,dst);plan['source_sha256'][name]=qa.base.digest(dst)
        qa.dump(root/'plan.json',plan);qa.dump(root/'status.json',dict(status='prepared'))
        cases.append(dict(label=f'window{window}',root=str(root),
            description=f'shared2GiB, effective window{(window//18)*18}MiB, depth{2048//((window//18)*18)}'))
    qa.dump(folder/'comparison_plan.json',dict(cases=cases,delta_key='window2048_vs_500_change_pct',
        chart_title='Mixed 80-session QA | DRAM OFF | shared2GiB | windows486MiB vs2034MiB'))
    qa.dump(folder/'status.json',dict(status='prepared',expected_requests=960))

def run(folder):
    with (folder/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        cases=qa.read(folder/'comparison_plan.json')['cases']
        env=dict(os.environ,DAOSGDS_TRANSPORT='object',DAOS_GDS_MULTI_PREFETCH='0',
            DAOSGDR_TIMING='0',HF_HOME='/home/hf/hf_cache',VLLM_PLUGINS='daos_resume_tokens',
            PYTHONPATH=str(qa.ROOT/'experiment_plugins')+os.pathsep+str(qa.ROOT))
        for c in cases:
            root=Path(c['root']);plan=check_sources(root);gate=root/'gpu_9gib_check'
            assert qa.early.idle_gpu()
            wait_space(root,plan['capacity']['stored_kv_upper_gib'])
            if not (gate/'result.json').exists():
                command=[str(qa.ROOT/'run_vllm.sh'),sys.executable,str(qa.ROOT/'tests/sync_retrieve_capacity.py'),
                    '--capacity','2','--chunks','512','--window-mib',str(plan['retrieve_window_mib']),
                    '--store-window-mib',str(plan['store_window_mib']),'--capacity-pipeline',
                    '--split-store-gib','0','--no-dram','--output',str(gate)]
                qa.dump(root/'gate_command.json',command)
                with (root/'gpu_9gib_check.log').open('x') as log:
                    subprocess.run(command,cwd=qa.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
            result=qa.read(gate/'result.json')
            assert result['status']=='passed' and result['returned_tokens']==65536
            assert result['all_returned_gpu_values_match'] and result['no_dram_validation']['passed']
            assert qa.read(gate/'cleanup.json')['remaining']==0
            print(c['label']+' DRAM-off GPU gate passed',flush=True)
        for i,c in enumerate(cases):
            root=Path(c['root']);plan=check_sources(root)
            wait_space(root,plan['capacity']['stored_kv_upper_gib'])
            qa.dump(folder/'status.json',dict(status='running',condition=c['label'],index=i+1,total=2))
            command=[sys.executable,str(qa.ROOT/'realqa_q4_cxs.py'),'run','--output',str(root)]
            qa.dump(root/'benchmark_command.json',dict(command=command,environment={k:env[k] for k in
                ('DAOSGDS_TRANSPORT','DAOS_GDS_MULTI_PREFETCH','DAOSGDR_TIMING','HF_HOME','VLLM_PLUGINS','PYTHONPATH')}))
            with (root/'benchmark.log').open('x') as log:
                subprocess.run(command,cwd=qa.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            assert qa.read(root/'status.json')['status']=='completed'
            assert qa.read(root/plan['cases'][0]['name']/'no_dram_check.json')['passed']
            print(c['label']+' completed480 and DAOS-only validation passed',flush=True)
            if i==0:
                output=root/'comparison_cache_cleanup'
                base=[str(qa.ROOT/'run_vllm.sh'),sys.executable,str(qa.ROOT/'cleanup_nodram_comparison.py'),
                      '--source-root',str(root),'--output',str(output)]
                with (root/'cleanup_process.log').open('x') as log:
                    for extra in ([],['--execute']):
                        subprocess.run(base+extra,cwd=qa.ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                assert qa.read(output/'result.json')['remaining_targets']==0
        from report_mixed_comparison import report
        report(folder)
        for c in cases:check_sources(Path(c['root']))
        assert qa.early.idle_gpu()
        qa.dump(folder/'execution_validation.json',dict(passed=True,requests=960,
            all_restored_chunks_from_daos=True,sources_match_snapshots=True,gpu_idle_after_run=True))
        qa.dump(folder/'status.json',dict(status='completed',conditions=2,requests=960))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run'])
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    try:globals()[a.action](a.output.resolve())
    except BaseException as exc:
        if a.output.exists():qa.dump(a.output/'status.json',dict(status='failed',error=repr(exc)))
        raise
