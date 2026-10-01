#!/usr/bin/env python3
"""Same completed C8 x S10 inputs/workload, experimental no-staging DAOS backend.

The workload is identical; DRAM tier and asynchronous store are NOT identical.
Hardware gate precedes server start. Existing workloads/results are untouched.
"""
import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

import yaml
import realqa_q4_cxs as qa

BASELINE = qa.ROOT/'realqa_q4_64k_window1024_d256_s2_u0835_resume_fix_20261001'


def prepare(root):
    assert root.parent == qa.ROOT and not root.exists()
    old=qa.read(BASELINE/'plan.json')
    assert qa.read(BASELINE/'status.json')['status']=='completed'
    root.mkdir()
    shutil.copytree(BASELINE/'documents',root/'documents')
    for name in ('documents.json','sessions.json'):
        shutil.copy2(BASELINE/name,root/name)
        assert qa.base.digest(root/name)==old['input_sha256'][name]
    plan=dict(old)
    plan.pop('retrieve_window_mib',None)
    plan.update(baseline=str(BASELINE),cpu_gib=0,staging_gib=0,
                cases=[dict(name='c8_s10_direct_scatter',policy='none')],
                source_sha256={},direct_scatter=True,
                comparability=dict(same_workload=True,single_variable_comparison=False,
                    changed=['No DRAM tier/promotion (baseline 256GiB)',
                             'Synchronous source-page stores (baseline asynchronous staged stores)',
                             'Byte ARRAY with IOM coverage validation (baseline SINGLE)',
                             'Zero staging; direct destination SGL (baseline 1GiB windows, 2GiB pool)']),
                notes=['Identical copied documents/session assignments and upstream C8 x S10 x six-turn workload.',
                       'Same model, 64K context, 256 output cap, seed, no vLLM prefix cache, GPU utilization 0.835.',
                       'Fresh private DAOS namespace; no workload warm-up. Hardware gate uses unrelated keys.',
                       'CPU key/descriptor metadata permitted; no CPU or GPU payload staging allocator.',
                       'Actual generated histories/arrival times may differ. Not a one-variable comparison.'])
    cfg=yaml.safe_load((qa.ROOT/'lmcache_config_daosgds_direct_scatter.yaml').read_text())
    cfg['extra_config'].update({'daosgds.object_namespace':'minji-cxs-direct-'+uuid.uuid4().hex+':',
                               'daosgds.io_workers':16,'daosgds.meta_workers':16})
    (root/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    files=set(old['source_sha256'])|{
        'realqa_direct_scatter_experiment.py','libdaosgdr_scatter.c','libdaosgdr_scatter.so',
        'lmcache_daos/direct_scatter_backend.py','lmcache_daos/direct_scatter_launch.py',
        'lmcache_daos/scatter_plan.py','lmcache_daos/scatter_binding.py',
        'experiment_plugins/direct_scatter/sitecustomize.py','tests/direct_scatter_roundtrip.py'}
    for name in sorted(files):
        dest=root/'executed_sources'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(qa.ROOT/name,dest)
        plan['source_sha256'][name]=qa.base.digest(dest)
    qa.dump(root/'plan.json',plan)
    qa.dump(root/'status.json',dict(status='prepared',expected_requests=480))


def report(root):
    plan=qa.read(root/'plan.json');case=root/plan['cases'][0]['name']
    calls=qa.read(case/'calls.json');window=qa.read(case/'workload.json')
    rows=[]
    for label,cc in [('all',calls)]+[(str(i),[c for c in calls if c['turn']==i]) for i in range(6)]:
        inp=sum(c['prompt_tokens'] for c in cc);cached=sum(c['cached_tokens'] for c in cc)
        rows.append(dict(turn=label,requests=len(cc),input_tokens=inp,cached_tokens=cached,
            cached_token_pct=100*cached/inp,ttft_http_ms=qa.stats([c['ttft_http_ms'] for c in cc]),
            ttft_upstream_ms=qa.stats([c['ttft']*1000 for c in cc]),
            completion_tokens=sum(c['completion_tokens'] for c in cc)))
    log=(case/'server.log').read_text()
    operations=re.findall(r'Direct scatter (store|fetch): tokens=(\d+) bytes=(\d+) cost_ms=([\d.]+) staging_bytes=0',log)
    op_stats={kind:dict(operations=len([v for v in operations if v[0]==kind]),
                       bytes=sum(int(v[2]) for v in operations if v[0]==kind),
                       cost_ms=qa.stats([float(v[3]) for v in operations if v[0]==kind]))
              for kind in ('store','fetch')}
    assert op_stats['store']['operations'] and op_stats['fetch']['operations'], 'Direct path not exercised'
    fixes=[json.loads(m) for m in re.findall(r'DAOS_RESUME_TOKEN_FIX (\{[^\n]+\})',log)]
    assert any(e['event']=='installed' for e in fixes), 'Resume patch missing'
    restored=[e for e in fixes if e['event']=='resume_full_history']
    assert all(e['tracker_after']>=e['expected_cache_tokens'] for e in restored)
    summary=dict(duration_s=(window['end_ns']-window['start_ns'])/1e9,by_turn=rows,
                 direct_scatter=op_stats,staging_pool_reserved_gib=0,
                 staging_measurement='NoPayloadAllocator + connector buffer disabled; not NVML peak',
                 gpu_kv_capacity_log=re.findall(r'[^\n]*(?:Available KV cache memory|GPU KV cache size)[^\n]*',log),
                 comparability=plan['comparability'])
    qa.dump(root/'summary.json',summary)
    qa.dump(case/'resume_token_fix_check.json',dict(passed=True,restored_requests=len(restored)))
    baseline=qa.read(BASELINE/'summary.json')
    qa.dump(root/'comparison.json',dict(
        baseline=str(BASELINE),single_variable_comparison=False,
        ttft_change_pct=100*(summary['by_turn'][0]['ttft_http_ms']['mean']/baseline['by_turn'][0]['ttft_http_ms']['mean']-1),
        duration_change_pct=100*(summary['duration_s']/baseline['duration_s']-1),
        baseline_ttft_ms=baseline['by_turn'][0]['ttft_http_ms']['mean'],
        direct_ttft_ms=summary['by_turn'][0]['ttft_http_ms']['mean'],
        baseline_duration_s=baseline['duration_s'],direct_duration_s=summary['duration_s'],
        changes=plan['comparability']['changed']))


class Health(qa.base.LogHealth):
    def __call__(self):
        super().__call__()
        for pattern in ('Direct scatter store failed','Direct scatter fetch failed',
                        'refusing fallback','Payload staging is forbidden','Traceback (most recent call last)'):
            if pattern in self.tail:
                raise RuntimeError('Direct-scatter server error: '+pattern)


def run(root):
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=qa.read(root/'plan.json')
        for name,digest in plan['source_sha256'].items():
            assert qa.base.digest(qa.ROOT/name)==digest, 'Source changed: '+name
        for name,digest in plan['input_sha256'].items():
            assert qa.base.digest(root/name)==digest
        assert qa.early.idle_gpu(), 'GPU busy; no other job will be stopped'
        case=root/plan['cases'][0]['name']
        assert not case.exists(), 'Never overwrite a partial run'
        try:
            qa.base.storage_guard(root,plan['capacity']['stored_kv_upper_gib'])
            qa.dump(root/'status.json',dict(status='gpu_correctness_gate',updated_ns=time.time_ns()))
            command=[str(qa.ROOT/'run_vllm.sh'),sys.executable,
                     str(qa.ROOT/'tests/direct_scatter_roundtrip.py'),
                     '--pool','discospool','--container','kvcache','--iterations','5']
            qa.dump(root/'gate_command.json',command)
            with (root/'gpu_gate.log').open('x') as log:
                subprocess.run(command,cwd=qa.ROOT,env=dict(os.environ,DAOSGDS_TRANSPORT='object',DAOSGDR_TIMING='0'),
                               stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
            assert '"status": "PASS"' in (root/'gpu_gate.log').read_text()
            case.mkdir()
            shutil.copy2(root/'config.yaml',case/'config.yaml')
            os.environ.update(DAOS_DIRECT_SCATTER_BOOTSTRAP='1',DAOS_VLLM_RESUME_TOKEN_FIX='1',
                              DAOSGDR_TIMING='0',DAOS_LOOKUP_READY_RETURN='0',DAOS_LOOKUP_LOCK_PROBE='0')
            os.environ['PYTHONPATH']=os.pathsep.join([str(qa.ROOT/'experiment_plugins/direct_scatter'),
                str(qa.ROOT/'experiment_plugins'),str(qa.ROOT),os.environ.get('PYTHONPATH','')])
            os.environ['VLLM_PLUGINS']='daos_resume_tokens'
            qa.dump(root/'status.json',dict(status='starting',updated_ns=time.time_ns()))
            args=qa.server_args(plan)
            with qa.base.server(args,case/'config.yaml',case) as client:
                assert 'DirectScatterBackend enabled: GPU staging=0' in (case/'server.log').read_text()
                health=Health(case/'server.log');health()
                window=dict(start_ns=time.time_ns());qa.dump(case/'workload.json',window)
                asyncio.run(qa.workload(root,case,health))
                window['end_ns']=time.time_ns();qa.dump(case/'workload.json',window)
                health()
            report(root)
            qa.dump(root/'status.json',dict(status='completed',requests=480,updated_ns=time.time_ns(),
                    cache_cleanup='Only gate keys removed; workload namespace retained'))
        except BaseException as e:
            qa.dump(root/'status.json',dict(status='failed',error=repr(e),updated_ns=time.time_ns()))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','run','report'])
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    globals()[a.action](a.output.resolve())
