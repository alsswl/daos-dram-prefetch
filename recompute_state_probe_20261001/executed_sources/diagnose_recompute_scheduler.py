"""Small read-only KV replay to isolate async scheduling from capacity recovery.

Reuse eight failed first-turn inputs and the failed run's retained namespace.
Fresh process/DRAM per arm; suppress stores in both arms. Not a performance run.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time

import yaml
import realqa_q4_cxs as qa
from recompute_experiment_support import ANSI, RecomputeHealth

SOURCE=qa.ROOT/'realqa_q4_64k_unwindowed_d256_s2_u0835_recompute_20261001'


def requests():
    case=SOURCE/'c8_s10_window0'
    calls=[json.loads(l) for l in (case/'calls.jsonl').read_text().splitlines()]
    failed=sorted((c for c in calls if c['status']!='success'),key=lambda c:c['http_start_ns'])
    assert len(failed)==8 and all(c['turn']==0 for c in failed)
    template=qa.upstream().FIRST_PROMPT
    rows=[]
    for c in failed:
        messages=[dict(role='user',content=template.format((SOURCE/'documents'/c['book']).read_text()))]
        digest=hashlib.sha256(json.dumps(messages,ensure_ascii=False).encode()).hexdigest()
        assert digest==c['messages_sha256']
        rows.append(dict(index=c['index'],book=c['book'],messages=messages,
                         sha256=digest,source_request_id=c.get('server_request_id')))
    return rows


def invoke(client,row,barrier):
    result=dict(index=row['index'],sha256=row['sha256'],start_ns=time.time_ns())
    texts=[]
    try:
        barrier.wait(timeout=30)
        with client.stream('POST','/v1/chat/completions',json=dict(
                model='comparison-model',messages=row['messages'],temperature=0,max_tokens=256,
                stream=True,stream_options={'include_usage':True},
                kv_transfer_params={'lmcache.skip_save':'true'})) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith('data: '):continue
                raw=line[6:]
                if raw=='[DONE]':break
                event=json.loads(raw)
                if event.get('error'):raise RuntimeError(str(event['error']))
                result['request_id']=event.get('id') or result.get('request_id')
                if event.get('usage'):result['usage']=event['usage']
                for c in event.get('choices',[]):
                    if c.get('finish_reason'):result['finish_reason']=c['finish_reason']
                    content=c.get('delta',{}).get('content')
                    if content:
                        result.setdefault('first_token_ns',time.time_ns())
                        texts.append(content)
        assert result.get('usage') and result.get('finish_reason') in ('stop','length'), 'Incomplete/error stream'
        result['status']='success'
    except Exception as exc:
        result.update(status='failed',error=repr(exc))
    result.update(end_ns=time.time_ns(),output=''.join(texts))
    return result


def run(root,probe=False):
    assert root.parent==qa.ROOT and not root.exists()
    assert qa.early.idle_gpu(),'GPU occupied; do not stop others'
    root.mkdir()
    rows=requests();qa.dump(root/'requests.json',rows)
    plan=qa.read(SOURCE/'plan.json')
    cfg=yaml.safe_load((SOURCE/'c8_s10_window0/config.yaml').read_text())
    assert cfg['extra_config']['daosgds.object_namespace']=='minji-windowed-b75ac64b95c24300a94a771fef30132e:'
    assert cfg['extra_config']['daosgds.retrieve_window_mib']==0
    cfg['extra_config']['daosgds.store']=False
    os.environ.update(DAOS_VLLM_RESUME_TOKEN_FIX='1',DAOS_GDS_PREFETCH_TIMING='1',
        DAOS_LOOKUP_READY_RETURN='0',DAOS_LOOKUP_LOCK_PROBE='0',VLLM_USE_V2_MODEL_RUNNER='1')
    os.environ['PYTHONPATH']=str(qa.ROOT/'experiment_plugins')+os.pathsep+os.environ.get('PYTHONPATH','')
    if probe:
        os.environ.update(DAOS_RECOMPUTE_STATE_PROBE='1',CUDA_LAUNCH_BLOCKING='1')
    plugins=os.environ.get('VLLM_PLUGINS')
    if plugins is not None:os.environ['VLLM_PLUGINS']=','.join(dict.fromkeys(plugins.split(',')+['daos_resume_tokens','daos_recompute_probe']))
    sources={}
    for name in ('diagnose_recompute_scheduler.py','compare_e2e.py','recompute_experiment_support.py',
                 'lmcache_daos/demand_read_backend.py','lmcache_daos/vllm_resume_patch.py',
                 'lmcache_daos/recompute_state_probe.py'):
        dest=root/'executed_sources'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(qa.ROOT/name,dest);sources[name]=qa.base.digest(dest)
    qa.dump(root/'plan.json',dict(source=str(SOURCE),source_sha256=sources,
        arms=['sync'] if probe else ['sync','async'],state_probe=probe,
        requests_per_arm=8,skip_store=True,force_same_v2_runner=True,
        note='Read-only fault diagnosis; retained DAOS cache, fresh DRAM/staging each arm. Not full CxS performance.'))
    summaries=[]
    for enabled in ((False,) if probe else (False,True)):
        assert qa.early.idle_gpu(),'GPU occupied before next diagnostic arm'
        name='async' if enabled else 'sync';case=root/name;case.mkdir()
        qa.dump(root/'status.json',dict(status='running',arm=name,updated_ns=time.time_ns()))
        (case/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
        args=qa.server_args(plan);args.async_scheduling=enabled
        error=None;calls=[]
        try:
            with qa.base.server(args,case/'config.yaml',case) as client:
                qa.dump(case/'initial_sample.json',qa.base.await_empty(case))
                barrier=threading.Barrier(8)
                with ThreadPoolExecutor(max_workers=8) as workers:
                    futures=[workers.submit(invoke,client,r,barrier) for r in rows]
                    for future in futures:
                        calls.append(future.result());qa.dump(case/'calls.json',calls)
                if all(c['status']=='success' for c in calls):
                    qa.dump(case/'final_sample.json',qa.base.drain(case,RecomputeHealth(case/'server.log'),timeout=120))
        except Exception as exc:error=repr(exc)
        log=ANSI.sub('',(case/'server.log').read_text())
        events=qa.base.read_events(case)
        outcomes=[e for e in events if e['event']=='daos_demand_outcome']
        recoveries=[(int(n),int(t)) for n,t in re.findall(r'Recovered from KV load failure: (\d+) request\(s\) rescheduled \((\d+) tokens affected\)',log)]
        summary=dict(arm=name,error=error,success=sum(c['status']=='success' for c in calls),
            failed=sum(c['status']!='success' for c in calls),recovery_events=len(recoveries),
            affected_tokens=sum(t for _,t in recoveries),capacity_failed_chunks=sum(e['capacity_failed_chunks'] for e in outcomes),
            returned_daos_chunks=sum(e['returned_chunks'] for e in outcomes),
            requested_daos_chunks=sum(e['requested_chunks'] for e in outcomes),
            cuda_error='CUDA error:' in log,
            v2_runner='Using V2 Model Runner' in log,
            scheduler_setting_confirmed=f'Asynchronous scheduling is {"enabled" if enabled else "disabled"}.' in log,
            daos_puts=max((e.get('daos_puts',0) for e in events),default=0))
        assert summary['daos_puts']==0,'Diagnostic unexpectedly changed retained KV cache'
        summaries.append(summary);qa.dump(root/'summary.json',summaries)
        print(json.dumps(summary),flush=True)
    qa.dump(root/'status.json',dict(status='completed_diagnostic',updated_ns=time.time_ns()))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--sync-probe',action='store_true')
    a=p.parse_args();run(a.output.resolve(),a.sync_probe)
