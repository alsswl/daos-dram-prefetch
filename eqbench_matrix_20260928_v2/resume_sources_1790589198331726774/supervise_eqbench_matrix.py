#!/usr/bin/env python3
"""Durable matrix supervisor: bounded retries, no workload/quality changes.

Launch with systemd as a service (not --scope/--pipe). No automatic reboot,
driver reset, broad cache removal, semantic code editing or favorable reruns.
"""
import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import uuid

import yaml
import eqbench_longform_matrix as matrix

ROOT=matrix.common.ROOT
PYTHON=ROOT/'venv/bin/python3'


def read(path, default=None):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return default


def dump(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)


def process_args(pid):
    try:return Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except FileNotFoundError:return []


def owned_group_members(pid,case):
    members=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            if os.getpgid(int(proc.name))!=pid or not process_args(int(proc.name)):continue
            env=(proc/'environ').read_bytes().split(b'\0')
            assert ('LMCACHE_CONFIG_FILE='+str(case/'server/config.yaml')).encode() in env, 'Process group identity mismatch'
            members.append(int(proc.name))
        except (FileNotFoundError,ProcessLookupError):continue
    return members


def stop_case_server(case):
    record=read(case/'server/pid.json',{})
    if not record:return
    pid=record['pid']; args=process_args(pid)
    if args:
        assert b'vllm.entrypoints.openai.api_server' in args, 'PID reused; do not signal'
        assert os.getpgid(pid)==pid
    if not owned_group_members(pid,case):return
    os.killpg(pid,signal.SIGTERM)
    deadline=time.monotonic()+35
    while owned_group_members(pid,case) and time.monotonic()<deadline:time.sleep(.2)
    if owned_group_members(pid,case):
        # Revalidate before escalation; only the recorded experiment group.
        os.killpg(pid,signal.SIGKILL)
        for _ in range(50):
            if not owned_group_members(pid,case):break
            time.sleep(.2)
    assert not owned_group_members(pid,case), 'Owned server did not stop'


def cleanup_rows(case):
    case=case.resolve(); root=case.parent
    assert root.parent==ROOT and root.name.startswith('eqbench_matrix_')
    assert case.name in {s['name'] for s in read(root/'plan.json')['cases']}
    assert read(root/'plan.json')['cleanup_completed_case_kv'] is True
    assert read(case/'status.json')['status'] in ('failed','completed')
    pid=read(case/'server/pid.json',{}).get('pid')
    assert pid is None or not owned_group_members(pid,case), 'Refuse cleanup of a live process group'
    config=case/'server/config.yaml';ec=yaml.safe_load(config.read_text())['extra_config']
    ns=ec['daosgds.object_namespace']
    assert re.fullmatch(r'minji-eqmatrix-[0-9a-f]{32}:',ns)
    assert ns==read(case/'identity.json')['namespace']
    assert ec['daosgds.pool']=='discospool' and ec['daosgds.container']=='kvcache'
    assert ec['daosgds.transport']=='object' and ec['daosgds.object_library']==str(ROOT/'libdaosgdr.so')
    import cleanup_experiment_cache as cleanup
    return [dict(namespace=ns,case=str(case),config_sha256=cleanup.digest(config),
                 status_sha256=cleanup.digest(case/'status.json'))]


def cleanup_cli(case, output, execute):
    import cleanup_experiment_cache as cleanup
    cleanup.eligible=lambda:cleanup_rows(case)
    assert output.resolve().parent==case.resolve()
    assert output.name.startswith('recovery_cleanup_')
    sys.argv=[sys.argv[0],'--output',str(output)]+(['--execute'] if execute else [])
    cleanup.main()


def cleanup_verified(case):
    namespace=read(case/'identity.json')['namespace']
    for folder in [case/'kv_cleanup',case/'interrupted_kv_cleanup',*case.glob('recovery_cleanup_*')]:
        r=read(folder/'result.json',{}); manifest=read(folder/'manifest.json',{})
        if (r.get('preserved_set_unchanged') and r.get('remaining_targets')==0
                and {x['namespace'] for x in manifest.get('namespaces',[])}=={namespace}):
            return folder/'result.json'
    return None


def cleanup_failed(case):
    existing=cleanup_verified(case)
    if existing:return existing
    output=case/('recovery_cleanup_'+uuid.uuid4().hex)
    for execute in (False,True):
        cmd=[str(ROOT/'run_vllm.sh'),str(PYTHON),str(Path(__file__).resolve()),
             '--cleanup-case',str(case),'--cleanup-output',str(output)]
        if execute:cmd.append('--execute')
        r=subprocess.run(cmd,env=dict(os.environ,DAOSGDS_TRANSPORT='object'),
                         capture_output=True,text=True,timeout=600)
        output.mkdir(exist_ok=True)
        (output/('execute.log' if execute else 'plan.log')).write_text(r.stdout+r.stderr)
        if r.returncode:raise RuntimeError('Scoped cleanup failed; inspect '+str(output))
    assert cleanup_verified(case)
    return output/'result.json'


def classify_failure(returncode, text, stalled=False):
    lower=text.lower()
    # Workload/data/code faults need investigation, not altered conditions.
    fatal=('cuda out of memory','outofmemoryerror','double free','negative ref',
           'context budget exceeded','tokenizer mismatch','assertionerror',
           'nameerror','typeerror','keyerror','syntaxerror','preserved key set changed')
    if any(x in lower for x in fatal):return 'needs_investigation'
    if any(x in lower for x in ('der_nospace','insufficient safe headroom','pool unhealthy')):
        return 'storage_unavailable'
    if stalled:return 'stalled'
    if returncode is not None and returncode<0:return 'terminated_by_signal'
    if any(x in lower for x in ('runner_disappeared','readtimeout','connecttimeout',
                               'connection reset','connectionerror','remoteprotocolerror',
                               'server exited','server startup timeout','incomplete stream')):
        return 'transport_or_runner_interruption'
    return 'needs_investigation'


def progress(root):
    status=read(root/'status.json',{})
    case=root/status.get('current','unknown')
    stamps=[];count=0
    for p in (case/'stories').glob('*/calls.json'):
        stamps.append(p.stat().st_mtime_ns)
        count+=len(read(p,[]))
    return dict(current=status.get('current'),completed=status.get('completed',[]),
                completed_calls=count,last_call_mtime_ns=max(stamps,default=0))


def archive_failed(root, case, reason):
    # Complete data can still be processed by --resume, never rerun it.
    if (case/'final_sample.json').exists() and 'end_ns' in read(case/'phase.json',{}):return
    stop_case_server(case)
    status=read(case/'status.json',{})
    if status.get('status')=='completed':return
    dump(case/'status.json',dict(**{k:v for k,v in status.items() if k not in ('status','supervisor_reason')},
        status='failed',supervisor_reason=reason))
    if (case/'identity.json').exists():cleanup_failed(case)
    target=root/'aborted_attempts'/(case.name+'__'+str(time.time_ns()))
    target.parent.mkdir(exist_ok=True)
    case.rename(target)
    return target


def tail(path, count=100_000):
    if not path.exists():return ''
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-count));return f.read().decode(errors='replace')


def ensure_no_runner(root):
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        args=process_args(int(proc.name))
        if (str(ROOT/'eqbench_longform_matrix.py').encode() in args
                and str(root).encode() in args):
            raise RuntimeError('Another matrix runner exists: '+proc.name)


def supervise(root, max_retries, idle_seconds):
    assert root.parent==ROOT and root.name=='eqbench_matrix_20260928_v2'
    assert read(root/'plan.json')['cases']==matrix.cases_for()
    meta=root/'supervision';meta.mkdir(exist_ok=True)
    lock=(meta/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ensure_no_runner(root)
    prior=read(root/'status.json')
    if prior['status']=='completed':return
    # Independently revalidate the owned case before recovering stale status.
    current=root/prior['current']
    if current.exists():
        stop_case_server(current)
        archive_failed(root,current,prior.get('failure_kind','service_recovered_stale_state'))
    failures=len(list((root/'aborted_attempts').glob(current.name+'__*')))
    if failures>max_retries:
        dump(meta/'status.json',dict(status='retry_limit',failures=failures,current=current.name));return
    dump(root/'status.json',dict(prior,status='failed'))
    stopped=False
    def stopping(signum,frame):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stopping);signal.signal(signal.SIGINT,stopping)
    while not stopped:
        ensure_no_runner(root)
        run_id=str(time.time_ns());log=meta/('runner_'+run_id+'.log')
        cmd=[str(PYTHON),'-u',str(ROOT/'eqbench_longform_matrix.py'),'--output',str(root),'--resume']
        with log.open('w') as stream:
            proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            dump(meta/'runner.json',dict(pid=proc.pid,started_ns=time.time_ns(),log=str(log),command=cmd))
            dump(meta/'status.json',dict(status='running',runner_pid=proc.pid,log=str(log)))
            marker=None;changed=time.monotonic();stalled=False
            while proc.poll() is None and not stopped:
                state=progress(root);new=(state['current'],len(state['completed']),state['last_call_mtime_ns'])
                if new!=marker:marker=new;changed=time.monotonic()
                dump(meta/'heartbeat.json',dict(time_ns=time.time_ns(),pid=proc.pid,
                    runner_alive=True,**state,last_progress_seconds_ago=time.monotonic()-changed))
                if time.monotonic()-changed>idle_seconds:
                    stalled=True;break
                time.sleep(5)
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=35)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            code=proc.wait()
        state=read(root/'status.json',{})
        if not state.get('current'):
            hb=read(meta/'heartbeat.json',{})
            state=dict(state,current=hb.get('current','unknown'),completed=hb.get('completed',[]))
        case=root/state.get('current','unknown')
        if case.exists():stop_case_server(case)
        record=dict(run_id=run_id,finished_ns=time.time_ns(),exit_code=code,log=str(log),
                    current=state.get('current'),completed=state.get('completed',[]),stalled=stalled)
        if code==0 and state.get('status')=='completed':
            dump(meta/'status.json',dict(status='completed',**record));return
        reason='user_or_service_stop' if stopped else classify_failure(code,tail(log)+'\n'+str(state),stalled)
        record['reason']=reason;dump(meta/('exit_'+run_id+'.json'),record)
        dump(root/'status.json',dict(state,status='failed',supervisor_reason=reason))
        if stopped:
            dump(meta/'status.json',dict(status='stopped',**record));return
        if reason in ('needs_investigation','storage_unavailable'):
            dump(meta/'status.json',dict(status='needs_investigation',**record));return
        if case.exists():archive_failed(root,case,reason)
        failures=len(list((root/'aborted_attempts').glob(case.name+'__*')))
        if failures>max_retries:
            dump(meta/'status.json',dict(status='retry_limit',failures=failures,**record));return
        dump(meta/'status.json',dict(status='retrying',failures=failures,**record))
        # No condition changes or performance-based retries.
        for _ in range(10):
            if stopped:return
            time.sleep(1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--max-retries',type=int,default=3)
    p.add_argument('--idle-seconds',type=int,default=900)
    p.add_argument('--cleanup-case',type=Path)
    p.add_argument('--cleanup-output',type=Path)
    p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if a.cleanup_case:
        cleanup_cli(a.cleanup_case,a.cleanup_output,a.execute);return
    root=a.output.resolve()
    try:supervise(root,a.max_retries,a.idle_seconds)
    except Exception as exc:
        dump(root/'supervision/status.json',dict(status='needs_investigation',error=repr(exc),time_ns=time.time_ns()))
        import traceback
        traceback.print_exc()
        # Unknown deterministic errors are recorded, not restarted indefinitely.


if __name__=='__main__':main()
