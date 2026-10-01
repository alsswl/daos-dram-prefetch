#!/usr/bin/env python3
"""Keep at most N requests in flight, replacing each completion independently.

Reuse the fixed replay HTTP payload and backend configuration. No wave barriers
after the initial launch; no model/backend/storage-policy changes.
"""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import threading
import time
import uuid

import yaml

import compare_e2e as common
from discovery_fixed_replay import await_empty, make_config, replay_one
from staging_mixed_pressure import latest_sample, server


def rolling_requests(records, concurrency, invoke, on_result, check_health=lambda: None):
    """Bound active futures; refill a slot without waiting for other requests.

    invoke(record, barrier) uses a shared barrier only for the initial requests.
    Subsequent requests get a one-party barrier, preserving replay_one unchanged.
    Persist completed in-flight calls on failure; never retry or admit more calls.
    """
    if concurrency < 1:
        raise ValueError('concurrency must be positive')
    records=iter(records)
    initial=[]
    for _ in range(concurrency):
        record=next(records,None)
        if record is None: break
        initial.append(record)
    if not initial: return
    barrier=threading.Barrier(len(initial))
    pending=set()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        pending.update(pool.submit(invoke,r,barrier) for r in initial)
        try:
            while pending:
                done,_=wait(pending,timeout=.1,return_when=FIRST_COMPLETED)
                check_health()
                # Futures finishing together need not be processed in input order.
                for future in sorted(done,key=lambda f:f.result().get('end_ns',0)):
                    pending.remove(future)
                    result=future.result()
                    on_result(result)
                    if 'error' in result:
                        raise RuntimeError('Replay request failed; no retry')
                    check_health()
                    record=next(records,None)
                    if record is not None:
                        pending.add(pool.submit(invoke,record,threading.Barrier(1)))
        except BaseException:
            for future in pending: future.cancel()
            # Already admitted calls drain under the HTTP timeout; preserve them.
            for future in pending:
                if not future.cancelled():
                    try: on_result(future.result())
                    except Exception: pass
            raise


class LogHealth:
    """Incremental error watcher, including failures hidden behind HTTP success."""
    def __init__(self,path):
        self.path=path; self.offset=0; self.tail=''

    def __call__(self):
        with self.path.open('rb') as stream:
            stream.seek(self.offset); chunk=stream.read().decode(errors='replace'); self.offset=stream.tell()
        text=self.tail+chunk; self.tail=text[-512:]
        patterns=('DER_NOSPACE','rc=-1007','LMCache ERROR','[libdaosgdr] daos_obj_update_gpu rc=',
                  'Double free','Double release','negative: -','CUDA error:','CUDA out of memory')
        for pattern in patterns:
            if pattern in text:
                raise RuntimeError(f'Server/storage error ({pattern}); benchmark invalid, stop admitting requests')


def run_case(a,case,records,concurrency,enabled):
    namespace='minji-rolling-replay-'+uuid.uuid4().hex
    cfg=make_config(enabled,namespace,cpu_gib=a.cpu_gb,staging_gib=a.staging_gib)
    cfg['extra_config'].update({'storage_plugin.daosgds.module_path':'lmcache_daos.capacity_probe_backend',
                               'storage_plugin.daosgds.class_name':'CapacityProbeBackend'})
    (case/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    calls=[]; phase=None
    try:
        with server(a,case/'config.yaml',case) as client:
            common.dump(case/'initial_sample.json',await_empty(case))
            phase=dict(index=0,concurrency=concurrency,start_ns=time.time_ns(),arrival_mode='rolling')
            common.dump(case/'phases.json',[phase])
            health=LogHealth(case/'server.log'); health()
            def save(row):
                calls.append(row)
                common.dump(case/'replay_calls.json',sorted(calls,key=lambda r:r['index']))
                if len(calls)%8==0 or len(calls)==len(records):
                    print(f'{case.name} {len(calls)}/{len(records)} completed (rolling concurrency={concurrency})',flush=True)
            rolling_requests(records,concurrency,lambda r,b:replay_one(client,r,b),save,health)
            phase['end_ns']=time.time_ns(); common.dump(case/'phases.json',[phase])
            (case/'metrics.txt').write_text(client.get('/metrics').text)
            deadline=time.monotonic()+60; stable=0; sample={}
            while time.monotonic()<deadline:
                health(); sample=latest_sample(case) or {}
                if sample and sample['used_bytes']==0 and sample['dram_mirror']['pending_bytes']==0: stable+=1
                else: stable=0
                if stable>=10: break
                time.sleep(.2)
            common.dump(case/'final_sample.json',sample)
            if stable<10: raise RuntimeError('Staging/mirror failed to drain')
            health()
        common.dump(case/'status.json',dict(status='completed',requests=len(calls),arrival_mode='rolling'))
    except BaseException as exc:
        if phase and 'end_ns' not in phase:
            phase.update(end_ns=time.time_ns(),failed=True); common.dump(case/'phases.json',[phase])
        common.dump(case/'status.json',dict(status='failed',error=repr(exc),requests=len(calls)))
        raise
