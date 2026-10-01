#!/usr/bin/env python3
"""36 live longform cases, scoped per-completed-case DAOS cleanup.

Sixteen conversations = first four original stories repeated four times.
No replay, padding, forced length, inter-turn barrier or intermediate drain.
Both prefetch-ON arms use early_ready=True to isolate queued cancellation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid

import yaml
import eqbench_longform_pilot as pilot
from eqbench_longform_pilot import common, read, pool_query, check_space, stats


def cases_for(cpu_values=(8,4,2), staging_values=(8,4)):
    assert len(cpu_values)==3 and len(set(cpu_values))==3
    assert all(v in (2,4,8,16,32) for v in cpu_values)
    assert len(staging_values)==2 and set(staging_values)=={4,8}
    modes = dict(off=(False, False, False), wait=(True, False, True),
                 cancel=(True, True, True))
    result = []
    for c in (8, 16):
        for d in cpu_values:
            for s in staging_values:
                names = list(modes); rotation = (len(result)//3) % 3
                for m in names[rotation:]+names[:rotation]:
                    pf, cancel, early = modes[m]
                    result.append(dict(name=f'c{c}_d{d}_s{s}_{m}', concurrency=c,
                        cpu_gib=d, staging_gib=s, mode=m, prefetch=pf,
                        cancel_queued=cancel, early_ready=early))
    return result


def cases_from_plan(plan):
    rows=plan['cases']
    cpu=list(dict.fromkeys(s['cpu_gib'] for s in rows))
    staging=list(dict.fromkeys(s['staging_gib'] for s in rows))
    expected=cases_for(cpu,staging)
    assert rows==expected, 'Plan differs from the declared factorial grid'
    return expected


def conversations(stories):
    return {str(i): dict(source_story_id=str((i-1)%4+1), repetition=(i-1)//4+1,
                        story=stories[str((i-1)%4+1)]) for i in range(1, 17)}


class FirstWave:
    """Synchronize only initial concurrency clients; subsequent jobs roll in."""
    def __init__(self, concurrency):
        self.remaining = concurrency
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(concurrency)

    def wait(self, timeout=30):
        with self.lock:
            first = self.remaining > 0
            if first: self.remaining -= 1
        if first: self.barrier.wait(timeout=timeout)


def config_for(spec, namespace):
    cfg = pilot.make_config(spec['prefetch'], namespace,
                           cpu_gib=spec['cpu_gib'], staging_gib=spec['staging_gib'])
    cfg['extra_config'].update({
        'storage_plugin.daosgds.module_path': 'lmcache_daos.capacity_probe_backend',
        'storage_plugin.daosgds.class_name': 'CapacityProbeBackend',
        'daosgds.dram_prefetch_workers': 1,
        'daosgds.dram_prefetch_cancel_queued': spec['cancel_queued'],
        'daosgds.dram_prefetch_early_ready': spec['early_ready']})
    return cfg


def cleanup_case(case, execute=False):
    """Reuse enumerated-key manifest workflow; never delete an entire OID."""
    import cleanup_experiment_cache as cleanup
    case = case.resolve()
    assert case.is_relative_to(common.ROOT) and case.parent.name.startswith('eqbench_matrix_')
    plan = read(case.parent/'plan.json')
    assert plan['cleanup_completed_case_kv'] is True
    assert case.name in {s['name'] for s in plan['cases']}
    assert read(case/'status.json')['status'] == 'completed'
    assert (case/'summary.json').exists() and (case/'staging_hits.png').exists()
    config = case/'server/config.yaml'
    ec = yaml.safe_load(config.read_text())['extra_config']
    ns = ec['daosgds.object_namespace']
    assert re.fullmatch(r'minji-eqmatrix-[0-9a-f]{32}:', ns)
    assert ns == read(case/'identity.json')['namespace']
    assert ec['daosgds.pool'] == 'discospool' and ec['daosgds.container'] == 'kvcache'
    assert ec['daosgds.transport'] == 'object'
    assert ec['daosgds.object_library'] == str(common.ROOT/'libdaosgdr.so')
    rows = [dict(namespace=ns, case=str(case), config_sha256=cleanup.digest(config),
                 status_sha256=cleanup.digest(case/'status.json'))]
    cleanup.eligible = lambda: rows
    sys.argv = [sys.argv[0], '--output', str(case/'kv_cleanup')]+(['--execute'] if execute else [])
    cleanup.main()


def prefetch_counters(initial, final, enabled):
    before, after = initial.get('cpu_prefetch') or {}, final.get('cpu_prefetch') or {}
    assert after.get('copy_errors',0) == after.get('deferred_pending_batches',0) == 0
    if enabled: assert after, 'Missing prefetch-ON counters'
    return {k:v-before.get(k,0) for k,v in after.items()}


def report_case(case, spec):
    calls = sorted([c for p in (case/'stories').glob('*/calls.json') for c in read(p)], key=lambda c:c['index'])
    assert len(calls) == 208 and not any('error' in c for c in calls)
    assert [c['index'] for c in calls] == list(range(208))
    events = pilot.read_events(case/'server')
    selected = pilot.select_events(events, calls)
    attributed = pilot.attribute_recomputation(calls, selected, 128)
    rows = []
    for c, attr in zip(calls, attributed):
        assert c['index'] == attr['index']
        row = {k:v for k,v in c.items() if k not in ('output','sampling','usage')}
        row.update(attr); rows.append(row)
    common.dump(case/'requests_summary.json', rows)
    phase = read(case/'phase.json')
    scoped = [e for e in events if phase['start_ns'] <= e['time_ns'] <= phase['end_ns']]
    samples = [e['used_bytes']/2**30 for e in scoped if e['event']=='occupancy_sample']
    initial, final = read(case/'initial_sample.json'), read(case/'final_sample.json')
    assert initial['used_bytes'] == initial['cpu_hot_bytes'] == initial['daos_puts'] == 0
    assert final['used_bytes'] == final['dram_mirror']['pending_bytes'] == 0
    assert final['dram_mirror']['errors'] == 0
    pf = prefetch_counters(initial, final, spec['prefetch'])
    candidates = sum(e['queried_chunks'] for e in selected if e['event']=='tier_lookup' and e['tier']=='dram')
    dram = sum(r['dram_lookup_chunks'] for r in rows); daos = sum(r['daos_lookup_chunks'] for r in rows)
    issues = [r['index'] for r in rows if r['attribution_issues'] or r['unattributed_shortfall_tokens']]
    cap = None if issues else sum(r['capacity_recomputed_tokens'] for r in rows)
    inputs = sum(r['prompt_tokens'] for r in rows)
    summary = dict(**spec, requests=208, stories=16,
        elapsed_seconds=(phase['end_ns']-phase['start_ns'])/1e9,
        ttft_ms=stats(r['ttft_ms'] for r in rows),
        total_input_tokens=inputs, total_output_tokens=sum(r['completion_tokens'] for r in rows),
        cached_tokens=sum(r['cached_tokens'] for r in rows),
        computed_input_tokens=sum(r['computed_prompt_tokens'] for r in rows),
        dram_hit_pct=100*dram/candidates, daos_hit_pct=100*daos/candidates,
        dram_lookup_chunks=dram, daos_lookup_chunks=daos, lookup_candidates=candidates,
        peak_staging_gib=max(e.get('used_bytes',0) for e in scoped)/2**30,
        mean_staging_gib=sum(samples)/len(samples),
        capacity_failed_chunks=sum(r['capacity_failed_chunks'] for r in rows),
        capacity_recomputed_tokens=cap, capacity_recomputed_input_pct=None if cap is None else 100*cap/inputs,
        attribution_issue_indices=issues, cpu_prefetch=pf,
        length_capped_requests=sum(r['finish_reason']=='length' for r in rows))
    common.dump(case/'summary.json', summary)
    pilot.timeline(case, events, phase['start_ns'], phase['end_ns'], spec['staging_gib'])
    return summary


def report_matrix(root):
    summaries = [read(root/s['name']/'summary.json') for s in read(root/'plan.json')['cases']
                 if (root/s['name']/'summary.json').exists()]
    common.dump(root/'summary.json', summaries)
    lines = ['# EQ-Bench Longform 36조건 비교', '',
        '진행 중인 조건은 표에서 제외한다. 조건마다 빈 DRAM·새 DAOS namespace로 시작한다.',
        '첫4개 주제를 각각4번: 총16대화×13단계=208호출/조건. 실제 답변 누적, 반복 주제 간 재사용 포함.',
        '전체 warm 반복이 아니라 cold 시작 후 자연스러운 재사용이다. 출력·계산량 차이도 확인해야 한다.',
        'wait/cancel은 모두 early_ready ON: 취소 외 설정 동일. 이전 wait(early_ready OFF) 실험과 다르다.',
        '취소는 retrieve 시 아직 시작하지 않은 DRAM 복사에만 적용. DAOS 프리페치는 모든 조건에서 ON.',
        '모델 Qwen3-14B BF16, 청크128, 작업자1, max-num-seqs16. 조건별1회, 공식 품질 채점 아님.', '',
        '|조건|시간(s)|평균 TTFT(ms)|DRAM hit %|DAOS hit %|staging 최대 GiB|용량 부족 재계산 토큰|대기 취소|생성 토큰|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    flat = []
    for s in summaries:
        lines.append(f"|[{s['name']}]({s['name']}/staging_hits.png)|{s['elapsed_seconds']:.2f}|{s['ttft_ms']['mean']:.2f}|{s['dram_hit_pct']:.2f}|{s['daos_hit_pct']:.2f}|{s['peak_staging_gib']:.3f}|{s['capacity_recomputed_tokens']}|{s['cpu_prefetch'].get('retrieve_cancelled_queued',0)}|{s['total_output_tokens']}|")
        flat.append({k:v for k,v in s.items() if not isinstance(v,(dict,list))} | {'ttft_mean_ms':s['ttft_ms']['mean'], 'ttft_p95_ms':s['ttft_ms']['p95']})
    (root/'RESULT_KO.md').write_text('\n'.join(lines)+'\n')
    if flat:
        with (root/'summary.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--port',type=int,default=8017)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--dram-gib',type=int,nargs=3,default=[8,4,2])
    p.add_argument('--staging-gib',type=int,nargs=2,default=[8,4])
    p.add_argument('--cleanup-case',type=Path)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--resume',action='store_true',help='Resume only after runner exited; never rerun a partly measured case')
    a=p.parse_args()
    if a.cleanup_case: cleanup_case(a.cleanup_case,a.execute);return
    assert a.output is not None
    root=a.output.resolve();assert root.parent==common.ROOT and root.name.startswith('eqbench_matrix_')
    if a.resume:
        assert root.is_dir() and read(root/'status.json')['status']=='failed'
    else:
        root.mkdir(exist_ok=False)
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=pilot.UPSTREAM,text=True).strip()==pilot.COMMIT
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=pilot.UPSTREAM,text=True).strip()
    stories=read(pilot.UPSTREAM/'data/longform_creative_writing_prompts_minimalist.json')
    jobs=conversations(stories);templates=pilot.load_templates(pilot.UPSTREAM/'data')
    if a.resume:
        assert read(root/'conversations.json')==jobs
    else:
        common.dump(root/'conversations.json',jobs);common.dump(root/'templates.json',templates)
    names=['eqbench_longform_matrix.py','eqbench_longform_pilot.py','cleanup_experiment_cache.py','supervise_eqbench_matrix.py',
        'cold_warm_prefetch.py','discovery_fixed_replay.py','discovery_rolling_replay.py',
        'staging_mixed_pressure.py','compare_e2e.py','run_vllm.sh','libdaosgdr.so','libdaosgdr.c',
        'prefetch_capacity_sweep.py','report_capacity_matrix.py','report_cold_warm_prefetch.py',
        'report_prefetch_timing.py','report_prefetch_capacity_sweep.py','analyze_discovery_staging.py',
        'lmcache_config_daosgds_async_dram.yaml','list_experiment_dkeys']
    names += [str(f.relative_to(common.ROOT)) for f in sorted((common.ROOT/'lmcache_daos').glob('*.py'))]
    hashes={}
    snapshot = 'resume_sources_'+str(time.time_ns()) if a.resume else 'executed_sources'
    for name in names:
        dest=root/snapshot/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(common.ROOT/name,dest);hashes[name]=hashlib.sha256(dest.read_bytes()).hexdigest()
    specs=cases_from_plan(read(root/'plan.json')) if a.resume else cases_for(a.dram_gib,a.staging_gib)
    new_plan = dict(cases=specs,requests_per_case=208,total_requests=7488,
        upstream_commit=pilot.COMMIT,source_sha256=hashes,cleanup_completed_case_kv=True,
        cleanup_authority='User approved DAOS cleanup 2026-09-28; scope only this matrix completed cases.',
        model='Qwen/Qwen3-14B',chunk_tokens=128,max_num_seqs=16,
        comparison='early_ready held True for both prefetch arms',repeats_per_condition=1)
    if a.resume:
        assert read(root/'plan.json')['cases']==specs
        common.dump(root/snapshot/'resume.json',dict(source_sha256=hashes,reason='Resume with preserved measurements; no partial-case rerun'))
    else: common.dump(root/'plan.json',new_plan)
    if a.dry_run:
        common.dump(root/'status.json',dict(status='dry_run'));print(root);return
    os.environ['HF_HOME']='/home/hf/hf_cache';os.environ['DAOS_GDS_PREFETCH_TIMING']='1'
    from transformers import AutoTokenizer
    a.model='Qwen/Qwen3-14B';a.max_model_len=32768
    tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True);assert len(tok)>100000
    completed=[]
    try:
        for spec in specs:
            case=root/spec['name']
            if a.resume and case.exists():
                # Only repair reporting/cleanup after a fully drained measurement.
                assert 'end_ns' in read(case/'phase.json') and (case/'final_sample.json').exists()
                report_case(case,spec)
                if read(case/'status.json')['status']!='completed':
                    common.dump(case/'status.json',dict(status='completed',requests=208,report_recovered=True))
                report_matrix(root)
                cleanup_result=case/'kv_cleanup/result.json'
                if not cleanup_result.exists():
                    assert not (case/'kv_cleanup').exists(), 'Partial cleanup requires manual audit'
                    for execute in (False,True):
                        cmd=[str(common.ROOT/'run_vllm.sh'),str(common.ROOT/'venv/bin/python3'),str(Path(__file__).resolve()),'--cleanup-case',str(case)]
                        if execute:cmd.append('--execute')
                        r=subprocess.run(cmd,env=dict(os.environ,DAOSGDS_TRANSPORT='object'),capture_output=True,text=True,timeout=600)
                        (case/('cleanup_execute.log' if execute else 'cleanup_plan.log')).write_text(r.stdout+r.stderr)
                        if r.returncode:raise RuntimeError('Scoped cleanup failed')
                assert read(cleanup_result)['preserved_set_unchanged']
                completed.append(spec['name']);continue
            case.mkdir();srv=case/'server';srv.mkdir()
            ns='minji-eqmatrix-'+uuid.uuid4().hex
            common.dump(case/'identity.json',dict(namespace=ns+':'))
            cfg=config_for(spec,ns);(srv/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
            q=pool_query();common.dump(case/'pool_before.json',q);check_space(q)
            common.dump(root/'status.json',dict(status='running',current=spec['name'],completed=completed))
            common.dump(case/'status.json',dict(status='running'))
            print(f"CASE {len(completed)+1}/36 {spec['name']}",flush=True)
            with pilot.server(a,srv/'config.yaml',srv) as client:
                common.dump(case/'initial_sample.json',pilot.await_empty(srv))
                health=pilot.LogHealth(srv/'server.log');health()
                phase=dict(start_ns=time.time_ns());common.dump(case/'phase.json',phase)
                stop=threading.Event();barrier=FirstWave(spec['concurrency'])
                with ThreadPoolExecutor(max_workers=spec['concurrency']) as pool:
                    pending={pool.submit(pilot.run_story,client,tok,k,v['story'],templates,case,barrier,stop) for k,v in jobs.items()}
                    try:
                        while pending:
                            done,pending=wait(pending,timeout=.2,return_when=FIRST_COMPLETED)
                            health()
                            for f in done:f.result()
                    except BaseException:
                        stop.set();raise
                phase['end_ns']=time.time_ns();common.dump(case/'phase.json',phase)
                (case/'metrics_after.txt').write_text(client.get('/metrics').text)
                common.dump(case/'final_sample.json',pilot.drain(srv,health))
            common.dump(case/'pool_after.json',pool_query())
            report_case(case,spec)
            common.dump(case/'status.json',dict(status='completed',requests=208))
            report_matrix(root)
            # Native DAOS subprocess starts after vLLM exits; exact namespace only.
            for execute in (False,True):
                cmd=[str(common.ROOT/'run_vllm.sh'),str(common.ROOT/'venv/bin/python3'),
                     str(Path(__file__).resolve()),'--cleanup-case',str(case)]
                if execute:cmd.append('--execute')
                result=subprocess.run(cmd,env=dict(os.environ,DAOSGDS_TRANSPORT='object'),capture_output=True,text=True,timeout=600)
                (case/('cleanup_execute.log' if execute else 'cleanup_plan.log')).write_text(result.stdout+result.stderr)
                if result.returncode:raise RuntimeError('Scoped cleanup failed; stop before next case')
            completed.append(spec['name'])
            print(f"COMPLETE {len(completed)}/36 {spec['name']} KV cleaned",flush=True)
        common.dump(root/'status.json',dict(status='completed',completed=completed))
    except BaseException as exc:
        common.dump(root/'status.json',dict(status='failed',current=spec['name'],completed=completed,error=repr(exc)))
        if read(case/'status.json')['status']!='completed':
            common.dump(case/'status.json',dict(status='failed',error=repr(exc)))
        raise


if __name__=='__main__':main()
