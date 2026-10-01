#!/usr/bin/env python3
"""Async lookup 1ms: no prefetch / DAOS-only / DAOS+DRAM, cold1+warm4.

Identical 1684-request history replay, fresh process/namespace per condition.
Retain evolving DRAM and DAOS cache within the five phases of each condition.
"""
import argparse
from copy import deepcopy
import csv
import fcntl
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time

import yaml
import sharegpt_async_payload_compare as previous
import sharegpt_early_lookup as early
from report_early_lookup import summarize_case

ROOT, base = early.ROOT, early.base
read, dump = early.read, early.dump
BASELINE = ROOT/'sharegpt_async_payload_d256_s8_20260930'
POLICIES = {'none': {'daos':False, 'dram':False},
            'daos': {'daos':True, 'dram':False},
            'both': {'daos':True, 'dram':True}}
PHASES = ['cold','warm1','warm2','warm3','warm4']
NEW_FILES = ('lmcache_daos/tier_payload_backend.py', 'sharegpt_async_threeway.py',
             'tests/test_tier_payload.py', 'tests/tier_payload_roundtrip.py')


def config(policy):
    flags = POLICIES[policy]
    cfg = previous.config(True)
    ec = cfg['extra_config']
    ec.pop('daosgds.early_payload_prefetch')
    ec.update({'storage_plugin.daosgds.module_path':'lmcache_daos.tier_payload_backend',
               'storage_plugin.daosgds.class_name':'TierPayloadBackend',
               'daosgds.early_daos_prefetch':flags['daos'],
               'daosgds.early_dram_prefetch':flags['dram']})
    return cfg


def comparable_config(cfg):
    result = deepcopy(cfg)
    for key in ('daosgds.object_namespace','daosgds.root',
                'daosgds.early_daos_prefetch','daosgds.early_dram_prefetch'):
        result['extra_config'].pop(key)
    return result


def prepare(root):
    assert root.parent == ROOT and not root.exists()
    assert read(BASELINE/'status.json')['status'] == 'completed'
    plan = read(BASELINE/'plan.json')
    for name,digest in plan['source_sha256'].items():
        assert base.digest(ROOT/name) == digest, f'Baseline source changed: {name}'
    assert base.digest(BASELINE/'requests.json') == plan['request_sha256']
    old = previous.comparable_config(yaml.safe_load((BASELINE/'c16_async_prefetch/config.yaml').read_text()))
    new = comparable_config(config('both'))
    for c in (old,new):
        for k in ('storage_plugin.daosgds.module_path','storage_plugin.daosgds.class_name'):
            c['extra_config'].pop(k)
    assert old == new, 'Unexpected change outside backend selector and per-tier switches'
    records = read(BASELINE/'requests.json')
    assert len(records) == 1684
    root.mkdir()
    for name in ('requests.json','sessions.json','capacity_estimate.json','params.json','runtime_versions.json'):
        shutil.copy2(BASELINE/name,root/name)
    plan.update(baseline=str(BASELINE), phases=PHASES, warm_repeats=4,
        requests_per_phase=len(records),requests_per_case=len(records)*len(PHASES),
        total_performance_requests=3*len(records)*len(PHASES),
        cases=[dict(name='c16_'+policy,policy=policy,prefetch=flags['daos'],
                    daos_prefetch=flags['daos'],dram_prefetch=flags['dram'],concurrency=16)
               for policy,flags in POLICIES.items()],
        notes=['All three arms use async metadata-first lookup, 1ms backoff, identical cache/store/promotion policy.',
               'none: both tiers demand-read at retrieve; daos: DAOS prefetch only; both: DAOS and DRAM prefetch.',
               'Same 421 ShareGPT conversations, first4 original turns, 1684 requests per phase, no padding.',
               'Fixed original histories/output caps; actual arrivals, generated text and tier placement can vary.',
               'DRAM256GiB, staging8GiB, C16, max_num_seqs16, Qwen3-14B BF16, chunk128, DRAM worker1.',
               'Five phases per arm = cold1+warm4, not five independently reset cold/warm pairs.',
               'New process/empty DRAM+staging/new namespace per arm; retain DRAM+DAOS across warm repeats.',
               'Drain staging and pending I/O between phases; do not flush OS or DAOS server caches.',
               'Order none->daos->both; warm repeats are correlated, not independent trials.',
               'No ready-return patch, lock probe, CUDA profiler, occupancy gate or artificial read delay.',
               'Live mixed DRAM/DAOS GPU byte validation for all three policies; model smoke for new DAOS-only mode.',
               'Per-phase results and staging/hit plots; pair cached token counts and report warm-only averages.',
               'Clean exact completed experiment UUID namespaces only after each full five-phase arm.'])
    for name in set(plan['source_sha256']) | set(NEW_FILES):
        dest = root/'executed_sources'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,dest)
        plan['source_sha256'][name] = base.digest(dest)
    dump(root/'plan.json',plan)
    dump(root/'status.json',dict(status='prepared',updated_ns=time.time_ns()))


def validate_policy(case,policy):
    flags = POLICIES[policy]
    events = base.read_events(case)
    enabled = [e for e in events if e['event']=='tier_payload_policy']
    assert len(enabled)==1
    assert all(enabled[0][tier] is flag for tier,flag in flags.items())
    retrieves = {e['request_id']:e['monotonic_ns'] for e in events if e['event']=='retrieve_start'}
    counts = {}
    for tier,flag in flags.items():
        starts = [e for e in events if e['event']=='early_payload_start' and e['tier']==tier]
        ready = [e for e in events if e['event']=='early_payload_ready' and e['tier']==tier]
        decisions = [e for e in events if e['event']=='early_retrieve_decision' and e['tier']==tier]
        counts[tier] = dict(prefetch_batches=len(starts),retrieved_batches=len(decisions))
        if not flag:
            assert not starts and not ready, f'{tier} speculative I/O while disabled'
            assert all(e['decision']=='queued_to_demand' for e in decisions)
    for e in events:
        if e['event']=='early_daos_demand_start':
            assert retrieves[e['request_id']] <= e['monotonic_ns']
    assert any(v['retrieved_batches'] for v in counts.values())
    dump(case/'payload_policy_check.json',dict(passed=True,policy=policy,counts=counts))


def run_one(root,spec):
    original = early.config
    early.config = lambda: config(spec['policy'])
    try:
        early.run_case(root,spec)
    finally:
        early.config = original
    summarize_case(root,spec)
    validate_policy(root/spec['name'],spec['policy'])


def aggregate_warm(rows):
    warm = [r for r in rows if r['phase'].startswith('warm')]
    assert warm
    queried = sum(r['queried_chunks'] for r in warm)
    inp = sum(r['input_tokens'] for r in warm)
    return dict(warm_repeats=len(warm),
        mean_ttft_ms=statistics.mean(r['ttft']['mean'] for r in warm),
        min_phase_mean_ttft_ms=min(r['ttft']['mean'] for r in warm),
        max_phase_mean_ttft_ms=max(r['ttft']['mean'] for r in warm),
        mean_elapsed_seconds=statistics.mean(r['elapsed_seconds'] for r in warm),
        dram_hit_pct=100*sum(r['hit_chunks']['dram'] for r in warm)/queried,
        daos_hit_pct=100*sum(r['hit_chunks']['daos'] for r in warm)/queried,
        input_reuse_pct=100*sum(r['cached_tokens'] for r in warm)/inp,
        computed_tokens=sum(r['computed_tokens'] for r in warm),
        output_tokens=sum(r['output_tokens'] for r in warm),
        peak_staging_gib=max(r['peak_staging_gib'] for r in warm))


def report(root,partial=False):
    plan = read(root/'plan.json')
    rows, summaries, comparisons, configs, commands, natives, namespaces = [], {}, [], [], [], [], []
    done = []
    for spec in plan['cases']:
        case = root/spec['name']
        if not (case/'summary.json').exists():
            assert partial, f'Missing completed case: {case}'
            continue
        assert read(case/'status.json')['status']=='completed'
        cfg = yaml.safe_load((case/'config.yaml').read_text())
        assert all(cfg['extra_config'][f'daosgds.early_{tier}_prefetch'] is flag
                   for tier,flag in POLICIES[spec['policy']].items())
        configs.append(comparable_config(cfg))
        commands.append(read(case/'command.json'))
        natives.append(read(case/'native_maps.json'))
        namespaces.append(cfg['extra_config']['daosgds.object_namespace'])
        current = read(case/'summary.json')
        assert [s['phase'] for s in current] == plan['phases']
        rows.extend(dict(policy=spec['policy'],**s) for s in current)
        summaries[spec['policy']] = aggregate_warm(current)
        done.append(spec)
    assert all(c==configs[0] for c in configs)
    assert all(c==commands[0] for c in commands)
    assert all(c==natives[0] for c in natives)
    assert len(set(namespaces))==len(namespaces)
    for left,right in (('none','daos'),('none','both'),('daos','both')):
        if left not in summaries or right not in summaries: continue
        paired = []
        for phase in plan['phases']:
            a,b = [read(root/('c16_'+p)/phase/'replay_calls.json') for p in (left,right)]
            keys = ('index','prompt_sha256','prompt_tokens','max_tokens')
            assert [tuple(x[k] for k in keys) for x in a]==[tuple(x[k] for k in keys) for x in b]
            x,y = [next(s for s in rows if s['policy']==p and s['phase']==phase) for p in (left,right)]
            paired.append(dict(phase=phase,requests=len(a),
                same_cached_requests=sum(x['cached_tokens']==y['cached_tokens'] for x,y in zip(a,b)),
                same_generated_output_requests=sum(x['output_sha256']==y['output_sha256'] for x,y in zip(a,b)),
                ttft_change_pct=100*(y['ttft']['mean']/x['ttft']['mean']-1),
                elapsed_change_pct=100*(y['elapsed_seconds']/x['elapsed_seconds']-1)))
        x,y = summaries[left],summaries[right]
        comparisons.append(dict(before=left,after=right,phases=paired,
            warm_mean_ttft_change_pct=100*(y['mean_ttft_ms']/x['mean_ttft_ms']-1),
            warm_mean_elapsed_change_pct=100*(y['mean_elapsed_seconds']/x['mean_elapsed_seconds']-1)))
    dump(root/'summary.json',rows)
    dump(root/'comparison.json',dict(completed_policies=[s['policy'] for s in done],
        warm_aggregates=summaries,comparisons=comparisons,
        limitations=['Warm4 is continued cache reuse, not independent trials; fixed arm order.',
                     'Cold is reported separately. Negative change means faster.',
                     'Same inputs and policy except prefetch tiers; actual arrivals, tier placement and generation can differ.']))
    fields = ['policy','phase','requests','ttft_mean_ms','ttft_p50_ms','ttft_p95_ms','elapsed_seconds',
              'dram_hit_pct','daos_hit_pct','input_reuse_pct','computed_tokens','output_tokens',
              'peak_staging_gib','mean_sampled_staging_gib']
    with (root/'summary.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {k:row[k] for k in fields if k in row}
            flat.update(ttft_mean_ms=row['ttft']['mean'],ttft_p50_ms=row['ttft']['p50'],ttft_p95_ms=row['ttft']['p95'])
            writer.writerow(flat)


def smoke(root,plan):
    # New mixed-tier dispatch has GPU validation for all modes. Previous OFF/ON
    # model paths already passed; exercise DAOS-only model integration once.
    folder = root.with_name(root.name+'_validation')
    folder.mkdir()
    spec = next(s for s in plan['cases'] if s['policy']=='daos')
    pilot = dict(plan,phases=['cold','warm1'],warm_repeats=1,requests_per_phase=8,
                 requests_per_case=16,cases=[spec])
    dump(folder/'plan.json',pilot)
    dump(folder/'requests.json',read(root/'requests.json')[:8])
    dump(root/'validation_location.json',dict(path=str(folder)))
    dump(root/'status.json',dict(status='vllm_validation',current=spec['name'],updated_ns=time.time_ns()))
    run_one(folder,spec)
    assert read(folder/spec['name']/'summary.json')[1]['cached_tokens']>0
    early.cleanup(folder/spec['name'])
    dump(folder/'status.json',dict(status='completed'))


def run(root):
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan = read(root/'plan.json')
        try:
            early.verify_sources(root,plan)
            assert early.idle_gpu(), 'GPU occupied; no unrelated job will be stopped'
            assert shutil.disk_usage(ROOT).free>8*2**30
            mem = {s.split(':')[0]:int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()}
            assert mem['MemAvailable']>=320*2**30
            os.environ.update(DAOS_GDS_PREFETCH_TIMING='1',DAOS_LOOKUP_READY_RETURN='0',DAOS_LOOKUP_LOCK_PROBE='0')
            dump(root/'status.json',dict(status='gpu_validation',updated_ns=time.time_ns()))
            with (root/'gpu_validation.log').open('x') as log:
                subprocess.run([str(ROOT/'run_vllm.sh'),str(ROOT/'venv/bin/python3'),
                    str(ROOT/'tests/tier_payload_roundtrip.py'),'--output',str(root/'gpu_validation')],
                    cwd=ROOT,env=dict(os.environ,DAOSGDS_TRANSPORT='object'),
                    stdout=log,stderr=subprocess.STDOUT,timeout=300,check=True)
            assert read(root/'gpu_validation/result.json')['result']=='PASS'
            smoke(root,plan)
            for index,spec in enumerate(plan['cases']):
                early.verify_sources(root,plan)
                assert shutil.disk_usage(ROOT).free>4*2**30
                dump(root/'status.json',dict(status='performance',current=spec['name'],
                    completed_cases=index,total_cases=len(plan['cases']),updated_ns=time.time_ns()))
                run_one(root,spec)
                report(root,partial=True)
                early.cleanup(root/spec['name'])
            report(root)
            dump(root/'status.json',dict(status='completed',updated_ns=time.time_ns()))
        except BaseException as exc:
            dump(root/'status.json',dict(status='failed',error=repr(exc),updated_ns=time.time_ns()))
            raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--report',action='store_true')
    args = parser.parse_args()
    root = args.output.resolve()
    assert root.parent==ROOT
    if args.prepare: prepare(root)
    elif args.report: report(root,partial=True)
    else: run(root)
