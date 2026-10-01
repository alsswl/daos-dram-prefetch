#!/usr/bin/env python3
"""One fresh full cold fill + 256 warm requests; mutex diagnostic, not TTFT A/B."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import yaml
import sharegpt_early_lookup as early
import sharegpt_lookup_backoff as backoff
from analyze_gpu_overlap import Intersections

ROOT, base, cw = early.ROOT, early.base, early.cw
read, dump = early.read, early.dump
BASELINE = ROOT/'sharegpt_ready_return_d256_s8_20260930'


def analyze(root):
    case = root/'c16_lock_probe'
    events = [json.loads(line) for path in case.glob('lookup_lock.*.jsonl') for line in path.open()]
    assert events, 'Lock probe produced no records'
    polls = {}
    for e in events:
        if e['operation'] == 'lookup_cache':
            polls.setdefault((e['pid'], e['client_id']), []).append((e['acquired_ns'], e['release_started_ns']))
    spans = {key: Intersections(intervals) for key, intervals in polls.items()}
    summaries = []
    for phase in ('cold', 'warm_lock_probe'):
        calls = read(case/phase/'replay_calls.json')
        ids = {c['server_request_id'] for c in calls}
        receivers = [e for e in events if e['request_id'] in ids and e['operation']=='process_responses_from_workers']
        assert receivers, f'No response lock timings for {phase}'
        rows = []
        for e in receivers:
            row = dict(e)
            row['acquire_ms'] = (e['acquired_ns']-e['requested_ns'])/1e6
            row['poll_hold_overlap_ms'] = spans.get((e['pid'],e['client_id']), Intersections([])).overlap(
                e['requested_ns'], e['acquired_ns'])/1e6
            rows.append(row)
        scoped_polls = [e for e in events if e['request_id'] in ids and e['operation']=='lookup_cache']
        summary = dict(phase=phase, requests=len(calls), responses=len(rows),
            response_lock_acquire_ms=base.stats(e['acquire_ms'] for e in rows),
            acquire_over_0_1ms=sum(e['acquire_ms']>0.1 for e in rows),
            acquire_over_1ms=sum(e['acquire_ms']>1 for e in rows),
            observed_locked_count=sum(e['observed_locked_before_acquire'] for e in rows),
            total_response_acquire_ms=sum(e['acquire_ms'] for e in rows),
            total_acquire_overlap_poll_hold_ms=sum(e['poll_hold_overlap_ms'] for e in rows),
            polling_lock_hold_ms=base.stats((e['release_started_ns']-e['acquired_ns'])/1e6 for e in scoped_polls),
            polling_holds_over_0_5ms=sum(e['release_started_ns']-e['acquired_ns']>500000 for e in scoped_polls))
        dump(case/phase/'response_lock_waits.json', rows)
        summaries.append(summary)
    dump(root/'summary.json', dict(phases=summaries, limitations=[
        'Instrumented diagnostic, not a performance A/B or pure mutex hardware wait.',
        'Acquire time includes OS scheduling, GIL and measurement overhead.',
        'Overlap with scheduler polling lock hold supports contention; it is not directly TTFT savings.',
        'Per-event JSON I/O happens after unlock but can perturb subsequent scheduling.',
        'Warm subset is first64 original conversations, not the complete 1684-request warm phase.']))


def run(root):
    assert root.parent == ROOT and not root.exists()
    assert early.idle_gpu(), 'GPU occupied; unrelated processes will not be stopped'
    assert read(BASELINE/'status.json')['status'] == 'completed'
    old = read(BASELINE/'plan.json')
    for name, digest in old['source_sha256'].items():
        assert base.digest(ROOT/name) == digest, f'Baseline source changed: {name}'
    assert shutil.disk_usage(ROOT).free > 4*2**30
    mem = {s.split(':')[0]: int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()}
    assert mem['MemAvailable'] >= 320*2**30
    root.mkdir()
    case = root/'c16_lock_probe'
    case.mkdir()
    records = read(BASELINE/'requests.json')
    sessions = sorted({r['session'] for r in records})[:64]
    warm = [r for r in records if r['session'] in sessions]
    assert len(records)==1684 and len(warm)==256
    plan = dict(old, cases=[dict(name=case.name, concurrency=16, prefetch=True)],
        phases=['cold','warm_lock_probe'], diagnostic=True, baseline=str(BASELINE),
        full_cold_requests=1684, warm_requests=256, requests_per_case=1940,
        notes=['Existing 1ms backoff + ready return retained; lock timing only, no CUDA profiler.',
               'Fresh cold full corpus then first64 original conversations x4 turns.',
               'No lock policy, sleep, timeout, prefetch or cache capacity change.',
               'Normal blocking acquire preserved; writes after release; observer effects possible.'])
    names = set(old['source_sha256']) | {'lookup_lock_probe.py', 'sharegpt_lookup_lock_probe.py',
        'tests/test_lookup_lock_probe.py',
        'experiment_plugins/lookup_lock_probe_hook-0.1.dist-info/METADATA',
        'experiment_plugins/lookup_lock_probe_hook-0.1.dist-info/entry_points.txt'}
    plan['source_sha256'] = {}
    for name in names:
        dest = root/'executed_sources'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
        plan['source_sha256'][name] = base.digest(dest)
    dump(root/'plan.json', plan)
    shutil.copy2(BASELINE/'requests.json', root/'requests.json')
    dump(root/'warm_requests.json', warm)
    cfg = backoff.config()
    (case/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    os.environ.update(DAOS_GDS_PREFETCH_TIMING='1', DAOS_LOOKUP_READY_RETURN='1', DAOS_LOOKUP_LOCK_PROBE='1')
    os.environ['PYTHONPATH'] = str(ROOT/'experiment_plugins')+os.pathsep+str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH','')
    assert not os.environ.get('VLLM_PLUGINS'), 'Plugin allow-list requires review'
    cap = plan['capacity']
    try:
        dump(root/'status.json', dict(status='preflight', updated_ns=time.time_ns()))
        cw.wait_space(case, 2*cap['conservative_stored_kv_gib']-cap['unique_input_kv_gib'])
        args = SimpleNamespace(model=plan['model'], max_model_len=16384, max_num_seqs=16, port=8017)
        dump(root/'status.json', dict(status='starting_server', updated_ns=time.time_ns()))
        with base.server(args, case/'config.yaml', case) as client:
            assert 'LOOKUP_LOCK_PROBE active' in (case/'server.log').read_text(), 'Lock probe not installed'
            initial = base.await_empty(case)
            assert initial['used_bytes']==initial['cpu_hot_bytes']==initial['daos_puts']==0
            dump(case/'initial_sample.json', initial)
            health = base.LogHealth(case/'server.log')
            dump(root/'status.json', dict(status='cold', updated_ns=time.time_ns()))
            state = cw.run_phases(case, records, 16, client, health, initial, phases=['cold'])
            dump(root/'status.json', dict(status='warm_lock_probe', updated_ns=time.time_ns()))
            final = cw.run_phases(case, warm, 16, client, health, state, phases=['warm_lock_probe'],
                warm_extra_gib=cap['conservative_stored_kv_gib']-cap['unique_input_kv_gib'])
            assert final['daos_alloc_fail']==final['dram_mirror']['errors']==final['cpu_prefetch']['copy_errors']==0
            dump(case/'final_sample.json', final)
        dump(case/'status.json', dict(status='completed', requests=1940))
        early.cleanup(case)
        analyze(root)
        dump(root/'status.json', dict(status='completed', updated_ns=time.time_ns()))
    except BaseException as exc:
        dump(root/'status.json', dict(status='failed', error=repr(exc), updated_ns=time.time_ns()))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    run(p.parse_args().output.resolve())
