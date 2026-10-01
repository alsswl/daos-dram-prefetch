#!/usr/bin/env python3
"""Cold-start ShareGPT history replay: DRAM256/staging8/C16, prefetch OFF/ON.

No backend edits. Full runs use fixed original histories, no synthetic padding,
no prefill warmup, no forced hit placement, no occupancy gate or queued cancel.
"""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace
import uuid

import yaml

import compare_e2e as common
from cold_warm_prefetch import drain
from discovery_fixed_replay import await_empty, make_config, replay_one
from discovery_rolling_replay import LogHealth
from prefetch_capacity_sweep import pool_query, check_space
from prepare_sharegpt_scale import digest, dump
from report_capacity_matrix import save_chart
from report_cold_warm_prefetch import select_events
from report_prefetch_capacity_sweep import timeline
from report_prefetch_timing import join, stats
from staging_mixed_pressure import latest_sample, read_events, server

ROOT = common.ROOT


def read(path):
    return json.loads(path.read_text())


def session_requests(records, concurrency, invoke, on_result, health=lambda: None):
    """Rolling global limit + one in-flight turn per conversation; no wave fence."""
    if concurrency < 1:
        raise ValueError('concurrency must be positive')
    remaining, active, pending = list(records), set(), {}
    next_turn = {}
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        def refill():
            i = 0
            while len(pending) < concurrency and i < len(remaining):
                r = remaining[i]
                sid = r['session']
                if sid not in active and r['turn'] == next_turn.get(sid, 0):
                    remaining.pop(i)
                    active.add(sid)
                    pending[executor.submit(invoke, r, threading.Barrier(1))] = r
                else:
                    i += 1
        refill()
        try:
            while pending:
                done, _ = wait(pending, timeout=.2, return_when=FIRST_COMPLETED)
                health()
                for future in sorted(done, key=lambda f: f.result()['end_ns']):
                    record = pending.pop(future)
                    row = future.result()
                    on_result(row)
                    if 'error' in row:
                        raise RuntimeError('Request failed; no retry or new admission')
                    active.remove(record['session'])
                    next_turn[record['session']] = record['turn']+1
                refill()
            if remaining:
                raise RuntimeError('Invalid or non-contiguous session turn schedule')
        except BaseException:
            for future in pending:
                future.cancel()
            for future in pending:
                if not future.cancelled():
                    try:
                        on_result(future.result())
                    except Exception:
                        pass
            raise


def prepare(root, data):
    if root.exists():
        raise FileExistsError(root)
    cap = read(data/'capacity_estimate.json')
    records = read(data/'requests.json')
    assert read(data/'status.json')['status'] == 'prepared'
    assert digest(data/'requests.json') == cap['request_sha256']
    assert cap['unique_input_kv_gib'] >= 320  # At least 1.25x the 256GiB CPU tier.
    assert cap['input_tokens_max'] <= 8192
    assert all(r['index'] == i and r['prompt_sha256'] == hashlib.sha256(r['prompt'].encode()).hexdigest()
               and r['expected_prompt_tokens'] + r['max_tokens'] <= 16384
               for i, r in enumerate(records))
    root.mkdir()
    for name in ('requests.json', 'sessions.json', 'capacity_estimate.json', 'params.json'):
        shutil.copy2(data/name, root/name)
    specs = [dict(name='pilot_on', prefetch=True, pilot=True, concurrency=4),
             dict(name='c16_off', prefetch=False, pilot=False, concurrency=16),
             dict(name='c16_on', prefetch=True, pilot=False, concurrency=16)]
    files = ['sharegpt_scale_experiment.py', 'prepare_sharegpt_scale.py', 'tests/test_sharegpt_scale.py',
             'discovery_occupancy_gate.py', 'cold_warm_prefetch.py', 'discovery_fixed_replay.py',
             'discovery_rolling_replay.py', 'staging_mixed_pressure.py', 'compare_e2e.py',
             'report_cold_warm_prefetch.py', 'report_prefetch_timing.py', 'report_capacity_matrix.py',
             'report_prefetch_capacity_sweep.py', 'prefetch_capacity_sweep.py',
             'cleanup_experiment_cache.py', 'list_experiment_dkeys', 'run_vllm.sh',
             'libdaosgdr.so', 'libdaosgdr.c', 'lmcache_config_daosgds_async_dram.yaml']
    files += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'lmcache_daos').glob('*.py'))]
    hashes = {}
    for name in files:
        dest = root/'executed_sources'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
        hashes[name] = digest(dest)
    dump(root/'plan.json', dict(model='Qwen/Qwen3-14B', cpu_gib=256, staging_gib=8,
        chunk_tokens=128, max_num_seqs=16, max_model_len=16384, workers=1,
        requests_per_case=len(records), cases=specs, capacity=cap,
        source_sha256=hashes, request_sha256=digest(root/'requests.json'),
        notes=['One cold-start history replay per arm, no separately preloaded warm phase.',
               'Same original histories, token/output caps and session schedule in OFF and ON.',
               'Rolling max16 requests with one in-flight turn per conversation; wall-clock arrivals may differ.',
               'Fresh process/empty DRAM/new namespace each arm; vLLM prefix cache OFF.',
               'Only DRAM prefetch changes. DAOS prefetch/store/read promotion remain ON.',
               'One copy worker; no occupancy gate, queued cancellation or early-ready policy.',
               'Reference answers form subsequent prompts; generated answers are measured but not fed back.',
               'Single trial: input/cached/output differences reported, no claim of statistical significance.',
               'Pilot is separate, cleaned before full arms. Only exact completed experiment namespaces deleted.']))
    import importlib.metadata
    dump(root/'runtime_versions.json', {name: importlib.metadata.version(name)
         for name in ('vllm', 'lmcache', 'torch', 'transformers')})
    (root/'EXPERIMENT_KO.md').write_text(
        '# ShareGPT DRAM 256GiB 규모 실험\n\n'
        'Qwen3-14B BF16 / DRAM 256GiB / GPU staging 8GiB / 청크128 / 동시요청16.\n\n'
        '원본 4턴 이상 대화의 앞 4턴을 사용한다. 마지막 입력4K~8K, 각 입력8K 이하. '
        '원본 답변을 붙인 고정 이력을 재생하며 새 생성 답변을 다음 입력에 넣지 않는다. '
        '패딩·문서 복제 없이 고정 시드로 선택한다. 출력은 원본 답변 토큰 수와256 중 작은 값이 상한이다.\n\n'
        '각 조건은 빈 DRAM·staging과 새 DAOS namespace에서 시작한다. 사전 캐시 채우기는 없다. '
        '대화별 이전 요청이 완료되면 다음 턴 투입이 가능하며 전역 동시성16을 유지한다. '
        'DRAM 프리페치만 OFF/ON, DAOS 프리페치는 모두 ON이다. 용량 외 조기 차단 정책은 없다.\n\n'
        f'고유 입력 KV {cap["unique_input_kv_gib"]:.2f}GiB는 중복 prefix 청크를 제거한 추정치이며 실제 DRAM 점유량이 아니다. '
        'ON/OFF의 실제 hit·재계산·생성량 차이를 별도로 기록한다.\n\n'
        '한 조건 완료·서버 종료 후 해당 UUID namespace의 KV만 삭제한다. 로그·그래프와 다른 namespace는 보존한다.\n')
    dump(root/'status.json', dict(status='prepared', completed=[]))


def storage_guard(case, required_gib=0):
    query = pool_query()
    nvme = check_space(query)
    if required_gib:
        # Account for bookkeeping/placement imbalance; conservative, not a guarantee.
        required = (required_gib*1.10+48)*2**30
        per_target = (required_gib*1.15/query['response']['active_targets']+4)*2**30
        if nvme['free'] < required or nvme['min'] < per_target:
            raise RuntimeError(f'Insufficient headroom for conservative KV estimate {required_gib:.1f}GiB')
    with (case/'pool_checks.jsonl').open('a') as f:
        f.write(json.dumps(dict(time_ns=time.time_ns(), query=query))+'\n')
    return query


def run_case(root, spec, records):
    case = root/spec['name']
    if case.exists():
        raise RuntimeError('Partial case requires diagnosis; never overwrite or mix runs')
    case.mkdir()
    plan = read(root/'plan.json')
    if spec['pilot']:
        records = [dict(r, index=i) for i, r in enumerate(r for r in records if r['session'] < 8)]
    required = 32 if spec['pilot'] else plan['capacity']['conservative_stored_kv_gib']
    storage_guard(case, required)
    cfg = make_config(spec['prefetch'], 'minji-cold-warm-'+uuid.uuid4().hex, cpu_gib=256, staging_gib=8)
    cfg['extra_config'].update({
        'storage_plugin.daosgds.module_path': 'lmcache_daos.capacity_probe_backend',
        'storage_plugin.daosgds.class_name': 'CapacityProbeBackend',
        'daosgds.dram_prefetch_workers': 1,
        'daosgds.dram_prefetch_cancel_queued': False,
        'daosgds.dram_prefetch_early_ready': False,
        'daosgds.dram_prefetch_stop_occupancy_ratio': None})
    (case/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    a = SimpleNamespace(model=plan['model'], max_model_len=16384, max_num_seqs=16, port=8017)
    calls, phase = [], None
    try:
        with server(a, case/'config.yaml', case) as client:
            initial = await_empty(case)
            dump(case/'initial_sample.json', initial)
            health = LogHealth(case/'server.log')
            health()
            (case/'metrics_before.txt').write_text(client.get('/metrics').text)
            phase = dict(start_ns=time.time_ns(), concurrency=spec['concurrency'],
                         arrival_mode='session-aware rolling', name='cold_start_multiturn')
            dump(case/'phase.json', phase)
            with (case/'calls.jsonl').open('x') as journal:
                def invoke(r, barrier):
                    row = replay_one(client, r, barrier, max_tokens=r['max_tokens'], stop=())
                    row.update(session=r['session'], turn=r['turn'], max_tokens=r['max_tokens'],
                               expected_prompt_tokens=r['expected_prompt_tokens'])
                    if 'error' not in row and row['prompt_tokens'] != r['expected_prompt_tokens']:
                        row['error'] = 'Server/tokenizer input count mismatch'
                    return row

                def save(row):
                    calls.append(row)
                    journal.write(json.dumps(row, ensure_ascii=False)+'\n')
                    journal.flush()
                    if len(calls) % 32 == 0 or len(calls) == len(records):
                        sample = latest_sample(case) or {}
                        dump(case/'progress.json', dict(completed=len(calls), total=len(records),
                            sample=sample, updated_ns=time.time_ns()))
                        print(f'{case.name}: {len(calls)}/{len(records)}, '
                              f'DRAM={sample.get("cpu_hot_bytes",0)/2**30:.2f}GiB, '
                              f'staging={sample.get("used_bytes",0)/2**30:.2f}GiB', flush=True)
                    if len(calls) % 128 == 0:
                        storage_guard(case)
                session_requests(records, spec['concurrency'], invoke, save, health)
            phase['end_ns'] = time.time_ns()
            dump(case/'phase.json', phase)
            dump(case/'replay_calls.json', sorted(calls, key=lambda r:r['index']))
            (case/'metrics_after.txt').write_text(client.get('/metrics').text)
            final = drain(case, health, timeout=180)
            dump(case/'final_sample.json', final)
            if final['daos_puts'] == 0:
                raise RuntimeError('No DAOS stores observed')
            if spec['pilot']:
                if not any(c.get('cached_tokens', 0) for c in calls if c['turn'] > 0):
                    raise RuntimeError('Pilot saw no follow-up cache reuse')
                if not (final.get('cpu_prefetch') or {}).get('staged_bytes', 0):
                    raise RuntimeError('Pilot DRAM prefetch was never exercised')
        dump(case/'status.json', dict(status='completed', requests=len(calls), updated_ns=time.time_ns()))
    except BaseException as exc:
        dump(case/'replay_calls.json', sorted(calls, key=lambda r:r['index']))
        if phase:
            phase.update(end_ns=time.time_ns(), failed=True)
            dump(case/'phase.json', phase)
        dump(case/'status.json', dict(status='failed', error=repr(exc), completed=len(calls)))
        raise


def report_case(case):
    calls, phase = read(case/'replay_calls.json'), read(case/'phase.json')
    events = read_events(case)
    start, end = phase['start_ns'], phase['end_ns']
    scoped = [e for e in events if start <= e['time_ns'] <= end]
    samples = [e for e in scoped if e['event'] == 'occupancy_sample']
    selected = select_events(events, calls)
    rows = join(calls, selected)
    assert not any('error' in c for c in calls)
    assert not any(r['other_failed_chunks'] for r in rows), 'Non-capacity DAOS read failures'
    q = sum(e['queried_chunks'] for e in selected if e['event'] == 'tier_lookup' and e['tier'] == 'dram')
    hits = {tier: sum(e['hit_chunks'] for e in selected if e['event'] == 'tier_lookup' and e['tier'] == tier)
            for tier in ('dram', 'daos')}
    assert q > 0 and 0 <= sum(hits.values()) <= q
    inp = sum(c['prompt_tokens'] for c in calls)
    initial, final = read(case/'initial_sample.json'), read(case/'final_sample.json')
    assert initial['used_bytes'] == initial['cpu_hot_bytes'] == initial['daos_puts'] == 0
    assert final['used_bytes'] == final['dram_mirror']['pending_bytes'] == final['dram_mirror']['errors'] == 0
    result = dict(case=case.name, requests=len(calls), elapsed_seconds=(end-start)/1e9,
        ttft=stats(c['ttft_ms'] for c in calls), input_tokens=inp,
        cached_tokens=sum(c['cached_tokens'] for c in calls),
        computed_tokens=inp-sum(c['cached_tokens'] for c in calls),
        output_tokens=sum(c['completion_tokens'] for c in calls),
        queried_chunks=q, hit_chunks=hits, dram_hit_pct=100*hits['dram']/q, daos_hit_pct=100*hits['daos']/q,
        input_reuse_pct=100*sum(c['cached_tokens'] for c in calls)/inp,
        peak_staging_gib=max(e['used_bytes'] for e in scoped)/2**30,
        sampled_peak_cpu_gib=max(e['cpu_hot_bytes'] for e in samples)/2**30,
        final_cpu_gib=final['cpu_hot_bytes']/2**30,
        mean_sampled_staging_gib=sum(e['used_bytes'] for e in samples)/len(samples)/2**30,
        sample_fraction_ge_60pct=sum(e['used_bytes'] >= .6*8*2**30 for e in samples)/len(samples),
        sample_fraction_ge_90pct=sum(e['used_bytes'] >= .9*8*2**30 for e in samples)/len(samples),
        capacity_recomputed_tokens=sum(r['capacity_recomputed_tokens'] for r in rows),
        prefetch_counters=final.get('cpu_prefetch'), mirror_counters=final.get('dram_mirror'),
        notes=['Hit ratios are lookup candidate chunk ratios; input reuse is actual server cached_tokens.',
               'Staging includes DAOS, DRAM prefetch and store lifetimes, not just DRAM reads.',
               'Sample averages are approximate (20ms sampler), not exact time-weighted GPU utilization.'])
    dump(case/'timing_by_request.json', rows)
    dump(case/'summary.json', result)
    timeline(case, events, start, end, 8)
    # Additional DRAM-residency graph: same 2-second bins as staging plot.
    bins = {}
    for e in samples:
        b = int((e['time_ns']-start)/2e9)
        bins.setdefault(b, []).append(e['cpu_hot_bytes']/2**30)
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="330">',
           '<rect width="1000" height="330" fill="white"/><g font-family="sans-serif" font-size="13">',
           f'<text x="55" y="28">{case.name}: DRAM residency (GiB), 2-second sampled peak</text>']
    for v in (0,64,128,192,256):
        y = 260-200*v/256
        svg += [f'<line x1="65" x2="970" y1="{y}" y2="{y}" stroke="#ddd"/>',
                f'<text x="20" y="{y+4}">{v}</text>']
    duration = (end-start)/1e9
    points = ' '.join(f'{65+905*min(2*b+1,duration)/duration:.2f},{260-200*max(v)/256:.2f}'
                      for b,v in sorted(bins.items()))
    svg += [f'<polyline points="{points}" stroke="#0072b2" fill="none" stroke-width="1.5"/>']
    for i in range(5):
        svg += [f'<text x="{65+905*i/4}" y="285">{duration*i/4:.0f}s</text>']
    svg += ['<text x="60" y="315">Configured DRAM cache capacity: 256GiB. This is resident KV, not process RSS.</text></g></svg>']
    save_chart(case, 'dram_residency', svg)
    return result


def report(root):
    plan = read(root/'plan.json')
    results = [read(root/s['name']/'summary.json') for s in plan['cases']
               if not s['pilot'] and (root/s['name']/'summary.json').exists()]
    dump(root/'summary.json', results)
    lines = ['# ShareGPT DRAM256 / staging8 / C16 결과', '',
             '각 조건은 빈 캐시에서 시작하는 원본 이력 재생 1회. 별도 warm 측정은 없음. DAOS 프리페치는 항상 ON.', '',
             '|정책|요청|전체 s|평균 TTFT ms|p95 ms|DRAM hit %|DAOS hit %|입력 재사용 %|최대 staging GiB|최대 DRAM GiB|',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in results:
        lines.append(f'|[{s["case"]}]({s["case"]}/staging_hits.png)|{s["requests"]}|{s["elapsed_seconds"]:.2f}|'
            f'{s["ttft"]["mean"]:.2f}|{s["ttft"]["p95"]:.2f}|{s["dram_hit_pct"]:.2f}|{s["daos_hit_pct"]:.2f}|'
            f'{s["input_reuse_pct"]:.2f}|{s["peak_staging_gib"]:.3f}|{s["sampled_peak_cpu_gib"]:.2f}|')
    lines += ['', '## 그래프', '']
    for s in results:
        name = s['case']
        lines += [f'- {name}: [staging·hit 비율]({name}/staging_hits.png), [DRAM 점유]({name}/dram_residency.png)']
    if len(results) == 2:
        configs, libraries = [], []
        for name in ('c16_off', 'c16_on'):
            cfg = yaml.safe_load((root/name/'config.yaml').read_text())
            ec = cfg['extra_config']
            assert ec.pop('daosgds.dram_prefetch') == (name == 'c16_on')
            ec.pop('daosgds.object_namespace')
            ec.pop('daosgds.root')
            configs.append(cfg)
            libraries.append(read(root/name/'native_maps.json'))
        assert configs[0] == configs[1], 'Unexpected server configuration difference'
        assert libraries[0] == libraries[1], 'Native library difference'
        left, right = [read(root/n/'replay_calls.json') for n in ('c16_off','c16_on')]
        assert len(left) == len(right)
        assert all((a['index'],a['prompt_sha256'],a['prompt_tokens'],a['max_tokens']) ==
                   (b['index'],b['prompt_sha256'],b['prompt_tokens'],b['max_tokens'])
                   for a,b in zip(left,right))
        paired = dict(requests=len(left), same_cached_requests=sum(a['cached_tokens']==b['cached_tokens'] for a,b in zip(left,right)),
            same_output_requests=sum(a['output_sha256']==b['output_sha256'] for a,b in zip(left,right)))
        dump(root/'paired_checks.json', paired)
        lines += ['', f'같은 요청별 cached_tokens: {paired["same_cached_requests"]}/{len(left)}. '
                  f'같은 생성 결과: {paired["same_output_requests"]}/{len(left)}.',
                  'TTFT 차이는 캐시 재사용량·계산량·출력 차이와 함께 해석해야 한다. 각 조건1회 예비실험이다.']
    (root/'RESULT_KO.md').write_text('\n'.join(lines)+'\n')


def run(root):
    import fcntl
    with (root/'runner.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read(root/'plan.json')
        records = read(root/'requests.json')
        for name, value in plan['source_sha256'].items():
            assert digest(ROOT/name) == value, f'Source changed: {name}'
        assert digest(root/'requests.json') == plan['request_sha256']
        # Never disturb another GPU workload.
        gpu = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True)
        if gpu.strip():
            raise RuntimeError('GPU is occupied; no unrelated process will be stopped')
        mem = {line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
        if mem['MemAvailable'] < 320*2**30:
            raise RuntimeError('Insufficient host memory headroom for DRAM256')
        os.environ['DAOS_GDS_PREFETCH_TIMING'] = '1'
        completed = []
        current = None
        try:
            if not (root/'storage_preflight.log').exists():
                probe = subprocess.run([str(ROOT/'run_vllm.sh'), str(ROOT/'venv/bin/python3'),
                    str(ROOT/'tests/object_gpu_roundtrip.py'), '--pool', 'discospool',
                    '--container', 'kvcache', '--size-mib', '20'], cwd=ROOT,
                    env=dict(os.environ, DAOSGDS_TRANSPORT='object'),
                    capture_output=True, text=True, timeout=120)
                (root/'storage_preflight.log').write_text(probe.stdout+probe.stderr)
                if probe.returncode:
                    raise RuntimeError('DAOS GPU roundtrip preflight failed')
            for spec in plan['cases']:
                current = spec['name']
                dump(root/'status.json', dict(status='running', current=current, completed=completed, updated_ns=time.time_ns()))
                case = root/current
                if not (case/'status.json').exists() or read(case/'status.json')['status'] != 'completed':
                    run_case(root, spec, records)
                if not (case/'summary.json').exists():
                    report_case(case)
                report(root)
                # Reuse audited exact-namespace cleanup; its assertions verify
                # completed case, parent plan, config, library and preserved keys.
                with (case/'cleanup.log').open('a') as stream:
                    subprocess.run([str(ROOT/'run_vllm.sh'), str(ROOT/'venv/bin/python3'),
                        str(ROOT/'discovery_occupancy_gate.py'), '--cleanup-case', str(case)],
                        cwd=ROOT, env=dict(os.environ, DAOSGDS_TRANSPORT='object'),
                        stdout=stream, stderr=subprocess.STDOUT, timeout=900, check=True)
                completed.append(current)
            dump(root/'status.json', dict(status='completed', completed=completed, updated_ns=time.time_ns()))
        except BaseException as exc:
            dump(root/'status.json', dict(status='failed', current=current, completed=completed,
                                         error=repr(exc), updated_ns=time.time_ns()))
            raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prepare-from', type=Path)
    p.add_argument('--report', action='store_true')
    a = p.parse_args()
    root = a.output.resolve()
    assert root.parent == ROOT
    if a.prepare_from:
        prepare(root, a.prepare_from.resolve())
    elif a.report:
        report(root)
    else:
        run(root)


if __name__ == '__main__':
    main()
