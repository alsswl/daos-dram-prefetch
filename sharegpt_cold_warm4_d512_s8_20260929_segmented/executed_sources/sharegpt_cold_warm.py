#!/usr/bin/env python3
"""ShareGPT: identical original-history replay cold then warm, per OFF/ON arm.

Reuse the earlier corpus and session-aware admission without changing backends.
Warm retains the live cold process and cache; no synthetic placement or padding.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace
import uuid

import yaml

import sharegpt_scale_experiment as base

ROOT = base.ROOT
read, dump = base.read, base.dump


def phase_names(warm_repeats):
    if type(warm_repeats) is not int or warm_repeats < 1:
        raise ValueError('warm_repeats must be a positive integer')
    return ['cold', 'warm'] if warm_repeats == 1 else ['cold'] + [f'warm{i}' for i in range(1, warm_repeats+1)]


def prepare(root, data, cpu_gib=256, warm_repeats=1, numa_interleave=False, segmented_pinned=False):
    phases = phase_names(warm_repeats)
    if cpu_gib not in (256, 512):
        raise ValueError('Supported DRAM sizes: 256 or 512 GiB')
    if segmented_pinned and cpu_gib != 512:
        raise ValueError('Segmented registration is only implemented for the 512GiB pool')
    base.prepare(root, data)
    plan = read(root/'plan.json')
    plan['cases'] = [s for s in plan['cases'] if not s['pilot']]
    plan.update(phases=phases, cpu_gib=cpu_gib, warm_repeats=warm_repeats, numa_interleave=numa_interleave,
                segmented_pinned=segmented_pinned,
                requests_per_phase=plan['requests_per_case'],
                requests_per_case=len(phases)*plan['requests_per_case'])
    plan['notes'][0] = 'Cold fill followed by identical warm replay in the SAME process and namespace.'
    plan['notes'][-1] = ('No new pilot: prior full cold test passed. Delete only exact completed '
                         'experiment namespace after ALL phases and server shutdown.')
    plan['notes'] += ['Staging/mirror drain between phases; DRAM and DAOS retained, no reset.',
                      'Warm hit placement is measured, not assumed; no resets between warm repeats.',
                      'Neither OS nor DAOS server caches are flushed.',
                      'Warm repeats share evolving caches; these are not independent cold-start trials.',
                      'Initial space guard covers cold+one warm allowance; before each later warm, '
                      'check additional suffix allowance and runtime free-space guards. No mid-arm cleanup.']
    if numa_interleave:
        plan['notes'].append('Both arms inherit process-local NUMA interleave nodes0,1. '
                             'The original 256GiB experiment used the default policy; '
                             'cross-capacity comparison also changes memory placement.')
    if segmented_pinned:
        plan['notes'].append('512GiB contiguous host tensor registered with CUDA in 7.8125GiB regions '
                             '(400 x 20MiB KV chunks), checked not to cross registration boundaries; '
                             'identical in OFF/ON, no pageable fallback, unchanged LRU/prefetch policy. '
                             'Opt-in local vLLM general plugin; installed library files unchanged.')
    for name in ('sharegpt_cold_warm.py', 'tests/test_sharegpt_cold_warm.py',
                 'with_numa_interleave.py', 'tests/pinned_512_probe.py', 'segmented_pinned.py',
                 'tests/segmented_pinned_probe.py', 'tests/test_segmented_pinned.py',
                 'experiment_plugins/segmented_pinned_hook-0.1.dist-info/METADATA',
                 'experiment_plugins/segmented_pinned_hook-0.1.dist-info/entry_points.txt'):
        dest = root/'executed_sources'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, dest)
        plan['source_sha256'][name] = base.digest(dest)
    dump(root/'plan.json', plan)
    (root/'EXPERIMENT_KO.md').write_text(
        '# ShareGPT cold → warm 비교\n\n'
        f'Qwen3-14B BF16 / DRAM{cpu_gib}GiB / staging8GiB / 동시요청16 / 청크128.\n\n'
        '기존 421개 대화 × 앞4턴 = 1,684개 고정 입력을 그대로 재생한다. '
        '원본 답변으로 구성된 이력이며 생성한 답변은 다음 입력에 반영하지 않는다. '
        '출력 상한은 기존과 동일하게 원본 답변 길이와256 중 작은 값이다.\n\n'
        'OFF, ON 각각 새 프로세스·빈 DRAM·새 DAOS namespace로 cold 1,684개를 실행한다. '
        '비동기 쓰기/복사와 staging 해제를 기다린 뒤, 같은 프로세스와 캐시를 유지한 채 '
        f'동일한 요청 순서를 warm {warm_repeats}회, 매회 1,684개씩 재생한다. '
        'warm 사이에도 캐시를 초기화하지 않는다. 따라서 warm 반복은 독립 실험이 아니라 연속 재사용이다. '
        '대화별 순서는 유지하고 전역16개 이하를 rolling 방식으로 투입한다. '
        '캐시를 임의로 특정 계층에 배치하지 않으며 실제 DRAM/DAOS hit를 측정한다.\n\n'
        'DRAM 프리페치만 OFF/ON이고 DAOS 프리페치는 항상 ON이다. '
        '복사 작업자1, 대기열 취소OFF, 점유율 기반 조기차단 없음. '
        'vLLM prefix caching OFF, max-num-seqs16, max-model-len16384.\n\n'
        '각 단계의 TTFT·입력재사용·재계산·출력량·staging 점유를 별도로 기록한다. '
        '1회씩의 예비 비교이며 벽시계 도착 시각과 생성 결과, 계층별 캐시 배치는 달라질 수 있다. '
        'OS/서버 캐시 강제 초기화는 하지 않는다. '
        '전체 단계 완료 후 해당 조건의 UUID namespace KV만 삭제하며 로그·그래프는 유지한다.\n'
        + ('\nOFF/ON 모두 실험 프로세스에 NUMA interleave(노드0,1)를 적용한다. '
           '기존256GiB 결과와의 비교에는 메모리 배치 차이도 포함된다.\n' if numa_interleave else '')
        + ('\n512GiB CPU 텐서의 CUDA 고정 메모리 등록만7.8125GiB(20MiB 청크400개) 구간으로 분할한다. '
           'OFF/ON 동일하게 적용하며 LRU·캐시 크기·프리페치 정책은 유지한다. '
           '설치된 라이브러리 파일은 수정하지 않고 명시적 환경변수가 켜진 실험에서만 적용한다.\n'
           if segmented_pinned else ''))


def invoke(client, record, barrier):
    row = base.replay_one(client, record, barrier, max_tokens=record['max_tokens'], stop=())
    row.update(session=record['session'], turn=record['turn'], max_tokens=record['max_tokens'],
               expected_prompt_tokens=record['expected_prompt_tokens'])
    if 'error' not in row and row['prompt_tokens'] != record['expected_prompt_tokens']:
        row['error'] = 'Server/tokenizer input count mismatch'
    return row


def run_phases(case, records, concurrency, client, health, initial,
               phases=('cold', 'warm'), warm_extra_gib=0):
    previous = initial
    for name in phases:
        # New generated suffixes may differ despite fixed input histories.
        # Do not reset caches to make space in the middle of an arm.
        if name != 'cold' and warm_extra_gib:
            base.storage_guard(case, warm_extra_gib)
        folder = case/name
        folder.mkdir()
        dump(case/'status.json', dict(status='running', phase=name))
        dump(folder/'initial_sample.json', previous)
        (folder/'metrics_before.txt').write_text(client.get('/metrics').text)
        phase = dict(name=name, concurrency=concurrency,
                     arrival_mode='session-aware rolling', start_ns=time.time_ns())
        dump(folder/'phase.json', phase)
        calls = []
        try:
            with (folder/'calls.jsonl').open('x') as journal:
                def save(row):
                    calls.append(row)
                    journal.write(json.dumps(row, ensure_ascii=False)+'\n')
                    journal.flush()
                    if len(calls) % 32 == 0 or len(calls) == len(records):
                        sample = base.latest_sample(case) or {}
                        dump(case/'progress.json', dict(phase=name, completed=len(calls),
                            total=len(records), sample=sample, updated_ns=time.time_ns()))
                        print(f'{case.name}/{name}: {len(calls)}/{len(records)}, '
                              f'DRAM={sample.get("cpu_hot_bytes",0)/2**30:.2f}GiB, '
                              f'staging={sample.get("used_bytes",0)/2**30:.2f}GiB', flush=True)
                    if len(calls) % 128 == 0:
                        base.storage_guard(case)
                base.session_requests(records, concurrency,
                    lambda r, b: invoke(client, r, b), save, health)
            phase['end_ns'] = time.time_ns()
            dump(folder/'phase.json', phase)
            dump(folder/'replay_calls.json', sorted(calls, key=lambda r: r['index']))
            (folder/'metrics_after.txt').write_text(client.get('/metrics').text)
            previous = base.drain(case, health, timeout=180)
            dump(folder/'final_sample.json', previous)
            assert previous['daos_puts'] > 0, 'No DAOS stores observed'
            dump(folder/'status.json', dict(status='completed', requests=len(calls)))
            print(f'{case.name}/{name}: completed and drained', flush=True)
        except BaseException as exc:
            dump(folder/'replay_calls.json', sorted(calls, key=lambda r: r['index']))
            phase.update(end_ns=time.time_ns(), failed=True)
            dump(folder/'phase.json', phase)
            dump(folder/'status.json', dict(status='failed', error=repr(exc)))
            raise
    return previous


def wait_space(case, required, timeout=900):
    """Wait only for physical DAOS reclamation; no forced GC or extra deletion."""
    deadline = time.monotonic()+timeout
    while True:
        try:
            return base.storage_guard(case, required)
        except RuntimeError as exc:
            if 'headroom' not in str(exc) or time.monotonic() >= deadline:
                raise
            print(f'{case.name}: waiting for DAOS free space: {exc}', flush=True)
            time.sleep(20)


def run_case(root, spec, records):
    case = root/spec['name']
    case.mkdir(exist_ok=False)
    plan = read(root/'plan.json')
    # Count output allowance twice: warm generations can introduce new suffix keys.
    cap = plan['capacity']
    required = 2*cap['conservative_stored_kv_gib']-cap['unique_input_kv_gib']
    wait_space(case, required)
    if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                '--format=csv,noheader'], text=True).strip():
        raise RuntimeError('GPU occupied; unrelated workloads will not be stopped')
    cfg = base.make_config(spec['prefetch'], 'minji-cold-warm-'+uuid.uuid4().hex,
                           cpu_gib=plan['cpu_gib'], staging_gib=8)
    cfg['extra_config'].update({
        'storage_plugin.daosgds.module_path': 'lmcache_daos.capacity_probe_backend',
        'storage_plugin.daosgds.class_name': 'CapacityProbeBackend',
        'daosgds.dram_prefetch_workers': 1, 'daosgds.dram_prefetch_cancel_queued': False,
        'daosgds.dram_prefetch_early_ready': False,
        'daosgds.dram_prefetch_stop_occupancy_ratio': None})
    (case/'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    args = SimpleNamespace(model=plan['model'], max_model_len=16384, max_num_seqs=16, port=8017)
    try:
        with base.server(args, case/'config.yaml', case) as client:
            initial = base.await_empty(case)
            assert initial['used_bytes'] == initial['cpu_hot_bytes'] == initial['daos_puts'] == 0
            dump(case/'initial_sample.json', initial)
            health = base.LogHealth(case/'server.log')
            health()
            final = run_phases(case, records, spec['concurrency'], client, health, initial,
                phases=plan['phases'],
                warm_extra_gib=cap['conservative_stored_kv_gib']-cap['unique_input_kv_gib'])
            dump(case/'final_sample.json', final)
        dump(case/'status.json', dict(status='completed', requests=len(plan['phases'])*len(records)))
    except BaseException as exc:
        dump(case/'status.json', dict(status='failed', error=repr(exc)))
        raise


def summarize_case(root, spec):
    case = root/spec['name']
    records = read(root/'requests.json')
    events = base.read_events(case)
    assert len({e['pid'] for e in events}) == 1, 'Process changed between cold/warm'
    initial = read(case/'initial_sample.json')
    assert initial['used_bytes'] == initial['cpu_hot_bytes'] == initial['daos_puts'] == 0
    results, seen, previous = [], set(), initial
    for name in read(root/'plan.json')['phases']:
        folder = case/name
        calls, phase = read(folder/'replay_calls.json'), read(folder/'phase.json')
        assert len(calls) == len(records) and not any('error' in c for c in calls)
        assert [(c['index'], c['prompt_sha256'], c['prompt_tokens'], c['max_tokens']) for c in calls] == [
            (r['index'], r['prompt_sha256'], r['expected_prompt_tokens'], r['max_tokens']) for r in records]
        ids = {c['server_request_id'] for c in calls}
        assert len(ids) == len(calls) and not ids & seen
        seen.update(ids)
        before, after = read(folder/'initial_sample.json'), read(folder/'final_sample.json')
        assert before == previous
        previous = after
        assert before['used_bytes'] == after['used_bytes'] == after['dram_mirror']['pending_bytes'] == 0
        assert after['dram_mirror']['errors'] == 0
        selected = base.select_events(events, calls)
        rows = base.join(calls, selected)
        assert not any(r['other_failed_chunks'] for r in rows)
        q = sum(e['queried_chunks'] for e in selected if e['event'] == 'tier_lookup' and e['tier'] == 'dram')
        hits = {t: sum(e['hit_chunks'] for e in selected if e['event'] == 'tier_lookup' and e['tier'] == t)
                for t in ('dram', 'daos')}
        assert q > 0 and 0 <= sum(hits.values()) <= q
        start, end = phase['start_ns'], phase['end_ns']
        scoped = [e for e in events if start <= e['time_ns'] <= end]
        samples = [e for e in scoped if e['event'] == 'occupancy_sample']
        inp = sum(c['prompt_tokens'] for c in calls)
        cached = sum(c['cached_tokens'] for c in calls)
        pf = {k: v-(before.get('cpu_prefetch') or {}).get(k, 0)
              for k,v in (after.get('cpu_prefetch') or {}).items()}
        assert pf.get('copy_errors', 0) == 0
        s = dict(case=case.name, phase=name, requests=len(calls),
            elapsed_seconds=(end-start)/1e9, ttft=base.stats(c['ttft_ms'] for c in calls),
            input_tokens=inp, cached_tokens=cached, computed_tokens=inp-cached,
            output_tokens=sum(c['completion_tokens'] for c in calls),
            queried_chunks=q, hit_chunks=hits, dram_hit_pct=100*hits['dram']/q,
            daos_hit_pct=100*hits['daos']/q, input_reuse_pct=100*cached/inp,
            peak_staging_gib=max(e['used_bytes'] for e in scoped)/2**30,
            mean_sampled_staging_gib=sum(e['used_bytes'] for e in samples)/len(samples)/2**30,
            sampled_peak_cpu_gib=max(e['cpu_hot_bytes'] for e in samples)/2**30,
            initial_cpu_gib=before['cpu_hot_bytes']/2**30, final_cpu_gib=after['cpu_hot_bytes']/2**30,
            capacity_recomputed_tokens=sum(r['capacity_recomputed_tokens'] for r in rows),
            prefetch_counter_deltas=pf)
        dump(folder/'timing_by_request.json', rows)
        dump(folder/'summary.json', s)
        base.timeline(folder, events, start, end, 8)
        results.append(s)
    dump(case/'summary.json', results)


def report(root):
    plan = read(root/'plan.json')
    results = []
    for spec in read(root/'plan.json')['cases']:
        path = root/spec['name']/'summary.json'
        if path.exists():
            results.extend(read(path))
    dump(root/'summary.json', results)
    lines = ['# ShareGPT cold / warm 비교', '',
        f'Qwen3-14B, DRAM{plan["cpu_gib"]}GiB / staging8GiB / 동시요청16. DRAM 프리페치만 OFF/ON.', '',
        '|정책|단계|요청|전체 s|평균 TTFT ms|p95 ms|DRAM hit %|DAOS hit %|입력 재사용 %|최대 staging GiB|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s in results:
        lines.append(f'|{s["case"]}|[{s["phase"]}]({s["case"]}/{s["phase"]}/staging_hits.png)|'
            f'{s["requests"]}|{s["elapsed_seconds"]:.2f}|{s["ttft"]["mean"]:.2f}|{s["ttft"]["p95"]:.2f}|'
            f'{s["dram_hit_pct"]:.2f}|{s["daos_hit_pct"]:.2f}|{s["input_reuse_pct"]:.2f}|{s["peak_staging_gib"]:.3f}|')
    if len(results) == len(plan['cases'])*len(plan['phases']):
        configs, libs = [], []
        for name in ('c16_off', 'c16_on'):
            cfg = yaml.safe_load((root/name/'config.yaml').read_text())
            ec = cfg['extra_config']
            assert ec.pop('daosgds.dram_prefetch') == (name == 'c16_on')
            ec.pop('daosgds.object_namespace'); ec.pop('daosgds.root')
            configs.append(cfg)
            libs.append(read(root/name/'native_maps.json'))
        assert configs[0] == configs[1] and libs[0] == libs[1]
        pairs = {}
        for name in plan['phases']:
            a, b = [read(root/m/name/'replay_calls.json') for m in ('c16_off', 'c16_on')]
            assert [(c['index'], c['prompt_sha256'], c['prompt_tokens'], c['max_tokens']) for c in a] == [
                (c['index'], c['prompt_sha256'], c['prompt_tokens'], c['max_tokens']) for c in b]
            pairs[name] = dict(requests=len(a), same_cached_requests=sum(
                x['cached_tokens'] == y['cached_tokens'] for x,y in zip(a,b)),
                same_output_requests=sum(x['output_sha256'] == y['output_sha256'] for x,y in zip(a,b)))
        dump(root/'paired_checks.json', pairs)
        lines += ['', '```json', json.dumps(pairs, indent=2), '```']
    lines += ['', 'warm은 해당 조건의 cold 완료 후 같은 프로세스·DRAM·DAOS 캐시를 유지한 재생이다. '
        '각 단계 사이 staging과 비동기 쓰기만 drain한다. warm 반복 사이에도 캐시를 초기화하지 않는다. '
        'DRAM/DAOS 배치는 실측하며 전부 DRAM hit라고 가정하지 않는다.',
        'hit 비율은 최초 lookup 후보 청크 기준, 입력 재사용률은 서버 cached_tokens 기준이다. '
        '그래프는20ms 샘플의2초 구간 최대/평균이며 짧은 event peak는 표에만 잡힐 수 있다. '
        'staging 점유에는 DAOS/DRAM 읽기와 쓰기가 모두 포함된다.',
        'warm 반복은 캐시를 유지한 연속 재사용이며 독립 시행이 아니다. 도착 시각·생성량·계층별 캐시 배치는 다를 수 있다. '
        '입력/캐시/출력 일치를 함께 확인한다. '
        '시작·초기화·drain·삭제 시간은 요청 처리 시간에서 제외한다. OS/서버 캐시를 강제로 비우지 않는다.']
    (root/'RESULT_KO.md').write_text('\n'.join(lines)+'\n')


def run(root):
    with (root/'runner.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = read(root/'plan.json')
        if plan.get('numa_interleave'):
            from with_numa_interleave import current_policy
            policy = current_policy()
            assert policy['mode'] == 3 and policy['node_mask'] == 3, 'Required NUMA policy not active'
            dump(root/'numa_policy.json', policy)
        if plan.get('segmented_pinned'):
            from importlib.metadata import entry_points
            assert os.environ.get('DAOS_SEGMENTED_PINNED') == '1'
            assert any(ep.name == 'segmented_pinned' for ep in entry_points(group='vllm.general_plugins'))
        for name, sha in plan['source_sha256'].items():
            assert base.digest(ROOT/name) == sha, f'Source changed: {name}'
        assert base.digest(root/'requests.json') == plan['request_sha256']
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True).strip():
            raise RuntimeError('GPU occupied; no unrelated process will be stopped')
        mem = {s.split(':')[0]: int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()}
        assert mem['MemAvailable'] >= (plan['cpu_gib']+64)*2**30, 'Insufficient host memory'
        os.environ['DAOS_GDS_PREFETCH_TIMING'] = '1'
        completed, current = [], None
        try:
            if not (root/'storage_preflight.log').exists():
                p = subprocess.run([str(ROOT/'run_vllm.sh'), str(ROOT/'venv/bin/python3'),
                    str(ROOT/'tests/object_gpu_roundtrip.py'), '--pool', 'discospool',
                    '--container', 'kvcache', '--size-mib', '20'], cwd=ROOT,
                    env=dict(os.environ, DAOSGDS_TRANSPORT='object'), capture_output=True, text=True, timeout=120)
                (root/'storage_preflight.log').write_text(p.stdout+p.stderr)
                assert p.returncode == 0, 'DAOS GPU roundtrip failed'
            for spec in plan['cases']:
                current = spec['name']
                case = root/current
                dump(root/'status.json', dict(status='running', current=current, completed=completed,
                                             updated_ns=time.time_ns()))
                if not (case/'status.json').exists() or read(case/'status.json')['status'] != 'completed':
                    run_case(root, spec, read(root/'requests.json'))
                if not (case/'summary.json').exists():
                    summarize_case(root, spec)
                report(root)
                with (case/'cleanup.log').open('a') as stream:
                    subprocess.run([str(ROOT/'run_vllm.sh'), str(ROOT/'venv/bin/python3'),
                        str(ROOT/'discovery_occupancy_gate.py'), '--cleanup-case', str(case)], cwd=ROOT,
                        env=dict(os.environ, DAOSGDS_TRANSPORT='object'), stdout=stream,
                        stderr=subprocess.STDOUT, timeout=900, check=True)
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
    p.add_argument('--cpu-gib', type=int, choices=(256, 512), default=256,
                   help='Preparation only; runtime uses the immutable plan')
    p.add_argument('--warm-repeats', type=int, default=1,
                   help='Preparation only; no cache reset between repeats')
    p.add_argument('--numa-interleave', action='store_true',
                   help='Preparation: require process-local interleave nodes0,1 at runtime')
    p.add_argument('--segmented-pinned', action='store_true',
                   help='Preparation: require the opt-in bounded CUDA registration plugin')
    a = p.parse_args()
    root = a.output.resolve()
    assert root.parent == ROOT
    if a.prepare_from:
        prepare(root, a.prepare_from.resolve(), a.cpu_gib, a.warm_repeats, a.numa_interleave, a.segmented_pinned)
    elif a.report:
        report(root)
    else:
        run(root)


if __name__ == '__main__':
    main()
