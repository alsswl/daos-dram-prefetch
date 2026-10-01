"""Metadata-first attribution: do not use completion-first prefetch timing join."""
from collections import Counter
from pathlib import Path
import yaml
import sharegpt_scale_experiment as base

read, dump = base.read, base.dump


def validate_events(events):
    notified, decisions = {}, Counter()
    for e in events:
        name, rid = e['event'], e.get('request_id')
        if name == 'early_lookup_notify':
            assert rid not in notified, 'Duplicate lookup notification'
            notified[rid] = e
        elif name in ('early_payload_start', 'early_daos_demand_start', 'early_retrieve_decision'):
            assert rid in notified, 'Read/resolve started before existence notification'
            assert notified[rid]['monotonic_ns'] <= e['monotonic_ns']
            if name == 'early_retrieve_decision':
                assert e['decision'] in ('ready', 'wait_running', 'queued_to_demand')
                decisions[e['tier']+'/'+e['decision']] += 1
        elif name in ('early_lookup_error', 'prefetch_error'):
            raise AssertionError(f'Early lookup failed: {e}')
    return dict(decisions)


def summarize_case(root, spec):
    case = root/spec['name']
    records, plan = read(root/'requests.json'), read(root/'plan.json')
    events = base.read_events(case)
    assert len({e['pid'] for e in events}) == 1
    assert any(e['event'] == 'early_lookup_enabled' for e in events)
    validate_events(events)
    previous = read(case/'initial_sample.json')
    assert previous['used_bytes'] == previous['cpu_hot_bytes'] == previous['daos_puts'] == 0
    results, seen = [], set()
    for name in plan['phases']:
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
        assert after['daos_alloc_fail'] == before['daos_alloc_fail'], 'GPU allocation failure'
        assert after['cpu_prefetch']['copy_errors'] == 0
        selected = base.select_events(events, calls)
        decisions = validate_events(selected)
        q = sum(e['queried_chunks'] for e in selected if e['event'] == 'tier_lookup' and e['tier'] == 'dram')
        hits = {t: sum(e['hit_chunks'] for e in selected if e['event'] == 'tier_lookup' and e['tier'] == t)
                for t in ('dram', 'daos')}
        assert q > 0 and 0 <= sum(hits.values()) <= q
        start, end = phase['start_ns'], phase['end_ns']
        scoped = [e for e in events if start <= e['time_ns'] <= end]
        samples = [e for e in scoped if e['event'] == 'occupancy_sample']
        inp, cached = sum(c['prompt_tokens'] for c in calls), sum(c['cached_tokens'] for c in calls)
        s = dict(case=case.name, phase=name, requests=len(calls), elapsed_seconds=(end-start)/1e9,
            ttft=base.stats(c['ttft_ms'] for c in calls), input_tokens=inp, cached_tokens=cached,
            computed_tokens=inp-cached, output_tokens=sum(c['completion_tokens'] for c in calls),
            queried_chunks=q, hit_chunks=hits, dram_hit_pct=100*hits['dram']/q,
            daos_hit_pct=100*hits['daos']/q, input_reuse_pct=100*cached/inp,
            peak_staging_gib=max(e['used_bytes'] for e in scoped)/2**30,
            mean_sampled_staging_gib=sum(e['used_bytes'] for e in samples)/len(samples)/2**30,
            initial_cpu_gib=before['cpu_hot_bytes']/2**30, final_cpu_gib=after['cpu_hot_bytes']/2**30,
            retrieve_decisions=decisions,
            resolve_wait_ms={tier: base.stats(e['wait_ms'] for e in selected
                if e['event'] == 'early_retrieve_decision' and e['tier'] == tier)
                for tier in ('dram', 'daos')})
        dump(folder/'summary.json', s)
        dump(folder/'retrieve_decisions.json', [e for e in selected if e['event'] == 'early_retrieve_decision'])
        base.timeline(folder, events, start, end, 8)
        results.append(s)
    dump(case/'summary.json', results)


def report(root):
    plan = read(root/'plan.json')
    results = read(root/'c16_early/summary.json')
    dump(root/'summary.json', results)
    old_root = Path(plan['baseline'])
    old = read(old_root/'c16_on/summary.json')
    configs = []
    for directory, early in ((old_root/'c16_on', False), (root/'c16_early', True)):
        cfg = yaml.safe_load((directory/'config.yaml').read_text())
        ec = cfg['extra_config']
        assert ec['daosgds.dram_prefetch'] is True and cfg['enable_async_loading'] is True
        assert ec.pop('daosgds.early_lookup', False) is early
        for key in ('storage_plugin.daosgds.module_path', 'storage_plugin.daosgds.class_name',
                    'daosgds.object_namespace', 'daosgds.root'):
            ec.pop(key)
        configs.append(cfg)
    assert configs[0] == configs[1], 'Unexpected config difference from baseline ON'
    assert read(old_root/'c16_on/command.json') == read(root/'c16_early/command.json')
    assert read(old_root/'c16_on/native_maps.json') == read(root/'c16_early/native_maps.json')
    pairs = {}
    for phase in plan['phases']:
        a, b = read(old_root/'c16_on'/phase/'replay_calls.json'), read(root/'c16_early'/phase/'replay_calls.json')
        assert [(x['index'], x['prompt_sha256'], x['max_tokens']) for x in a] == [
            (x['index'], x['prompt_sha256'], x['max_tokens']) for x in b]
        pairs[phase] = dict(requests=len(a), same_cached_requests=sum(
            x['cached_tokens'] == y['cached_tokens'] for x,y in zip(a,b)),
            same_output_requests=sum(x['output_sha256'] == y['output_sha256'] for x,y in zip(a,b)))
    dump(root/'paired_checks.json', pairs)
    lines = ['# 존재 확인 먼저 알림: 결과', '',
        'Qwen3-14B / DRAM256GiB / staging8GiB / C16. DRAM·DAOS 프리페치 ON.', '',
        '|단계|기존 TTFT ms|조기 알림 TTFT ms|변화 %|전체 s|DRAM hit %|DAOS hit %|입력 재사용 %|peak staging GiB|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for before, s in zip(old, results, strict=True):
        assert before['phase'] == s['phase']
        delta = 100*(s['ttft']['mean']/before['ttft']['mean']-1)
        lines.append(f'|[{s["phase"]}](c16_early/{s["phase"]}/staging_hits.png)|'
            f'{before["ttft"]["mean"]:.2f}|{s["ttft"]["mean"]:.2f}|{delta:+.2f}|'
            f'{s["elapsed_seconds"]:.2f}|{s["dram_hit_pct"]:.2f}|{s["daos_hit_pct"]:.2f}|'
            f'{s["input_reuse_pct"]:.2f}|{s["peak_staging_gib"]:.3f}|')
    lines += ['', '음수 변화는 TTFT 감소이다. 같은 입력/출력상한을 재생하지만 실제 캐시 배치·생성량·'
              '도착 시각은 달라질 수 있다. paired_checks.json을 함께 확인한다. '
              'warm4회는 캐시를 유지한 연속 재사용이며 독립 시행이 아니다.',
              'hit 비율은 최초 lookup 후보 청크 기준. staging은 읽기·쓰기 합계. '
              '그래프는20ms 샘플의2초 구간 최대/평균이며 표의 event peak와 다를 수 있다.',
              'retrieve_decisions.json의 ready/wait_running/queued_to_demand는 배치 수이지 청크 비율이 아니다. '
              '기존 완료-first 전용 타이밍 집계기를 재사용하지 않는다.']
    (root/'RESULT_KO.md').write_text('\n'.join(lines)+'\n')
