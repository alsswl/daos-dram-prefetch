"""Read-only validation/aggregation of demand-mode events, not async attribution."""
import sharegpt_scale_experiment as base

read, dump = base.read, base.dump


def validate_no_prefetch(events):
    forbidden = {'prefetch_start', 'prefetch_ready', 'prefetch_error', 'serializer_queued',
                 'cpu_prefetch_timing', 'daos_prefetch_outcome', 'cpu_get_start', 'cpu_get_ready'}
    assert not any(e['event'] in forbidden for e in events), 'Unexpected prefetch activity'
    active, batches = set(), 0
    for e in events:
        name, rid = e['event'], e.get('request_id')
        if name == 'retrieve_start':
            assert rid not in active
            active.add(rid)
        elif name in ('daos_demand_start', 'daos_demand_outcome'):
            assert rid in active, 'DAOS payload read outside retrieve'
            if name == 'daos_demand_start': batches += 1
        elif name == 'retrieve_return':
            assert rid in active
            active.remove(rid)
    assert not active
    return batches


def summarize_case(root, spec):
    case = root/spec['name']
    records, plan = read(root/'requests.json'), read(root/'plan.json')
    events = base.read_events(case)
    assert len({e['pid'] for e in events}) == 1
    assert any(e['event'] == 'demand_read_enabled' for e in events)
    validate_no_prefetch(events)
    previous = read(case/'initial_sample.json')
    assert previous['used_bytes'] == previous['cpu_hot_bytes'] == previous['daos_puts'] == 0
    results = []
    for name in plan['phases']:
        folder = case/name
        calls, phase = read(folder/'replay_calls.json'), read(folder/'phase.json')
        assert len(calls) == len(records) and not any('error' in c for c in calls)
        assert [(c['index'], c['prompt_sha256'], c['prompt_tokens'], c['max_tokens']) for c in calls] == [
            (r['index'], r['prompt_sha256'], r['expected_prompt_tokens'], r['max_tokens']) for r in records]
        before, after = read(folder/'initial_sample.json'), read(folder/'final_sample.json')
        assert before == previous
        previous = after
        assert before['used_bytes'] == after['used_bytes'] == after['dram_mirror']['pending_bytes'] == 0
        assert after['dram_mirror']['errors'] == 0 and after['cpu_prefetch'] is None
        selected = base.select_events(events, calls)
        q = sum(e['queried_chunks'] for e in selected if e['event'] == 'tier_lookup' and e['tier'] == 'dram')
        hits = {t: sum(e['hit_chunks'] for e in selected if e['event'] == 'tier_lookup' and e['tier'] == t)
                for t in ('dram', 'daos')}
        assert q > 0 and 0 <= sum(hits.values()) <= q
        outcomes = [e for e in selected if e['event'] == 'daos_demand_outcome']
        # Stop/report if a batch lost data; don't label unmeasured loss as capacity recomputation.
        assert all(e['returned_chunks'] == e['requested_chunks'] for e in outcomes), 'Demand read shortfall'
        assert after['daos_alloc_fail'] == before['daos_alloc_fail'], 'GPU allocation failure'
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
            sampled_peak_cpu_gib=max(e['cpu_hot_bytes'] for e in samples)/2**30,
            demand_batches=len(outcomes), no_prefetch_verified=True,
            capacity_recomputed_tokens=0)
        dump(folder/'summary.json', s)
        base.timeline(folder, events, start, end, 8)
        results.append(s)
    dump(case/'summary.json', results)
