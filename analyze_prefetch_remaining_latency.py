#!/usr/bin/env python3
"""Read-only trace attribution for matched warm1 requests; writes analysis outputs only."""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from report_prefetch_timing import stats

ROOT = Path(__file__).resolve().parent
CASES = {
    'off': 'sharegpt_demand_d256_s8_20260930_v2/c16_demand',
    'on_10ms': 'sharegpt_early_lookup_d256_s8_20260930/c16_early',
    'on_1ms': 'sharegpt_backoff_1ms_d256_s8_20260930/c16_backoff_1ms',
    'on_ready': 'sharegpt_ready_return_d256_s8_20260930/c16_ready_return',
}


def read(path):
    return json.loads(path.read_text())


def api_id(value):
    match = re.fullmatch(r'(chatcmpl-[0-9a-f]{16})-[0-9a-f]{8}', value or '')
    return match.group(1) if match else value


def case_rows(folder):
    calls = read(folder/'warm1/replay_calls.json')
    by_id = {c['server_request_id']: c for c in calls}
    events = defaultdict(list)
    starts = sorted(c['start_ns'] for c in calls)
    ends = sorted(c['end_ns'] for c in calls)
    samples = []
    phase = read(folder/'warm1/phase.json')
    for f in folder.glob('trace.*.jsonl'):
        for line in f.open():
            e = json.loads(line)
            rid = api_id(e.get('request_id'))
            if rid in by_id:
                events[rid].append(e)
            if e['event']=='occupancy_sample' and phase['start_ns']<=e['time_ns']<=phase['end_ns']:
                samples.append(e)
    rows = []
    for c in calls:
        group = events[c['server_request_id']]
        row = {k: c[k] for k in ('index','session','turn','prompt_sha256','prompt_tokens','cached_tokens',
                                  'completion_tokens','ttft_ms','start_ns','end_ns')}
        row['new_tokens'] = c['prompt_tokens']-c['cached_tokens']
        row['active_client_calls_at_start'] = bisect_right(starts,c['start_ns'])-bisect_right(ends,c['start_ns'])
        row['prior_calls_finishing_within_10ms'] = bisect_right(ends,c['start_ns'])-bisect_left(ends,c['start_ns']-10_000_000)
        def one(name):
            out = [e for e in group if e['event']==name]
            assert len(out)<=1, (name,c['index'],len(out))
            return out[0] if out else None
        def ms(a,b):
            return (b['monotonic_ns']-a['monotonic_ns'])/1e6
        def after_http(e):
            return (e['time_ns']-c['start_ns'])/1e6
        tiers = [e for e in group if e['event']=='tier_lookup']
        for tier in ('dram','daos'):
            t = [e for e in tiers if e['tier']==tier]
            assert len(t)<=1
            row[tier+'_chunks'] = t[0]['hit_chunks'] if t else 0
        if tiers:
            first, last = min(tiers,key=lambda e:e['monotonic_ns']), max(tiers,key=lambda e:e['monotonic_ns'])
            row['http_to_first_lookup_result_ms'] = after_http(first)
            row['http_to_lookup_result_ms'] = after_http(last)
            row['between_tier_results_ms'] = ms(first,last)
        rs, re_ = one('retrieve_start'), one('retrieve_return')
        if rs and re_:
            row.update(http_to_retrieve_ms=after_http(rs), retrieve_ms=ms(rs,re_),
                after_retrieve_ms=c['ttft_ms']-after_http(re_), http_to_retrieve_end_ms=after_http(re_))
            row['lookup_result_to_retrieve_ms'] = ms(last,rs)
            assert abs(row['http_to_retrieve_ms']+row['retrieve_ms']+row['after_retrieve_ms']-c['ttft_ms'])<0.01
        ps, pe = one('prefetch_start'), one('prefetch_ready')
        if ps and pe:
            row['daos_read_ms'] = ms(ps,pe)
            row['lookup_result_to_daos_start_ms'] = ms(last,ps)
            row['http_to_daos_start_ms'] = after_http(ps)
            if rs:
                row['daos_ready_to_retrieve_ms'] = ms(pe,rs)
                row['daos_prefetch_lead_ms'] = ms(ps,rs)
                row['daos_read_before_retrieve_ms'] = max(0,min(row['daos_read_ms'],ms(ps,rs)))
        ds,de = one('daos_demand_start'),one('daos_demand_outcome')
        if ds and de:
            row['daos_read_ms'] = ms(ds,de)
            row['http_to_daos_start_ms'] = after_http(ds)
        for tier in ('dram','daos'):
            dec = [e for e in group if e['event']=='early_retrieve_decision' and e['tier']==tier]
            assert len(dec)<=1
            if dec:
                row[tier+'_decision'] = dec[0]['decision']
                row[tier+'_resolve_wait_ms'] = dec[0]['wait_ms']
        timing=one('cpu_prefetch_timing')
        if timing and 'copy_start_ns' in timing:
            row['dram_queue_ms'] = (timing['worker_start_ns']-timing['queued_ns'])/1e6
            row['dram_host_copy_ms'] = (timing['copy_end_ns']-timing['copy_start_ns'])/1e6
        rows.append(row)
    peak=max(e['used_bytes'] for e in samples)/2**30
    return rows, dict(sampled_peak_staging_gib=peak, phases=read(folder/'summary.json'))


FIELDS=('ttft_ms','http_to_first_lookup_result_ms','http_to_lookup_result_ms','between_tier_results_ms',
        'lookup_result_to_retrieve_ms','http_to_retrieve_ms','retrieve_ms','after_retrieve_ms',
        'http_to_retrieve_end_ms','daos_read_ms','http_to_daos_start_ms','lookup_result_to_daos_start_ms',
        'daos_ready_to_retrieve_ms','daos_prefetch_lead_ms','daos_read_before_retrieve_ms',
        'daos_resolve_wait_ms','dram_resolve_wait_ms','dram_queue_ms','dram_host_copy_ms')


def summarize(rows):
    return dict(requests=len(rows), timings={k:stats(r.get(k) for r in rows) for k in FIELDS},
        decisions={tier:dict(Counter(r.get(tier+'_decision') for r in rows if tier+'_decision' in r)) for tier in ('dram','daos')},
        average_prompt_tokens=stats(r['prompt_tokens'] for r in rows),
        active_client_calls=stats(r['active_client_calls_at_start'] for r in rows),
        daos_ready_before_retrieve=sum(r.get('daos_ready_to_retrieve_ms',-1)>=0 for r in rows))


def analyze_response_delivery(output):
    """Reuse mutex diagnostic to separate response publication from scheduler polling."""
    case=ROOT/'sharegpt_lookup_lock_probe_d256_s8_20260930/c16_lock_probe'
    locks,hosts=defaultdict(list),defaultdict(list)
    for path in case.glob('lookup_lock.*.jsonl'):
        for line in path.open():
            e=json.loads(line)
            locks[api_id(e['request_id'])].append(e)
    for path in case.glob('trace.*.jsonl'):
        for line in path.open():
            e=json.loads(line)
            if e.get('request_id'):
                hosts[api_id(e['request_id'])].append(e)
    summaries=[]
    for phase in ('cold','warm_lock_probe'):
        rows=[]
        for call in read(case/phase/'replay_calls.json'):
            rid=call['server_request_id']
            group=locks[rid]
            replies=[e for e in group if e['operation']=='process_responses_from_workers']
            assert len(replies)==1
            reply=replies[0]
            checks=[e for e in group if e['operation'] in ('lookup_cache','lookup')
                    and e['acquired_ns']>=reply['release_started_ns']]
            assert checks
            check=min(checks,key=lambda e:e['acquired_ns'])
            notifications=[e for e in hosts[rid] if e['event']=='early_lookup_notify']
            assert len(notifications)==1
            notify=notifications[0]
            row=dict(index=call['index'], cached_tokens=call['cached_tokens'],
                response_lock_acquire_ms=(reply['acquired_ns']-reply['requested_ns'])/1e6,
                notify_to_response_recorded_ms=(reply['release_started_ns']-notify['monotonic_ns'])/1e6,
                response_recorded_to_scheduler_check_ms=(check['acquired_ns']-reply['release_started_ns'])/1e6,
                check_operation=check['operation'])
            retrieved=[e for e in hosts[rid] if e['event']=='retrieve_start']
            assert len(retrieved)<=1
            if retrieved:
                row['check_to_retrieve_ms']=(retrieved[0]['monotonic_ns']-check['acquired_ns'])/1e6
            for tier in ('dram','daos'):
                t=[e for e in hosts[rid] if e['event']=='tier_lookup' and e['tier']==tier]
                assert len(t)<=1
                row[tier+'_chunks']=t[0]['hit_chunks'] if t else 0
            rows.append(row)
        fields=['response_lock_acquire_ms','notify_to_response_recorded_ms',
                'response_recorded_to_scheduler_check_ms','check_to_retrieve_ms']
        cohorts=dict(all=rows,dram_only=[r for r in rows if r['dram_chunks']>0 and r['daos_chunks']==0],
                     daos_only=[r for r in rows if r['daos_chunks']>0 and r['dram_chunks']==0])
        summaries.append(dict(phase=phase,cohorts={name:dict(requests=len(rs),
            timings={k:stats(r.get(k) for r in rs) for k in fields},
            check_operations=dict(Counter(r['check_operation'] for r in rs))) for name,rs in cohorts.items()}))
        (output/f'{phase}_response_delivery_rows.json').write_text(json.dumps(rows)+'\n')
    result=dict(phases=summaries,limitations=[
        'Separate instrumented run with ready-return enabled, first64 warm conversations only.',
        'Response publication timestamp is just before releasing the lookup mutex; includes all worker results (TP1).',
        'Polling lag can include useful model execution as well as CPU/scheduling delays; not wholly wasted time.',
        'Publication-to-check time is distinct from mutex acquisition wait.'])
    (output/'response_delivery.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


def run(output):
    output.mkdir(exist_ok=False)
    rows, all_metadata = {}, {}
    for mode,path in CASES.items():
        rows[mode],all_metadata[mode]=case_rows(ROOT/path)
        (output/f'{mode}_rows.json').write_text(json.dumps(rows[mode])+'\n')
    maps={mode:{r['index']:r for r in rs} for mode,rs in rows.items()}
    common=sorted(set.intersection(*(set(m) for m in maps.values())))
    for index in common:
        values=[m[index] for m in maps.values()]
        assert len({(r['prompt_sha256'],r['prompt_tokens'],r['cached_tokens']) for r in values})==1
    common_retrieve=[i for i in common if all('retrieve_ms' in m[i] for m in maps.values())]
    pair_same_tiers=[i for i in common if all(maps['off'][i][k]==maps['on_1ms'][i][k] for k in ('dram_chunks','daos_chunks'))]
    cohorts={
        'all':common,
        'common_retrieved':common_retrieve,
        'paired_daos_only': [i for i in pair_same_tiers if maps['off'][i]['daos_chunks']>0 and maps['off'][i]['dram_chunks']==0],
        'paired_dram_only': [i for i in pair_same_tiers if maps['off'][i]['dram_chunks']>0 and maps['off'][i]['daos_chunks']==0],
        'paired_mixed': [i for i in pair_same_tiers if maps['off'][i]['dram_chunks']>0 and maps['off'][i]['daos_chunks']>0],
        'no_reuse':[i for i in common if maps['off'][i]['cached_tokens']==0],
        'initial16':[i for i in common if i<16],
        'after_initial16':[i for i in common if i>=16],
    }
    for label,low,high in [('small',1,8),('medium',9,32),('large',33,1000)]:
        cohorts['paired_daos_only_'+label]=[i for i in cohorts['paired_daos_only'] if low<=maps['off'][i]['daos_chunks']<=high]
    summaries={name:{mode:summarize([m[i] for i in indices]) for mode,m in maps.items()}
               for name,indices in cohorts.items()}
    (output/'summary.json').write_text(json.dumps(dict(cohorts=summaries,metadata=all_metadata,
        notes=['Tier-matched cohorts match OFF/ON1ms exactly; other variants use same indices, may have different tier placement.',
               'TTFT decomposition uses identical request cohort and sums exactly.',
               'DAOS read spans are host intervals, not NIC DMA.',
               'Client active calls are not GPU batch concurrency.',
               'Single historical runs, fixed inputs but variable generated output and arrival timing.']),indent=2)+'\n')
    for name in ('common_retrieved','paired_daos_only','paired_dram_only','paired_mixed','no_reuse'):
        print(name)
        for mode in rows:
            s=summaries[name][mode]
            print(mode,s['requests'],{k:round(s['timings'][k]['mean'],3) if s['timings'][k]['mean'] is not None else None
                                     for k in ('ttft_ms','http_to_retrieve_ms','retrieve_ms','after_retrieve_ms','daos_read_ms','daos_ready_to_retrieve_ms')})


if __name__=='__main__':
    run(ROOT/'prefetch_remaining_latency_20260930')
