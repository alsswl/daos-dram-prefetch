"""Audit the full CXS async/drop run and retain request-level evidence."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re

import realqa_q4_cxs as qa
from report_prefetch_timing import stats
from report_capacity_matrix import attribute_recomputation
from report_cold_warm_prefetch import select_events
from recompute_experiment_support import ANSI, RecomputeHealth


def analyze(root):
    plan=qa.read(root/'plan.json');case=root/'serving';phase=case/'cold'
    assert qa.read(root/'status.json')['status']=='completed'
    RecomputeHealth(case/'server.log')()
    calls=qa.read(phase/'calls.json');events=qa.base.read_events(case)
    assert len(calls)==480 and all(c['status']=='success' for c in calls)
    reference=qa.read(Path(plan['reference_histories']).parent/'calls.json')
    expected={(c['session_index'],c['turn']):c for c in reference}
    assert all(c['messages_sha256']==expected[c['session_index'],c['turn']]['messages_sha256'] for c in calls)
    assert all(c['prompt_tokens']==expected[c['session_index'],c['turn']]['prompt_tokens'] for c in calls)
    assert all(c['prompt_tokens']+c['completion_tokens']<=plan['max_model_len'] for c in calls)
    scoped=select_events(events,calls)
    rows=attribute_recomputation([dict(c,start_ns=c['http_start_ns'],ttft_ms=c['ttft_http_ms']) for c in calls],scoped)
    qa.dump(root/'recomputation_by_request.json',rows)
    assert all(r['other_failed_chunks']==0 for r in rows)
    samples=[e for e in events if e['event']=='occupancy_sample']
    assert samples and all(e['cpu_hot_bytes']==e['cpu_ready_bytes']==0 for e in samples)
    assert all(e['dram_mirror']['enabled'] is False for e in samples)
    assert not any(e['event'].startswith(('window_', 'store_window_')) for e in events)
    final=qa.read(case/'final_sample.json');assert final['used_bytes']==0
    enabled=[e for e in events if e['event']=='async_drop_enabled'];assert len(enabled)==1
    fs=[e['stats'] for e in events if e['event']=='async_drop_final_stats']
    outcomes=[e for e in events if e['event']=='daos_prefetch_outcome']
    assert len(outcomes)==sum(e['event']=='prefetch_start' for e in events)
    assert all(e['requested_chunks']==e['returned_chunks']+e['capacity_failed_chunks']+
               e['other_failed_chunks']+e['successful_tail_discarded_chunks'] for e in outcomes)
    # vLLM's process-group shutdown can exit before backend.close telemetry.
    # Every completed payload fetch is either returned or explicitly discarded.
    # Successful PUT counts are sampled after a stable zero-occupancy drain.
    gets=sum(e['returned_chunks']+e['successful_tail_discarded_chunks'] for e in outcomes)
    puts=final['daos_puts'];chunk_bytes=plan['chunk_tokens']*qa.KV_BYTES
    payload=dict(get=gets,put=puts,get_bytes=gets*chunk_bytes,put_bytes=puts*chunk_bytes)
    assert len(fs)<=1
    if fs:
        assert all(fs[0][k]==v for k,v in payload.items())
    window=qa.read(phase/'workload.json');start,end=window['start_ns'],window['end_ns']
    log=ANSI.sub('',(case/'server.log').read_text())
    stores=[(rid,int(a),int(b)) for rid,a,b in re.findall(r'\[req_id=(\S+)\] Stored (\d+) out of total (\d+) tokens\.',log)]
    store_pressure=[int(n) for n in re.findall(
        r'Local cpu memory under pressure so choosing to store only\s+(\d+) total chunks',log)]
    assert all(a<=b for _,a,b in stores)
    qa.dump(root/'store_admission.json',[dict(request_id=rid,admitted_tokens=a,offered_tokens=b,
        unadmitted_tokens=b-a) for rid,a,b in stores])
    summaries=[]
    for label,cc in [('all',calls),('first_turn',[c for c in calls if c['turn']==0]),('followup',[c for c in calls if c['turn']>0])]:
        summaries.append(dict(group=label,requests=len(cc),ttft_ms=stats(c['ttft_http_ms'] for c in cc),
            prompt_tokens=sum(c['prompt_tokens'] for c in cc),cached_tokens=sum(c['cached_tokens'] for c in cc),
            cached_token_pct=100*sum(c['cached_tokens'] for c in cc)/sum(c['prompt_tokens'] for c in cc),
            output_tokens=sum(c['completion_tokens'] for c in cc)))
    metrics=(phase/'metrics_after.txt').read_text()
    summary=dict(status='completed',requests=480,groups=summaries,elapsed_seconds=(end-start)/1e9,
        input_tokens=sum(c['prompt_tokens'] for c in calls),
        computed_input_tokens=sum(c['prompt_tokens']-c['cached_tokens'] for c in calls),
        daos_read_gib=payload['get_bytes']/2**30,daos_write_gib=payload['put_bytes']/2**30,
        daos_put_chunks=payload['put'],daos_get_chunks=payload['get'],
        payload_accounting='Completed fetch outcomes (including discarded tails) and drained successful PUT counter, times full chunk bytes',
        peak_staging_gib=max(e['used_bytes'] for e in events)/2**30,
        allocation_failure_events=sum(e['event'] in ('allocate','batched_allocate') and e.get('failed',False) for e in events),
        read_allocation_failures=sum(r['capacity_failed_chunks'] for r in rows),
        discarded_successful_tail_chunks=sum(r['successful_tail_discarded_chunks'] for r in rows),
        capacity_recomputed_tokens=sum(r['capacity_recomputed_tokens'] for r in rows),
        daos_lookup_available_token_pct=100*sum(min(r['prompt_tokens']-1,r['daos_lookup_chunks']*128) for r in rows)/sum(c['prompt_tokens'] for c in calls),
        unattributed_prefix_shortfall_tokens=sum(r['unattributed_shortfall_tokens'] for r in rows),
        attribution_issue_requests=[dict(index=r['index'],issues=r['attribution_issues']) for r in rows if r['attribution_issues']],
        logged_nonempty_store_calls=len(stores),partial_store_calls=sum(a<b for _,a,b in stores),
        store_capacity_failure_calls=len(store_pressure),store_zero_admission_calls=store_pressure.count(0),
        store_offered_tokens=sum(b for _,_,b in stores),store_admitted_tokens=sum(a for _,a,_ in stores),
        store_unadmitted_tokens_logged_nonempty=sum(b-a for _,a,b in stores),
        scheduler_recovery_events=len(re.findall('Recovered from KV load failure:',log)),
        identical_input_hashes=True,dram_kv_bytes=0,staging_final_bytes=0,
        notes=['Computed input = prompt minus reported cached tokens; not GPU timing or unique tokens.',
               'Store token totals cover nonempty logged stores only; native zero-admission early returns omit requested token counts.',
               'Unadmitted store token totals are therefore a lower bound and can include later retried or repeated prefixes.',
               'Async lookup reports actual loaded prefix before scheduling, so normal prefill need not log recovery.',
               'C8 HTTP concurrency, original single async serializer, 16 per-chunk IO workers.',
               'Historical comparison changes both loading and store/drop policy; not a single-variable async comparison.'])
    old=dict(requests=len(reference),ttft_ms=stats(c['ttft_http_ms'] for c in reference),
             elapsed_seconds=(qa.read(Path(plan['reference_histories']).parent/'workload.json')['end_ns']-
                              qa.read(Path(plan['reference_histories']).parent/'workload.json')['start_ns'])/1e9,
             cached_token_pct=100*sum(c['cached_tokens'] for c in reference)/sum(c['prompt_tokens'] for c in reference),
             output_tokens=sum(c['completion_tokens'] for c in reference))
    old['daos_read_gib']=qa.read(Path(plan['reference_histories']).parent/'no_dram_check.json')['daos_returned_gib']
    summary['historical_window500_baseline']=old
    qa.dump(root/'results.json',summary)
    with (root/'requests.csv').open('w') as f:
        fields=['index','session_index','turn','prompt_tokens','cached_tokens','completion_tokens','ttft_http_ms']
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(sorted(calls,key=lambda c:c['index']))
    lines=[f'Full CXS C8 x S10 x 6: {len(calls)} successful requests',
        'Qwen3-4B-Instruct-2507, 61440-token documents, output cap 256, max model length 65536.',
        '2GiB shared staging; DRAM off; original dkey hash; native async loading; store/read immediate capacity miss.',
        f'Mean TTFT: {summaries[0]["ttft_ms"]["mean"]:.2f} ms; p95: {summaries[0]["ttft_ms"]["p95"]:.2f} ms.',
        f'Elapsed: {summary["elapsed_seconds"]:.2f} s; cached input: {summaries[0]["cached_token_pct"]:.3f}%.',
        f'DAOS read/write: {summary["daos_read_gib"]:.3f}/{summary["daos_write_gib"]:.3f} GiB.',
        f'Read allocation failures: {summary["read_allocation_failures"]}; store capacity-failure calls: {len(store_pressure)} (zero admission {store_pressure.count(0)}).',
        f'Peak staging: {summary["peak_staging_gib"]:.6f} GiB; final staging: 0.',
        f'Historical window500 mean TTFT: {old["ttft_ms"]["mean"]:.2f} ms, cached {old["cached_token_pct"]:.3f}%.',
        *summary['notes']]
    (root/'report.txt').write_text('\n'.join(lines)+'\n')
    assert sum(n>0 for n in store_pressure)==summary['partial_store_calls']
    assert summary['allocation_failure_events']==summary['read_allocation_failures']+len(store_pressure)
    assert not summary['attribution_issue_requests'] and summary['unattributed_prefix_shortfall_tokens']==0
    qa.dump(root/'validation.json',dict(passed=True,successful_requests=480,
        identical_reference_inputs=True,context_within_65536=True,
        actual_cached_tokens_match_trace=True,all_prefix_losses_attributed=True,
        read_noncapacity_failures=0,dram_kv_bytes=0,staging_after_bytes=0,
        gate=qa.read(root/'gate/result.json')['passed'],
        owned_object_cleanup=qa.read(root/'owned_object_cleanup.json')))
    # Export a standalone scientific figure. One-second peaks retain short
    # occupancy bursts while avoiding an unreadable multi-million-point plot.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    peaks=defaultdict(float)
    for e in events:
        if start<=e['time_ns']<=end:
            sec=int((e['time_ns']-start)/1e9);peaks[sec]=max(peaks[sec],e['used_bytes']/2**30)
    fig,axes=plt.subplots(2,1,figsize=(11,6),constrained_layout=True)
    xx=sorted(peaks);axes[0].plot(xx,[peaks[x] for x in xx],lw=.8)
    axes[0].axhline(2,color='red',ls='--',lw=1);axes[0].set(ylabel='Staging GiB (1s peak)',ylim=(0,2.1))
    for turn,color,label in [(True,'tab:orange','First turn'),(False,'tab:blue','Follow-up')]:
        cc=[c for c in calls if (c['turn']==0)==turn]
        axes[1].scatter([(c['http_start_ns']-start)/1e9 for c in cc],[c['ttft_http_ms']/1000 for c in cc],s=10,c=color,label=label)
    axes[1].set(xlabel='Elapsed seconds',ylabel='HTTP TTFT (s)');axes[1].legend()
    fig.suptitle('Full CXS C8 | shared 2GiB | async load | store/read drop on capacity')
    fig.savefig(root/'timeline.png',dpi=160);fig.savefig(root/'timeline.svg');plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);analyze(p.parse_args().root.resolve())
