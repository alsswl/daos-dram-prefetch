"""Validate identical CXS inputs and report target-count cold/warm results."""
import argparse
import json
import re
from pathlib import Path
import statistics


def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')


def stats(xs):
    xs=sorted(xs)
    if not xs:return dict(count=0)
    pos=(len(xs)-1)*.95;lo=int(pos);hi=min(lo+1,len(xs)-1)
    return dict(count=len(xs),mean=statistics.mean(xs),median=statistics.median(xs),
                p95=xs[lo]+(pos-lo)*(xs[hi]-xs[lo]),max=max(xs))


def main(folder):
    comparison=read(folder/'comparison_plan.json')
    assert read(folder/'status.json')['status']=='completed'
    reference=read(Path(comparison['reference_histories']).parent/'calls.json')
    ref={(c['session_index'],c['turn']):(c['messages_sha256'],c['prompt_tokens']) for c in reference}
    results=[];configs=[]
    for n in comparison['target_counts']:
        root=folder/f't{n:02d}';case=root/'serving';plan=read(root/'plan.json')
        assert read(root/'status.json')['status']=='completed'
        assert read(root/'owned_object_cleanup.json')['rc']==0
        assert read(root/'placement_validation.json')['passed']
        assert read(case/'no_dram_check.json')['passed']
        resumes=[json.loads(s) for s in re.findall(r'DAOS_RESUME_TOKEN_FIX (\{[^\n]+\})',(case/'server.log').read_text())]
        assert any(e['event']=='installed' for e in resumes)
        restored=[e for e in resumes if e['event']=='resume_full_history']
        assert all(e['tracker_after']>=e['expected_cache_tokens'] for e in restored)
        save(root/'resume_validation.json',dict(passed=True,restored_requests=len(restored)))
        import yaml
        cfg=yaml.safe_load((case/'config.yaml').read_text())
        for k in ['daosgds.root','daosgds.object_namespace','daosgds.placement_manifest']:cfg['extra_config'].pop(k)
        configs.append(cfg)
        events=[]
        for p in case.glob('trace.*.jsonl'):events.extend(json.loads(line) for line in p.read_text().splitlines())
        events.sort(key=lambda e:e['monotonic_ns'])
        for phase in ['cold','warm']:
            dest=case/phase;calls=read(dest/'calls.json');w=read(dest/'workload.json')
            assert len(calls)==48 and all(c['status']=='success' for c in calls)
            if phase=='cold':assert all(c['cached_tokens']==0 for c in calls if c['turn']==0)
            assert {(c['session_index'],c['turn']):(c['messages_sha256'],c['prompt_tokens']) for c in calls}==ref
            assert all(c['prompt_tokens']+256<=65536 for c in calls)
            es=[e for e in events if w['start_ns']<=e['time_ns']<=w['end_ns']]
            started={};retrieves=[]
            for e in es:
                if e['event']=='window_retrieve_start':
                    assert e['request_id'] not in started;started[e['request_id']]=e
                elif e['event']=='window_retrieve_done':
                    first=started.pop(e['request_id'])
                    retrieves.append(dict(request_id=e['request_id'],chunks=e['chunks'],bytes=e['bytes'],
                                          ms=(e['monotonic_ns']-first['monotonic_ns'])/1e6))
            assert not started
            normalize=lambda rid:re.sub(r'-[0-9a-f]{8}$','',rid)
            by_http={}
            for restore in retrieves:by_http.setdefault(normalize(restore['request_id']),[]).append(restore)
            http_ids={c['server_request_id'] for c in calls}
            assert set(by_http)<=http_ids
            phase_resumes=[e for e in restored if normalize(e['request_id']) in http_ids]
            extra=[r for group in by_http.values() for r in group[1:]]
            if phase=='warm':
                assert len(by_http)==48 and len(extra)==len(phase_resumes)
                assert {normalize(e['request_id']) for e in phase_resumes}=={rid for rid,group in by_http.items() if len(group)>1}
            outcomes=[e for e in es if e['event']=='daos_demand_outcome']
            assert all(e['other_failed_chunks']==0 for e in outcomes)
            assert sum(r['chunks'] for r in retrieves)==sum(e['returned_chunks'] for e in outcomes)
            initial=read(dest/'initial_sample.json');final=read(dest/'final_sample.json')
            assert initial['used_bytes']==final['used_bytes']==0
            inp=sum(c['prompt_tokens'] for c in calls);cached=sum(c['cached_tokens'] for c in calls)
            result=dict(targets=n,phase=phase,requests=len(calls),duration_s=(w['end_ns']-w['start_ns'])/1e9,
                        ttft_ms=stats([c['ttft_http_ms'] for c in calls]),
                        initial_turn_ttft_ms=stats([c['ttft_http_ms'] for c in calls if c['turn']==0]),
                        followup_ttft_ms=stats([c['ttft_http_ms'] for c in calls if c['turn']>0]),
                        retrieve_ms=stats([r['ms'] for r in retrieves]),
                        retrieve_ms_per_gib=stats([r['ms']/(r['bytes']/2**30) for r in retrieves]),
                        read_gib=sum(r['bytes'] for r in retrieves)/2**30,
                        first_retrieve_gib=sum(group[0]['bytes'] for group in by_http.values())/2**30,
                        repeated_retrieve_gib=sum(r['bytes'] for r in extra)/2**30,
                        repeated_retrieve_calls=len(extra),scheduler_resume_records=len(phase_resumes),
                        read_chunks=sum(r['chunks'] for r in retrieves),input_tokens=inp,cached_tokens=cached,
                        cached_token_pct=100*cached/inp,computed_tokens=inp-cached,
                        output_tokens=sum(c['completion_tokens'] for c in calls),
                        daos_puts=final['daos_puts']-initial['daos_puts'],
                        read_allocation_failures=sum(e['capacity_failed_chunks'] for e in outcomes),
                        store_allocation_failures=read(case/'store_window_policy_check.json')['store_allocation_failures'])
            results.append(result);save(dest/'retrieve_requests.json',retrieves)
    assert all(c==configs[0] for c in configs)
    report=dict(passed=True,requests=480,identical_prompt_hashes_and_token_counts=True,config_only_routing_namespace_differs=True,
                warm_read_volume_identical=len({r['read_gib'] for r in results if r['phase']=='warm'})==1,
                warm_first_retrieve_volume_identical=len({r['first_retrieve_gib'] for r in results if r['phase']=='warm'})==1,
                gpu_hardware=(folder/'gpu_hardware.txt').read_text().strip(),
                daos_version=(folder/'daos_version.txt').read_text().strip(),
                execution_order=comparison['execution_order'],results=results,
                limitations=['One cold/warm pair per condition, not independent repeated measurements.',
                             'Cold means empty namespace at phase start, with natural within-phase follow-up cache hits.',
                             'Warm retains DAOS and process state; DRAM and native GPU prefix caches are disabled.',
                             'Reference conversations generated in the first cold phase; subsequent outputs do not alter replay inputs.',
                             'Cold includes first-use overhead; server caches were not flushed.',
                             '8 sessions and C8 are retained; S1 means no multiple sessions interleaved inside a lane.',
                             '1-to-2 targets also changes engine count. 2/4/8/16 use two engines.',
                             'Scheduler preemption/resume can produce more retrieve calls than HTTP requests; actual transferred bytes include re-restoration.',
                             'Retrieve is client-observed restoration time, including allocation, metadata, I/O and GPU scatter.'])
    save(folder/'results.json',report)
    lines=['CXS target-count cold/warm comparison','Qwen3-4B-Instruct-2507; 8 Gutenberg documents, 61440 document tokens; C8 x S1 x 6 turns.',
           report['gpu_hardware'],report['daos_version'],
           '48 requests/phase. 16 I/O workers. 2GiB shared staging, 500MiB windows. DRAM and native prefix cache OFF.',
           'All 480 actual request hashes and input token counts match the 48-request reference.', '',
           'targets | phase | mean TTFT ms | p95 TTFT ms | mean retrieve ms | DAOS read GiB | cached input % | duration s']
    for r in results:
        lines.append(f"{r['targets']:7d} | {r['phase']:5s} | {r['ttft_ms']['mean']:12.2f} | {r['ttft_ms']['p95']:11.2f} | {r['retrieve_ms'].get('mean',0):16.2f} | {r['read_gib']:13.3f} | {r['cached_token_pct']:14.3f} | {r['duration_s']:10.2f}")
    lines+=['','Limitations:']+['- '+x for x in report['limitations']]
    lines+=['','Warm per-request first retrieval vs scheduler re-restoration:']
    for r in results:
        if r['phase']=='warm':lines.append(f"{r['targets']} targets: first retrieval {r['first_retrieve_gib']:.6f}GiB; {r['repeated_retrieve_calls']} repeated calls add {r['repeated_retrieve_gib']:.6f}GiB; matched {r['scheduler_resume_records']} scheduler resume records.")
    (folder/'report.txt').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    xs=np.arange(5)
    for phase,color in [('cold','#0072b2'),('warm','#d55e00')]:
        rs=[r for r in results if r['phase']==phase]
        axes[0].plot(xs,[r['ttft_ms']['mean'] for r in rs],'-o',color=color,label=phase)
        axes[1].plot(xs,[r['retrieve_ms'].get('mean',0) for r in rs],'-o',color=color,label=phase)
        axes[2].plot(xs,[r['read_gib'] for r in rs],'-o',color=color,label=phase)
    for ax in axes:
        ax.set_xticks(xs,['1','2','4','8','16']);ax.set_xlabel('Targets used');ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Mean HTTP TTFT (ms)');axes[0].set_title('End-to-end first-token latency')
    axes[1].set_ylabel('Mean retrieve completion (ms)');axes[1].set_title('Client KV restoration time')
    axes[2].set_ylabel('DAOS read volume (GiB)');axes[2].set_title('Actual bytes read per phase')
    fig.suptitle('CXS: fixed 16 I/O workers, identical 48-request inputs per phase\nCold = empty namespace at start; warm = exact replay after writes drain',fontsize=11)
    for suffix in ['png','svg']:fig.savefig(folder/f'comparison.{suffix}',dpi=180)
    plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);main(p.parse_args().folder)
