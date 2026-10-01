"""Serving comparison: report read volume and request times alongside TTFT."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import statistics


def read(path):return json.loads(path.read_text())


def stats(values):
    values=sorted(values)
    if not values:return dict(count=0)
    position=(len(values)-1)*.95;lower=int(position);upper=min(lower+1,len(values)-1)
    p95=values[lower]+(position-lower)*(values[upper]-values[lower])
    return dict(count=len(values),mean=statistics.mean(values),median=statistics.median(values),
                p95=p95,max=max(values))


def main(folder):
    results=[];all_calls=[]
    for spec in read(folder/'comparison_plan.json')['cases']:
        root=Path(spec['root']);plan=read(root/'plan.json');case=root/plan['cases'][0]['name']
        assert read(root/'status.json')['status']=='completed'
        calls=read(case/'calls.json');all_calls.append(calls)
        assert len(calls)==480 and all(c['status']=='success' for c in calls)
        events=[]
        for path in case.glob('trace.*.jsonl'):events.extend(json.loads(l) for l in path.open())
        events.sort(key=lambda e:e['monotonic_ns'])
        starts={};retrievals=[]
        outcomes=[]
        for e in events:
            if e['event']=='window_retrieve_start':
                assert e['request_id'] not in starts;starts[e['request_id']]=e
            elif e['event']=='window_retrieve_done':
                start=starts.pop(e['request_id'])
                retrievals.append(dict(request_id=e['request_id'],bytes=e['bytes'],chunks=e['chunks'],
                                       ms=(e['monotonic_ns']-start['monotonic_ns'])/1e6))
            elif e['event']=='daos_demand_outcome':outcomes.append(e)
        assert not starts
        assert all(e['other_failed_chunks']==0 for e in outcomes)
        assert sum(r['chunks'] for r in retrievals)==sum(e['returned_chunks'] for e in outcomes)
        by_http={re.sub(r'-[0-9a-f]{8}$','',r['request_id']):r for r in retrievals}
        (folder/f"retrieve_requests_{spec['label']}.json").write_text(json.dumps(retrievals,indent=2)+'\n')
        matches=[(c,by_http[c['server_request_id']]) for c in calls if c['server_request_id'] in by_http]
        final=read(case/'final_sample.json')
        positions=Path(plan['placement_manifest']).with_suffix('.positions.sqlite')
        import sqlite3
        db=sqlite3.connect(f'file:{positions}?mode=ro',uri=True)
        indices=db.execute('SELECT idx FROM positions WHERE key LIKE ?', (plan['experiment_namespace']+'%',)).fetchall();db.close()
        manifest=read(Path(plan['placement_manifest']));n=len(manifest['addressing']['placement_keys'])
        assert read(case/'no_dram_check.json')['passed'] and final['used_bytes']==0
        assert read(root/'owned_object_cleanup.json')['rc']==0
        summary=read(root/'summary.json')
        result=dict(mode=spec['label'],requests=480,duration_s=summary['duration_s'],
            ttft_ms=stats([c['ttft_http_ms'] for c in calls]),
            followup_ttft_ms=stats([c['ttft_http_ms'] for c in calls if c['turn']>0]),
            new_document_ttft_ms=stats([c['ttft_http_ms'] for c in calls if c['turn']==0]),
            retrieve_ms=stats([r['ms'] for r in retrievals]),
            normalized_retrieve_ms_per_gib=stats([r['ms']/(r['bytes']/2**30) for r in retrievals]),
            retrieved_gib=sum(r['bytes'] for r in retrievals)/2**30,
            restored_chunks=sum(r['chunks'] for r in retrievals),
            daos_puts=final['daos_puts'],stored_gib=final['daos_puts']*18/1024,
            input_tokens=sum(c['prompt_tokens'] for c in calls),output_tokens=sum(c['completion_tokens'] for c in calls),
            cached_tokens=sum(c.get('cached_tokens',0) for c in calls),
            matched_retrieve_http_requests=len(matches),
            matched_http_ttft_ms=stats([c['ttft_http_ms'] for c,_ in matches]),
            allocation_failed_attempts=sum(e['capacity_failed_chunks'] for e in outcomes),
            persisted_keys=len(indices),
            index_modulo_shard_counts=dict(Counter(i[0]%n for i in indices)) if spec['label']=='balanced' else None)
        results.append(result)
    a,b=results
    originals={(c['session_index'],c['turn']):c for c in all_calls[0]}
    matched_prompts=sum(c['messages_sha256']==originals[c['session_index'],c['turn']]['messages_sha256'] for c in all_calls[1])
    report=dict(passed=True,results=results,identical_actual_prompt_hashes=matched_prompts,
                actual_prompts_compared=480,
                balanced_change_percent={
                    'mean_ttft':100*(b['ttft_ms']['mean']/a['ttft_ms']['mean']-1),
                    'p95_ttft':100*(b['ttft_ms']['p95']/a['ttft_ms']['p95']-1),
                    'mean_followup_ttft':100*(b['followup_ttft_ms']['mean']/a['followup_ttft_ms']['mean']-1),
                    'mean_retrieve':100*(b['retrieve_ms']['mean']/a['retrieve_ms']['mean']-1),
                    'mean_normalized_retrieve':100*(b['normalized_retrieve_ms_per_gib']['mean']/a['normalized_retrieve_ms_per_gib']['mean']-1),
                    'duration':100*(b['duration_s']/a['duration_s']-1),
                    'read_volume':100*(b['retrieved_gib']/a['retrieved_gib']-1)},
                limitations=['One serving run per condition; execution order fixed.',
                             'Same seeded documents/schedule; generated histories and wall arrivals may differ.',
                             'Both placement and dkey/akey hierarchy change; not a pure scheduler experiment.',
                             'Server target service time and network contention are not directly measured.'])
    (folder/'placement_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();main(a.folder)
