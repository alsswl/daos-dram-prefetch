import json,statistics
from pathlib import Path
root=Path(__file__).resolve().parent
plan=json.loads((root/'comparison_plan.json').read_text());results=[]
for spec in plan['cases']:
    run=Path(spec['root']);cfg=json.loads((run/'plan.json').read_text())
    case=run/cfg['cases'][0]['name']
    assert json.loads((run/'status.json').read_text())['status']=='completed'
    starts={};durations=[];counts={};chunks=total=0
    for path in case.glob('trace.*.jsonl'):
        for line in path.open():
            e=json.loads(line);kind=e['event'];counts[kind]=counts.get(kind,0)+1
            if kind=='retrieve_start':starts[e['request_id']]=e['monotonic_ns']
            elif kind=='retrieve_return':durations.append((e['monotonic_ns']-starts.pop(e['request_id']))/1e6)
            elif kind=='daos_demand_outcome':chunks+=e['returned_chunks']
            elif kind=='window_retrieve_done':total+=e['chunks']
    assert not starts and chunks==total
    results.append(dict(condition=spec['label'],retrieve_requests=len(durations),
        retrieve_wall_mean_ms=statistics.mean(durations),retrieve_wall_median_ms=statistics.median(durations),
        retrieve_wall_total_s=sum(durations)/1000,retrieve_copy_windows=counts['window_copy_start'],
        daos_read_batches=counts['daos_demand_outcome'],daos_returned_chunks=chunks,
        daos_returned_gib=chunks*18/1024,store_windows=counts['store_window_submitted']))
result=dict(results=results,note='Application retrieve wall times include metadata, orchestration and final scatter; these are not hardware RDMA operation timings. DAOS payload bytes equal all restored bytes in each condition.')
(root/'io_window_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
