import json, statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parent
results=[]
for root in (ROOT.parent/'realqa_mixed_capacity2g_20261001_shared',ROOT):
    plan=json.loads((root/'plan.json').read_text());case=root/plan['cases'][0]['name']
    assert json.loads((root/'status.json').read_text())['status']=='completed'
    pending={};times=[];counts={};daos_chunks=store_chunks=0
    for path in case.glob('trace.*.jsonl'):
        for line in path.open():
            e=json.loads(line);kind=e['event'];counts[kind]=counts.get(kind,0)+1
            if kind=='retrieve_start':pending[e['request_id']]=e['monotonic_ns']
            elif kind=='retrieve_return':
                times.append((e['monotonic_ns']-pending.pop(e['request_id']))/1e6)
            elif kind=='daos_demand_outcome':daos_chunks+=e['returned_chunks']
            elif kind=='store_window_submitted':store_chunks+=e['chunks']
    assert not pending
    results.append(dict(root=str(root),configured_window_mib=plan['retrieve_window_mib'],
        retrieve_requests=len(times),retrieve_wall_total_s=sum(times)/1000,
        retrieve_wall_mean_ms=statistics.mean(times),retrieve_wall_median_ms=statistics.median(times),
        retrieve_copy_windows=counts.get('window_copy_start',0),
        daos_read_batches=counts.get('daos_demand_outcome',0),daos_returned_chunks=daos_chunks,
        daos_returned_gib=daos_chunks*18/1024,store_windows=counts.get('store_window_submitted',0),
        store_submitted_chunks=store_chunks,store_submitted_gib=store_chunks*18/1024))
(ROOT/'io_window_metrics.json').write_text(json.dumps(dict(results=results,
    note='Retrieve wall time includes DRAM/DAOS, orchestration and scatter. Chunk counts describe application payloads, not hardware RDMA operations. Actual generated histories and cache tier hit patterns can differ.'),indent=2)+'\n')
print(json.dumps(results,indent=2))
