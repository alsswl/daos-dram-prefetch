#!/usr/bin/env python3
"""Print aggregates and validity checks from compare_gpu_store.py artifacts."""
import argparse
from collections import defaultdict
import json
import re
import statistics as st


def main():
    from pathlib import Path
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder', type=Path)
    a = p.parse_args()
    records = json.loads((a.folder/'results.json').read_text())
    summary = {}
    for mode in ['host_staged', 'gpu_direct']:
        for concurrency in [1, 4]:
            for phase in ['cold', 'warm']:
                part = [r for r in records if (r['mode'], r['concurrency'], r['phase']) ==
                        (mode, concurrency, phase)]
                if not part:
                    continue
                responses = [x for r in part for x in r['responses']]
                key = f'{mode}/c{concurrency}/{phase}'
                summary[key] = dict(repeats=len(part),
                    batch_ms=st.mean(r['http_batch_ms'] for r in part),
                    batch_range_ms=[min(r['http_batch_ms'] for r in part), max(r['http_batch_ms'] for r in part)],
                    e2e_ms=st.mean(x['e2e_ms'] for x in responses),
                    ttft_ms=st.mean(x['first_token_ms'] for x in responses),
                    drain_inclusive_batch_ms=st.mean(r['http_plus_drain_ms'] for r in part),
                    staging_peak_gib=max(r['peak_staging_gib'] for r in part),
                    manager_copy_ms_per_request=st.mean(r['manager_copy_ms']/4 for r in part),
                    manager_copy_gib=sum(r['manager_copy_bytes'] for r in part)/2**30,
                    cached_tokens=sorted(set(x['cached_tokens'] for x in responses)))
    store = defaultdict(list)
    native = []
    for folder in sorted(a.folder.glob('r*_*')):
        groups = {}
        pattern = (r'\[req_id=(.*?)\] Stored (\d+) out of total (\d+) tokens\. '
                   r'size: .*?cost ([\d.]+) ms, .*?offload_time: ([\d.]+) ms, put_time: ([\d.]+) ms')
        for req, n, total, cost, offload, put in re.findall(pattern, (folder/'server.log').read_text()):
            v = groups.setdefault(req, [0, 0., 0., 0.])
            assert n == total, 'Partial store'
            for i, value in enumerate([int(n), float(cost), float(offload), float(put)]):
                v[i] += value
        vals = list(groups.values())
        if len(vals) != 9:
            continue  # Live/incomplete process. Never treat it as a full sample.
        assert all(v[0] == 8192 for v in vals)
        mode = folder.name.split('_', 1)[1]
        for concurrency, part in [(1, vals[1:5]), (4, vals[5:9])]:
            store[f'{mode}/c{concurrency}'].extend(part)
        native.append(json.loads((folder/'native_maps.json').read_text()))
    store_summary = {k: dict(requests=len(v), store_submit_ms=st.mean(x[1] for x in v),
                        offload_ms=st.mean(x[2] for x in v), put_submit_ms=st.mean(x[3] for x in v))
                     for k, v in store.items()}
    outputs = {}
    for r in records:
        for x in r['responses']:
            outputs[(r['repeat'], r['mode'], r['phase'], x['request_id'])] = x['output_token_sha256']
    comparisons = defaultdict(lambda: [0, 0])
    for (repeat, mode, phase, req), value in outputs.items():
        if mode == 'host_staged':
            other = outputs.get((repeat, 'gpu_direct', phase, req))
            if other:
                comparisons['host_vs_gpu_' + phase][0] += 1
                comparisons['host_vs_gpu_' + phase][1] += value == other
        if phase == 'cold':
            other = outputs.get((repeat, mode, 'warm', req))
            if other:
                comparisons[mode + '_cold_vs_warm'][0] += 1
                comparisons[mode + '_cold_vs_warm'][1] += value == other
    print(json.dumps(dict(http=summary, store_submit=store_summary,
        token_equality_checks_total_and_equal=dict(comparisons),
        native_stacks_equal=all(x == native[0] for x in native) if native else None,
        native_stack=native[0] if native else None), indent=2))


if __name__ == '__main__':
    main()
