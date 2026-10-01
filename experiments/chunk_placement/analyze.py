"""Validate completed runs and report paired layout results; optional plot."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args(); root = args.root
    plan = json.loads((root/'plan.json').read_text())
    status = json.loads((root/'status.json').read_text())
    assert status['status'] == 'completed' and status['all_verified']
    for name, digest in plan['source_sha256'].items():
        assert hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() == digest, name
    rows = [json.loads(s) for s in (root/'measurements.jsonl').read_text().splitlines()]
    assert len(rows) == plan['layouts']*(plan['repeats']+1)*len(plan['workers'])*2
    identities = {(r['seed'], r['mode'], r['workers'], r['repeat']) for r in rows}
    assert len(identities) == len(rows)
    for r in rows:
        assert r['verified'] and r['calls'] == plan['chunks'] and len(r['events']) == plan['chunks']
        assert [e['chunk'] for e in r['events']] == list(range(plan['chunks']))
        assert all(0 <= e['start_ns'] < e['end_ns'] <= r['ms']*1e6+1 for e in r['events'])
    cleanup = json.loads((root/'cleanup.json').read_text())
    assert len(cleanup) == plan['layouts']
    assert all(c['punch_and_close_rc'] == 0 and not c['production_object_touched'] for c in cleanup)
    distributions = []
    for seed in range(plan['layouts']):
        pair = root/f'pair_{seed}'
        assert all(json.loads((pair/'validation.json').read_text()).values())
        placement = json.loads((pair/'placements.json').read_text())
        balanced = Counter(p['shard'] for p in placement['balanced'])
        n = len(balanced)
        assert all(p['shard'] == p['chunk'] % n for p in placement['balanced'])
        for start in range(0, plan['chunks']-n+1):
            assert len({p['shard'] for p in placement['balanced'][start:start+n]}) == n
        distributions.append({mode: [Counter(p['shard'] for p in ps).get(i, 0) for i in range(n)]
                              for mode, ps in placement.items()})
    comparisons = []
    for w in plan['workers']:
        pairs = []
        for seed in range(plan['layouts']):
            medians = {mode: statistics.median(r['ms'] for r in rows if r['seed']==seed and
                       r['workers']==w and r['mode']==mode and r['repeat']>=0) for mode in plan['modes']}
            pairs.append(dict(seed=seed, **medians,
                              reduction_percent=100*(1-medians['balanced']/medians['baseline'])))
        comparisons.append(dict(workers=w, per_pair=pairs,
                                median_pair_reduction_percent=statistics.median(p['reduction_percent'] for p in pairs)))
    report = dict(passed=True, verified_timed_reads=len(rows), measured_reads=sum(r['repeat']>=0 for r in rows),
                  placement_verified=True, balanced_every_contiguous_shard_count_chunks=True,
                  isolated_objects_removed=len(cleanup), comparisons=comparisons,
                  counts_by_shard=distributions,
                  limitations=['Native payload benchmark, not CXS TTFT or integrated LMCache.',
                               'Only three fresh layouts by default; repeat timings are not independent deployments.',
                               'Client call intervals include local registration and scheduling delays.',
                               'balanced changes both placement and dkey/akey indexing hierarchy.'])
    (root/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
    if args.plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
        for result in comparisons:
            if result['workers'] != 16: continue
            for pair in result['per_pair']:
                axes[0].plot([0, 1], [pair['baseline'], pair['balanced']], 'o-', label=f"Layout {pair['seed']+1}")
        axes[0].set_xticks([0, 1], ['Existing hash placement', 'Balanced placement'])
        axes[0].set_ylabel('Median read latency (ms)'); axes[0].set_title('96 x 18 MiB, 16 I/O workers')
        axes[0].legend(); axes[0].grid(axis='y', alpha=.3)
        distribution = distributions[0]; n = len(distribution['balanced'])
        axes[1].bar([i-.2 for i in range(n)], distribution['baseline'], .4, label='Existing')
        axes[1].bar([i+.2 for i in range(n)], distribution['balanced'], .4, label='Balanced')
        axes[1].set_xticks(range(n)); axes[1].set_xlabel('Shard (one distinct target each)')
        axes[1].set_ylabel('Chunks'); axes[1].set_title('Layout 1: stored chunk distribution'); axes[1].legend()
        fig.savefig(root/'comparison.png', dpi=170); fig.savefig(root/'comparison.svg')


if __name__ == '__main__': main()
