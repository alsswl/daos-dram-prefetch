"""Validate and plot the controlled target-count experiment."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics as st
import numpy as np


def save(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('root', type=Path)
    root = parser.parse_args().root
    plan = json.loads((root/'plan.json').read_text())
    status = json.loads((root/'status.json').read_text()); assert status['status']=='completed'
    rows = [json.loads(line) for line in (root/'measurements.jsonl').read_text().splitlines()]
    assert len(rows)==plan['layouts']*(plan['repeats']+1)*5
    assert len({(r['layout'], r['repeat'], r['target_count']) for r in rows})==len(rows)
    cleanup = json.loads((root/'cleanup.json').read_text())
    assert len(cleanup)==plan['layouts'] and all(r['punch_and_close_rc']==0 for r in cleanup)
    for name, digest in plan['source_sha256'].items():
        assert hashlib.sha256((root/'sources'/name).read_bytes()).hexdigest()==digest
    target_stats = []
    for row in rows:
        assert row['verified'] and row['calls']==96 and row['workers']==16
        assert row['read_bytes']==96*18*2**20
        events = row['events']; assert sorted(e['chunk'] for e in events)==list(range(96))
        placements = json.loads((root/f"layout_{row['layout']}"/'placements.json').read_text())[str(row['target_count'])]
        for e, p in zip(events, placements):
            assert all(e[k]==p[k] for k in ['chunk', 'rank', 'target', 'shard'])
            assert 0<=e['start_ns']<e['end_ns']<=row['ms']*1e6+1
        counts = Counter((e['rank'], e['target']) for e in events)
        assert len(counts)==row['target_count'] and set(counts.values())=={96//row['target_count']}
        current=0; peak=0
        for _, delta in sorted([(e['start_ns'], 1) for e in events]+[(e['end_ns'], -1) for e in events]):
            current+=delta; peak=max(peak, current)
        assert current==0 and peak<=16
        row['peak_client_calls']=peak
        durations=[(e['end_ns']-e['start_ns'])/1e6 for e in events]
        row['fetch_median_ms']=float(np.median(durations)); row['fetch_p95_ms']=float(np.percentile(durations,95))
        for (rank, target), count in counts.items():
            es=[e for e in events if (e['rank'], e['target'])==(rank,target)]
            ds=[(e['end_ns']-e['start_ns'])/1e6 for e in es]
            target_stats.append(dict(layout=row['layout'], repeat=row['repeat'], target_count=row['target_count'],
                                     rank=rank, target=target, chunks=count, median_fetch_ms=float(np.median(ds)),
                                     p95_fetch_ms=float(np.percentile(ds,95)),
                                     last_completion_ms=max(e['end_ns'] for e in es)/1e6))
    measured = [r for r in rows if r['repeat']>=0]
    summary=[]
    for count in plan['target_counts']:
        rs=[r for r in measured if r['target_count']==count]
        summary.append(dict(targets=count, samples=len(rs), median_ms=st.median(r['ms'] for r in rs),
                            median_gib_s=st.median(r['gib_s'] for r in rs),
                            per_layout_median_ms=[st.median(r['ms'] for r in rs if r['layout']==s) for s in range(plan['layouts'])],
                            median_per_run_fetch_median_ms=st.median(r['fetch_median_ms'] for r in rs),
                            median_per_run_fetch_p95_ms=st.median(r['fetch_p95_ms'] for r in rs),
                            mean_targets_with_outstanding_client_calls=st.mean(r['mean_targets_with_outstanding_client_calls'] for r in rs),
                            min_peak_client_calls=min(r['peak_client_calls'] for r in rs),
                            max_peak_client_calls=max(r['peak_client_calls'] for r in rs)))
    save(root/'summary.json', summary); save(root/'per_target_latency.json', target_stats)
    report = [
        'Controlled DAOS/GDR target-spread experiment',
        'Fixed: 16 worker threads, 96 distinct dkeys, one 18 MiB SINGLE value and one fetch per dkey.',
        'Total per read: 1.6875 GiB. GPU slab: 2 GiB. No LMCache window/scatter or model execution.',
        f"Sampling: {plan['layouts']} target selections/object layouts, {plan['repeats']} measured reads per condition per layout, plus warmup.",
        'The five conditions share one object per layout. Execution order is randomized every round.',
        '',
        'Targets | Median whole-read ms | Median GiB/s | Mean targets with outstanding client fetches | Layout medians ms',
    ]
    for s in summary:
        report.append(f"{s['targets']:7d} | {s['median_ms']:20.3f} | {s['median_gib_s']:12.3f} | {s['mean_targets_with_outstanding_client_calls']:42.3f} | " + ', '.join(f'{x:.3f}' for x in s['per_layout_median_ms']))
    report += [
        '',
        f"1-to-16 target speedup: {summary[0]['median_ms']/summary[-1]['median_ms']:.3f}x.",
        f"2-to-16 target speedup (both use two engines): {summary[1]['median_ms']/summary[-1]['median_ms']:.3f}x.",
        '',
        'Interpretation: fixed client concurrency and fetch count do not ensure equal performance; physical target spread matters.',
        'Limitations:',
        '- Warm reads; server caches were not flushed. This is not a cold NVMe benchmark.',
        '- All timings are client observed and include MR, progress, network, and server work; not target service time.',
        '- One-to-two targets also changes engine count; 2/4/8/16 use both engines equally.',
        '- Repeats within one layout are not independent object/target selections.',
        '- Targets can share engine CPU, NIC, and storage resources; linear scaling is not assumed.',
        '- This demonstrates target-spread sensitivity, not the effect size of natural hashing versus balanced placement in serving.',
        '- Only fresh experiment objects were removed; existing production/cache objects were never opened by the benchmark.',
        '',
        'Artifacts: measurements.jsonl (all call intervals); per_target_latency.json (per-target fetch latency and last completion);',
        'comparison.png/svg; target_timeline.png/svg; plan.json; validation.json; cleanup.json; sources/.',
    ]
    (root/'report.txt').write_text('\n'.join(report)+'\n')
    save(root/'validation.json', dict(passed=True, measured_reads=len(measured), verified_reads=len(rows),
                                     same_fetch_count_size_and_workers=True, all_placements_verified=True,
                                     no_more_than_16_client_calls=True, sources_match=True, own_objects_removed=len(cleanup)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10})
    fig, axes=plt.subplots(1,3,figsize=(14,4.3),layout='constrained')
    xs=np.arange(len(summary)); labels=[str(s['targets']) for s in summary]
    axes[0].plot(xs,[s['median_ms'] for s in summary],'-o',label=f"Median of {summary[0]['samples']} reads")
    for x,s in zip(xs,summary):
        axes[0].scatter([x]*len(s['per_layout_median_ms']),s['per_layout_median_ms'],color='gray',marker='x',alpha=.7)
    axes[0].set_ylabel('Whole-read completion time (ms)'); axes[0].set_title('Same 96 fetches; varying placement')
    axes[0].legend(fontsize=8)
    axes[1].plot(xs,[s['median_gib_s'] for s in summary],'-o',color='#258365')
    axes[1].set_ylabel('Read throughput (GiB/s)'); axes[1].set_title('96 chunks × 18 MiB = 1.6875 GiB')
    axes[2].plot(xs,[s['mean_targets_with_outstanding_client_calls'] for s in summary],'-o',label='Measured client intervals')
    axes[2].plot(xs,plan['target_counts'],'--',color='gray',label='Selected target count')
    axes[2].set_ylabel('Time-average targets with pending fetch'); axes[2].set_title('Client-observed target concurrency')
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(xs,labels); ax.set_xlabel('Targets used'); ax.grid(alpha=.2)
    fig.suptitle(f"DAOS + GDR: fixed 16 workers, one dkey and one fetch per chunk\n{plan['layouts']} target selections/object layouts × {plan['repeats']} measured warm reads; gray crosses: layout medians",fontsize=11)
    for suffix in ['png','svg']: fig.savefig(root/f'comparison.{suffix}',dpi=180)
    plt.close(fig)
    # All panels use one object layout and identical physical target row order.
    representatives=[]
    for count in plan['target_counts']:
        rs=[r for r in measured if r['target_count']==count and r['layout']==0]
        med=st.median(r['ms'] for r in rs)
        representatives.append(min(rs,key=lambda r:abs(r['ms']-med)))
    alltargets=sorted({(e['rank'],e['target']) for e in representatives[-1]['events']})
    tmax=max(r['ms'] for r in representatives); bins=700
    grid=(np.arange(bins)+.5)*tmax/bins
    fig,axes=plt.subplots(5,1,figsize=(11,11),sharex=True,layout='constrained')
    for ax,row in zip(axes,representatives):
        values=np.zeros((16,bins))
        for e in row['events']:
            ix=alltargets.index((e['rank'],e['target']))
            values[ix]+=(grid>=e['start_ns']/1e6)&(grid<e['end_ns']/1e6)
        im=ax.imshow(values,aspect='auto',origin='lower',extent=[0,tmax,-.5,15.5],vmin=0,vmax=16,cmap='viridis',interpolation='nearest')
        ax.set_yticks(range(16),[f'{r}:{t}' for r,t in alltargets],fontsize=6)
        ax.set_ylabel('Engine:target')
        ax.set_title(f"{row['target_count']} targets, read completes at {row['ms']:.1f} ms",fontsize=10)
        ax.axvline(row['ms'],color='red',lw=.8,ls='--')
    axes[-1].set_xlabel('Time since read submission (ms)')
    fig.colorbar(im,ax=axes,label='Outstanding client fetch calls (not server service concurrency)',shrink=.75)
    fig.suptitle('Target request timelines: layout 0, median-time repeat per condition',fontsize=12)
    for suffix in ['png','svg']: fig.savefig(root/f'target_timeline.{suffix}',dpi=180)
    plt.close(fig)
    save(root/'timeline_selection.json',[dict(layout=r['layout'], repeat=r['repeat'], targets=r['target_count'], ms=r['ms']) for r in representatives])
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
