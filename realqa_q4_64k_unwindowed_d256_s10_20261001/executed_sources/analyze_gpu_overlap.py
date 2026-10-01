#!/usr/bin/env python3
"""Measured CUDA copy/kernel overlap; DAOS host spans are explicitly not DMA time."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import re
import statistics


def merge(intervals):
    out = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if out and start <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], end))
        else:
            out.append((start, end))
    return out


class Intersections:
    def __init__(self, spans):
        self.spans = merge(spans)
        self.ends = [end for _, end in self.spans]

    def overlap(self, start, end):
        total = 0
        for i in range(bisect_right(self.ends, start), len(self.spans)):
            a, b = self.spans[i]
            if a >= end:
                break
            total += max(0, min(end, b)-max(start, a))
        return total


def kernel_gap_stats(intervals):
    """Between first/last kernel only; no kernel does not imply no GPU DMA."""
    spans = merge(intervals)
    if len(spans) < 2:
        return dict(gaps_over_1ms=0, total_gaps_over_1ms_ms=0, mean_gap_over_1ms_ms=None)
    gaps = [(right[0]-left[1])/1000 for left, right in zip(spans, spans[1:])
            if right[0]-left[1] >= 1000]
    return dict(gaps_over_1ms=len(gaps), total_gaps_over_1ms_ms=sum(gaps),
                mean_gap_over_1ms_ms=statistics.mean(gaps) if gaps else None,
                maximum_gap_ms=max(gaps, default=0),
                active_window_seconds=(spans[-1][1]-spans[0][0])/1e6)


def analyze(case, output=None):
    traces = list((case/'gpu_trace').glob('*.pt.trace.json.gz'))
    assert len(traces) == 1, f'Expected one worker trace, got {traces}'
    with gzip.open(traces[0], 'rt') as f:
        trace = json.load(f)
    events = trace['traceEvents']
    epoch = trace['baseTimeNanoseconds']
    spans = [e for e in events if e.get('cat') == 'Trace' and e.get('ph') == 'X']
    assert len(spans) == 1
    window = (spans[0]['ts'], spans[0]['ts']+spans[0]['dur'])
    kernels = [e for e in events if e.get('cat') == 'kernel']
    assert kernels
    # Conservative recognized model compute, not KV packing/gather/copy kernels.
    compute = [e for e in kernels if re.search(r'gemm|nvjet|cutlass|flash|fmha|attention', e['name'], re.I)
               and not re.search(r'cache|transfer|gather|reshape', e['name'], re.I)]
    assert compute, 'No recognized model compute kernels'
    busy = Intersections((e['ts'], e['ts']+e['dur']) for e in compute)
    any_kernel = Intersections((e['ts'], e['ts']+e['dur']) for e in kernels)
    runtime = {e['args']['correlation']: e for e in events
               if e.get('cat') == 'cuda_runtime' and 'Memcpy' in e['name']
               and 'correlation' in e.get('args', {})}
    copies = [e for e in events if e.get('cat') == 'gpu_memcpy'
              and 'HtoD' in e['name'] and e.get('args', {}).get('bytes') == 20*2**20]
    # Convert host monotonic timestamps to Kineto's documented epoch-relative us.
    def host_us(e, mono=None):
        return (e['time_ns']-epoch+(0 if mono is None else mono-e['monotonic_ns']))/1000
    host, copy_batches, daos, starts = [], [], [], {}
    for path in case.glob('trace.*.jsonl'):
        for line in path.open():
            e = json.loads(line)
            ts = host_us(e)
            if window[0]-1e6 <= ts <= window[1]+1e6:
                host.append(e)
            if e['event'] == 'cpu_prefetch_timing' and 'copy_start_ns' in e:
                a, b = host_us(e, e['copy_start_ns']), host_us(e, e['copy_end_ns'])
                if a >= window[0] and b <= window[1]:
                    copy_batches.append(dict(rid=e['request_id'], start=a, end=b,
                                             chunks=e['chunks'], bytes=e['bytes']))
            if e['event'] == 'prefetch_start':
                starts[e['request_id']] = e
            elif e['event'] == 'prefetch_ready' and e['request_id'] in starts:
                first = starts.pop(e['request_id'])
                a, b = host_us(first), ts
                if a >= window[0] and b <= window[1]:
                    daos.append(dict(rid=e['request_id'], start=a, end=b,
                                     chunks=first['chunks']))
    copy_batches.sort(key=lambda e:e['start'])
    batch_starts = [e['start'] for e in copy_batches]
    matched, excluded = [], Counter()
    for e in copies:
        api = runtime.get(e['args']['correlation'])
        if api is None:
            excluded['missing_runtime_correlation'] += 1
            continue
        i = bisect_right(batch_starts, api['ts'])-1
        if i < 0 or not copy_batches[i]['start'] <= api['ts'] <= copy_batches[i]['end']:
            excluded['outside_complete_prefetch_batch'] += 1
            continue
        batch = copy_batches[i]
        assert e['ts'] >= batch['start']-100 and e['ts']+e['dur'] <= batch['end']+100
        matched.append(dict(rid=batch['rid'], start=e['ts'], end=e['ts']+e['dur'],
                            bytes=e['args']['bytes'], stream=e['args']['stream'],
                            correlation=e['args']['correlation']))
    counts = Counter(e['rid'] for e in matched)
    valid = {b['rid'] for b in copy_batches if counts[b['rid']] == b['chunks']}
    incomplete = [dict(b, observed=counts[b['rid']]) for b in copy_batches if b['rid'] not in valid]
    complete_copies = [e for e in matched if e['rid'] in valid]
    assert complete_copies, 'No complete CPU-prefetch DMA batches; cannot measure requested overlap'
    def summarize(rows):
        total = sum(e['end']-e['start'] for e in rows)
        overlap = sum(busy.overlap(e['start'], e['end']) for e in rows)
        all_overlap = sum(any_kernel.overlap(e['start'], e['end']) for e in rows)
        return dict(intervals=len(rows), summed_interval_ms=total/1000,
                    overlap_recognized_compute_ms=overlap/1000,
                    recognized_overlap_pct=100*overlap/total if total else None,
                    overlap_any_kernel_ms=all_overlap/1000,
                    any_kernel_overlap_pct=100*all_overlap/total if total else None,
                    intervals_with_compute_overlap=sum(busy.overlap(e['start'],e['end'])>0 for e in rows))
    gpu = summarize(complete_copies)
    gpu.update(request_batches=len(valid), bytes=sum(e['bytes'] for e in complete_copies),
               streams=sorted({e['stream'] for e in complete_copies}))
    retrievals = {e['request_id']: host_us(e) for e in host if e['event'] == 'retrieve_start'}
    last_copy = {}
    for e in complete_copies:
        last_copy[e['rid']] = max(last_copy.get(e['rid'], e['end']), e['end'])
    lead = [(retrievals[rid]-end)/1000 for rid,end in last_copy.items() if rid in retrievals]
    gpu['copy_finish_to_retrieve_start'] = dict(
        requests=len(lead), mean_ms=statistics.mean(lead) if lead else None,
        positive_means_ready_before_retrieve=True,
        ready_before_retrieve=sum(v>=0 for v in lead))
    summary = dict(trace=str(traces[0]), profiler_window_seconds=(window[1]-window[0])/1e6,
        gpu_kernels=len(kernels), recognized_compute_kernels=len(compute),
        recognized_kernel_examples=Counter(e['name'] for e in compute).most_common(8),
        dram_h2d_actual_dma=gpu, daos_host_prefetch_intervals=summarize(daos),
        recognized_compute_busy_pct=100*busy.overlap(*window)/(window[1]-window[0]),
        any_kernel_busy_pct=100*any_kernel.overlap(*window)/(window[1]-window[0]),
        first_kernel_delay_ms=(any_kernel.spans[0][0]-window[0])/1000,
        between_kernel_gaps=kernel_gap_stats(any_kernel.spans),
        incomplete_batches=incomplete, excluded_copies=dict(excluded),
        limitations=['Diagnostic profiled run, not a speedup benchmark.',
                     'Only completely captured prefetch batches are counted.',
                     'Recognized compute is a conservative subset of model kernels; any-kernel also includes KV transforms.',
                     'DAOS host interval includes submission, allocation, network and completion. Not measured NIC DMA duration.',
                     'Intervals overlap across requests: summed ms are not total wall time.',
                     'Temporal overlap does not establish equal milliseconds of TTFT reduction.'])
    analysis = output if output is not None else case.parent/'overlap_analysis'
    analysis.mkdir(exist_ok=True)
    (analysis/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (analysis/'dram_copies.json').write_text(json.dumps(complete_copies)+'\n')
    (analysis/'daos_host_spans.json').write_text(json.dumps(daos)+'\n')
    # Choose the naturally observed 500ms window with most CPU-prefetch copy time.
    candidates = [e['start'] for e in complete_copies]
    chosen = max(candidates, key=lambda t: sum(max(0,min(t+500000,e['end'])-max(t,e['start'])) for e in complete_copies))
    start, end = chosen, min(chosen+500000, window[1])
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="350">',
           '<rect width="1200" height="350" fill="white"/>',
           '<g font-family="sans-serif" font-size="14">',
           '<text x="20" y="28">Observed GPU overlap: metadata-first, DRAM256 / staging8 GiB, C16</text>',
           '<text x="20" y="52">Diagnostic trace; 500ms window selected for most DRAM DMA activity, not representative utilization.</text>']
    rows = [("Model GEMM / attention", busy.spans, '#0072b2'),
            ("DRAM to GPU DMA", [(e['start'], e['end']) for e in complete_copies], '#d55e00'),
            ("DAOS read host span", [(e['start'], e['end']) for e in daos], '#009e73')]
    for idx,(label,intervals,color) in enumerate(rows):
        y = 95+idx*65
        svg.append(f'<text x="15" y="{y+16}">{label}</text>')
        svg.append(f'<line x1="205" x2="1180" y1="{y+28}" y2="{y+28}" stroke="#ccc"/>')
        for a,b in merge(intervals):
            a,b=max(start,a),min(end,b)
            if b <= a: continue
            x=205+975*(a-start)/(end-start); width=975*(b-a)/(end-start)
            svg.append(f'<rect x="{x:.3f}" y="{y}" width="{width:.3f}" height="25" fill="{color}"/>')
    for i in range(6):
        svg.append(f'<text x="{205+975*i/5:.1f}" y="300">{(end-start)/1000*i/5:.0f}ms</text>')
    svg.append('<text x="20" y="334">Aligned time axis. Green is a host I/O interval, NOT measured DAOS NIC DMA.</text></g></svg>')
    (analysis/'timeline.svg').write_text('\n'.join(svg))
    try:
        import cairosvg
        cairosvg.svg2png(url=str(analysis/'timeline.svg'), write_to=str(analysis/'timeline.png'))
    except ImportError:
        pass
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('case', type=Path)
    analyze(p.parse_args().case.resolve())
