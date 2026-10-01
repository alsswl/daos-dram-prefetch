"""Paired native DAOS/GDR placement benchmark, with full byte verification.

Same isolated OID for both layouts in each pair avoids object-placement bias;
the two key spaces are disjoint. Production cache OIDs are never opened.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import hashlib
import json
from pathlib import Path
import random
import re
import statistics
import subprocess
import threading
import time
import uuid

import numpy as np
import torch
from lmcache.utils import CacheEngineKey
from addressing import Addressing

HERE = Path(__file__).resolve().parent


def save(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def check(rc):
    if rc:
        raise RuntimeError(f'DAOS rc={rc}')


def pool_query():
    result = json.loads(subprocess.check_output(['daos', 'pool', 'query', 'discospool', '--json']))
    if result['status']:
        raise RuntimeError(result)
    return result


def load_library():
    lib = C.CDLL(str(HERE / 'placement.so'))
    signatures = {
        'placement_open': [C.c_char_p, C.c_char_p, C.c_uint64, C.POINTER(C.c_void_p), C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)],
        'placement_predict': [C.c_char_p, C.c_uint],
        'placement_shard': [C.c_void_p, C.c_char_p, C.POINTER(C.c_uint)],
        'placement_io': [C.c_void_p, C.c_char_p, C.c_char_p, C.c_void_p, C.c_size_t, C.c_int],
        'placement_size': [C.c_void_p, C.c_char_p, C.c_char_p, C.POINTER(C.c_uint64)],
        'placement_close': [C.c_void_p, C.c_int],
    }
    for name, signature in signatures.items():
        getattr(lib, name).argtypes = signature
        getattr(lib, name).restype = C.c_uint if name == 'placement_predict' else C.c_int
    return lib


def query_layout(oid):
    result = subprocess.run(['daos', 'object', 'query', 'discospool', 'kvcache', oid],
                            text=True, capture_output=True, check=True)
    layout = {int(g): (int(rank), int(target)) for g, rank, target in
              re.findall(r'grp: (\d+)\s+replica 0 (\d+):(\d+)', result.stdout)}
    if sorted(layout) != list(range(len(layout))) or not layout:
        raise RuntimeError('Unrecognized object layout')
    if len(set(layout.values())) != len(layout):
        raise RuntimeError('Experiment requires distinct targets for all S-class shards')
    return layout, result.stdout


def active_metrics(events, elapsed_ns):
    # Client call intervals, not proof of concurrent storage service or DMA.
    points = sorted([(e['start_ns'], e['shard'], 1) for e in events] +
                    [(e['end_ns'], e['shard'], -1) for e in events])
    counts = Counter(); last = 0; area = 0; peak = 0
    for timestamp, shard, delta in points:
        area += (timestamp-last) * sum(v > 0 for v in counts.values())
        counts[shard] += delta; peak = max(peak, counts[shard]); last = timestamp
    assert all(v == 0 for v in counts.values())
    return dict(mean_targets_with_outstanding_client_calls=area/elapsed_ns,
                peak_calls_to_one_target=peak)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--layouts', type=int, default=3)
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 16])
    parser.add_argument('--chunks', type=int, default=96)
    args = parser.parse_args()
    if min(args.repeats, args.layouts, args.chunks, *args.workers) < 1 or args.chunks > 113:
        raise ValueError('Positive counts required; all chunks must fit in 2 GiB')
    args.out.mkdir(parents=True, exist_ok=False)
    pool_before = pool_query(); save(args.out/'pool_before.json', pool_before)
    nvme = next(x for x in pool_before['response']['tier_stats'] if x['media_type'] == 'nvme')
    if nvme['free'] < 30*2**30 or nvme['min'] < 2*2**30:
        raise RuntimeError('Insufficient free capacity; no existing data will be removed')
    if pool_before['response']['disabled_targets'] or pool_before['response']['rebuild']['state'] != 'idle':
        raise RuntimeError('Requires healthy, idle-rebuild pool')
    lib = load_library()
    chunk_bytes = 18*2**20; total_bytes = args.chunks*chunk_bytes
    plan = dict(chunks=args.chunks, chunk_bytes=chunk_bytes, gpu_slab_bytes=2*2**30,
                workers=args.workers, repeats=args.repeats, layouts=args.layouts,
                modes=['baseline', 'balanced'], read_bytes=total_bytes,
                source_sha256={n: hashlib.sha256((HERE/n).read_bytes()).hexdigest()
                               for n in ['placement.c', 'placement.so', 'addressing.py', 'run.py']},
                notes=['Native payload-only benchmark matching the earlier 96-chunk grouping experiment.',
                       'One 18 MiB SINGLE/IOD/fetch per chunk in both modes; no batching or target-aware reordering.',
                       '2 GiB GPU buffer; no LMCache allocator/window/scatter, DRAM payload cache or model execution.',
                       'Both modes share a fresh isolated OID per pair, using disjoint key spaces.',
                       'balanced uses absolute chunk index modulo frozen shard count, and full CacheEngineKey as akey.',
                       'Balanced also changes dkey/akey hierarchy; this is not a pure hash-function comparison.',
                       'Warm reads without server cache flush. Timed calls include registration and client progress.',
                       'CPU oracle and full-byte verification outside timed region.',
                       'Interval concurrency measures outstanding CLIENT calls, not server-side service concurrency.',
                       'Only newly created experiment OIDs are punched at exit.'])
    save(args.out/'plan.json', plan); save(args.out/'status.json', dict(status='running'))
    torch.cuda.set_device(0); torch.manual_seed(20261002)
    slab = torch.empty(2*2**30, dtype=torch.uint8, device='cuda'); buf = slab[:total_bytes]
    buf.random_(0, 256); torch.cuda.synchronize()
    oracle = buf.cpu(); expected = oracle.numpy()
    save(args.out/'payload.json', dict(sha256=hashlib.sha256(expected).hexdigest()))
    rows = []; cleanup = []; rng = random.Random(20261002)
    try:
        for seed in range(args.layouts):
            pair_dir = args.out/f'pair_{seed}'; pair_dir.mkdir()
            ctx, hi, lo = C.c_void_p(), C.c_uint64(), C.c_uint64()
            nonce = (uuid.uuid4().int & ((1 << 63)-1)) | (1 << 62)
            check(lib.placement_open(b'discospool', b'kvcache', nonce, C.byref(ctx), C.byref(hi), C.byref(lo)))
            oid = f'{hi.value}.{lo.value}'
            # Record ownership immediately; cleanup is restricted to this handle.
            save(pair_dir/'ownership.json', dict(oid=oid, created_by_this_run=True))
            try:
                layout, raw_layout = query_layout(oid)
                (pair_dir/'object_layout.txt').write_text(raw_layout)
                count = len(layout)
                keys = [None]*count
                for salt in range(100000):
                    key = f'placement-v1/{seed}/{salt}'
                    bucket = lib.placement_predict(key.encode(), count)
                    if keys[bucket] is None: keys[bucket] = key
                    if all(k is not None for k in keys): break
                if not all(k is not None for k in keys): raise RuntimeError('Salt search exhausted')
                schemes = {mode: Addressing(mode, tuple(keys)) for mode in plan['modes']}
                save(pair_dir/'addressing.json', {mode: scheme.to_dict() for mode, scheme in schemes.items()})
                # Reconstruct the read mapping from the persisted descriptor.
                schemes = {mode: Addressing.from_dict(data) for mode, data in
                           json.loads((pair_dir/'addressing.json').read_text()).items()}
                full_keys = []; previous = f'prefix-seed-{seed}'.encode()
                for i in range(args.chunks):
                    previous = hashlib.sha256(previous + i.to_bytes(8, 'little')).digest()
                    key = CacheEngineKey(model_name='qwen4b-shaped-placement-v1', world_size=1,
                                         worker_id=0, chunk_hash=int.from_bytes(previous[:8], 'big'), dtype=torch.bfloat16)
                    full_keys.append('placement-experiment:' + key.to_string())
                cases = {}
                for mode, scheme in schemes.items():
                    placements = []
                    for i, full_key in enumerate(full_keys):
                        dk, ak = scheme.address(full_key, i)
                        predicted = lib.placement_predict(dk, count)
                        actual = C.c_uint(); check(lib.placement_shard(ctx, dk, C.byref(actual)))
                        if actual.value != predicted: raise RuntimeError('Installed DAOS routing differs from predictor')
                        if mode == 'balanced' and actual.value != i % count: raise RuntimeError('Unbalanced mapping')
                        placements.append(dict(chunk=i, full_key=full_key, dkey=dk.decode(), akey=ak.decode(),
                                               shard=actual.value, rank=layout[actual.value][0], target=layout[actual.value][1]))
                    cases[mode] = placements
                addresses = [{(p['dkey'], p['akey']) for p in cases[mode]} for mode in plan['modes']]
                assert not addresses[0].intersection(addresses[1])
                save(pair_dir/'placements.json', cases)
                executors = {w: ThreadPoolExecutor(max_workers=w, initializer=lambda: torch.cuda.set_device(0)) for w in args.workers}
                try:
                    for w, executor in executors.items():
                        barrier = threading.Barrier(w)
                        list(executor.map(lambda _: barrier.wait(), range(w)))
                    writer = executors[max(args.workers)]
                    buf.copy_(oracle); torch.cuda.synchronize()
                    def write(p):
                        check(lib.placement_io(ctx, p['dkey'].encode(), p['akey'].encode(),
                                               buf.data_ptr()+p['chunk']*chunk_bytes, chunk_bytes, 1))
                    store_order = list(plan['modes']); rng.shuffle(store_order)
                    for mode in store_order: list(writer.map(write, cases[mode]))
                    for mode, placements in cases.items():
                        for p in placements:
                            size = C.c_uint64()
                            check(lib.placement_size(ctx, p['dkey'].encode(), p['akey'].encode(), C.byref(size)))
                            assert size.value == chunk_bytes
                        size = C.c_uint64()
                        check(lib.placement_size(ctx, placements[0]['dkey'].encode(), b'absent-key', C.byref(size)))
                        assert size.value == 0
                    # Wrong absolute index must not silently find a different chunk.
                    if count > 1:
                        dk, ak = schemes['balanced'].address(full_keys[0], 1)
                        size = C.c_uint64(); check(lib.placement_size(ctx, dk, ak, C.byref(size)))
                        assert size.value == 0
                    print(json.dumps(dict(event='stored_and_lookup_verified', seed=seed, oid=oid,
                                          per_shard={m: dict(Counter(p['shard'] for p in ps)) for m, ps in cases.items()})), flush=True)
                    for rep in range(-1, args.repeats):
                        order = [(mode, w) for mode in plan['modes'] for w in args.workers]; rng.shuffle(order)
                        for mode, workers in order:
                            # Index is absolute even when lookup starts in the middle.
                            jobs = [(p, schemes[mode].address(p['full_key'], p['chunk'])) for p in cases[mode]]
                            buf.zero_(); torch.cuda.synchronize()
                            def fetch(job):
                                p, (dk, ak) = job
                                begin = time.perf_counter_ns()
                                rc = lib.placement_io(ctx, dk, ak, buf.data_ptr()+p['chunk']*chunk_bytes, chunk_bytes, 0)
                                end = time.perf_counter_ns(); check(rc)
                                return dict(chunk=p['chunk'], shard=p['shard'], rank=p['rank'], target=p['target'],
                                            start_ns=begin, end_ns=end)
                            start = time.perf_counter_ns()
                            events = list(executors[workers].map(fetch, jobs))
                            torch.cuda.synchronize(); elapsed = time.perf_counter_ns()-start
                            for e in events: e['start_ns']-=start; e['end_ns']-=start
                            assert np.array_equal(buf.cpu().numpy(), expected), (seed, mode, workers, rep)
                            row = dict(seed=seed, mode=mode, workers=workers, repeat=rep, calls=len(events),
                                       ms=elapsed/1e6, gib_s=total_bytes/2**30/(elapsed/1e9), verified=True,
                                       events=events, **active_metrics(events, elapsed))
                            rows.append(row)
                            with (args.out/'measurements.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                        print(json.dumps(dict(event='round_complete', seed=seed, repeat=rep)), flush=True)
                    # Fetch a suffix with absolute positions to catch accidental window-local indexing.
                    suffix_start = min(27, args.chunks-1)
                    for mode in plan['modes']:
                        buf.zero_(); torch.cuda.synchronize()
                        for p in cases[mode][suffix_start:]:
                            dk, ak = schemes[mode].address(p['full_key'], p['chunk'])
                            check(lib.placement_io(ctx, dk, ak, buf.data_ptr()+p['chunk']*chunk_bytes, chunk_bytes, 0))
                        torch.cuda.synchronize()
                        assert np.array_equal(buf[suffix_start*chunk_bytes:].cpu().numpy(), expected[suffix_start*chunk_bytes:])
                    after_layout, _ = query_layout(oid)
                    assert after_layout == layout
                    save(pair_dir/'validation.json', dict(placement_predictions_match=True, lookup_sizes_match=True,
                                                         missing_akey_rejected=True, suffix_read_verified=True,
                                                         layout_unchanged=True))
                finally:
                    for executor in executors.values(): executor.shutdown(wait=True)
            finally:
                rc = lib.placement_close(ctx, 1)
                cleanup.append(dict(oid=oid, punch_and_close_rc=rc, production_object_touched=False))
                save(args.out/'cleanup.json', cleanup); check(rc)
        summary = []
        for workers in args.workers:
            for mode in plan['modes']:
                selected = [r for r in rows if r['repeat']>=0 and r['workers']==workers and r['mode']==mode]
                summary.append(dict(mode=mode, workers=workers, samples=len(selected),
                                    median_ms=statistics.median(r['ms'] for r in selected),
                                    per_seed_median_ms=[statistics.median(r['ms'] for r in selected if r['seed']==s) for s in range(args.layouts)],
                                    mean_targets_with_outstanding_client_calls=statistics.mean(r['mean_targets_with_outstanding_client_calls'] for r in selected),
                                    mean_peak_calls_to_one_target=statistics.mean(r['peak_calls_to_one_target'] for r in selected)))
        save(args.out/'summary.json', summary)
        pool_after = pool_query(); save(args.out/'pool_after.json', pool_after)
        assert pool_after['response']['version'] == pool_before['response']['version']
        save(args.out/'status.json', dict(status='completed', verified_reads=len(rows),
                                         measured_reads=sum(r['repeat']>=0 for r in rows),
                                         objects_removed=len(cleanup), all_verified=True))
        print(json.dumps(dict(event='completed', summary=summary)), flush=True)
    except BaseException as e:
        save(args.out/'status.json', dict(status='failed', error=repr(e), completed_reads=len(rows)))
        raise


if __name__ == '__main__':
    main()
