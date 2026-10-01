"""Control target count without changing workers, fetch size, or key hierarchy."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import ctypes as C
import hashlib
import json
from pathlib import Path
import random
import shutil
import threading
import time
import uuid

import numpy as np
import torch
from run import HERE, active_metrics, check, load_library, pool_query, query_layout, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--layouts', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    assert args.layouts > 0 and args.repeats > 0
    args.out.mkdir(parents=True, exist_ok=False)
    sources = args.out / 'sources'; sources.mkdir()
    source_names = ['target_spread.py', 'target_spread.sh', 'run.py', 'addressing.py', 'placement.c', 'placement.so']
    for name in source_names:
        shutil.copy2(HERE / name, sources / name)
    chunk_bytes = 18 * 2**20; chunks = 96; total_bytes = chunks * chunk_bytes
    target_counts = [1, 2, 4, 8, 16]
    plan = dict(workers=16, chunks=chunks, chunk_bytes=chunk_bytes, read_bytes=total_bytes,
                gpu_slab_bytes=2*2**30, target_counts=target_counts,
                layouts=args.layouts, repeats=args.repeats,
                source_sha256={n: hashlib.sha256((sources/n).read_bytes()).hexdigest() for n in source_names},
                notes=['Each condition uses 96 distinct dkeys, one fixed kv akey, SINGLE value, and 96 fetch calls.',
                       'One fresh OC_SX object per layout; all five conditions share its physical layout.',
                       'Selected targets receive equal chunk counts in cyclic order; no target-aware runtime scheduling.',
                       'Targets alternate between the two engines; 2/4/8/16 target conditions use both engines equally.',
                       'One-target condition alternates engine across layouts. 1-to-2 also changes engine count.',
                       'Five conditions execute in shuffled order each round; first round is untimed warmup.',
                       'Warm reads; no server cache flush. No LMCache windows, scatter, model, or DRAM payload cache.',
                       'Client fetch timing includes registration, network, progress, and server work; not server service time.',
                       'CPU full-byte verification is outside timing. Only run-owned objects are removed.'])
    save(args.out/'plan.json', plan)
    before = pool_query(); save(args.out/'pool_before.json', before)
    info = before['response']; nvme = next(t for t in info['tier_stats'] if t['media_type']=='nvme')
    assert info['active_targets']==16 and not info['disabled_targets'] and info['rebuild']['state']=='idle'
    assert nvme['free'] > 30*2**30 and nvme['min'] > 8*2**30
    lib = load_library(); rng = random.Random(2026100202)
    torch.cuda.set_device(0); torch.manual_seed(20261002)
    slab = torch.empty(2*2**30, device='cuda', dtype=torch.uint8)
    buf = slab[:total_bytes]; buf.random_(0, 256); torch.cuda.synchronize()
    oracle = buf.cpu(); expected = oracle.numpy()
    save(args.out/'payload.json', dict(sha256=hashlib.sha256(expected).hexdigest()))
    rows = []; cleanup = []
    save(args.out/'status.json', dict(status='running'))
    try:
        for seed in range(args.layouts):
            dest = args.out/f'layout_{seed}'; dest.mkdir()
            ctx, hi, lo = C.c_void_p(), C.c_uint64(), C.c_uint64()
            nonce = (uuid.uuid4().int & ((1<<63)-1)) | (1<<62)
            check(lib.placement_open(b'discospool', b'kvcache', nonce, C.byref(ctx), C.byref(hi), C.byref(lo)))
            oid = f'{hi.value}.{lo.value}'
            save(dest/'ownership.json', dict(oid=oid, nonce=nonce, created_by_this_run=True))
            try:
                layout, raw = query_layout(oid); (dest/'object_layout.txt').write_text(raw)
                assert len(layout)==16
                ranks = sorted({rank for rank, target in layout.values()}); assert len(ranks)==2
                rank_shards = {rank: sorted((s for s, p in layout.items() if p[0]==rank), key=lambda s: layout[s][1]) for rank in ranks}
                assert all(len(ss)==8 for ss in rank_shards.values())
                # Rotate physical target IDs, including one-target selection, across layouts.
                for rank, ss in rank_shards.items():
                    offset = (seed//2 + 3*ranks.index(rank)) % len(ss)
                    rank_shards[rank] = ss[offset:] + ss[:offset]
                rank_order = ranks[seed%2:] + ranks[:seed%2]
                target_order = [rank_shards[rank][i] for i in range(8) for rank in rank_order]
                cases = {}
                for count in target_counts:
                    selected = target_order[:count]; placements = []
                    for i in range(chunks):
                        shard = selected[i % count]
                        for salt in range(100000):
                            dk = f'spread/{seed:02d}/{count:02d}/{i:03d}/{salt:06d}'.encode()
                            if lib.placement_predict(dk, 16)==shard: break
                        else: raise RuntimeError('Salt search exhausted')
                        actual = C.c_uint(); check(lib.placement_shard(ctx, dk, C.byref(actual)))
                        assert actual.value==shard
                        rank, target = layout[shard]
                        placements.append(dict(chunk=i, dkey=dk.decode(), akey='kv', shard=shard, rank=rank, target=target))
                    assert sorted(Counter(p['shard'] for p in placements).values())==[chunks//count]*count
                    assert len({p['dkey'] for p in placements})==chunks
                    cases[count] = placements
                save(dest/'placements.json', cases)
                with ThreadPoolExecutor(max_workers=16, initializer=lambda: torch.cuda.set_device(0)) as executor:
                    barrier = threading.Barrier(16)
                    list(executor.map(lambda _: barrier.wait(), range(16)))
                    buf.copy_(oracle); torch.cuda.synchronize()
                    def put(p):
                        check(lib.placement_io(ctx, p['dkey'].encode(), b'kv', buf.data_ptr()+p['chunk']*chunk_bytes, chunk_bytes, 1))
                    order = target_counts.copy(); rng.shuffle(order)
                    for count in order: list(executor.map(put, cases[count]))
                    for placements in cases.values():
                        for p in placements:
                            size = C.c_uint64()
                            check(lib.placement_size(ctx, p['dkey'].encode(), b'kv', C.byref(size)))
                            assert size.value==chunk_bytes
                    print(json.dumps(dict(event='stored', layout=seed, oid=oid)), flush=True)
                    for rep in range(-1, args.repeats):
                        order = target_counts.copy(); rng.shuffle(order)
                        for count in order:
                            buf.zero_(); torch.cuda.synchronize()
                            jobs = [(p, p['dkey'].encode()) for p in cases[count]]
                            def fetch(job):
                                p, dk = job
                                begin = time.perf_counter_ns()
                                rc = lib.placement_io(ctx, dk, b'kv', buf.data_ptr()+p['chunk']*chunk_bytes, chunk_bytes, 0)
                                end = time.perf_counter_ns(); check(rc)
                                return dict(chunk=p['chunk'], shard=p['shard'], rank=p['rank'], target=p['target'], start_ns=begin, end_ns=end)
                            start = time.perf_counter_ns()
                            events = list(executor.map(fetch, jobs))
                            torch.cuda.synchronize(); elapsed = time.perf_counter_ns()-start
                            for e in events: e['start_ns']-=start; e['end_ns']-=start
                            assert np.array_equal(buf.cpu().numpy(), expected), (seed, rep, count)
                            row = dict(layout=seed, repeat=rep, target_count=count, workers=16, calls=96,
                                       read_bytes=total_bytes, ms=elapsed/1e6, gib_s=total_bytes/2**30/(elapsed/1e9),
                                       verified=True, events=events, **active_metrics(events, elapsed))
                            rows.append(row)
                            with (args.out/'measurements.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
                        print(json.dumps(dict(event='round_complete', layout=seed, repeat=rep,
                                              ms_by_targets={r['target_count']:round(r['ms'],2) for r in rows[-5:]})), flush=True)
                assert query_layout(oid)[0]==layout
                save(dest/'validation.json', dict(layout_unchanged=True, distinct_dkeys_per_condition=96,
                                                 equal_chunks_per_selected_target=True, predicted_shards_verified=True))
            finally:
                rc = lib.placement_close(ctx, 1)
                cleanup.append(dict(oid=oid, punch_and_close_rc=rc, production_object_touched=False))
                save(args.out/'cleanup.json', cleanup); check(rc)
        after = pool_query(); save(args.out/'pool_after.json', after)
        assert after['response']['version']==info['version']
        assert after['response']['active_targets']==16 and after['response']['rebuild']['state']=='idle'
        save(args.out/'status.json', dict(status='completed', verified_reads=len(rows), measured_reads=sum(r['repeat']>=0 for r in rows),
                                         objects_removed=len(cleanup), all_verified=True))
    except BaseException as e:
        save(args.out/'status.json', dict(status='failed', error=repr(e), completed_reads=len(rows)))
        raise


if __name__=='__main__':
    main()
