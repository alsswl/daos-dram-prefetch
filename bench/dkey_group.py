"""Small isolated native-object GDR grouping experiment (no LMCache/vLLM).

Run via run_vllm.sh. CPU payload is retained only as an untimed correctness
oracle; every measured byte is fetched from DAOS into the same 2 GiB GPU slab.
"""
import argparse
import ctypes as C
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics
import subprocess
import time
import uuid

import numpy as np
import torch


def check(rc):
    if rc:
        raise RuntimeError(f'DAOS rc={rc}')


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=5)
    p.add_argument('--layouts', type=int, default=3)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    pool = json.loads(subprocess.check_output(
        ['daos', 'pool', 'query', 'discospool', '--json']))
    save(args.out / 'pool_before.json', pool)
    assert pool['status'] == 0
    nvme = next(x for x in pool['response']['tier_stats'] if x['media_type'] == 'nvme')
    assert nvme['free'] > 60 * 2**30 and nvme['min'] > 6 * 2**30
    lib = C.CDLL(str(here / 'dkey_group.so'))
    lib.group_open.argtypes = [C.c_char_p, C.c_char_p, C.c_uint64,
                              C.POINTER(C.c_void_p), C.POINTER(C.c_uint64), C.POINTER(C.c_uint64)]
    lib.group_io.argtypes = [C.c_void_p, C.c_char_p, C.c_void_p, C.c_size_t, C.c_uint, C.c_int]
    lib.group_shard.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.c_uint)]
    lib.group_close.argtypes = [C.c_void_p, C.c_int]
    for name in ('group_open', 'group_io', 'group_shard', 'group_close'):
        getattr(lib, name).restype = C.c_int
    torch.cuda.set_device(0)
    chunk_bytes, chunks = 18 * 2**20, 96
    total_bytes = chunk_bytes * chunks
    slab = torch.empty(2 * 2**30, dtype=torch.uint8, device='cuda')
    buf = slab[:total_bytes]
    torch.manual_seed(20261002)
    buf.random_(0, 256)
    torch.cuda.synchronize()
    expected = buf.cpu().numpy()
    ctx, hi, lo = C.c_void_p(), C.c_uint64(), C.c_uint64()
    nonce = (uuid.uuid4().int & ((1 << 63) - 1)) | (1 << 62)
    check(lib.group_open(b'discospool', b'kvcache', nonce, C.byref(ctx), C.byref(hi), C.byref(lo)))
    plan = dict(pool='discospool', container='kvcache', oid=f'{hi.value}.{lo.value}',
                object_class='OC_SX', chunk_bytes=chunk_bytes, chunks=chunks,
                read_bytes=total_bytes, gpu_slab_bytes=slab.numel(), groups=[1,2,4,8,16,32],
                workers=[1,16], layouts=args.layouts, repeats=args.repeats,
                payload_sha256=hashlib.sha256(expected).hexdigest(),
                measured_path='native daos_obj_fetch_gpu; g SINGLE akeys/IODs per dkey',
                source_sha256={f: hashlib.sha256((here/f).read_bytes()).hexdigest()
                               for f in ['dkey_group.py','dkey_group.c','dkey_group.so']},
                notes=['No client payload cache, scatter copy, metadata lookup, or LLM compute in timing.',
                       '2 GiB GPU slab; CPU reference used only after timing for full byte validation.',
                       'Warm repeated reads; server caches are not flushed.',
                       'Native API call counts are not physical RPC/RDMA/NVMe counts.',
                       'Random dkeys: target collision is measured, not controlled away.',
                       'Cleanup punches only the newly generated experiment OID.'])
    save(args.out/'plan.json', plan)
    save(args.out/'status.json', {'status':'running'})
    cases, rows = [], []
    try:
        layout = subprocess.run(['daos','object','query','discospool','kvcache',plan['oid']],
                                text=True, capture_output=True)
        (args.out/'object_layout.txt').write_text(layout.stdout + layout.stderr)
        if layout.returncode: raise RuntimeError('Object layout query failed')
        for seed in range(args.layouts):
            for g in plan['groups']:
                jobs = [(f'seed{seed}/group{g}/key{i}'.encode(), i*g)
                        for i in range(chunks//g)]
                def write(job):
                    key, offset = job
                    torch.cuda.set_device(0)
                    check(lib.group_io(ctx, key, buf.data_ptr()+offset*chunk_bytes, chunk_bytes, g, 1))
                with ThreadPoolExecutor(max_workers=16) as ex:
                    list(ex.map(write, jobs))
                placements = []
                for key, offset in jobs:
                    shard = C.c_uint()
                    check(lib.group_shard(ctx, key, C.byref(shard)))
                    placements.append(dict(key=key.decode(), chunk_start=offset, shard=shard.value))
                case = dict(seed=seed, group=g, calls=len(jobs), placements=placements,
                            active_shards=len({x['shard'] for x in placements}),
                            chunks_per_shard=dict(Counter(x['shard'] for x in placements)))
                case['chunks_per_shard'] = {k:v*g for k,v in case['chunks_per_shard'].items()}
                cases.append(case)
                print(json.dumps(dict(event='stored',seed=seed,group=g,active_shards=case['active_shards'])), flush=True)
        save(args.out/'placements.json', cases)
        maps = [line for line in Path('/proc/self/maps').read_text().splitlines()
                if any(x in line for x in ['libdaos.so','libfabric.so','libmercury.so'])]
        (args.out/'native_maps.txt').write_text('\n'.join(maps)+'\n')
        executors = {w:ThreadPoolExecutor(max_workers=w, initializer=lambda:torch.cuda.set_device(0))
                     for w in plan['workers']}
        try:
            rng = random.Random(20261002)
            for rep in range(-1, args.repeats):
                schedule = [(c,w) for c in cases for w in plan['workers']]
                rng.shuffle(schedule)
                for case,w in schedule:
                    g=case['group']
                    buf.zero_();torch.cuda.synchronize()
                    jobs = [(x['key'].encode(), x['chunk_start']) for x in case['placements']]
                    def fetch(job):
                        key, offset = job
                        start=time.perf_counter_ns()
                        check(lib.group_io(ctx,key,buf.data_ptr()+offset*chunk_bytes,chunk_bytes,g,0))
                        return (time.perf_counter_ns()-start)/1e6
                    start=time.perf_counter_ns()
                    times=list(executors[w].map(fetch,jobs))
                    torch.cuda.synchronize()
                    ms=(time.perf_counter_ns()-start)/1e6
                    assert np.array_equal(buf.cpu().numpy(), expected), (case['seed'],g,w,rep)
                    row=dict(seed=case['seed'],group=g,workers=w,repeat=rep,ms=ms,
                             gib_s=total_bytes/2**30/(ms/1000),calls=len(jobs),
                             active_shards=case['active_shards'],call_ms=times,verified=True)
                    rows.append(row)
                    with (args.out/'measurements.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(dict(event='round_complete',repeat=rep,reads=len(rows))),flush=True)
        finally:
            for ex in executors.values():ex.shutdown(wait=True)
        summary=[]
        for w in plan['workers']:
            for g in plan['groups']:
                selected=[r for r in rows if r['repeat']>=0 and r['workers']==w and r['group']==g]
                per_seed=[statistics.median(r['ms'] for r in selected if r['seed']==s)
                          for s in range(args.layouts)]
                summary.append(dict(group=g,workers=w,calls=chunks//g,fetch_mib=g*18,
                                    samples=len(selected),median_ms=statistics.median(r['ms'] for r in selected),
                                    median_gib_s=statistics.median(r['gib_s'] for r in selected),
                                    per_seed_median_ms=per_seed,
                                    active_shards=[c['active_shards'] for c in cases if c['group']==g]))
        save(args.out/'summary.json',summary)
        save(args.out/'status.json',dict(status='completed',verified_reads=len(rows),measured_reads=sum(r['repeat']>=0 for r in rows)))
    except BaseException as e:
        save(args.out/'status.json',dict(status='failed',error=repr(e),completed_reads=len(rows)))
        raise
    finally:
        rc=lib.group_close(ctx,1)
        save(args.out/'cleanup.json',dict(oid=plan['oid'],punch_and_close_rc=rc,production_object_touched=False))
        check(rc)
    print(json.dumps(dict(event='completed',summary=summary)),flush=True)


if __name__=='__main__':main()
