"""Hardware gate: real GPU final-page SGL -> DAOS -> different GPU final pages.

Run only on a configured GDR client. Uses unique dkeys and removes ONLY its
own keys in finally. Reference gathering is verification, not the I/O path.
"""
import argparse
import json
from pathlib import Path
import random
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from lmcache_daos.scatter_binding import ScatterObjectStore
from lmcache_daos.scatter_plan import plan_segments


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pool',required=True)
    p.add_argument('--container',required=True)
    p.add_argument('--library',default=str(Path(__file__).resolve().parents[1]/'libdaosgdr_scatter.so'))
    p.add_argument('--device',type=int,default=0)
    p.add_argument('--iterations',type=int,default=5)
    args=p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit('BLOCKED: CUDA device/driver unavailable; no DAOS objects were created')
    if args.iterations < 1:
        p.error('--iterations must be positive')
    torch.cuda.set_device(args.device)
    device=f'cuda:{args.device}'
    # Match the Qwen3-4B experiment, including 576 page-sized IOVs per chunk.
    layers,nb,bs,nh,hs,n=36,24,16,8,128,128
    source=[torch.empty((nb,2,bs,nh,hs),dtype=torch.bfloat16,device=device).uniform_(-1,1)
            for _ in range(layers)]
    dest=[torch.empty_like(t) for t in source]
    rng=random.Random(31)
    store=ScatterObjectStore(args.pool,args.container,library_path=args.library)
    run='direct-scatter-gate:'+uuid.uuid4().hex+':'
    keys=[];results=[]
    try:
        for iteration in range(args.iterations):
            src_blocks=rng.sample(range(nb),8)
            dst_blocks=rng.sample(range(nb),8)
            src_slots=[b*bs+t for b in src_blocks for t in range(bs)]
            dst_slots=[b*bs+t for b in dst_blocks for t in range(bs)]
            sp=plan_segments([t.data_ptr() for t in source],nb,bs,nh*hs,2,src_slots)
            key=run+str(iteration);keys.append(key)
            torch.cuda.synchronize()
            start=time.perf_counter()
            store.putv(key,sp,b'{"schema":"direct-scatter-hardware-gate-v1"}',args.device)
            put_ms=(time.perf_counter()-start)*1000
            assert store.stat(key) is not None
            # Exercise a complete fetch, an unaligned prefix, and one-token tail.
            for skip in (0,5,127):
                for t in dest:t.fill_(-4)
                dp=plan_segments([t.data_ptr() for t in dest],nb,bs,nh*hs,2,dst_slots,skip=skip)
                torch.cuda.synchronize()
                start=time.perf_counter()
                store.getv(key,dp,args.device)
                torch.cuda.synchronize()
                get_ms=(time.perf_counter()-start)*1000
                # Independent token-by-token reference; checks ALL guard bytes,
                # skipped prefix tokens, K/V, layers and unallocated blocks.
                for src,got in zip(source,dest):
                    actual=got.cpu();expected=torch.full_like(actual,-4)
                    cpu_src=src.cpu()
                    for i in range(skip,n):
                        sb,su=divmod(src_slots[i],bs)
                        db,du=divmod(dst_slots[i],bs)
                        expected[db,:,du]=cpu_src[sb,:,su]
                    assert torch.equal(actual,expected), (iteration,skip)
                results.append(dict(iteration=iteration,skip=skip,iovs=len(dp),
                                    bytes=sum(s.length for s in dp),put_ms=put_ms,get_ms=get_ms))
        print(json.dumps({'status':'PASS','checks':len(results),'results':results},indent=2))
    finally:
        try:
            for key in keys:store.remove(key)
        finally:
            store.close()


if __name__=='__main__':main()
