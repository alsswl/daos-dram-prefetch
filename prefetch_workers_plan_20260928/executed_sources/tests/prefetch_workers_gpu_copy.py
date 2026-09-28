"""Real CUDA integrity preflight for independent per-worker streams."""
from concurrent.futures import wait
from types import SimpleNamespace as NS
import json
import threading

import torch

from lmcache_daos.dram_prefetch_backend import CPUHitPrefetch


def main():
    torch.cuda.set_device(0)
    cpu = NS(batched_get_non_blocking=lambda *a: None, close=lambda: None)
    pf = CPUHitPrefetch(NS(device_id=0), cpu, None, workers=2)
    barrier = threading.Barrier(2)
    records = []
    try:
        for trial in range(3):
            pairs = []
            for i in range(2):
                n = 64 * 2**20
                src = torch.randint(0, 256, (n,), dtype=torch.uint8, pin_memory=True)
                dst = torch.empty(n, dtype=torch.uint8, device='cuda:0')
                pairs.append((NS(raw_data=src, get_size=lambda: n),
                              NS(raw_data=dst, get_size=lambda: n)))
            torch.cuda.synchronize()
            def copy(pair):
                torch.cuda.set_device(0)
                barrier.wait(timeout=10)  # Preflight only, NOT benchmark behavior.
                pf.copy([pair[0]], [pair[1]])
                return dict(thread=threading.get_ident(), stream=int(pf.stream.cuda_stream))
            futures = [pf.worker.submit(copy, pair) for pair in pairs]
            for f in futures: records.append(f.result(timeout=30))
            for src, dst in pairs:
                assert torch.equal(src.raw_data, dst.raw_data.cpu()), 'GPU copy byte mismatch'
        assert len({r['thread'] for r in records}) == 2
        assert len({r['stream'] for r in records}) == 2
        print(json.dumps(dict(status='passed',copies=6,size_mib=64,workers=2,streams=2)))
    finally:
        pf.close()


if __name__ == '__main__': main()
