#!/usr/bin/env python3
"""Allocate/release a 512GiB LMCache pinned buffer, validate small GPU roundtrips."""
import json
from pathlib import Path
import subprocess
import time


def main():
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader'], text=True).strip(), 'GPU occupied'
    mem = {s.split(':')[0]: int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()}
    assert mem['MemAvailable'] > 600*2**30
    import torch
    from lmcache.v1.memory_management import _allocate_cpu_memory, _free_cpu_memory
    size = 512*2**30
    print('BEGIN LMCache pinned allocation 512GiB', flush=True)
    started = time.monotonic()
    buffer = _allocate_cpu_memory(size)
    try:
        assert buffer.is_pinned() and buffer.numel() == size
        print(json.dumps(dict(allocated_gib=512, seconds=time.monotonic()-started,
                              pinned=buffer.is_pinned())), flush=True)
        for i in range(16):
            start = i*(size//16)
            sample = buffer[start:start+2**20]
            sample.fill_(i+1)
            gpu = sample.to('cuda', non_blocking=True)
            torch.cuda.synchronize()
            assert torch.equal(sample, gpu.cpu())
        ptr = buffer.data_ptr()
        maps = Path('/proc/self/maps').read_text().splitlines()
        regions = [line.split()[0].split('-')[0] for line in maps
                   if int(line.split('-')[0], 16) <= ptr < int(line.split()[0].split('-')[1], 16)]
        for line in Path('/proc/self/numa_maps').read_text().splitlines():
            if line.split()[0] in regions:
                print('ALLOCATION_NUMA '+line, flush=True)
        print('PASS: 16 spread-out 1MiB GPU roundtrips', flush=True)
    finally:
        _free_cpu_memory(buffer, size)
        print('RELEASED 512GiB pinned allocation', flush=True)


if __name__ == '__main__':
    main()
