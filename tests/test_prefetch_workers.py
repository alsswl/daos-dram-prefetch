import asyncio
from contextlib import contextmanager
import threading
from types import SimpleNamespace as NS

import pytest

from lmcache_daos import dram_prefetch_backend as mod
from test_dram_prefetch import fixture, discard


@pytest.mark.parametrize('workers', [0, -1, True, 1.5, '2'])
def test_reject_invalid_workers(workers):
    with pytest.raises(ValueError, match='positive integer'):
        mod.CPUHitPrefetch(None, None, None, workers=workers)


def test_two_workers_have_distinct_streams_and_keep_them_until_copy_done(monkeypatch):
    pf, cpu, inner, sources = fixture(workers=2)
    pf.backend.device_id = 0
    pf.copy = mod.CPUHitPrefetch.copy.__get__(pf)
    local, barrier = threading.local(), threading.Barrier(2)
    streams, copies = [], []

    class Stream:
        def __init__(self, device):
            self.owner = threading.get_ident()
            self.synced = 0
            streams.append(self)
        def synchronize(self):
            assert self.owner == threading.get_ident()
            self.synced += 1

    @contextmanager
    def stream_context(stream):
        assert stream.owner == threading.get_ident()
        local.current = stream
        yield

    class Raw:
        device = NS(type='cpu')
        def __getitem__(self, key): return self
        def copy_(self, source, non_blocking):
            assert non_blocking
            copies.append(local.current)
            barrier.wait(timeout=5)  # Test only: proves two workers enter copy.

    sources[0].raw_data = Raw()
    original_allocate = inner.allocate
    def allocate(*args):
        obj = original_allocate(*args)
        obj.raw_data = Raw()
        return obj
    inner.allocate = allocate
    monkeypatch.setattr(mod.torch.cuda, 'Stream', Stream)
    monkeypatch.setattr(mod.torch.cuda, 'stream', stream_context)

    async def run():
        for _ in range(2):
            outputs = await asyncio.gather(cpu.batched_get_non_blocking('a', ['a']),
                                           cpu.batched_get_non_blocking('b', ['b']))
            assert inner.total_allocated_size == 64
            for out in outputs: discard(out)
            assert inner.total_allocated_size == 0
    try:
        asyncio.run(run())
        assert len(streams) == 2 and len({s.owner for s in streams}) == 2
        assert all(s.synced == 2 for s in streams)
        assert pf.stream is None  # Main thread never owns a copy stream.
        assert pf.stats['staged_requests'] == 4 and sources[0].refs == 1
    finally:
        pf.close()


def test_counter_updates_from_two_workers_are_not_lost():
    pf, _, _, _ = fixture(workers=2)
    def increment():
        for _ in range(5000): pf.increment_stats(staged_requests=1)
    try:
        futures = [pf.worker.submit(increment) for _ in range(2)]
        for f in futures: f.result()
        assert pf.stats['staged_requests'] == 10000
    finally:
        pf.close()
