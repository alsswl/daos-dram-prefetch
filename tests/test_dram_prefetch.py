import asyncio
import threading
from types import SimpleNamespace as NS

import pytest

from lmcache_daos.dram_prefetch_backend import CPUHitPrefetch, StagedCPUObject, discard
from run_dram_cache import build_profile
from test_dram_cache import base


class Obj:
    def __init__(self, size=32, device='cpu', parent=None):
        self.size, self.parent = size, parent
        self.refs, self.pins = 1, 0
        self.raw_data = NS(device=NS(type=device))
        self.metadata = NS(fmt='fmt')
    def get_size(self): return self.size
    def get_shapes(self): return [self.size]
    def get_dtypes(self): return ['uint8']
    def ref_count_up(self): self.refs += 1
    def ref_count_down(self):
        self.refs -= 1
        assert self.refs >= 0
        if self.refs == 0 and self.parent:
            self.parent.total_allocated_size -= self.size
    @property
    def is_pinned(self): return self.pins > 0
    def pin(self): self.pins += 1
    def unpin(self):
        self.pins -= 1
        assert self.pins >= 0


class Inner:
    total_allocated_size = 0
    def __init__(self):
        self.address_manager = NS(compute_aligned_size=lambda n: n)
        self.calls, self.fail_on = 0, None
    def allocate(self, shapes, dtypes, fmt):
        self.calls += 1
        if self.calls == self.fail_on: return None
        n = shapes[0]
        self.total_allocated_size += n
        return Obj(n, 'cuda', self)


def fixture(limit=128, sources=None, workers=1):
    sources = sources or [Obj()]
    inner = Inner()
    async def get(*args):
        for src in sources: src.ref_count_up()
        return sources
    cpu = NS(batched_get_non_blocking=get, close=lambda: None)
    backend = NS(memory_allocator=NS(allocator=inner, device_mem_lock=threading.Lock()),
                 _ensure_cuda_ctx=lambda: None, _release_memory_obj=lambda o: o.ref_count_down())
    pf = CPUHitPrefetch(backend, cpu, limit, workers=workers)
    pf.copy = lambda sources, destinations: None
    return pf, cpu, inner, sources


def test_profile_defaults_and_rollback():
    original = build_profile(base(), 'on', 4)
    enabled = build_profile(base(), 'on', 4, 'on')
    assert enabled['extra_config']['storage_plugin.daosgds.class_name'] == 'DaosDramPrefetchBackend'
    assert build_profile(enabled, 'on', 4, 'off') == original
    with pytest.raises(ValueError): build_profile(base(), 'off', 4, 'on')
    with pytest.raises(ValueError): build_profile(base(), 'on', 4, 'on', 999)


def test_gpu_result_holds_memory_until_consumed_and_keeps_cpu_cache():
    pf, cpu, inner, sources = fixture()
    try:
        sources[0].pin()
        out = asyncio.run(cpu.batched_get_non_blocking('request', ['key']))
        assert isinstance(out[0], StagedCPUObject)
        assert out[0].raw_data.device.type == 'cuda' and out[0].is_pinned
        assert inner.total_allocated_size == 32 and sources[0].refs == 2
        out[0].ref_count_up()
        out[0].ref_count_down()
        assert inner.total_allocated_size == 32
        discard(out)
        assert inner.total_allocated_size == 0 and sources[0].refs == 1
        assert sources[0].pins == 0
        with pytest.raises(RuntimeError): out[0].ref_count_down()
    finally: pf.close()


def test_shared_daos_occupancy_causes_cpu_fallback_without_miss():
    pf, cpu, inner, sources = fixture(limit=64)
    inner.total_allocated_size = 48  # another DAOS/ready allocation
    try:
        out = asyncio.run(cpu.batched_get_non_blocking('request', ['key']))
        assert out is sources and inner.total_allocated_size == 48
        discard(out)
        assert sources[0].refs == 1
    finally: pf.close()


def test_partial_allocation_is_rolled_back():
    pf, cpu, inner, sources = fixture(sources=[Obj(), Obj()])
    inner.fail_on = 2
    try:
        out = asyncio.run(cpu.batched_get_non_blocking('request', ['a','b']))
        assert out is sources and inner.total_allocated_size == 0
        discard(out)
        assert all(s.refs == 1 for s in sources)
    finally: pf.close()


def test_capacity_policy_ignores_watermark_and_keeps_physical_failure_safe():
    pf, cpu, inner, sources = fixture(limit=None, sources=[Obj(), Obj()])
    inner.total_allocated_size = 96
    try:
        out = asyncio.run(cpu.batched_get_non_blocking('no-watermark', ['a','b']))
        assert all(isinstance(o, StagedCPUObject) for o in out)
        assert inner.total_allocated_size == 160
        discard(out)
        assert inner.total_allocated_size == 96
        inner.fail_on = inner.calls + 2
        out = asyncio.run(cpu.batched_get_non_blocking('physical-full', ['a','b']))
        assert out is sources and inner.total_allocated_size == 96
        discard(out)
        assert all(s.refs == 1 for s in sources)
        assert pf.stats['capacity_rejections'] == 1
        assert pf.stats['watermark_rejections'] == 0
    finally:
        pf.close()


def test_copy_exception_releases_gpu_and_cpu_caller_references():
    pf, cpu, inner, sources = fixture()
    def fail(*args): raise RuntimeError('injected copy failure')
    pf.copy = fail
    try:
        with pytest.raises(RuntimeError, match='injected'):
            asyncio.run(cpu.batched_get_non_blocking('request', ['key']))
        assert inner.total_allocated_size == 0 and sources[0].refs == 1
    finally: pf.close()


def test_cancel_during_copy_never_recycles_inflight_buffers():
    pf, cpu, inner, sources = fixture()
    started, finish = threading.Event(), threading.Event()
    def copy(*args):
        started.set()
        assert finish.wait(5)
    pf.copy = copy
    async def run():
        task = asyncio.create_task(cpu.batched_get_non_blocking('cancel', ['key']))
        while not started.is_set(): await asyncio.sleep(.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        assert inner.total_allocated_size == 32 and sources[0].refs == 2
        finish.set()
        for _ in range(1000):
            if inner.total_allocated_size == 0: break
            await asyncio.sleep(.001)
        assert inner.total_allocated_size == 0 and sources[0].refs == 1
    try: asyncio.run(run())
    finally:
        finish.set()
        pf.close()
