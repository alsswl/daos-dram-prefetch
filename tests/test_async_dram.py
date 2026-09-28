import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
import threading
import time

import pytest

from lmcache_daos.async_dram_backend import AsyncDramMirror
from lmcache_daos import gpu_store


class Obj:
    def __init__(self):
        self.refs = 1
        self.metadata = NS(fmt='fmt')
        self.pins = 0
    def ref_count_up(self): self.refs += 1
    def ref_count_down(self): self.refs -= 1
    def get_size(self): return 10
    def get_shapes(self): return ['shape']
    def get_dtypes(self): return ['dtype']
    def pin(self): self.pins += 1


class CPU:
    def __init__(self):
        self.hot_cache = OrderedDict()
        self.cpu_lock = threading.Lock()
        self.cache_policy = NS(update_on_hit=lambda k, cache: cache.move_to_end(k))
        self.closed = False
        self.allocated = []
    def close(self): self.closed = True
    async def batched_async_contains(self, *args): return 0
    def contains(self, key):
        with self.cpu_lock: return key in self.hot_cache
    def allocate(self, *args, **kwargs):
        assert kwargs['busy_loop'] is False
        obj = Obj()
        self.allocated.append(obj)
        return obj
    def submit_put_task(self, key, obj):
        with self.cpu_lock:
            obj.ref_count_up()
            self.hot_cache[key] = obj


def mirror(limit=20, age=1000):
    cpu = CPU()
    backend = NS(_ensure_cuda_ctx=lambda: None, _release_memory_obj=lambda obj: obj.ref_count_down())
    m = AsyncDramMirror(backend, cpu, limit, age)
    m.copy_payload = lambda *args: None
    return m, cpu


def test_publish_after_copy_and_bounded_refs():
    m, cpu = mirror(limit=10)
    entered, release = threading.Event(), threading.Event()
    def copy(*args):
        entered.set()
        assert release.wait(3)
    m.copy_payload = copy
    obj, skipped = Obj(), Obj()
    try:
        assert m.offer('a', obj)
        assert entered.wait(3)
        assert obj.refs == 2 and not cpu.contains('a')
        assert not m.offer('b', skipped)
        assert skipped.refs == 1
        assert m.snapshot()['peak_pending_bytes'] == 10
    finally:
        release.set()
        m.close()
    assert cpu.contains('a') and obj.refs == 1
    assert m.snapshot()['pending_bytes'] == 0
    assert m.snapshot()['copied'] == 1


def test_expired_and_failure_release():
    m, cpu = mirror(age=.001)
    gate = threading.Event()
    m.worker.submit(lambda: gate.wait(3))
    obj = Obj()
    assert m.offer('a', obj)
    time.sleep(.01)
    gate.set()
    m.close()
    assert not cpu.contains('a') and obj.refs == 1
    assert m.snapshot()['skipped_expired'] == 1
    m, cpu = mirror()
    m.copy_payload = lambda *args: (_ for _ in ()).throw(RuntimeError('DMA failure'))
    obj = Obj()
    assert m.offer('a', obj)
    m.close()
    assert not cpu.contains('a') and obj.refs == 1
    assert cpu.allocated[0].refs == 0
    assert m.snapshot()['errors'] == 1


def test_cpu_close_drains_and_prevents_new_admission():
    m, cpu = mirror()
    obj = Obj()
    assert m.offer('a', obj)
    cpu.close()
    assert cpu.closed and obj.refs == 1
    assert not m.offer('b', Obj())
    m.close()


def test_full_pinned_cpu_pool_skips_without_waiting():
    m, cpu = mirror()
    cpu.allocate = lambda *args, **kwargs: None
    obj = Obj()
    assert m.offer('a', obj)
    m.close()
    assert obj.refs == 1 and not cpu.contains('a')
    assert m.snapshot()['skipped_allocation'] == 1
    assert m.snapshot()['pending_bytes'] == 0


def test_async_hit_updates_lru_without_skipping_prefix_hole():
    m, cpu = mirror()
    try:
        cpu.hot_cache.update((k, Obj()) for k in ['a', 'b', 'c'])
        assert asyncio.run(cpu.batched_async_contains('r', ['a', 'b', 'missing', 'c'], True)) == 2
        assert list(cpu.hot_cache) == ['c', 'b', 'a']
        assert cpu.hot_cache['a'].pins == 1
        assert m.snapshot()['cpu_hit_chunks'] == 2
    finally:
        m.close()


def test_config_and_stream_sync(monkeypatch):
    config = NS(local_cpu=True, extra_config={'daosgds.async_dram': True})
    gpu_store.validate_config(config, 'gpu_direct')
    with pytest.raises(ValueError): gpu_store.validate_config(config, 'host_staged')
    config.local_cpu = False
    with pytest.raises(ValueError): gpu_store.validate_config(config, 'gpu_direct')
    events = []
    monkeypatch.setattr(gpu_store.torch.cuda, 'synchronize',
                        lambda *args: (_ for _ in ()).throw(AssertionError('device-wide sync')))
    be = NS(dram_mirror=object(), device_id=0,
            batched_submit_put_task=lambda *args, **kw: events.append(kw))
    manager = NS(_bypass_lock=threading.Lock(), _bypassed_backends=set())
    obj = Obj()
    obj.tensor = NS(is_cuda=True, device=NS(index=0))
    token = gpu_store._store_context.set((manager, NS(synchronize=lambda: events.append('gather_sync'))))
    try:
        gpu_store.put_direct(manager, ('daos', be), ['k'], [obj])
    finally:
        gpu_store._store_context.reset(token)
    assert events == ['gather_sync', {'transfer_spec': None, 'payload_ready': True}]
    assert obj.refs == 0


@pytest.mark.parametrize('enabled,hit', [(True, True), (False, True), (True, False)])
def test_read_promotion_only_on_enabled_success(monkeypatch, enabled, hit):
    from lmcache_daos.async_dram_backend import DaosAsyncDramBackend, DaosGdsBackend
    source = Obj() if hit else None
    monkeypatch.setattr(DaosGdsBackend, 'get_blocking', lambda self, key: source)
    be = DaosAsyncDramBackend.__new__(DaosAsyncDramBackend)
    be.promote_on_read = enabled
    offers = []
    be.dram_mirror = NS(offer=lambda *args, **kw: offers.append((args, kw)))
    assert be.get_blocking('key') is source
    assert len(offers) == int(enabled and hit)
    if offers:
        assert offers[0] == (('key', source), {'origin': 'read'})


def test_read_promotion_does_not_wait_and_retains_source(monkeypatch):
    from lmcache_daos.async_dram_backend import DaosAsyncDramBackend, DaosGdsBackend
    m, cpu = mirror()
    entered, release = threading.Event(), threading.Event()
    def copy(*args):
        entered.set()
        assert release.wait(3)
    m.copy_payload = copy
    source = Obj()
    monkeypatch.setattr(DaosGdsBackend, 'get_blocking', lambda self, key: source)
    be = DaosAsyncDramBackend.__new__(DaosAsyncDramBackend)
    be.promote_on_read, be.dram_mirror = True, m
    try:
        assert be.get_blocking('a') is source  # Returns despite blocked D2H.
        assert entered.wait(3)
        source.ref_count_down()  # Caller already finished consuming GPU data.
        assert source.refs == 1 and not cpu.contains('a')
    finally:
        release.set()
        m.close()
    assert source.refs == 0 and cpu.contains('a')
    assert m.snapshot()['read_copied'] == 1
    assert m.snapshot()['write_copied'] == 0


def test_failed_promotion_submission_preserves_read(monkeypatch):
    from lmcache_daos.async_dram_backend import DaosAsyncDramBackend, DaosGdsBackend
    m, cpu = mirror()
    monkeypatch.setattr(m.worker, 'submit', lambda *a: (_ for _ in ()).throw(RuntimeError('failed submit')))
    source = Obj()
    monkeypatch.setattr(DaosGdsBackend, 'get_blocking', lambda self, key: source)
    be = DaosAsyncDramBackend.__new__(DaosAsyncDramBackend)
    be.promote_on_read, be.dram_mirror = True, m
    try:
        assert be.get_blocking('a') is source
        assert source.refs == 1
        assert m.snapshot()['pending_bytes'] == 0
        assert m.snapshot()['read_submit_errors'] == 1
    finally:
        m.close()


@pytest.mark.parametrize('enabled', [False, True])
def test_combined_prefetch_opt_in_and_cpu_shutdown_chain(monkeypatch, enabled):
    from lmcache_daos import async_dram_backend as module
    cpu = CPU()
    async def get(*args): return []
    cpu.batched_get_non_blocking = get
    config = NS(local_cpu=True, enable_async_loading=True, use_layerwise=False,
                cache_policy='LRU', extra_config={'daosgds.async_dram':True,
                'daosgds.dram_prefetch':enabled})
    monkeypatch.setattr(module.DaosGdsBackend, '__init__', lambda *a: None)
    monkeypatch.setattr(module.DaosGdsBackend, 'close', lambda self: None)
    monkeypatch.setattr(module, 'install_store_context', lambda: None)
    backend = module.DaosAsyncDramBackend(config, local_cpu_backend=cpu)
    assert (backend.cpu_prefetch is not None) == enabled
    assert asyncio.run(cpu.batched_get_non_blocking('r', [])) == []
    cpu.close()
    assert cpu.closed and backend.dram_mirror.closed
    if enabled:
        assert backend.cpu_prefetch.closed
    assert cpu.batched_get_non_blocking is get
    backend.close()  # StorageManager closes CPU before DAOS; both are safe.
