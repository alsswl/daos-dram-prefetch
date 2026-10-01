import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS
import threading

import pytest

from lmcache_daos.early_lookup_state import EarlyReadBatch
from lmcache_daos.early_lookup_backend import EarlyLookupCoordinator
from lmcache.v1.event_manager import EventManager, EventType


class Obj:
    def __init__(self, pinned=False):
        self.refs = 1
        self.pins = int(pinned)
        self.tensor = object()
    def ref_count_down(self):
        self.refs -= 1
        assert self.refs >= 0
    def unpin(self):
        self.pins -= 1
        assert self.pins >= 0
    def get_size(self): return 20


def batch(tier='daos', count=1, demand=None):
    sources = [Obj(True) for _ in range(count)] if tier == 'dram' else None
    calls = []
    def read():
        calls.append('demand')
        return [Obj() for _ in range(count)] if tier == 'daos' else None
    b = EarlyReadBatch('req', tier, count, sources, lambda o:o.ref_count_down(),
                       demand or read, lambda *a, **k: None)
    return b, calls


def test_queued_daos_read_becomes_demand_without_duplicate_io():
    b, calls = batch()
    b.resolve()
    assert calls == ['demand'] and not b.claim()
    result = b.selected[0]
    b.views[0].ref_count_down()
    assert result.refs == 0 and b.is_retired


def test_queued_cpu_copy_returns_original_dram_and_unpins_once():
    b, _ = batch('dram')
    b.resolve()
    assert b.selected is b.sources and not b.claim()
    b.views[0].unpin()
    b.views[0].ref_count_down()
    assert b.sources[0].pins == b.sources[0].refs == 0


def test_ready_prefetch_used_without_demand():
    b, calls = batch()
    result = Obj()
    assert b.claim()
    b.finish([result])
    b.resolve()
    assert calls == [] and b.views[0].tensor is result.tensor
    b.views[0].ref_count_down()
    assert result.refs == 0


def test_running_read_waits_and_never_duplicates():
    b, calls = batch()
    assert b.claim()
    with ThreadPoolExecutor(1) as worker:
        waiting = worker.submit(b.resolve)
        assert not waiting.done()
        b.finish([Obj()])
        waiting.result(timeout=2)
    assert calls == []
    b.abort()


def test_abort_inflight_cpu_copy_does_not_release_dma_source():
    b, _ = batch('dram')
    assert b.claim()
    b.views[0].ref_count_down()
    assert b.sources[0].refs == 1 and b.sources[0].pins == 0
    dest = Obj()
    b.finish([dest])
    assert dest.refs == b.sources[0].refs == 0 and b.is_retired


def test_abort_queued_never_starts_read():
    b, calls = batch()
    b.abort()
    assert not b.claim() and b.is_retired and calls == []


def test_partial_load_is_failure_not_fake_cache_hit():
    b, _ = batch(count=2)
    obj = Obj()
    assert b.claim()
    b.finish([obj])
    with pytest.raises(RuntimeError, match='fully loaded'):
        b.resolve()
    assert obj.refs == 0
    b.abort()


def test_abort_during_demand_read_retains_active_lifetime():
    entered, release = threading.Event(), threading.Event()
    result = Obj()
    def read():
        entered.set()
        assert release.wait(2)
        return [result]
    b, _ = batch(demand=read)
    with ThreadPoolExecutor(1) as worker:
        f = worker.submit(b.resolve)
        assert entered.wait(2)
        b.abort()
        assert not b.work_finished.is_set() and not b.is_retired
        release.set()
        with pytest.raises(RuntimeError, match='aborted'):
            f.result(timeout=2)
    assert b.work_finished.is_set() and result.refs == 0


def test_payload_cannot_be_accessed_before_resolve():
    b, _ = batch()
    with pytest.raises(RuntimeError, match='before retrieve'):
        _ = b.views[0].tensor
    b.abort()


def test_metadata_notification_precedes_io_and_queued_takeover():
    async def run():
        log = []
        async def contains(rid, keys, pin): return len(keys)
        async def read(*args):
            log.append('unexpected_prefetch')
            return [Obj()]
        async def serialize(coro, n): return await coro
        be = NS(batched_async_contains=contains, batched_get_non_blocking=read,
                batched_get_blocking=lambda keys: (log.append('demand') or [Obj()]),
                _release_memory_obj=lambda o:o.ref_count_down(), cpu_prefetch=None,
                _staging_trace=NS(emit=lambda name, **kw: log.append(name)))
        coordinator = EarlyLookupCoordinator(be, object())
        manager = NS(event_manager=EventManager(), async_serializer=NS(run=serialize),
                     get_active_storage_backends=lambda **kw: [('daos',be)],
                     async_lookup_server=NS(send_response_to_scheduler=lambda rid,n: log.append(('notify',n))))
        await coordinator.lookup(manager, 'r', ['key'], [0,128], pin=True)
        assert ('notify',128) in log and 'unexpected_prefetch' not in log
        future = manager.event_manager.get_event_future(EventType.LOADING,'r')
        view = future.result()[0][0][1]
        view.batch.resolve()  # before the async worker gets a turn
        await asyncio.sleep(0)
        assert log.index(('notify',128)) < log.index('demand')
        assert 'unexpected_prefetch' not in log
        view.ref_count_down()
        coordinator.close()
    asyncio.run(run())


def test_real_lmcache_consumer_resolves_plan_before_reading_payload(monkeypatch):
    import torch
    from lmcache.utils import CacheEngineKey
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.storage_backend.storage_manager import StorageManager
    from lmcache.v1.event_manager import EventStatus
    from lmcache_daos.early_lookup_backend import install_hooks
    # Record originals with monkeypatch so process-local hooks are restored.
    monkeypatch.setattr(StorageManager, 'async_lookup_and_prefetch', StorageManager.async_lookup_and_prefetch)
    monkeypatch.setattr(LMCacheEngine, '_async_process_tokens_internal', LMCacheEngine._async_process_tokens_internal)
    monkeypatch.setattr(StorageManager, '_daos_early_lookup_installed', False, raising=False)
    install_hooks()
    engine = LMCacheEngine.__new__(LMCacheEngine)
    engine.storage_manager = NS(_daos_gpu_store=('daos', NS(early_lookup=object())))
    engine.event_manager = EventManager()
    key = CacheEngineKey('test', 1, 0, 42, torch.bfloat16)
    engine.token_database = NS(process_tokens=lambda **kw: [(0, 128, key)])
    b, calls = batch()
    loop = asyncio.new_event_loop()
    try:
        future = loop.create_future()
        future.set_result([[(key, b.views[0])]])
        engine.event_manager.add_event(EventType.LOADING, 'req', future)
        engine.event_manager.update_event_status(EventType.LOADING, 'req', EventStatus.DONE)
        mask = torch.zeros(128, dtype=torch.bool)
        chunks, size = engine._async_process_tokens_internal(list(range(128)), None, mask, req_id='req')
        assert calls == ['demand'] and mask.all() and size == 20
        assert chunks[0][1].tensor is b.selected[0].tensor
        chunks[0][1].ref_count_down()
        assert b.is_retired
    finally:
        loop.close()


def test_hooks_delegate_when_option_is_disabled(monkeypatch):
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.storage_backend.storage_manager import StorageManager
    from lmcache_daos.early_lookup_backend import install_hooks
    async def old_lookup(manager, *args, **kwargs): return 'old-lookup'
    def old_process(engine, *args, **kwargs): return 'old-process'
    monkeypatch.setattr(StorageManager, 'async_lookup_and_prefetch', old_lookup)
    monkeypatch.setattr(LMCacheEngine, '_async_process_tokens_internal', old_process)
    monkeypatch.setattr(StorageManager, '_daos_early_lookup_installed', False, raising=False)
    install_hooks()
    manager = NS(_daos_gpu_store=('daos', object()))
    assert asyncio.run(StorageManager.async_lookup_and_prefetch(manager)) == 'old-lookup'
    assert LMCacheEngine._async_process_tokens_internal(NS(storage_manager=manager)) == 'old-process'
