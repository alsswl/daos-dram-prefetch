import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
import threading
from types import SimpleNamespace as NS

import pytest

from test_dram_prefetch import fixture, discard
from lmcache_daos.queued_prefetch import DeferredCPUObject, resolve_event
from lmcache_daos.prefetch_timing import install


def setup(cancel=True):
    pf, cpu, inner, sources = fixture()
    # Unit tests need no installed engine hook or GPU. Integration tested below.
    pf.early_ready, pf.cancel_queued = True, cancel
    return pf, cpu, inner, sources


def block_worker(pf):
    entered, release = threading.Event(), threading.Event()
    def block():
        entered.set()
        assert release.wait(5)
    pf.worker.submit(block)
    assert entered.wait(2)
    return release


def test_queued_retrieve_cancels_without_waiting_or_allocating():
    pf, cpu, inner, sources = setup()
    release = block_worker(pf)
    rows = []
    install(pf, NS(emit=lambda event, **row: rows.append(row)))
    try:
        sources[0].pin()
        out = asyncio.run(cpu.batched_get_non_blocking('queued', ['k']))
        assert isinstance(out[0], DeferredCPUObject)
        assert not release.is_set() and not out[0].batch.future.done()
        out[0].batch.resolve()
        assert out[0].raw_data.device.type == 'cpu'
        assert out[0].batch.future.cancelled()
        assert inner.calls == 0 and pf.stats['retrieve_cancelled_queued'] == 1
        discard(out)
        assert sources[0].refs == 1 and sources[0].pins == 0
        assert pf.stats['deferred_pending_batches'] == 0
        release.set()
        pf.close()
        assert inner.calls == 0 and rows == []  # cancelled task never executes
    finally:
        release.set(); pf.close()


def test_running_copy_is_not_cancelled_and_buffers_remain_live():
    pf, cpu, inner, sources = setup()
    entered, release = threading.Event(), threading.Event()
    def copy(*args):
        entered.set()
        assert release.wait(5)
    pf.copy = copy
    try:
        out = asyncio.run(cpu.batched_get_non_blocking('running', ['k']))
        assert entered.wait(2)
        with ThreadPoolExecutor(1) as executor:
            resolving = executor.submit(out[0].batch.resolve)
            # State is RUNNING, so cancellation is impossible by Future contract.
            assert not out[0].batch.future.cancel()
            assert inner.total_allocated_size == 32 and sources[0].refs == 2
            release.set()
            resolving.result(timeout=2)
        assert out[0].raw_data.device.type == 'cuda'
        assert pf.stats['retrieve_cancelled_queued'] == 0
        discard(out)
        assert inner.total_allocated_size == 0 and sources[0].refs == 1
    finally:
        release.set(); pf.close()


def test_completed_copy_uses_staging_and_capacity_failure_uses_cpu():
    for capacity_fail in (False, True):
        pf, cpu, inner, sources = setup()
        if capacity_fail: inner.fail_on = 1
        try:
            out = asyncio.run(cpu.batched_get_non_blocking('done', ['k']))
            out[0].batch.future.result(timeout=2)
            out[0].batch.resolve()
            assert out[0].raw_data.device.type == ('cpu' if capacity_fail else 'cuda')
            assert pf.stats['retrieve_capacity_cpu' if capacity_fail else 'retrieve_ready_gpu'] == 1
            discard(out)
            assert sources[0].refs == 1 and inner.total_allocated_size == 0
        finally: pf.close()


def test_early_wait_control_does_not_cancel_queued_work():
    pf, cpu, inner, sources = setup(cancel=False)
    release = block_worker(pf)
    try:
        out = asyncio.run(cpu.batched_get_non_blocking('control', ['k']))
        with ThreadPoolExecutor(1) as executor:
            resolving = executor.submit(out[0].batch.resolve)
            assert not out[0].batch.future.cancelled()
            release.set()
            resolving.result(timeout=2)
        assert out[0].raw_data.device.type == 'cuda'
        assert pf.stats['retrieve_cancelled_queued'] == 0
        discard(out)
        assert inner.total_allocated_size == 0 and sources[0].refs == 1
    finally:
        release.set(); pf.close()


@pytest.mark.parametrize('running', [False, True])
def test_request_abort_cleans_after_dma_or_cancels_before_start(running):
    pf, cpu, inner, sources = setup()
    entered, finish = threading.Event(), threading.Event()
    def copy(*args):
        entered.set()
        assert finish.wait(5)
    pf.copy = copy
    release = None if running else block_worker(pf)
    try:
        sources[0].pin()
        out = asyncio.run(cpu.batched_get_non_blocking('abort', ['k']))
        if running: assert entered.wait(2)
        discard(out)
        if running:
            assert sources[0].refs == 2 and inner.total_allocated_size == 32
            finish.set(); out[0].batch.future.result(timeout=2)
        else:
            assert out[0].batch.future.cancelled()
        assert sources[0].refs == 1 and sources[0].pins == 0
        assert inner.total_allocated_size == 0
        assert pf.stats['deferred_pending_batches'] == 0
    finally:
        finish.set()
        if release: release.set()
        pf.close()


def test_copy_error_propagates_and_abort_releases_once():
    pf, cpu, inner, sources = setup()
    def fail(*args): raise RuntimeError('injected')
    pf.copy = fail
    try:
        out = asyncio.run(cpu.batched_get_non_blocking('error', ['k']))
        with pytest.raises(RuntimeError, match='injected'): out[0].batch.future.result(timeout=2)
        with pytest.raises(RuntimeError, match='injected'): out[0].batch.resolve()
        discard(out)
        assert sources[0].refs == 1 and inner.total_allocated_size == 0
        assert pf.stats['copy_errors'] == 1 and pf.stats['deferred_pending_batches'] == 0
    finally: pf.close()


def test_event_hook_resolves_cpu_once_and_does_not_touch_daos():
    pf, cpu, inner, sources = setup()
    release = block_worker(pf)
    try:
        out = asyncio.run(cpu.batched_get_non_blocking('event', ['k']))
        future = Future()
        daos = object()
        future.set_result([[('k', out[0])], [('daos', daos)]])
        engine = NS(event_manager=NS(get_event_future=lambda *args: future))
        resolve_event(engine, 'event')
        resolve_event(engine, 'event')
        assert pf.stats['retrieve_cancelled_queued'] == 1
        discard(out)
        assert sources[0].refs == 1
    finally:
        release.set(); pf.close()


def test_cancel_start_race_repeated_no_double_release():
    pf, cpu, inner, sources = setup()
    try:
        for i in range(100):
            out = asyncio.run(cpu.batched_get_non_blocking(str(i), ['k']))
            out[0].batch.resolve()
            discard(out)
            assert sources[0].refs == 1 and inner.total_allocated_size == 0
        assert pf.stats['deferred_pending_batches'] == 0
    finally: pf.close()


def test_installed_hook_resolves_before_native_token_processing(monkeypatch):
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache_daos.queued_prefetch import install_retrieve_hook
    pf, cpu, inner, sources = setup()
    release = block_worker(pf)
    try:
        out = asyncio.run(cpu.batched_get_non_blocking('hook', ['k']))
        future = Future(); future.set_result([[('k', out[0])]])
        engine = NS(event_manager=NS(get_event_future=lambda *args: future))
        def native(self, **kwargs):
            assert out[0].batch.future.cancelled()
            return 'native result'
        monkeypatch.setattr(LMCacheEngine, '_async_process_tokens_internal', native)
        monkeypatch.setattr(LMCacheEngine, '_daos_deferred_cpu_installed', False, raising=False)
        install_retrieve_hook()
        patched = LMCacheEngine._async_process_tokens_internal
        install_retrieve_hook()
        assert LMCacheEngine._async_process_tokens_internal is patched
        assert patched(engine, req_id='hook') == 'native result'
        discard(out)
    finally:
        release.set(); pf.close()


def test_partial_consumer_cleanup_holds_all_sources_until_copy_done():
    from test_dram_prefetch import Obj
    pf, cpu, inner, sources = fixture(sources=[Obj(), Obj()])
    pf.early_ready = pf.cancel_queued = True
    entered, finish = threading.Event(), threading.Event()
    def copy(*args):
        entered.set()
        assert finish.wait(5)
    pf.copy = copy
    try:
        out = asyncio.run(cpu.batched_get_non_blocking('partial', ['a','b']))
        assert entered.wait(2)
        out[0].ref_count_down()
        assert inner.total_allocated_size == 64 and all(s.refs == 2 for s in sources)
        finish.set(); out[1].batch.future.result(timeout=2)
        assert inner.total_allocated_size == 32 and sources[0].refs == 1
        out[1].batch.resolve()
        discard([out[1]])
        assert inner.total_allocated_size == 0 and all(s.refs == 1 for s in sources)
    finally:
        finish.set(); pf.close()
