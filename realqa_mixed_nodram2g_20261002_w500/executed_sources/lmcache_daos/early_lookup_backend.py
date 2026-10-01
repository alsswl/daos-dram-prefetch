"""Opt-in metadata-first scheduler notification, then speculative GPU reads.

Select EarlyLookupBackend + daosgds.early_lookup=true. Existing backends and
installed LMCache files are unchanged. Event DONE means a READY PLAN here,
not DMA completion; the retrieve hook resolves every view before consumption.
"""
import asyncio
import functools
import threading
import time

from .capacity_probe_backend import CapacityProbeBackend
from .early_lookup_state import EarlyReadBatch, EarlyReadView
from .gds_backend import _cfg


class EarlyLookupCoordinator:
    def __init__(self, backend, cpu):
        self.backend, self.cpu = backend, cpu
        self.condition = threading.Condition()
        self.batches = set()
        self.closed = False
        self.lookups = 0
        self.tasks = set()

    def emit(self, event, **fields):
        self.backend._staging_trace.emit(event, **fields)

    def retired(self, batch):
        with self.condition:
            self.batches.discard(batch)
            self.condition.notify_all()

    def register(self, batch):
        with self.condition:
            if self.closed:
                batch.abort()
                raise RuntimeError('Early lookup closing')
            self.batches.add(batch)

    async def lookup(self, manager, lookup_id, keys, cum_chunk_lengths,
                     search_range=None, pin=False, keys_per_chunk=1):
        from lmcache.v1.event_manager import EventType, EventStatus
        if keys_per_chunk != 1 or not pin or len(cum_chunk_lengths) != len(keys)+1:
            raise ValueError('Early lookup requires pinned, non-layerwise prefix keys')
        with self.condition:
            if self.closed:
                raise RuntimeError('Early lookup closing')
            self.lookups += 1
        jobs, keyed, hit = [], [], 0
        registered = notified = False
        try:
            tiers = list(manager.get_active_storage_backends(search_range=search_range))
            if any(b is not self.cpu and b is not self.backend for _, b in tiers):
                raise ValueError('Early lookup supports only CPU + one DAOS backend')
            for _, backend in tiers:
                remaining = keys[hit:]
                n = await backend.batched_async_contains(lookup_id, remaining, pin)
                if not 0 <= n <= len(remaining):
                    raise ValueError('Invalid lookup prefix length')
                if not n:
                    continue
                selected = remaining[:n]
                sources = None
                if backend is self.cpu:
                    sources = []
                    try:
                        for key in selected:
                            obj = self.cpu.get_blocking(key)
                            if obj is None:
                                raise RuntimeError('Pinned CPU lookup disappeared')
                            sources.append(obj)
                    except BaseException:
                        for key in selected:
                            self.cpu.unpin(key)
                        for obj in sources:
                            obj.ref_count_down()
                        raise
                batch = EarlyReadBatch(lookup_id, 'dram' if sources is not None else 'daos',
                    n, sources, self.backend._release_memory_obj,
                    (lambda: None) if sources is not None else
                    functools.partial(self.demand_read, lookup_id, selected),
                    self.emit, self.retired)
                self.register(batch)
                jobs.append((batch, selected))
                keyed.append(list(zip(selected, batch.views)))
                hit += n
                if hit == len(keys):
                    break
            with self.condition:
                if self.closed:
                    raise RuntimeError('Early lookup closed before publication')
            if hit:
                # Publish the complete plan before notifying: retrieve/abort can
                # race with starting jobs and still owns a valid, cancellable plan.
                future = asyncio.get_running_loop().create_future()
                future.set_result(keyed)
                manager.event_manager.add_event(EventType.LOADING, lookup_id, future)
                manager.event_manager.update_event_status(EventType.LOADING, lookup_id, EventStatus.DONE)
                registered = True
            self.emit('early_lookup_notify', request_id=lookup_id, hit_chunks=hit,
                      hit_tokens=cum_chunk_lengths[hit])
            manager.async_lookup_server.send_response_to_scheduler(lookup_id, cum_chunk_lengths[hit])
            notified = True
            for batch, selected in jobs:
                self.start(manager, batch, selected)
        except BaseException as exc:
            for batch, _ in jobs:
                batch.abort()
            if registered and not notified:
                manager.event_manager.pop_event(EventType.LOADING, lookup_id)
            self.emit('early_lookup_error', request_id=lookup_id, error=repr(exc), notified=notified)
            if not notified:
                # No reuse was promised: ordinary recomputation remains safe.
                manager.async_lookup_server.send_response_to_scheduler(lookup_id, 0)
            else:
                raise
        finally:
            with self.condition:
                self.lookups -= 1
                self.condition.notify_all()

    def demand_read(self, rid, keys):
        self.emit('early_daos_demand_start', request_id=rid, chunks=len(keys))
        return self.backend.batched_get_blocking(keys)

    def start(self, manager, batch, keys):
        if batch.future.cancelled():
            return  # retrieve or abort won before submission
        if batch.tier == 'dram':
            pf = self.backend.cpu_prefetch
            if pf is None:
                if batch.claim():
                    batch.finish()
                return
            try:
                future = pf.worker.submit(functools.partial(self.cpu_work, batch), batch.sources, batch.rid)
                future.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
            except BaseException as exc:
                if batch.claim():
                    batch.finish(error=exc)
        else:
            task = asyncio.create_task(self.daos_work(manager, batch, keys))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
            task.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)

    def cpu_work(self, batch, sources, lookup_id):
        if not batch.claim():
            return
        pf = self.backend.cpu_prefetch
        destinations = None
        try:
            self.emit('early_payload_start', request_id=lookup_id, tier='dram')
            self.backend._ensure_cuda_ctx()
            destinations = pf.reserve(sources)
            if destinations is not None:
                pf.copy(sources, destinations)  # synchronizes even on error
                pf.increment_stats(staged_requests=1, staged_bytes=sum(s.get_size() for s in sources))
                trace = self.backend._staging_trace
                with trace.lock:
                    for obj in destinations:
                        trace.ready[id(obj)] = (lookup_id, obj.get_size())
                        self.backend._probe_cpu_ready[id(obj)] = lookup_id
            else:
                pf.increment_stats(fallback_requests=1)
            self.emit('early_payload_ready', request_id=lookup_id, tier='dram', staged=destinations is not None)
            batch.finish(destinations)
        except BaseException as exc:
            pf.increment_stats(copy_errors=1)
            for obj in destinations or []:
                self.backend._release_memory_obj(obj)
            batch.finish(error=exc)

    async def daos_work(self, manager, batch, keys):
        entered = False
        async def read():
            nonlocal entered
            entered = True
            if not batch.claim():
                return
            try:
                self.emit('early_payload_start', request_id=batch.rid, tier='daos')
                objects = await self.backend.batched_get_non_blocking(batch.rid, keys)
                self.emit('early_payload_ready', request_id=batch.rid, tier='daos', chunks=len(objects))
                batch.finish(objects)
            except BaseException as exc:
                batch.finish(error=exc)
        coro = read()
        try:
            # Preserve existing weighted admission + parallel chunk I/O. A queued
            # demand read bypasses this speculative wait, not physical allocation.
            await manager.async_serializer.run(coro, len(keys))
        except BaseException as exc:
            if not entered:
                coro.close()
                # Budget errors (e.g. a batch > pool) defer the read to retrieve;
                # the eventual allocation/IO failure must never be hidden.
                batch.future.cancel()
                self.emit('early_prefetch_deferred', request_id=batch.rid, error=repr(exc))

    def close(self):
        with self.condition:
            self.closed = True
            if not self.condition.wait_for(lambda: self.lookups == 0, timeout=30):
                raise RuntimeError('Early lookup still active; refusing to release CPU pool')
            batches = list(self.batches)
        for batch in batches:
            batch.abort()
        for batch in batches:
            batch.work_finished.wait()
        # All payload work has now finished or was cancelled before start.
        # Retire serializer waiters too, without ever cancelling a running DMA.
        for task in list(self.tasks):
            if not task.done():
                task.get_loop().call_soon_threadsafe(task.cancel)


def selected_coordinator(manager):
    selected = getattr(manager, '_daos_gpu_store', None)
    return getattr(selected[1], 'early_lookup', None) if selected else None


def install_hooks():
    from lmcache.v1.cache_engine import LMCacheEngine
    from lmcache.v1.storage_backend.storage_manager import StorageManager
    from lmcache.v1.event_manager import EventType
    if getattr(StorageManager, '_daos_early_lookup_installed', False):
        return
    original_lookup = StorageManager.async_lookup_and_prefetch
    original_process = LMCacheEngine._async_process_tokens_internal

    @functools.wraps(original_lookup)
    async def lookup(manager, *args, **kwargs):
        coordinator = selected_coordinator(manager)
        if coordinator is None:
            return await original_lookup(manager, *args, **kwargs)
        return await coordinator.lookup(manager, *args, **kwargs)

    @functools.wraps(original_process)
    def process(engine, *args, **kwargs):
        if selected_coordinator(engine.storage_manager) is None:
            return original_process(engine, *args, **kwargs)
        rid = kwargs['req_id']
        try:
            future = engine.event_manager.get_event_future(EventType.LOADING, rid)
            seen = set()
            for tier in future.result():
                for _, obj in tier:
                    if isinstance(obj, EarlyReadView) and id(obj.batch) not in seen:
                        seen.add(id(obj.batch))
                        obj.batch.resolve()
        except BaseException:
            engine.cleanup_memory_objs(rid)
            raise  # Never run the model with missing announced KV.
        return original_process(engine, *args, **kwargs)

    StorageManager.async_lookup_and_prefetch = lookup
    LMCacheEngine._async_process_tokens_internal = process
    StorageManager._daos_early_lookup_installed = True


class EarlyLookupBackend(CapacityProbeBackend):
    def __init__(self, config, dst_device='cuda', metadata=None, local_cpu_backend=None, loop=None):
        enabled = _cfg(config, 'early_lookup', False)
        if type(enabled) is not bool:
            raise ValueError('early_lookup must be a YAML boolean')
        if enabled and (not config.enable_async_loading
                or not config.local_cpu or config.use_layerwise
                or _cfg(config, 'dram_prefetch_early_ready', False)
                or _cfg(config, 'dram_prefetch_cancel_queued', False)):
            raise ValueError('EarlyLookupBackend requires early_lookup=true, async+CPU, '
                             'non-layerwise; disable legacy early_ready/cancel options')
        super().__init__(config, dst_device, metadata, local_cpu_backend, loop)
        if not enabled:
            return  # exact original CapacityProbeBackend path
        self.early_lookup = EarlyLookupCoordinator(self, local_cpu_backend)
        original_close = local_cpu_backend.close
        def close_cpu():
            self.early_lookup.close()  # drain read/DMA ownership before pinned CPU free
            return original_close()
        local_cpu_backend.close = close_cpu
        install_hooks()
        self._staging_trace.emit('early_lookup_enabled', notification='metadata',
                                 cancel_queued=True, running_read='wait', failure='raise')
