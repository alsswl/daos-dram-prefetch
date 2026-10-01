"""Bounded staging for stores as well as reads, with asynchronous DAOS puts.

Only this opt-in backend changes store admission. Preserve native key hashes,
absolute token/slot offsets, GPU gather, DAOS worker ownership and DRAM mirror.
Wait for free staging BEFORE gathering a window; never drop an unallocated tail.
The last window may remain in flight after store returns, as in the baseline.
"""
import functools
import math
import time

from .gds_backend import _cfg, logger
from .gpu_store import _store_context
from .windowed_demand_backend import WindowedDemandBackend
from .windowed_transfer import window_chunks


def wait_for_space(backend, required, emit, *, clock=time.monotonic, sleep=time.sleep):
    """Mirror references count as occupied until their final release."""
    started = clock()
    allocator = backend.memory_allocator
    while True:
        with allocator.device_mem_lock:
            free = backend.gpu_buffer_bytes - allocator.allocator.total_allocated_size
        if free >= required:
            elapsed = clock() - started
            if elapsed >= .001:
                emit('store_window_wait', required_bytes=required, wait_ms=elapsed*1000)
            return
        if clock() - started >= backend.store_window_timeout_s:
            raise TimeoutError('Windowed store timed out waiting for staging; no tail may be skipped')
        sleep(.001)


def store_windows(engine, backend, tokens, mask, kwargs):
    if not engine.is_healthy() or engine._is_passive() or engine.is_frozen():
        return
    if not backend.store_enabled:
        return
    if tokens is None:
        raise ValueError('Windowed store requires tokens, not hash-only inputs')
    from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2
    connector = engine.gpu_connector
    if type(connector) is not VLLMPagedMemGPUConnectorV2:
        raise ValueError('Windowed store requires the native V2 paged connector')
    rid = engine._get_req_id(kwargs)
    emit = lambda event, **fields: backend._staging_trace.emit(event, request_id=rid, **fields)
    required = int(mask.sum()) if mask is not None else len(tokens)
    stats = engine.stats_monitor.on_store_request(required)
    completed = total_bytes = 0
    started = time.perf_counter()
    try:
        with stats.profile_process_tokens():
            infos = list(engine.token_database.process_tokens(
                tokens=tokens, mask=mask, request_configs=kwargs.get('request_configs')))
        if not infos:
            return
        # Read and write admissions share this lock. Existing asynchronous puts
        # and D2H mirrors can finish and release buffers while admission waits.
        with backend.window_lock:
            emit('store_window_request_start', chunks=len(infos), window_mib=backend.store_window_mib)
            for at in range(0, len(infos), backend.store_window_chunks):
                group = infos[at:at+backend.store_window_chunks]
                need = len(group) * backend.store_chunk_bytes
                wait_for_space(backend, need, emit)
                objects = []
                handed_off = False
                try:
                    for start, end, key in group:
                        if end-start != engine.config.chunk_size:
                            raise ValueError('Windowed store requires full-size chunks')
                        obj = engine.storage_manager.allocate(
                            engine.metadata.get_shapes(end-start), engine.metadata.get_dtypes(),
                            fmt=engine.fmt, busy_loop=False)
                        if obj is None:
                            # With equal-size chunks and serialized admission,
                            # the preceding capacity reservation must suffice.
                            # Reject instead of silently truncating the store.
                            raise RuntimeError('Windowed store allocation failed despite available capacity')
                        objects.append(obj)
                    size = sum(obj.get_size() for obj in objects)
                    emit('store_window_copy_start', chunks=len(group), bytes=size,
                         start=group[0][0], end=group[-1][1])
                    with stats.profile_from_gpu():
                        connector.batched_from_gpu(objects, [x[0] for x in group],
                                                   [x[1] for x in group], **kwargs)
                    # put_direct synchronizes this gather stream, then retains
                    # references for DAOS and the existing asynchronous mirror.
                    context = _store_context.set((engine.storage_manager, connector.store_stream))
                    try:
                        with stats.profile_put():
                            handed_off = True  # put_direct releases even on errors
                            engine.storage_manager.batched_put(
                                [x[2] for x in group], objects,
                                transfer_spec=kwargs.get('transfer_spec'), location=engine.store_location)
                    finally:
                        _store_context.reset(context)
                    completed += sum(e-s for s,e,_ in group)
                    total_bytes += size
                    emit('store_window_submitted', chunks=len(group), bytes=size,
                         start=group[0][0], end=group[-1][1])
                finally:
                    if not handed_off:
                        # The gather may already have enqueued a copy when an
                        # exception occurred. Never release its backing early.
                        connector.store_stream.synchronize()
                        for obj in objects:
                            backend._release_memory_obj(obj)
            emit('store_window_request_done', tokens=completed, chunks=len(infos), bytes=total_bytes)
    finally:
        engine.stats_monitor.on_store_finished(stats, completed)
    logger.info('[req_id=%s] Stored %d out of total %d tokens. Windowed staging: bytes=%d cost_ms=%.3f',
                rid, completed, required, total_bytes, (time.perf_counter()-started)*1000)


def install_store_window_hook():
    from lmcache.v1.cache_engine import LMCacheEngine
    if getattr(LMCacheEngine, '_daos_windowed_store_installed', False):
        return
    original = LMCacheEngine.store

    @functools.wraps(original)
    def store(engine, tokens=None, hashes=None, offsets=None, mask=None, **kwargs):
        selected = getattr(engine.storage_manager, '_daos_gpu_store', None)
        if not selected or not isinstance(selected[1], WindowedStoreBackend):
            return original(engine, tokens=tokens, hashes=hashes, offsets=offsets, mask=mask, **kwargs)
        if hashes is not None or offsets is not None:
            raise ValueError('Windowed store does not support hash-only inputs')
        return store_windows(engine, selected[1], tokens, mask, kwargs)

    LMCacheEngine.store = store
    LMCacheEngine._daos_windowed_store_installed = True


class WindowedStoreBackend(WindowedDemandBackend):
    def __init__(self, config, dst_device='cuda', metadata=None, local_cpu_backend=None, loop=None):
        if (getattr(config, 'save_unfull_chunk', False) or config.use_layerwise
                or getattr(config, 'enable_kv_events', False)):
            raise ValueError('Windowed store requires full chunks, non-layerwise, no KV events')
        window = _cfg(config, 'store_window_mib', 500)
        timeout = _cfg(config, 'store_window_timeout_s', 5.0)
        if isinstance(timeout, bool) or not math.isfinite(float(timeout)) or timeout <= 0:
            raise ValueError('store_window_timeout_s must be positive and finite')
        size = sum(math.prod(s)*d.itemsize for s,d in zip(metadata.get_shapes(), metadata.get_dtypes()))
        size = (size+4095)//4096*4096
        limit = window_chunks(window, size, int(_cfg(config, 'gpu_buffer_gb', 1)*2**30))
        if not limit or not _cfg(config, 'retrieve_window_mib', 0):
            raise ValueError('Windowed store requires positive read and write windows')
        self.store_window_mib, self.store_window_chunks = window, limit
        self.store_window_timeout_s, self.store_chunk_bytes = float(timeout), size
        super().__init__(config, dst_device, metadata, local_cpu_backend, loop)
        install_store_window_hook()
        self._staging_trace.emit('windowed_store_enabled', window_mib=window,
                                 effective_window_bytes=limit*size, async_put=True)
