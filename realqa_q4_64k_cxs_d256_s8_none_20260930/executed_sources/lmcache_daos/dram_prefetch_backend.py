"""Opt-in CPU-hit -> shared GPU staging prefetch, without changing DAOS I/O.

The original LocalCPUBackend still owns admission, lookup, pinning and eviction.
Only its async GET result is optionally staged. No installed LMCache files change.
Space exhaustion falls back to the original CPU objects (never a cache miss).
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import math
import threading
import time

import torch

from .gds_backend import DaosGdsBackend, _cfg, logger


class StagedCPUObject:
    """GPU view with the original CPU caller's reference/pin lifetime.

Normal retrieve/abort paths unpin and release this object exactly as they would
the CPU result. The CPU cache keeps its separate ownership reference. Keeping
the source until consumption avoids changing native CPU pin/eviction semantics.
"""
    def __init__(self, source, destination, release_gpu):
        self.source = source
        self.destination = destination
        self._release_gpu = release_gpu
        self._refs = 1
        self._lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self.destination, name)

    @property
    def is_pinned(self):
        return self.source.is_pinned

    def pin(self):
        return self.source.pin()

    def unpin(self):
        return self.source.unpin()

    def get_ref_count(self):
        with self._lock:
            return self._refs

    def ref_count_up(self):
        with self._lock:
            if self._refs <= 0:
                raise RuntimeError('Cannot retain a released staged CPU object')
            self._refs += 1

    def ref_count_down(self):
        with self._lock:
            if self._refs <= 0:
                raise RuntimeError('Double release of staged CPU object')
            self._refs -= 1
            release = self._refs == 0
        if release:
            try:
                self._release_gpu(self.destination)
            finally:
                self.source.ref_count_down()


def discard(objects):
    """Same pin/ref cleanup convention as LMCache's aborted-prefetch path."""
    for obj in objects:
        if obj.is_pinned:
            obj.unpin()
        obj.ref_count_down()


class CPUHitPrefetch:
    def __init__(self, backend, cpu, limit_bytes, *, cancel_queued=False, early_ready=False, workers=1,
                 occupancy_stop_bytes=None):
        if type(workers) is not int or workers < 1:
            raise ValueError('dram_prefetch_workers must be a positive integer')
        if occupancy_stop_bytes is not None and (type(occupancy_stop_bytes) is not int or occupancy_stop_bytes <= 0):
            raise ValueError('occupancy_stop_bytes must be a positive integer or None')
        self.backend = backend
        self.cpu = cpu
        self.limit_bytes = limit_bytes
        self.occupancy_stop_bytes = occupancy_stop_bytes
        self.original_get = cpu.batched_get_non_blocking
        self.original_close = cpu.close
        self.workers = workers
        self.worker = ThreadPoolExecutor(max_workers=workers, thread_name_prefix='dram-gpu-prefetch')
        self._worker_state = threading.local()
        self._stats_lock = threading.Lock()
        self.closed = False
        self.cancel_queued = cancel_queued
        self.early_ready = early_ready or cancel_queued
        self._deferred_stats_lock = threading.Lock()
        self.stats = dict(staged_requests=0, staged_bytes=0, fallback_requests=0,
                          cancelled_requests=0, copy_errors=0,
                          watermark_rejections=0, capacity_rejections=0,
                          occupancy_rejections=0, occupancy_rejected_bytes=0)
        self.stats.update(deferred_pending_batches=0, deferred_submitted=0,
                          deferred_abort_cancelled=0, retrieve_cancelled_queued=0,
                          retrieve_ready_gpu=0, retrieve_waited_gpu=0,
                          retrieve_capacity_cpu=0)
        if self.early_ready:
            from .queued_prefetch import install_retrieve_hook
            install_retrieve_hook()
        # Bound methods assigned to this instance only; no global CPU class patch.
        cpu.batched_get_non_blocking = self.get
        cpu.close = self.close_cpu

    @property
    def stream(self):
        """Each executor thread owns its stream, including its completion wait."""
        return getattr(self._worker_state, 'stream', None)

    @stream.setter
    def stream(self, value):
        self._worker_state.stream = value

    def increment_stats(self, **deltas):
        with self._stats_lock:
            for key, value in deltas.items():
                self.stats[key] += value

    def reserve(self, sources):
        """Atomic all-or-nothing allocation against the real shared pool.

        The limit is an admission watermark for TOTAL pool use, not a new pool
        and not a coroutine counter. Existing ready buffers count until freed.
        DAOS may use the remaining pool independently; no wait/deadlock here.
        A None limit disables the admission watermark, not the physical pool
        limit: attempt allocations and roll back on actual capacity exhaustion.
        """
        pool = self.backend.memory_allocator
        inner = pool.allocator
        required = sum(inner.address_manager.compute_aligned_size(s.get_size()) for s in sources)
        with pool.device_mem_lock:
            # Current TOTAL occupancy (DRAM prefetch + DAOS + other staging
            # allocations), not projected occupancy after this request. A batch
            # admitted below the threshold may cross it. Check and allocation
            # share the pool lock so concurrent workers cannot use stale state.
            if (self.occupancy_stop_bytes is not None and
                    inner.total_allocated_size >= self.occupancy_stop_bytes):
                self.increment_stats(occupancy_rejections=1, occupancy_rejected_bytes=required)
                trace = getattr(self.backend, '_staging_trace', None)
                if trace is not None:
                    trace.emit('cpu_prefetch_occupancy_reject',
                               threshold_bytes=self.occupancy_stop_bytes,
                               requested_bytes=required, chunks=len(sources))
                return None
            if self.limit_bytes is not None and inner.total_allocated_size + required > self.limit_bytes:
                self.increment_stats(watermark_rejections=1)
                return None
            allocated = []
            try:
                for src in sources:
                    obj = inner.allocate(src.get_shapes(), src.get_dtypes(), src.metadata.fmt)
                    if obj is None:
                        self.increment_stats(capacity_rejections=1)
                        for old in allocated:
                            old.ref_count_down()  # already hold outer allocator lock
                        return None
                    allocated.append(obj)
                return allocated
            except BaseException:
                for old in allocated:
                    old.ref_count_down()
                raise

    def copy(self, sources, destinations):
        # Each worker owns a separate stream. The async event-loop/model thread never
        # waits on CUDA here. Publish GPU objects only after DMA completion.
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=self.backend.device_id)
        try:
            with torch.cuda.stream(self.stream):
                for source, destination in zip(sources, destinations, strict=True):
                    n = source.get_size()
                    if destination.get_size() != n:
                        raise ValueError('CPU/GPU prefetch layout byte-size mismatch')
                    destination.raw_data[:n].copy_(source.raw_data[:n], non_blocking=True)
        finally:
            # Also drain previously enqueued copies before error cleanup/reuse.
            self.stream.synchronize()

    def stage(self, sources, lookup_id):
        started = time.perf_counter()
        destinations = None
        try:
            self.backend._ensure_cuda_ctx()
            if any(s.raw_data.device.type != 'cpu' for s in sources):
                destinations = None
            else:
                destinations = self.reserve(sources)
            if destinations is None:
                self.increment_stats(fallback_requests=1)
                logger.info('CPU staging fallback[%s]: %d objects (shared pool admission/allocation)',
                            lookup_id[-8:], len(sources))
                return sources
            self.copy(sources, destinations)
            result = [StagedCPUObject(src, dst, self.backend._release_memory_obj)
                      for src, dst in zip(sources, destinations, strict=True)]
            n = sum(s.get_size() for s in sources)
            self.increment_stats(staged_requests=1, staged_bytes=n)
            logger.info('CPU staging prefetch[%s]: %d objects, %.1f MiB in %.2f ms',
                        lookup_id[-8:], len(sources), n / 2**20,
                        (time.perf_counter() - started) * 1000)
            return result
        except BaseException:
            self.increment_stats(copy_errors=1)
            for obj in destinations or []:
                self.backend._release_memory_obj(obj)
            discard(sources)
            raise

    async def get(self, lookup_id, keys, transfer_spec=None):
        sources = await self.original_get(lookup_id, keys, transfer_spec)
        if not sources:
            return sources
        if self.closed:
            return sources
        if self.early_ready:
            from .queued_prefetch import DeferredBatch, DeferredCPUObject
            batch = DeferredBatch(self, sources, lookup_id)
            return [DeferredCPUObject(batch, i) for i in range(len(sources))]
        try:
            future = self.worker.submit(self.stage, sources, lookup_id)
        except BaseException:
            discard(sources)
            raise
        wrapped = asyncio.wrap_future(future)
        # Consume exceptions even if the awaiting coroutine is cancelled.
        wrapped.add_done_callback(lambda f: f.exception() if not f.cancelled() else None)
        try:
            return await asyncio.shield(wrapped)
        except asyncio.CancelledError:
            self.increment_stats(cancelled_requests=1)
            # Do not release/reuse host/GPU buffers while DMA is outstanding.
            # concurrent Future callback still runs if the asyncio loop stops.
            def cleanup(completed):
                if completed.exception() is None:
                    discard(completed.result())
            future.add_done_callback(cleanup)
            raise

    def close(self):
        if not self.closed:
            self.closed = True
            self.worker.shutdown(wait=True)
            self.cpu.batched_get_non_blocking = self.original_get
            self.cpu.close = self.original_close
            logger.info('CPU staging final stats: %s', self.stats)

    def close_cpu(self):
        # StorageManager closes CPU before DAOS. Drain host->GPU work before
        # the CPU allocator can release its pinned backing allocation.
        self.close()
        return self.original_close()


class DaosDramPrefetchBackend(DaosGdsBackend):
    def __init__(self, config, dst_device='cuda', metadata=None, local_cpu_backend=None, loop=None):
        if not config.local_cpu or not config.enable_async_loading or config.use_layerwise:
            raise ValueError('CPU staging prefetch requires local_cpu, async and non-layerwise mode')
        if local_cpu_backend is None:
            raise ValueError('CPU staging prefetch requires LocalCPUBackend')
        capacity = float(_cfg(config, 'gpu_buffer_gb', 6))
        limit = float(_cfg(config, 'cpu_prefetch_gpu_gb', capacity / 2))
        if not math.isfinite(limit) or not 0 < limit <= capacity:
            raise ValueError('cpu_prefetch_gpu_gb must be positive and <= GPU staging capacity')
        super().__init__(config, dst_device, metadata, local_cpu_backend, loop)
        self.cpu_prefetch = CPUHitPrefetch(self, local_cpu_backend, int(limit * 2**30))
        logger.info('CPU staging prefetch enabled: shared-pool admission watermark %.2f GiB', limit)

    def close(self):
        self.cpu_prefetch.close()
        super().close()
