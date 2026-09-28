"""DAOS-first GPU writes and opt-in reads with an asynchronous CPU side cache.

Only publish CPU objects after pinned D2H completes. No CPU payload is used as
the source of a DAOS put. No installed LMCache file is changed.
"""
from concurrent.futures import ThreadPoolExecutor
import math
import threading
import time

import torch

from .gds_backend import DaosGdsBackend, _cfg, logger
from .gpu_store import install_store_context


class AsyncDramMirror:
    def __init__(self, backend, cpu, limit_bytes, max_age_ms):
        self.backend, self.cpu = backend, cpu
        self.limit_bytes, self.max_age_ms = limit_bytes, max_age_ms
        self.lock = threading.Lock()
        self.pending = set()
        self.pending_bytes = 0
        self.closed = False
        self.stream = None
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='daos-dram-mirror')
        self.stats = dict(admitted=0, copied=0, copied_bytes=0, skipped_budget=0,
                          skipped_present=0, skipped_closed=0, skipped_expired=0,
                          skipped_allocation=0, errors=0, peak_pending_bytes=0,
                          copy_ms=0., cpu_hit_chunks=0)
        for origin in ('write', 'read'):
            self.stats.update({f'{origin}_offered': 0, f'{origin}_admitted': 0,
                               f'{origin}_copied': 0, f'{origin}_copied_bytes': 0})
        self.stats['read_submit_errors'] = 0
        self.original_close = cpu.close
        self.original_contains = cpu.batched_async_contains
        cpu.close = self.close_cpu
        cpu.batched_async_contains = self.contains

    async def contains(self, lookup_id, keys, pin=False):
        # Native async lookup doesn't call touch_cache(). Update this instance
        # at the actual hit, under the same lock as pin/eviction. Prefix first
        # becomes most recent; don't accumulate native keys_in_request forever.
        hits = []
        with self.cpu.cpu_lock:
            for key in keys:
                obj = self.cpu.hot_cache.get(key)
                if obj is None:
                    break
                if pin:
                    obj.pin()
                hits.append(key)
            for key in reversed(hits):
                self.cpu.cache_policy.update_on_hit(key, self.cpu.hot_cache)
        with self.lock:
            self.stats['cpu_hit_chunks'] += len(hits)
        return len(hits)

    def offer(self, key, source, origin='write'):
        """Called after successful DAOS put/get; never wait for CPU space."""
        if origin not in ('write', 'read'):
            raise ValueError('Unknown DRAM copy origin')
        size = source.get_size()
        with self.lock:
            self.stats[f'{origin}_offered'] += 1
            if self.closed:
                self.stats['skipped_closed'] += 1
                return False
            if key in self.pending or self.cpu.contains(key):
                self.stats['skipped_present'] += 1
                return False
            if self.pending_bytes + size > self.limit_bytes:
                self.stats['skipped_budget'] += 1
                return False
            source.ref_count_up()
            self.pending.add(key)
            self.pending_bytes += size
            self.stats['peak_pending_bytes'] = max(self.stats['peak_pending_bytes'], self.pending_bytes)
            try:
                self.worker.submit(self.copy_one, key, source, size, time.monotonic(), origin)
            except BaseException:
                self.pending.remove(key)
                self.pending_bytes -= size
                source.ref_count_down()
                raise
            self.stats['admitted'] += 1
            self.stats[f'{origin}_admitted'] += 1
            return True

    def copy_payload(self, source, destination, size):
        if destination.raw_data.device.type != 'cpu' or not destination.raw_data.is_pinned():
            raise ValueError('DRAM mirror destination must be pinned CPU memory')
        if source.raw_data.device.type != 'cuda' or destination.get_size() != size:
            raise ValueError('DRAM mirror source/size mismatch')
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=self.backend.device_id, priority=0)
        try:
            with torch.cuda.stream(self.stream):
                destination.raw_data[:size].copy_(source.raw_data[:size], non_blocking=True)
        finally:
            # Including exception paths: no reuse/free until outstanding DMA ends.
            self.stream.synchronize()

    def copy_one(self, key, source, size, admitted_at, origin='write'):
        destination = None
        started = time.monotonic()
        try:
            if (started - admitted_at) * 1000 > self.max_age_ms:
                with self.lock:
                    self.stats['skipped_expired'] += 1
                return
            self.backend._ensure_cuda_ctx()
            destination = self.cpu.allocate(source.get_shapes(), source.get_dtypes(),
                                            fmt=source.metadata.fmt, eviction=True, busy_loop=False)
            if destination is None:
                with self.lock:
                    self.stats['skipped_allocation'] += 1
                return
            self.copy_payload(source, destination, size)
            # Only now does lookup see the CPU entry. CPU cache takes its own ref.
            self.cpu.submit_put_task(key, destination)
            with self.lock:
                self.stats['copied'] += 1
                self.stats['copied_bytes'] += size
                self.stats[f'{origin}_copied'] += 1
                self.stats[f'{origin}_copied_bytes'] += size
                self.stats['copy_ms'] += (time.monotonic() - started) * 1000
        except Exception:
            with self.lock:
                self.stats['errors'] += 1
            logger.exception('Asynchronous DRAM copy failed; DAOS copy remains valid')
        finally:
            if destination is not None:
                destination.ref_count_down()
            self.backend._release_memory_obj(source)
            with self.lock:
                self.pending.discard(key)
                self.pending_bytes -= size

    def snapshot(self):
        with self.lock:
            return dict(self.stats, pending_bytes=self.pending_bytes, pending_chunks=len(self.pending))

    def close(self):
        with self.lock:
            self.closed = True
        self.worker.shutdown(wait=True)

    def close_cpu(self):
        # StorageManager closes CPU before DAOS. Stop new admission and drain
        # DMA before the pinned backing allocation can be released.
        self.close()
        self.cpu.batched_async_contains = self.original_contains
        return self.original_close()


class DaosAsyncDramBackend(DaosGdsBackend):
    def __init__(self, config, dst_device='cuda', metadata=None, local_cpu_backend=None, loop=None):
        if not _cfg(config, 'async_dram', False):
            raise ValueError('DaosAsyncDramBackend requires daosgds.async_dram: true')
        self.promote_on_read = bool(_cfg(config, 'dram_promote_on_read', False))
        if not _cfg(config, 'store', True) and not self.promote_on_read:
            raise ValueError('async_dram requires DAOS store or read promotion enabled')
        if (not config.enable_async_loading or config.cache_policy != 'LRU'
                or local_cpu_backend is None):
            raise ValueError('async_dram requires async loading, LRU and LocalCPUBackend')
        capacity = float(_cfg(config, 'gpu_buffer_gb', 6))
        limit = float(_cfg(config, 'dram_mirror_max_pending_gb', min(1., capacity / 4)))
        age = float(_cfg(config, 'dram_mirror_max_age_ms', 100))
        if not math.isfinite(limit) or not 0 < limit <= capacity:
            raise ValueError('DRAM mirror pending budget must be positive and <= GPU staging capacity')
        if not math.isfinite(age) or age <= 0:
            raise ValueError('DRAM mirror max age must be finite and positive')
        super().__init__(config, dst_device, metadata, local_cpu_backend, loop)
        self.dram_mirror = AsyncDramMirror(self, local_cpu_backend, int(limit * 2**30), age)
        install_store_context()
        logger.info('Async DRAM mirror: after DAOS success, pending <= %.3f GiB, queue age <= %.1fms',
                    limit, age)
        logger.info('Async DRAM read promotion enabled=%s', self.promote_on_read)

    def _put_one(self, key, obj, cb, payload_ready=False):
        def completed(k):
            self.dram_mirror.offer(k, obj)
            if cb is not None:
                cb(k)
        return super()._put_one(key, obj, completed, payload_ready)

    def get_blocking(self, key):
        # Both DFS/object and batched async prefetch use this completed GET.
        # The returned GPU payload is immutable; vLLM and D2H may read it in
        # parallel. The mirror retains its own ref before the caller sees it.
        obj = super().get_blocking(key)
        if obj is not None and self.promote_on_read:
            try:
                self.dram_mirror.offer(key, obj, origin='read')
            except Exception:
                # Admission is best effort. Never turn a valid DAOS read into
                # an inference failure because CPU caching couldn't be queued.
                with self.dram_mirror.lock:
                    self.dram_mirror.stats['read_submit_errors'] += 1
                logger.exception('DRAM promotion admission failed; returning valid GPU data')
        return obj

    def close(self):
        # Stop mirror admission before draining DAOS work; CPU may already be closed.
        self.dram_mirror.close()
        super().close()
        logger.info('Async DRAM mirror final stats: %s', self.dram_mirror.snapshot())
