"""Diagnostic-only attribution of async DAOS prefix loss to GPU allocation failure.

Preserves the base parallel GET + contiguous-prefix return algorithm. No admission,
retry, eviction, or scheduling policy is added. Only experiment YAML selects this.
"""
import asyncio
import threading
import time

from .gds_backend import logger
from .store_probe_backend import StoreProbeAsyncDramBackend


class CapacityProbeBackend(StoreProbeAsyncDramBackend):
    def __init__(self, *args, **kwargs):
        self._read_probe_local = threading.local()
        super().__init__(*args, **kwargs)
        original = self.memory_allocator.allocate

        def allocate(*args, **kwargs):
            obj = original(*args, **kwargs)
            current = getattr(self._read_probe_local, 'current', None)
            if current is not None and obj is None:
                current['allocation_failed'] = True
            return obj

        self.memory_allocator.allocate = allocate

    def _observed_get(self, key):
        current = dict(allocation_failed=False)
        self._read_probe_local.current = current
        try:
            obj = self.get_blocking(key)
            cause = None if obj is not None else (
                'capacity' if current['allocation_failed'] else 'other')
            return obj, cause
        finally:
            del self._read_probe_local.current

    async def batched_get_non_blocking(self, lookup_id, keys, transfer_spec=None):
        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        outcomes = await asyncio.gather(*(
            loop.run_in_executor(self._pool, self._observed_get, key) for key in keys))
        out, first_failure, discarded = [], None, 0
        for obj, cause in outcomes:
            if obj is None or first_failure is not None:
                if first_failure is None:
                    first_failure = cause
                if obj is not None:
                    self._release_memory_obj(obj)
                    discarded += 1
            else:
                out.append(obj)
        self._staging_trace.emit('daos_prefetch_outcome', request_id=lookup_id,
            requested_chunks=len(keys), returned_chunks=len(out),
            capacity_failed_chunks=sum(cause == 'capacity' for _, cause in outcomes),
            other_failed_chunks=sum(cause == 'other' for _, cause in outcomes),
            first_failure=first_failure, successful_tail_discarded_chunks=discarded)
        nbytes = sum(obj.get_size() for obj in out)
        elapsed = time.perf_counter()-started
        if nbytes:
            logger.info('DaosGdsBackend prefetch[%s]: %d/%d objects, %.1f MiB in %.1f ms (%.2f GB/s, GPU-direct)',
                lookup_id[-8:], len(out), len(keys), nbytes/2**20, elapsed*1e3, nbytes/elapsed/1e9)
        return out
