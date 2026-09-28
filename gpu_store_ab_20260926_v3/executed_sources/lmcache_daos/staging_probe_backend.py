"""Experiment-only tier/occupancy instrumentation; no admission-policy changes.

Selected only by a generated experiment YAML. Never retain payload references.
The existing allocator lifecycle trace remains the source of event-wise peaks.
"""
import threading

from .gds_backend import DaosGdsBackend, _cfg
from .dram_prefetch_backend import DaosDramPrefetchBackend, StagedCPUObject


class ProbeMixin:
    def __init__(self, config, dst_device='cuda', metadata=None,
                 local_cpu_backend=None, loop=None):
        super().__init__(config, dst_device, metadata, local_cpu_backend, loop)
        trace = self._staging_trace
        if trace is None or local_cpu_backend is None:
            raise ValueError('Probe requires staging tracing and LocalCPUBackend')
        self._probe_stop = threading.Event()
        self._probe_cpu_ready = {}
        interval = float(_cfg(config, 'probe_interval_ms', 5)) / 1000
        if not .001 <= interval <= 1:
            raise ValueError('probe_interval_ms must be between 1 and 1000')
        cpu = local_cpu_backend
        for tier, backend in [('dram', cpu), ('daos', self)]:
            original = backend.batched_async_contains

            async def contains(lookup_id, keys, pin=False, _get=original, _tier=tier):
                hits = await _get(lookup_id, keys, pin)
                trace.emit('tier_lookup', request_id=lookup_id, tier=_tier,
                           queried_chunks=len(keys), hit_chunks=hits)
                return hits

            backend.batched_async_contains = contains
        original_get = cpu.batched_get_non_blocking

        async def cpu_get(lookup_id, keys, transfer_spec=None):
            trace.emit('cpu_get_start', request_id=lookup_id, chunks=len(keys))
            objects = await original_get(lookup_id, keys, transfer_spec)
            staged = [o.destination for o in objects if isinstance(o, StagedCPUObject)]
            with trace.lock:
                for obj in staged:
                    oid = id(obj)
                    trace.ready[oid] = (lookup_id, obj.get_size())
                    self._probe_cpu_ready[oid] = lookup_id
                trace.emit('cpu_get_ready', request_id=lookup_id,
                           chunks=len(objects), staged_chunks=len(staged),
                           staged_bytes=sum(o.get_size() for o in staged))
            return objects

        cpu.batched_get_non_blocking = cpu_get

        def sample():
            while not self._probe_stop.is_set():
                with cpu.cpu_lock:
                    hot_chunks = len(cpu.hot_cache)
                    hot_bytes = sum(o.get_size() for o in cpu.hot_cache.values())
                with self.memory_allocator.device_mem_lock:
                    with trace.lock:
                        # ID reuse after a free must not attribute a later DAOS
                        # object to an earlier CPU-prefetch request.
                        self._probe_cpu_ready = {oid: rid for oid, rid in self._probe_cpu_ready.items()
                            if oid in trace.ready and trace.ready[oid][0] == rid}
                        cpu_ready = sum(trace.ready[i][1] for i in self._probe_cpu_ready)
                        trace.emit('occupancy_sample', cpu_hot_chunks=hot_chunks,
                                   cpu_hot_bytes=hot_bytes, cpu_ready_bytes=cpu_ready,
                                   daos_ready_bytes=sum(v[1] for k, v in trace.ready.items()
                                                        if k not in self._probe_cpu_ready),
                                   daos_puts=self.stats['put'], daos_alloc_fail=self.stats['alloc_fail'])
                self._probe_stop.wait(interval)

        self._probe_thread = threading.Thread(target=sample, name='staging-probe', daemon=True)
        self._probe_thread.start()

    def close(self):
        self._probe_stop.set()
        self._probe_thread.join(timeout=5)
        super().close()


class ProbeBackend(ProbeMixin, DaosGdsBackend):
    pass


class ProbePrefetchBackend(ProbeMixin, DaosDramPrefetchBackend):
    pass
