"""Experiment-only store instrumentation; no extra synchronization or copies."""
import time

from .staging_probe_backend import ProbeBackend
from .staging_probe_backend import ProbeMixin
from .async_dram_backend import DaosAsyncDramBackend


class StoreProbeMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from lmcache.v1.gpu_connector.gpu_connectors import VLLMPagedMemGPUConnectorV2
        from lmcache.v1.storage_backend import storage_manager
        trace = self._staging_trace
        cls = VLLMPagedMemGPUConnectorV2
        # Dedicated benchmark process, exactly one backend per worker.
        original = cls.batched_from_gpu

        def gather(connector, objects, starts, ends, **kw):
            trace.emit('store_gather_start', devices=[str(o.tensor.device) for o in objects],
                       chunks=len(objects), bytes=sum(o.get_size() for o in objects))
            try:
                return original(connector, objects, starts, ends, **kw)
            finally:
                trace.emit('store_gather_return')

        cls.batched_from_gpu = gather
        copy = storage_manager.allocate_and_copy_objects

        def observed_copy(allocator, keys, objects, stream):
            started = time.perf_counter()
            result = copy(allocator, keys, objects, stream)
            trace.emit('store_manager_copy', sources=[str(o.tensor.device) for o in objects],
                       destinations=[str(o.tensor.device) for o in result[1]],
                       bytes=sum(o.get_size() for o in result[1]),
                       ms=(time.perf_counter()-started)*1000)
            return result

        storage_manager.allocate_and_copy_objects = observed_copy


class StoreProbeBackend(StoreProbeMixin, ProbeBackend):
    pass


class StoreProbeAsyncDramBackend(StoreProbeMixin, ProbeMixin, DaosAsyncDramBackend):
    pass
