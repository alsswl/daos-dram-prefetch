"""Opt-in synchronous demand reads: no lookup-time GPU payload prefetch.

Preserves GPU-direct stores and asynchronous, bounded DRAM mirroring. This is
NOT a windowed streaming implementation: a synchronous batch can hold several
GPU chunks until the normal connector consumes them.
"""
import contextvars
import functools
import inspect

from .capacity_probe_backend import CapacityProbeBackend
from .gds_backend import _cfg

_operation = contextvars.ContextVar('daos_demand_operation', default=None)


class LookupPinnedCPUObject:
    """Caller ref belongs to retrieve; lookup pin belongs to lookup_unpin.

    The native sync retrieve cleanup otherwise unpins the shared CPU object,
    then lookup_unpin unpins it a second time. Keep the lookup's pin until the
    request owner cleans it up; cancellation still uses that same owner.
    """
    def __init__(self, source):
        self.source = source

    def __getattr__(self, name):
        return getattr(self.source, name)

    @property
    def is_pinned(self):
        return False


def validate_demand_config(config):
    if (config.enable_async_loading or config.use_layerwise or not config.local_cpu
            or _cfg(config, 'dram_prefetch', False)
            or not _cfg(config, 'demand_read_only', False)):
        raise ValueError('Demand reads require demand_read_only=true, local_cpu=true, '
                         'enable_async_loading=false, use_layerwise=false, dram_prefetch=false')


def protect_cpu_cache(cpu, trace):
    """Native sync write-back stores the supplied object WITHOUT a device copy.

    Never retain a GPU object in the CPU hot cache. The DAOS get path has already
    offered the immutable GPU payload to our bounded async D2H mirror. Only that
    mirror's completed CPU object may enter the cache. A skipped mirror offer
    remains skipped (do not introduce a second admission attempt here).
    """
    original = cpu.submit_put_task

    def submit(key, obj, on_complete_callback=None):
        if obj.tensor is not None and obj.tensor.is_cuda:
            trace.emit('sync_gpu_writeback_suppressed', bytes=obj.get_size())
            return None
        return original(key, obj, on_complete_callback=on_complete_callback)

    cpu.submit_put_task = submit


def install_context():
    from lmcache.v1.cache_engine import LMCacheEngine
    if getattr(LMCacheEngine, '_daos_demand_context_installed', False):
        return
    for name in ('lookup', 'retrieve'):
        original = getattr(LMCacheEngine, name)
        signature = inspect.signature(original)

        def make_wrapper(original, signature, name):
            @functools.wraps(original)
            def wrapper(engine, *args, **kwargs):
                selected = getattr(engine.storage_manager, '_daos_gpu_store', None)
                if not selected or not getattr(selected[1], '_supports_sync_read', False):
                    return original(engine, *args, **kwargs)
                bound = signature.bind_partial(engine, *args, **kwargs)
                rid = (bound.arguments.get('lookup_id') if name == 'lookup'
                       else engine._get_req_id(kwargs))
                token = _operation.set((name, rid))
                try:
                    return original(engine, *args, **kwargs)
                finally:
                    _operation.reset(token)
            return wrapper

        setattr(LMCacheEngine, name, make_wrapper(original, signature, name))
    LMCacheEngine._daos_demand_context_installed = True


class DemandReadBackend(CapacityProbeBackend):
    _supports_sync_read = True

    def __init__(self, config, dst_device='cuda', metadata=None, local_cpu_backend=None, loop=None):
        validate_demand_config(config)
        super().__init__(config, dst_device, metadata, local_cpu_backend, loop)
        assert self.cpu_prefetch is None
        trace = self._staging_trace
        protect_cpu_cache(local_cpu_backend, trace)
        original_cpu_get = local_cpu_backend.get_blocking

        def cpu_get(key):
            obj = original_cpu_get(key)
            return None if obj is None else LookupPinnedCPUObject(obj)

        local_cpu_backend.get_blocking = cpu_get
        for tier, backend in (('dram', local_cpu_backend), ('daos', self)):
            original = backend.batched_contains

            def contains(keys, pin=False, _original=original, _tier=tier):
                hits = _original(keys, pin)
                operation = _operation.get()
                if operation and operation[0] == 'lookup':
                    trace.emit('tier_lookup', request_id=operation[1], tier=_tier,
                               queried_chunks=len(keys), hit_chunks=hits)
                    if _tier == 'dram':
                        with self.dram_mirror.lock:
                            self.dram_mirror.stats['cpu_hit_chunks'] += hits
                return hits

            backend.batched_contains = contains
        install_context()
        trace.emit('demand_read_enabled', async_loading=False, dram_prefetch=False,
                   daos_prefetch=False, async_dram_mirror=True)

    async def batched_get_non_blocking(self, *args, **kwargs):
        raise RuntimeError('Lookup-time DAOS prefetch is forbidden in demand-read mode')

    def batched_get_blocking(self, keys):
        operation = _operation.get()
        if not operation or operation[0] != 'retrieve':
            raise RuntimeError('Demand batch must be called inside retrieve, not lookup')
        rid = operation[1]
        trace = self._staging_trace
        trace.emit('daos_demand_start', request_id=rid, requested_chunks=len(keys))
        outcomes = list(self._pool.map(self._observed_get, keys))
        objects, first_failure, discarded = [], None, 0
        for obj, cause in outcomes:
            if obj is None or first_failure is not None:
                if first_failure is None:
                    first_failure = cause
                if obj is not None:
                    self._release_memory_obj(obj)
                    discarded += 1
                objects.append(None)
            else:
                objects.append(obj)
        trace.emit('daos_demand_outcome', request_id=rid, requested_chunks=len(keys),
                   returned_chunks=sum(obj is not None for obj in objects),
                   capacity_failed_chunks=sum(cause == 'capacity' for _, cause in outcomes),
                   other_failed_chunks=sum(cause == 'other' for _, cause in outcomes),
                   first_failure=first_failure, successful_tail_discarded_chunks=discarded)
        return objects
