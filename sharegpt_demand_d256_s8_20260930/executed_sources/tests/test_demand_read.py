import asyncio
from types import SimpleNamespace as NS

import pytest

from lmcache_daos.demand_read_backend import (
    DemandReadBackend, _operation, protect_cpu_cache, validate_demand_config,
)
from report_demand_read import validate_no_prefetch


def config():
    return NS(enable_async_loading=False, use_layerwise=False, local_cpu=True,
              extra_config={'daosgds.demand_read_only': True, 'daosgds.dram_prefetch': False})


def test_demand_mode_requires_explicit_safe_config():
    validate_demand_config(config())
    for attr in ('enable_async_loading', 'use_layerwise'):
        cfg = config(); setattr(cfg, attr, True)
        with pytest.raises(ValueError): validate_demand_config(cfg)
    cfg = config(); cfg.extra_config['daosgds.dram_prefetch'] = True
    with pytest.raises(ValueError): validate_demand_config(cfg)


def test_cpu_cache_never_retains_gpu_writeback():
    stored, logged = [], []
    cpu = NS(submit_put_task=lambda key, obj, **kw: stored.append((key, obj)))
    trace = NS(emit=lambda *a, **kw: logged.append((a, kw)))
    protect_cpu_cache(cpu, trace)
    gpu = NS(tensor=NS(is_cuda=True), get_size=lambda: 20)
    host = NS(tensor=NS(is_cuda=False))
    cpu.submit_put_task('gpu', gpu)
    assert stored == [] and len(logged) == 1
    cpu.submit_put_task('cpu', host)
    assert stored == [('cpu', host)]


def test_async_calls_and_lookup_reads_are_rejected():
    backend = DemandReadBackend.__new__(DemandReadBackend)
    with pytest.raises(RuntimeError, match='forbidden'):
        asyncio.run(backend.batched_get_non_blocking('r', []))
    token = _operation.set(('lookup', 'r'))
    try:
        with pytest.raises(RuntimeError, match='inside retrieve'):
            backend.batched_get_blocking([])
    finally:
        _operation.reset(token)


def test_demand_batch_prefix_release_and_attribution():
    backend = DemandReadBackend.__new__(DemandReadBackend)
    first, tail = object(), object()
    released, logged = [], []
    backend._pool = NS(map=lambda fn, keys: [(first, None), (None, 'capacity'), (tail, None)])
    backend._release_memory_obj = released.append
    backend._staging_trace = NS(emit=lambda name, **kw: logged.append(dict(event=name, **kw)))
    token = _operation.set(('retrieve', 'r'))
    try:
        assert backend.batched_get_blocking(['a', 'b', 'c']) == [first, None, None]
    finally:
        _operation.reset(token)
    assert released == [tail]
    assert logged[-1]['capacity_failed_chunks'] == 1
    assert logged[-1]['successful_tail_discarded_chunks'] == 1


def test_trace_proves_reads_inside_retrieve_and_no_prefetch():
    events = [dict(event=e, request_id='r') for e in (
        'retrieve_start', 'daos_demand_start', 'daos_demand_outcome', 'retrieve_return')]
    assert validate_no_prefetch(events) == 1
    with pytest.raises(AssertionError): validate_no_prefetch(events[1:])
    with pytest.raises(AssertionError): validate_no_prefetch(events + [dict(event='prefetch_start')])
