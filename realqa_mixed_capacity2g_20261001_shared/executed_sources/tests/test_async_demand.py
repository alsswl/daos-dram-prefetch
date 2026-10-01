import asyncio
from types import SimpleNamespace as NS

import pytest
from lmcache.v1.event_manager import EventManager, EventType
from lmcache_daos.async_demand_backend import PayloadTimingCoordinator, AsyncDemandBackend
from test_early_lookup import Obj


@pytest.mark.parametrize('prefetch', [False, True])
@pytest.mark.parametrize('tier', ['dram', 'daos'])
@pytest.mark.parametrize('abort', [False, True])
def test_metadata_then_payload_policy(prefetch, tier, abort):
    async def run():
        log = []
        source = Obj(True)
        async def contains(*args): return 1
        async def read(*args):
            log.append('speculative')
            return [Obj()]
        async def serialize(coro, n): return await coro
        def demand(keys):
            log.append('demand')
            return [Obj()]
        cpu = NS(batched_async_contains=contains, get_blocking=lambda key: source)
        be = NS(early_payload_prefetch=prefetch, batched_async_contains=contains,
                batched_get_non_blocking=read, batched_get_blocking=demand,
                _release_memory_obj=lambda obj: obj.ref_count_down(), cpu_prefetch=None,
                _staging_trace=NS(emit=lambda event, **kw: log.append(event)))
        coordinator = PayloadTimingCoordinator(be, cpu)
        manager = NS(event_manager=EventManager(), async_serializer=NS(run=serialize),
                     get_active_storage_backends=lambda **kw: [(tier, cpu if tier == 'dram' else be)],
                     async_lookup_server=NS(send_response_to_scheduler=lambda rid, n: log.append(('notify', n))))
        await coordinator.lookup(manager, 'r', ['k'], [0,128], pin=True)
        view = manager.event_manager.get_event_future(EventType.LOADING, 'r').result()[0][0][1]
        if abort:
            view.batch.abort()
        await asyncio.sleep(0)
        if not prefetch:
            assert 'speculative' not in log and 'demand' not in log
            assert not coordinator.tasks
            if not abort:
                assert not view.batch.future.done()
        if not abort:
            view.batch.resolve()
            if tier == 'dram':
                assert view.batch.selected is view.batch.sources
            elif not prefetch:
                assert log.count('demand') == 1
                assert log.index(('notify',128)) < log.index('demand')
            else:
                assert log.count('speculative') == 1 and 'demand' not in log
            view.ref_count_down()
        coordinator.close()
        assert not coordinator.batches
        if tier == 'dram':
            assert source.refs == source.pins == 0
        if abort:
            assert 'demand' not in log and 'speculative' not in log
    asyncio.run(run())


@pytest.mark.parametrize('value', ['false', 0, None])
def test_policy_rejects_non_boolean_before_backend_allocation(value):
    cfg = NS(extra_config={'daosgds.early_payload_prefetch': value})
    with pytest.raises(ValueError, match='YAML boolean'):
        AsyncDemandBackend(cfg)


def test_two_arm_configs_only_differ_in_payload_policy():
    import sharegpt_async_payload_compare as runner
    a, b = runner.config(False), runner.config(True)
    assert a['enable_async_loading'] is b['enable_async_loading'] is True
    assert a['extra_config']['lookup_backoff_time'] == .001
    assert runner.comparable_config(a) == runner.comparable_config(b)
