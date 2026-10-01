import asyncio
from concurrent.futures import Future
from types import SimpleNamespace as NS

import pytest
from lmcache.v1.event_manager import EventManager, EventType
from lmcache_daos.tier_payload_backend import TierPayloadBackend, TierPayloadCoordinator
from test_early_lookup import Obj


@pytest.mark.parametrize('daos,dram', [(False,False), (True,False), (True,True)])
@pytest.mark.parametrize('abort', [False,True])
def test_mixed_prefix_policy_and_lifetime(daos, dram, abort):
    async def run():
        log, queued_cpu = [], []
        source = Obj(True)
        async def cpu_contains(*args): return 1
        async def daos_contains(*args): return 1
        async def read(rid, keys):
            log.append('daos_prefetch')
            assert keys == ['daos-key']
            return [Obj()]
        async def serialize(coro,n): return await coro
        def submit(fn,sources,rid):
            assert sources == [source]
            queued_cpu.append((fn,sources,rid))
            return Future()
        def demand(keys):
            assert keys == ['daos-key']
            log.append('daos_demand')
            return [Obj()]
        cpu = NS(batched_async_contains=cpu_contains, get_blocking=lambda k: source)
        be = NS(payload_prefetch_tiers={'daos':daos,'dram':dram},
                batched_async_contains=daos_contains, batched_get_non_blocking=read,
                batched_get_blocking=demand, _release_memory_obj=lambda o:o.ref_count_down(),
                cpu_prefetch=NS(worker=NS(submit=submit)),
                _staging_trace=NS(emit=lambda event, **kw: log.append(event)))
        coordinator = TierPayloadCoordinator(be,cpu)
        def copy(batch,sources,rid):
            if batch.claim():
                log.append('dram_prefetch')
                batch.finish([Obj()])
        coordinator.cpu_work = copy
        manager = NS(event_manager=EventManager(),async_serializer=NS(run=serialize),
                     get_active_storage_backends=lambda **kw:[('cpu',cpu),('daos',be)],
                     async_lookup_server=NS(send_response_to_scheduler=lambda rid,n:log.append(('notify',n))))
        await coordinator.lookup(manager,'r',['cpu-key','daos-key'],[0,128,256],pin=True)
        views = [tier[0][1] for tier in manager.event_manager.get_event_future(EventType.LOADING,'r').result()]
        assert ('notify',256) in log
        assert len(queued_cpu) == int(dram)
        if abort:
            for view in views: view.batch.abort()
        for fn,sources,rid in queued_cpu: fn(sources,rid)
        await asyncio.sleep(0)
        if not abort:
            assert ('daos_prefetch' in log) == daos
            assert ('dram_prefetch' in log) == dram
            for view in views: view.batch.resolve()
            assert (views[0].batch.selected is views[0].batch.sources) == (not dram)
            assert log.count('daos_demand') == int(not daos)
            for view in views: view.ref_count_down()
        else:
            assert not any(e in log for e in ('daos_prefetch','dram_prefetch','daos_demand'))
        coordinator.close()
        assert source.refs == source.pins == 0
        assert not coordinator.batches
    asyncio.run(run())


@pytest.mark.parametrize('key', ['early_daos_prefetch','early_dram_prefetch'])
def test_non_boolean_policy_rejected_before_allocation(key):
    with pytest.raises(ValueError, match='YAML booleans'):
        TierPayloadBackend(NS(extra_config={f'daosgds.{key}':'false'}))


def test_three_configs_only_change_payload_tiers():
    import sharegpt_async_threeway as runner
    configs = [runner.config(policy) for policy in runner.POLICIES]
    assert all(cfg['enable_async_loading'] for cfg in configs)
    assert all(cfg['extra_config']['lookup_backoff_time'] == .001 for cfg in configs)
    assert all(runner.comparable_config(c) == runner.comparable_config(configs[0]) for c in configs)
    assert runner.PHASES == ['cold','warm1','warm2','warm3','warm4']


def test_warm_aggregation_excludes_cold():
    from sharegpt_async_threeway import aggregate_warm
    def row(phase,ttft,elapsed):
        return dict(phase=phase, ttft={'mean':ttft},elapsed_seconds=elapsed,queried_chunks=100,
                    hit_chunks={'dram':40,'daos':60},input_tokens=1000,cached_tokens=900,
                    computed_tokens=100,output_tokens=20,peak_staging_gib=2)
    a = aggregate_warm([row('cold',1000,2000),row('warm1',100,400),row('warm2',120,420)])
    assert a['mean_ttft_ms'] == 110 and a['mean_elapsed_seconds'] == 410
    assert a['warm_repeats'] == 2 and a['dram_hit_pct'] == 40
