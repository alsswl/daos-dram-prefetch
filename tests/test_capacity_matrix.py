import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace as NS

import pytest

from discovery_fixed_replay import make_config
from lmcache_daos.capacity_probe_backend import CapacityProbeBackend
from report_capacity_matrix import attribute_recomputation


@pytest.mark.parametrize('dram',[8,4,2])
@pytest.mark.parametrize('staging',[8,4])
@pytest.mark.parametrize('enabled',[True,False])
def test_matrix_config(dram,staging,enabled):
    cfg=make_config(enabled,'matrix-test',cpu_gib=dram,staging_gib=staging)
    assert cfg['max_local_cpu_size']==dram
    assert cfg['extra_config']['daosgds.gpu_buffer_gb']==staging
    assert cfg['extra_config']['daosgds.dram_prefetch_policy']=='capacity'
    assert cfg['extra_config']['daosgds.dram_prefetch'] is enabled


@pytest.mark.parametrize('reason',['capacity','other'])
def test_probe_preserves_parallel_prefix_and_releases_tail(reason):
    b=object.__new__(CapacityProbeBackend)
    b._read_probe_local=threading.local()
    events=[]; released=[]
    b._staging_trace=NS(emit=lambda event,**fields:events.append(dict(event=event,**fields)))
    objs=[NS(get_size=lambda:20) for _ in range(3)]
    def get(key):
        if key==1:
            b._read_probe_local.current['allocation_failed']=reason=='capacity'
            return None
        return objs[key]
    b.get_blocking=get
    b._release_memory_obj=lambda obj:released.append(obj)
    with ThreadPoolExecutor(max_workers=3) as pool:
        b._pool=pool
        result=asyncio.run(b.batched_get_non_blocking('test',[0,1,2]))
    assert result==[objs[0]] and released==[objs[2]]
    e=events[0]
    assert e['returned_chunks']==1 and e['requested_chunks']==3
    assert e['first_failure']==reason
    assert e['capacity_failed_chunks']==int(reason=='capacity')
    assert e['other_failed_chunks']==int(reason=='other')
    assert e['successful_tail_discarded_chunks']==1


def example(first_failure='capacity',cached=256):
    calls=[dict(index=0,server_request_id='chatcmpl-test',start_ns=1,
                ttft_ms=20,prompt_tokens=641,cached_tokens=cached)]
    base=dict(request_id='chatcmpl-test-suffix')
    events=[dict(base,event='tier_lookup',tier='dram',hit_chunks=1),
            dict(base,event='tier_lookup',tier='daos',hit_chunks=3),
            dict(base,event='cpu_get_ready',chunks=1),
            dict(base,event='daos_prefetch_outcome',requested_chunks=3,returned_chunks=1,
                 capacity_failed_chunks=int(first_failure=='capacity'),
                 other_failed_chunks=int(first_failure=='other'),
                 successful_tail_discarded_chunks=1,first_failure=first_failure)]
    return calls,events


def test_attribution_excludes_cold_miss_and_includes_discarded_tail():
    r=attribute_recomputation(*example())[0]
    assert r['capacity_recomputed_tokens']==256
    assert r['computed_prompt_tokens']==385
    assert r['attribution_issues']==[]
    assert r['capacity_failed_chunks']==1  # Not 2: tail discard adds lost reuse.


def test_attribution_does_not_claim_other_failures_or_usage_mismatch():
    for calls,events in (example('other'),example(cached=128)):
        r=attribute_recomputation(calls,events)[0]
        assert r['capacity_recomputed_tokens']==0
        assert r['unattributed_shortfall_tokens']==256
