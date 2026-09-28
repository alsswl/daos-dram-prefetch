from contextlib import contextmanager
from types import SimpleNamespace as NS

import pytest
import yaml

import cold_warm_prefetch as cold
from prefetch_capacity_sweep import cases_for, check_space
from discovery_fixed_replay import make_config


def test_exact_matrix_and_no_cancel_when_prefetch_off():
    cases=cases_for()
    assert len(cases)==len({s['name'] for s in cases})==36
    assert len({(s['concurrency'],s['cpu_gib'],s['staging_gib'],s['mode']) for s in cases})==36
    for s in cases:
        assert not s['cancel_queued'] or s['prefetch']
        cfg=make_config(s['prefetch'],'test-only',s['cpu_gib'],s['staging_gib'])
        assert cfg['max_local_cpu_size']==s['cpu_gib']
        assert cfg['extra_config']['daosgds.gpu_buffer_gb']==s['staging_gib']
        assert cfg['extra_config']['daosgds.dram_prefetch_policy']=='capacity'


def test_pool_guard_checks_min_target_not_only_aggregate():
    q=dict(response=dict(uuid='f973c142-2353-41da-b154-5079ba6969f2',disabled_targets=0,
          state='Ready',rebuild=dict(status=0),tier_stats=[dict(media_type='nvme',free=1_500_000_000_000,min=90_000_000_000)]))
    assert check_space(q)['min']==90_000_000_000
    q['response']['tier_stats'][0]['min']=6_000_000_000
    with pytest.raises(RuntimeError,match='headroom'):check_space(q)


def test_run_case_honors_requested_capacity_and_preserves_warm(tmp_path,monkeypatch):
    empty=dict(used_bytes=0,cpu_hot_bytes=0,daos_puts=0,dram_mirror=dict(pending_bytes=0,errors=0))
    filled=dict(empty,cpu_hot_bytes=32,daos_puts=1)
    @contextmanager
    def server(a,config,case):
        cfg=yaml.safe_load(config.read_text())
        assert cfg['max_local_cpu_size']==2
        assert cfg['extra_config']['daosgds.gpu_buffer_gb']==4
        (case/'server.log').write_text('healthy')
        yield NS(get=lambda path:NS(text='metrics'))
    monkeypatch.setattr(cold,'server',server)
    monkeypatch.setattr(cold,'await_empty',lambda path:empty)
    monkeypatch.setattr(cold,'drain',lambda *args:filled)
    monkeypatch.setattr(cold,'rolling_requests',lambda rs,c,invoke,save,health:save(dict(index=0)))
    cold.run_case(NS(cpu_gib=2,staging_gib=4,cancel_queued=False,early_ready=False),tmp_path,[dict(index=0)],16,False)
