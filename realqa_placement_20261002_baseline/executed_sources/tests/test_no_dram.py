from types import SimpleNamespace
import pytest
from lmcache_daos.no_dram import configure_no_dram,validate_no_dram,DisabledDramMirror,validate_no_dram_events
from lmcache_daos.demand_read_backend import validate_demand_config
from lmcache_daos.gpu_store import validate_config

def config():
    d=configure_no_dram(dict(local_cpu=True,max_local_cpu_size=256,extra_config={
        'daosgds.async_dram':True,'daosgds.dram_promote_on_read':True,
        'daosgds.demand_read_only':True}))
    return SimpleNamespace(**d,enable_async_loading=False,use_layerwise=False)

def test_no_dram_config_and_inert_mirror():
    cfg=config();validate_no_dram(cfg);validate_demand_config(cfg);validate_config(cfg,'gpu_direct')
    mirror=DisabledDramMirror(object())
    # Sentinel source has no tensor or ref-count methods: disabling copies must
    # never access payloads, allocate a buffer, or retain references.
    assert mirror.offer('key',object()) is False
    assert mirror.offer('key',object(),origin='read') is False
    assert not hasattr(mirror,'worker') and not hasattr(mirror,'stream')
    mirror.close();assert all(v==0 for k,v in mirror.snapshot().items() if k!='enabled')

@pytest.mark.parametrize('field',['local_cpu','max_local_cpu_size','async_dram','dram_promote_on_read','dram_prefetch'])
def test_reject_accidental_dram_paths(field):
    cfg=config()
    if field in ('local_cpu','max_local_cpu_size'):setattr(cfg,field,1)
    else:cfg.extra_config['daosgds.'+field]=True
    with pytest.raises(ValueError):validate_no_dram(cfg)

def test_validation_requires_all_retrievals_from_daos():
    events=[dict(event='occupancy_sample',cpu_hot_bytes=0,cpu_hot_chunks=0,cpu_ready_bytes=0,
                 dram_mirror=DisabledDramMirror(None).snapshot()),
            dict(event='window_retrieve_done',chunks=3),
            dict(event='daos_demand_outcome',returned_chunks=3)]
    assert validate_no_dram_events(events)['passed']
    events[-1]['returned_chunks']=2
    with pytest.raises(AssertionError):validate_no_dram_events(events)
