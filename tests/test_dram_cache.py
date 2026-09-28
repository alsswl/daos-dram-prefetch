import asyncio
import copy
from pathlib import Path
import threading

import pytest
import yaml

from run_dram_cache import build_profile, profile_environment
from dram_cache_bench import metric
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.v1.storage_backend.cache_policy import get_cache_policy


def base():
    return yaml.safe_load((Path(__file__).parents[1]/'lmcache_config_daosgds_unified.yaml').read_text())


def test_on_off_only_changes_retention_and_explicit_capacity():
    original = base()
    saved = copy.deepcopy(original)
    assert build_profile(original) == original
    off = build_profile(original, 'off', 4)
    on = build_profile(original, 'on', 4)
    assert off['local_cpu'] is False and on['local_cpu'] is True
    on['local_cpu'] = False
    assert on == off
    assert original == saved
    assert on['extra_config'] == saved['extra_config']


@pytest.mark.parametrize('capacity', [0, -1, float('inf'), float('nan')])
def test_reject_bad_capacity(capacity):
    with pytest.raises(ValueError):
        build_profile(base(), 'on', capacity)


@pytest.mark.parametrize('change', ['store_off', 'sync', 'layerwise', 'location', 'no_plugin'])
def test_reject_unvalidated_or_nonmirrored_on_paths(change):
    cfg = base()
    if change == 'store_off':
        cfg['extra_config']['daosgds.store'] = False
    elif change == 'sync':
        cfg['enable_async_loading'] = False
    elif change == 'layerwise':
        cfg['use_layerwise'] = True
    elif change == 'location':
        cfg['store_location'] = 'LocalCPUBackend'
    else:
        cfg['storage_plugins'] = []
    with pytest.raises(ValueError):
        build_profile(cfg, 'on')


def test_environment_cannot_silently_invert_profile():
    env = dict(LMCACHE_LOCAL_CPU='True', LMCACHE_MAX_LOCAL_CPU_SIZE='999', KEEP='yes')
    result = profile_environment(env, '/test.yaml', build_profile(base(), 'off', 4))
    assert result['LMCACHE_LOCAL_CPU'] == 'False'
    assert result['LMCACHE_MAX_LOCAL_CPU_SIZE'] == '4'
    assert result['KEEP'] == 'yes' and env['LMCACHE_LOCAL_CPU'] == 'True'
    assert metric('lmcache:local_cpu_hot_cache_count{worker="0"} 32\n',
                  'lmcache:local_cpu_hot_cache_count') == 32
    assert metric('# no LMCache worker metrics\n', 'lmcache:local_cpu_hot_cache_count') is None


@pytest.mark.parametrize('enabled', [False, True])
def test_native_cpu_cache_retains_original_object_without_extra_copy(enabled):
    class Buffer:
        refs = 1
        payload = b'known KV bytes'
        def ref_count_up(self):
            self.refs += 1
        def ref_count_down(self):
            self.refs -= 1
            assert self.refs >= 0

    cpu = LocalCPUBackend.__new__(LocalCPUBackend)
    cpu.cache_policy = get_cache_policy('LRU')
    cpu.hot_cache = cpu.cache_policy.init_mutable_mapping()
    cpu.use_hot = enabled
    cpu.cpu_lock = threading.Lock()
    cpu.batched_msg_sender = None
    obj = Buffer()
    cpu.batched_submit_put_task(['k'], [obj])
    obj.ref_count_down()  # StorageManager releases its own reference.
    assert cpu.contains('k') is enabled
    assert obj.refs == int(enabled)
    if enabled:
        received = asyncio.run(cpu.batched_get_non_blocking('lookup', ['k']))
        assert received[0] is obj and received[0].payload == b'known KV bytes'
        assert obj.refs == 2
        received[0].ref_count_down()
        assert cpu.remove('k') is True
        assert obj.refs == 0
