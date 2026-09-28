from types import SimpleNamespace as NS
import threading

import pytest

from lmcache_daos import gpu_store


class Obj:
    def __init__(self, cuda=True, device=0):
        self.tensor = NS(is_cuda=cuda, device=NS(index=device))
        self.refs = 1

    def ref_count_down(self):
        self.refs -= 1


def setup(monkeypatch, fail=False):
    events = []
    monkeypatch.setattr(gpu_store.torch.cuda, 'synchronize', lambda dev: events.append('sync'))
    def submit(keys, objects, **kwargs):
        events.append('submit')
        if fail:
            raise RuntimeError('submit failed')
        for obj in objects:
            obj.refs += 1
    be = NS(store_path='gpu_direct', device_id=0, batched_submit_put_task=submit)
    manager = NS(storage_backends={'daos': be, 'LocalCPUBackend': NS(use_hot=False)},
                 _bypass_lock=threading.Lock(), _bypassed_backends=set())
    return manager, be, events


@pytest.mark.parametrize('name', ['local_cpu', 'local_disk', 'remote_url', 'enable_pd',
                                  'use_layerwise', 'enable_blending'])
def test_reject_incompatible(name):
    with pytest.raises(ValueError):
        gpu_store.validate_config(NS(**{name: True}), 'gpu_direct')
    gpu_store.validate_config(NS(**{name: True}), 'host_staged')


def test_selection(monkeypatch):
    manager, be, _ = setup(monkeypatch)
    assert gpu_store.select_backend(manager) == ('daos', be)
    manager.storage_backends['Other'] = NS()
    with pytest.raises(ValueError):
        gpu_store.select_backend(manager)
    be.store_path = 'host_staged'
    assert gpu_store.select_backend(manager) is None


def test_direct_lifetime(monkeypatch):
    manager, be, events = setup(monkeypatch)
    obj = Obj()
    gpu_store.put_direct(manager, ('daos', be), ['key'], [obj])
    assert events == ['sync', 'submit']
    assert obj.refs == 1  # worker owns the only remaining reference


@pytest.mark.parametrize('branch', ['bypass', 'submit_failure', 'host', 'device', 'location', 'counts'])
def test_release_on_skip_or_error(monkeypatch, branch):
    manager, be, events = setup(monkeypatch, fail=branch == 'submit_failure')
    obj = Obj(cuda=branch != 'host', device=1 if branch == 'device' else 0)
    if branch == 'bypass':
        manager._bypassed_backends.add('daos')
        gpu_store.put_direct(manager, ('daos', be), ['key'], [obj])
    else:
        with pytest.raises((ValueError, RuntimeError)):
            gpu_store.put_direct(manager, ('daos', be), [] if branch == 'counts' else ['key'],
                                 [obj], location='CPU' if branch == 'location' else None)
    assert obj.refs == 0
    assert events[0] == 'sync'


def test_install_scoped_and_idempotent(monkeypatch):
    from lmcache.v1.storage_backend import storage_manager
    class Manager:
        def _get_allocator_backend(self, config):
            return 'original allocator'
        def batched_put(self, *args, **kwargs):
            return 'original put'
    monkeypatch.setattr(storage_manager, 'StorageManager', Manager)
    gpu_store.install()
    wrapped = Manager.batched_put
    gpu_store.install()
    assert Manager.batched_put is wrapped
    manager, be, _ = setup(monkeypatch)
    direct = Manager()
    direct.__dict__.update(manager.__dict__)
    assert direct._get_allocator_backend(NS()) is be
    obj = Obj()
    direct.batched_put(['k'], [obj])
    assert obj.refs == 1
    legacy = Manager()
    legacy.storage_backends = {'LocalCPUBackend': NS()}
    assert legacy._get_allocator_backend(NS()) == 'original allocator'
    assert legacy.batched_put(['k'], [Obj()]) == 'original put'


def test_submit_failure_releases_worker_reference():
    from lmcache_daos.gds_backend import DaosGdsBackend
    class BadExecutor:
        def submit(self, *args):
            raise RuntimeError('executor stopped')
    backend = DaosGdsBackend.__new__(DaosGdsBackend)
    backend.store_enabled = True
    backend._put_lock = threading.Lock()
    backend._put_tasks = set()
    backend._pool = BadExecutor()
    obj = Obj()
    obj.ref_count_up = lambda: setattr(obj, 'refs', obj.refs + 1)
    with pytest.raises(RuntimeError):
        backend.batched_submit_put_task(['k'], [obj])
    assert obj.refs == 1
    assert not backend._put_tasks


def test_engine_none_format_gets_nonlayerwise_default():
    from lmcache_daos.gds_backend import DaosGdsBackend
    from lmcache.v1.memory_management import MemoryFormat
    calls = []
    backend = DaosGdsBackend.__new__(DaosGdsBackend)
    backend.memory_allocator = NS(
        allocate=lambda *args: calls.append(args) or Obj(),
        batched_allocate=lambda *args: calls.append(args) or [Obj()])
    backend.allocate('shapes', 'dtypes', fmt=None)
    backend.batched_allocate('shapes', 'dtypes', 2, fmt=None)
    assert calls[0][-1] == MemoryFormat.KV_2LTD
    assert calls[1][-1] == MemoryFormat.KV_2LTD
