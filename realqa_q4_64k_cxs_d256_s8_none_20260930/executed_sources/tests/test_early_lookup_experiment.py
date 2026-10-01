import pytest

import sharegpt_early_lookup as runner
from report_early_lookup import validate_events


def test_waits_for_predecessor_even_if_its_status_is_completed():
    assert not runner.predecessor_done(dict(ActiveState='active', Result='success'),
                                       dict(status='completed'))


def test_failed_predecessor_never_launches_next_job():
    with pytest.raises(RuntimeError, match='did not complete'):
        runner.predecessor_done(dict(ActiveState='failed', Result='exit-code'), dict(status='failed'))
    with pytest.raises(RuntimeError, match='did not complete'):
        runner.predecessor_done(dict(ActiveState='inactive', Result='success'), dict(status='running'))


def test_successful_predecessor_releases_queue():
    assert runner.predecessor_done(dict(ActiveState='inactive', Result='success'), dict(status='completed'))


def test_configuration_only_adds_metadata_first_to_existing_on_arm():
    cfg = runner.config()
    baseline = runner.base.make_config(True, 'test-ns', cpu_gib=256, staging_gib=8)
    baseline['extra_config'].update({
        'storage_plugin.daosgds.module_path': 'lmcache_daos.capacity_probe_backend',
        'storage_plugin.daosgds.class_name': 'CapacityProbeBackend',
        'daosgds.dram_prefetch_workers': 1, 'daosgds.dram_prefetch_cancel_queued': False,
        'daosgds.dram_prefetch_early_ready': False, 'daosgds.dram_prefetch_stop_occupancy_ratio': None})
    assert cfg['extra_config'].pop('daosgds.early_lookup') is True
    for c in (cfg, baseline):
        for key in ('storage_plugin.daosgds.module_path', 'storage_plugin.daosgds.class_name',
                    'daosgds.object_namespace', 'daosgds.root'):
            c['extra_config'].pop(key)
    assert cfg == baseline


def event(name, ns, **kw):
    return dict(event=name, monotonic_ns=ns, request_id='r', **kw)


def test_attribution_requires_notification_before_payload():
    with pytest.raises(AssertionError, match='before existence'):
        validate_events([event('early_payload_start', 1)])
    rows = [event('early_lookup_notify', 1), event('early_payload_start', 2),
            event('early_retrieve_decision', 3, tier='daos', decision='wait_running')]
    assert validate_events(rows) == {'daos/wait_running': 1}


def test_metadata_errors_not_reported_as_success():
    with pytest.raises(AssertionError, match='failed'):
        validate_events([event('early_lookup_error', 1)])


def test_prepare_uses_separate_root_and_unchanged_requests(tmp_path, monkeypatch):
    previous, output = tmp_path/'old', tmp_path/'new'
    previous.mkdir()
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'EXTRA_SOURCES', ())
    for name in ('requests.json', 'sessions.json', 'capacity_estimate.json', 'params.json',
                 'runtime_versions.json'):
        runner.dump(previous/name, {})
    runner.dump(previous/'plan.json', dict(cpu_gib=256, warm_repeats=4,
        request_sha256=runner.base.digest(previous/'requests.json'), source_sha256={}))
    runner.prepare(output, previous, 'old.service')
    assert (output/'requests.json').read_bytes() == (previous/'requests.json').read_bytes()
    assert runner.read(output/'status.json')['status'] == 'queued'
    assert runner.read(output/'plan.json')['cases'][0]['prefetch'] is True
