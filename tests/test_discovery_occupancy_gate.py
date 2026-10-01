from types import SimpleNamespace as NS

import yaml

import cold_warm_prefetch
from discovery_occupancy_gate import cases
from staging_mixed_pressure import inference_command


def test_gate_plan_covers_six_cases_without_changing_concurrency():
    rows = cases()
    assert len(rows) == 6
    assert {(r['concurrency'], r['mode']) for r in rows} == {
        (c, m) for c in (8, 16) for m in ('off', 'on', 'gate60')}
    for r in rows:
        assert r['prefetch'] == (r['mode'] != 'off')
        assert r['occupancy_stop_ratio'] == (.6 if r['mode'] == 'gate60' else None)


def test_c24_plan_and_server_limit_preserve_legacy_default(monkeypatch):
    import staging_mixed_pressure as staging
    import pytest
    rows = cases([24])
    assert len(rows) == 3 and {r['concurrency'] for r in rows} == {24}
    assert {r['mode'] for r in rows} == {'off', 'on', 'gate60'}
    with pytest.raises(ValueError):
        cases([24, 24])
    monkeypatch.setattr(staging.common, 'server_command', lambda a: ['vllm', 'serve'])
    assert inference_command(NS()) == ['vllm', 'serve', '--max-num-seqs', '16']
    assert inference_command(NS(max_num_seqs=24)) == ['vllm', 'serve', '--max-num-seqs', '24']
    with pytest.raises(ValueError):
        inference_command(NS(max_num_seqs=0))


def test_run_case_passes_gate_and_explicit_historical_worker_settings(tmp_path, monkeypatch):
    class Captured(Exception):
        pass

    def server(args, config, folder):
        cfg = yaml.safe_load(config.read_text())
        ec = cfg['extra_config']
        assert cfg['max_local_cpu_size'] == ec['daosgds.gpu_buffer_gb'] == 8
        assert ec['daosgds.dram_prefetch_stop_occupancy_ratio'] == .6
        assert ec['daosgds.dram_prefetch_policy'] == 'capacity'
        assert ec['daosgds.dram_prefetch_workers'] == 1
        assert ec['daosgds.dram_prefetch_cancel_queued'] is False
        assert ec['daosgds.dram_prefetch_early_ready'] is False
        raise Captured()

    monkeypatch.setattr(cold_warm_prefetch, 'server', server)
    import pytest
    with pytest.raises(Captured):
        cold_warm_prefetch.run_case(NS(cpu_gib=8, staging_gib=8, prefetch_workers=1,
            occupancy_stop_ratio=.6, cancel_queued=False, early_ready=False),
            tmp_path, [], 16, True)
