from contextlib import contextmanager
import json
from types import SimpleNamespace as NS

import pytest

import cold_warm_prefetch as bench


def test_cold_warm_reuses_process_client_and_cache(tmp_path, monkeypatch):
    cold = dict(used_bytes=0, cpu_hot_bytes=0, daos_puts=0, dram_mirror=dict(pending_bytes=0, errors=0))
    filled = dict(used_bytes=0, cpu_hot_bytes=4096, daos_puts=7, dram_mirror=dict(pending_bytes=0, errors=0))
    client = NS(get=lambda path: NS(text='metrics'))
    starts, invocations, drains = [], [], []
    @contextmanager
    def server(*args):
        starts.append(True)
        (tmp_path/'server.log').write_text('healthy')
        yield client
    def request(c, record, barrier):
        assert c is client
        invocations.append(record['index'])
        return dict(index=record['index'])
    def rolling(records, concurrency, invoke, save, health):
        for r in records:
            save(invoke(r, None))
    def drain(*args):
        drains.append(True)
        return filled
    monkeypatch.setattr(bench, 'server', server)
    monkeypatch.setattr(bench, 'await_empty', lambda path: cold)
    monkeypatch.setattr(bench, 'replay_one', request)
    monkeypatch.setattr(bench, 'rolling_requests', rolling)
    monkeypatch.setattr(bench, 'drain', drain)
    bench.run_case(NS(), tmp_path, [dict(index=0), dict(index=1)], 8, True)
    assert len(starts) == 1 and len(drains) == 2
    assert invocations == [0, 1, 0, 1]
    assert json.loads((tmp_path/'cold/initial_sample.json').read_text()) == cold
    assert json.loads((tmp_path/'warm/initial_sample.json').read_text()) == filled
    assert json.loads((tmp_path/'status.json').read_text())['requests'] == 4


def test_drain_requires_gpu_and_mirror_empty(monkeypatch):
    samples = iter([dict(used_bytes=32, dram_mirror=dict(pending_bytes=0, errors=0)),
                    dict(used_bytes=0, dram_mirror=dict(pending_bytes=32, errors=0))] +
                   [dict(used_bytes=0, dram_mirror=dict(pending_bytes=0, errors=0))]*10)
    monkeypatch.setattr(bench, 'latest_sample', lambda path: next(samples))
    monkeypatch.setattr(bench.time, 'sleep', lambda n: None)
    result = bench.drain(None, lambda: None)
    assert result['used_bytes'] == result['dram_mirror']['pending_bytes'] == 0
    with pytest.raises(StopIteration): next(samples)


def test_drain_stops_on_health_error():
    def bad(): raise RuntimeError('DAOS failure')
    with pytest.raises(RuntimeError, match='DAOS failure'):
        bench.drain(None, bad)
