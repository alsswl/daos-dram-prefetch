from types import SimpleNamespace

import pytest

import sharegpt_cold_warm as runner


def test_two_phases_keep_client_and_inherit_drained_cache(tmp_path, monkeypatch):
    records = [dict(index=0, session=0, turn=0, max_tokens=12, expected_prompt_tokens=42)]
    initial = dict(used_bytes=0, cpu_hot_bytes=0, daos_puts=0)
    cold = dict(used_bytes=0, cpu_hot_bytes=4096, daos_puts=1)
    warm = dict(used_bytes=0, cpu_hot_bytes=8192, daos_puts=2)
    states = iter([cold, warm])
    client = SimpleNamespace(get=lambda path: SimpleNamespace(text='metrics'))
    clients, schedules, drains = [], [], []

    def replay(c, r, b, **kwargs):
        clients.append(c)
        assert kwargs == dict(max_tokens=12, stop=())
        return dict(index=r['index'], prompt_tokens=42, end_ns=1)

    def schedule(rs, n, fn, save, health):
        schedules.append((rs, n))
        save(fn(rs[0], None))

    def drain(case, health, timeout):
        drains.append(case)
        return next(states)

    monkeypatch.setattr(runner.base, 'replay_one', replay)
    monkeypatch.setattr(runner.base, 'session_requests', schedule)
    monkeypatch.setattr(runner.base, 'drain', drain)
    monkeypatch.setattr(runner.base, 'latest_sample', lambda case: cold)
    final = runner.run_phases(tmp_path, records, 16, client, lambda: None, initial)
    assert final == warm
    assert clients == [client, client]
    assert schedules == [(records, 16), (records, 16)]
    assert len(drains) == 2
    assert runner.read(tmp_path/'cold/initial_sample.json') == initial
    assert runner.read(tmp_path/'warm/initial_sample.json') == cold
    assert runner.read(tmp_path/'warm/final_sample.json') == warm
    assert runner.read(tmp_path/'cold/replay_calls.json') == runner.read(tmp_path/'warm/replay_calls.json')


def test_failed_cold_never_enters_warm(tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError('failure')
    monkeypatch.setattr(runner.base, 'session_requests', fail)
    client = SimpleNamespace(get=lambda path: SimpleNamespace(text='metrics'))
    with pytest.raises(RuntimeError, match='failure'):
        runner.run_phases(tmp_path, [], 16, client, lambda: None, {})
    assert not (tmp_path/'warm').exists()
    assert runner.read(tmp_path/'cold/status.json')['status'] == 'failed'


def test_input_token_mismatch_is_not_silently_accepted(monkeypatch):
    monkeypatch.setattr(runner.base, 'replay_one', lambda *a, **k: dict(prompt_tokens=41))
    r = dict(session=1, turn=0, max_tokens=12, expected_prompt_tokens=42)
    assert 'error' in runner.invoke(None, r, None)


def test_phase_names_and_invalid_repeats():
    assert runner.phase_names(1) == ['cold', 'warm']
    assert runner.phase_names(4) == ['cold', 'warm1', 'warm2', 'warm3', 'warm4']
    with pytest.raises(ValueError):
        runner.phase_names(0)


def test_four_warms_inherit_immediately_previous_cache(tmp_path, monkeypatch):
    phases = runner.phase_names(4)
    states = [dict(used_bytes=0, cpu_hot_bytes=i*4096, daos_puts=i) for i in range(6)]
    iterator = iter(states[1:])
    client = SimpleNamespace(get=lambda path: SimpleNamespace(text='metrics'))
    clients, space_checks = [], []
    records = [dict(index=0, session=0, turn=0, max_tokens=12, expected_prompt_tokens=42)]
    def replay(c, r, b, **kwargs):
        clients.append(c)
        return dict(index=0, prompt_tokens=42, end_ns=1)
    monkeypatch.setattr(runner.base, 'replay_one', replay)
    monkeypatch.setattr(runner.base, 'session_requests', lambda rs,n,fn,save,h: save(fn(rs[0],None)))
    monkeypatch.setattr(runner.base, 'drain', lambda *a, **k: next(iterator))
    monkeypatch.setattr(runner.base, 'latest_sample', lambda case: states[-1])
    monkeypatch.setattr(runner.base, 'storage_guard', lambda case,n: space_checks.append(n))
    final = runner.run_phases(tmp_path, records, 16, client, lambda: None, states[0],
                              phases=phases, warm_extra_gib=82)
    assert final == states[-1] and clients == [client]*5 and space_checks == [82]*4
    for i, name in enumerate(phases):
        assert runner.read(tmp_path/name/'initial_sample.json') == states[i]
        assert runner.read(tmp_path/name/'final_sample.json') == states[i+1]
