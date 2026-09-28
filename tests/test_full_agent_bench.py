import pytest

from full_agent_bench import run_schedule


def test_default_preserves_original_schedule():
    assert run_schedule(1) == [('dfs', 0), ('dfs', 1), ('object', 0), ('object', 1)]


def test_ten_restarts_have_one_fill_and_ten_replays_per_mode():
    schedule = run_schedule(10, interleave_modes=True)
    assert len(schedule) == 22
    for mode in ('dfs', 'object'):
        assert [index for selected, index in schedule if selected == mode] == list(range(11))
    assert schedule[:4] == [('dfs', 0), ('object', 0), ('object', 1), ('dfs', 1)]
    assert len(set(schedule)) == 22


@pytest.mark.parametrize('count', [0, -1])
def test_invalid_restart_count(count):
    with pytest.raises(ValueError):
        run_schedule(count)
