from copy import deepcopy
import pytest

import sharegpt_lookup_backoff as runner
from analyze_gpu_overlap import kernel_gap_stats


def test_only_backoff_changes_and_default_is_preserved():
    before = runner.original_config()
    after = runner.config()
    runner.check_config(before, after)
    assert 'lookup_backoff_time' not in before['extra_config']
    assert runner.early.config is runner.original_config
    assert before['enable_async_loading'] and after['enable_async_loading']


def test_other_config_change_rejected():
    before, after = runner.original_config(), runner.config()
    after['extra_config']['daosgds.dram_prefetch_workers'] = 2
    with pytest.raises(AssertionError, match='Unexpected config'):
        runner.check_config(before, after)


def test_normalization_does_not_mutate_input():
    cfg = runner.config()
    copy = deepcopy(cfg)
    runner.comparable_config(cfg)
    assert cfg == copy


def test_kernel_gaps_merge_overlaps_exclude_startup():
    s = kernel_gap_stats([(100000, 101000), (100500, 102000), (103000, 104000), (104500, 105000)])
    assert s['gaps_over_1ms'] == 1
    assert s['total_gaps_over_1ms_ms'] == 1
    assert s['active_window_seconds'] == 0.005


def test_no_kernel_gap():
    assert kernel_gap_stats([])['gaps_over_1ms'] == 0
