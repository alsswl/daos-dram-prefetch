from report_prefetch_timing import stats


def test_empty_and_single_sample_statistics():
    assert stats([])['mean'] is None
    assert stats([None, 3]) == dict(n=1, mean=3, p50=3, p95=3, maximum=3)


def test_quantiles_interpolate_and_ignore_missing_values():
    result = stats([10, None, 0])
    assert result['n'] == 2
    assert result['mean'] == result['p50'] == 5
    assert result['p95'] == 9.5
