import pytest

from dram_cold_compare import CONDITIONS, aggregate, schedule, validate_cold


def test_cold_plan_has_fresh_case_per_mode_concurrency_repeat():
    cases = list(schedule(3))
    assert len(cases) == len(set(cases)) == 18
    assert cases[:3] == [(1, 1, name) for name in CONDITIONS]
    for concurrency in (1, 4):
        groups = [[name for r, c, name in cases if r == repeat and c == concurrency]
                  for repeat in (1, 2, 3)]
        for position in range(3):
            assert {group[position] for group in groups} == set(CONDITIONS)


def test_cold_gate_rejects_hits_and_prefetches():
    sample = dict(rows=[dict(cached_tokens=0)], daos_prefetch_calls=0,
                  cpu_gpu_prefetch_calls=0, cpu_gpu_fallback_calls=0)
    validate_cold(sample)
    for field in ('daos_prefetch_calls', 'cpu_gpu_prefetch_calls', 'cpu_gpu_fallback_calls'):
        with pytest.raises(RuntimeError):
            validate_cold(dict(sample, **{field: 1}))
    with pytest.raises(RuntimeError):
        validate_cold(dict(sample, rows=[dict(cached_tokens=128)]))


def test_cold_and_warm_results_never_combined():
    samples = [dict(tag=phase, ttft_mean_ms=latency,
                   rows=[dict(ttft_ms=latency, e2e_ms=latency+10, cached_tokens=hit)])
               for phase, latency, hit in [('cold', 100, 0), ('warm_1', 20, 128)]]
    result = aggregate([dict(condition='dram', concurrency=1, samples=samples)])
    assert len(result) == 2
    assert [r['mean_ttft_ms'] for r in result] == [100, 20]


def test_three_conditions_do_not_merge_dram_and_prefetch_results():
    sample = dict(tag='warm_1', ttft_mean_ms=20,
                  rows=[dict(ttft_ms=20, e2e_ms=30, cached_tokens=4095)])
    result = aggregate([dict(condition=name, concurrency=4, samples=[sample])
                        for name in CONDITIONS])
    assert len(result) == 3
    assert {r['condition'] for r in result} == set(CONDITIONS)


def test_profile_comparison_keeps_resource_differences_visible():
    from summarize_dram_threeway import normalize_config
    from run_dram_cache import build_profile
    from test_dram_cache import base
    original = base()
    profiles = [build_profile(original, dram, 4, gpu) for dram, gpu in CONDITIONS.values()]
    assert all(normalize_config(p) == normalize_config(profiles[0]) for p in profiles)
    profiles[1]['extra_config']['daosgds.gpu_buffer_gb'] = 99
    assert normalize_config(profiles[1]) != normalize_config(profiles[0])
