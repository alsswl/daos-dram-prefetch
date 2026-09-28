from staging_mixed_pressure import ReadOnlyRequests, batches, summarize


def test_workload_mix_and_same_request_count():
    prompts = [{'id': i} for i in range(16)]
    workloads = batches(prompts)
    assert all(len(batch) == 16 for batch in workloads.values())
    assert all(p['id'] >= 14 for p in workloads['dram_heavy'])
    assert all(p['id'] < 8 for p in workloads['daos_heavy'])
    assert sum(p['id'] >= 14 for p in workloads['mixed']) == 8


def test_readonly_request_uses_native_skip_save_without_mutating_input():
    class Client:
        def stream(self, method, url, **kwargs):
            return kwargs['json']
    body = {'prompt': [1, 2]}
    result = ReadOnlyRequests(Client()).stream('POST', '/', json=body)
    assert result['kv_transfer_params'] == {'lmcache.skip_save': 'true'}
    assert 'kv_transfer_params' not in body


def test_occupancy_average_is_time_weighted_and_hit_share_is_tier_based():
    def row(time, used, event='occupancy_sample', **extra):
        return dict(time_ns=time, used_bytes=used*2**30, ready_bytes=0,
                    cpu_ready_bytes=0, daos_ready_bytes=0, event=event, **extra)
    events = [row(0, 0), row(10, 5),
              row(15, 8, 'allocate', failed=False),
              row(16, 5, 'tier_lookup', tier='dram', hit_chunks=3),
              row(17, 5, 'tier_lookup', tier='daos', hit_chunks=1), row(20, 0)]
    result = summarize(events, 0, 30, 10)
    assert result['peak_used_gib'] == 8
    assert abs(result['mean_used_gib'] - 5/3) < 1e-8
    assert result['dram_share_of_hit_chunks'] == .75
    assert result['fraction_time_ge_5gib'] == 1/3
