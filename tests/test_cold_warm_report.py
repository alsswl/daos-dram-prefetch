from report_cold_warm_prefetch import metric_means, select_events


def test_phase_events_exclude_other_phase_even_with_same_input_index():
    events = [dict(event='cpu_get_ready', request_id='chatcmpl-cold-xyz'),
              dict(event='cpu_get_ready', request_id='chatcmpl-warm-xyz'),
              dict(event='occupancy_sample'), dict(event='serializer_queued', request_id=None)]
    assert select_events(events, [dict(server_request_id='chatcmpl-warm')]) == [events[1]]


def test_warm_metrics_subtract_cold_cumulative_totals(tmp_path):
    before, after = tmp_path/'before', tmp_path/'after'
    keys = ['time_to_first_token_seconds','request_queue_time_seconds','request_prefill_time_seconds']
    def make(count, total):
        return '\n'.join(f'vllm:{k}_{suffix}{{engine="0"}} {v}' for k in keys
                         for suffix, v in [('count', count), ('sum', total)])
    before.write_text(make(256, 25.6)); after.write_text(make(512, 38.4))
    values = metric_means(before, after)
    assert all(abs(value-50) < .000001 for value in values.values())
