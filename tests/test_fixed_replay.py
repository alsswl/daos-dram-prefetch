import json
from pathlib import Path

import pytest

from discovery_fixed_replay import extract_requests, make_config, replay_one


def test_extraction_order_and_no_errored_records(tmp_path):
    path = tmp_path/'agents/job_00000/llm_calls.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({
        'later': dict(start_ns=20,end_ns=22,messages=[['B']]),
        'first': dict(start_ns=10,end_ns=12,messages=[['A']]),
        'failed': dict(start_ns=1,end_ns=2,messages=[['bad']],error='failed')}))
    records = extract_requests(tmp_path,2)
    assert [r['prompt'] for r in records] == ['A','B']
    assert [r['index'] for r in records] == [0,1]
    with pytest.raises(ValueError): extract_requests(tmp_path,3)


def test_configuration_diff_is_only_flag_and_namespace():
    off, on = make_config(False,'test-off'), make_config(True,'test-on')
    for cfg in (off,on):
        ex = cfg['extra_config']
        assert ex['daosgds.gpu_buffer_gb'] == 8
        assert ex['daosgds.dram_prefetch_policy'] == 'capacity'
        assert cfg['max_local_cpu_size'] == 8
        assert ex['daosgds.dram_promote_on_read'] is True
        for key in ('daosgds.dram_prefetch','daosgds.object_namespace','daosgds.root'):
            ex.pop(key)
    assert off == on


def test_four_gib_dram_experiment_preserves_eight_gib_staging():
    for enabled in (False, True):
        cfg = make_config(enabled, 'test-4g', cpu_gib=4)
        assert cfg['max_local_cpu_size'] == 4
        assert cfg['extra_config']['daosgds.gpu_buffer_gb'] == 8
        assert cfg['extra_config']['daosgds.dram_prefetch'] is enabled


def test_stream_keeps_original_chat_and_stop_settings():
    class Barrier:
        def wait(self,timeout): pass
    class Client:
        def stream(self,method,url,json):
            assert url == '/v1/chat/completions'
            assert json['stop'] == ['\nObservation:']
            assert json['messages'] == [{'role':'user','content':'hello'}]
            assert json['chat_template_kwargs']['enable_thinking'] is False
            return self
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def raise_for_status(self): pass
        def iter_lines(self):
            yield 'data: '+json.dumps(dict(id='test',choices=[dict(delta=dict(content='OK'),finish_reason=None)]))
            yield 'data: '+json.dumps(dict(choices=[dict(delta={},finish_reason='stop')],
                usage=dict(prompt_tokens=20,completion_tokens=1,prompt_tokens_details=dict(cached_tokens=0))))
            yield 'data: [DONE]'
    row = replay_one(Client(),dict(index=0,prompt='hello',prompt_sha256='hash'),Barrier())
    assert 'error' not in row and row['cached_tokens'] == 0
    assert row['output'] == 'OK' and row['completion_tokens'] == 1


@pytest.mark.parametrize('capacity,near_full_fraction', [(8, .5), (10, 0)])
def test_analysis_uses_saved_capacity_not_current_default(tmp_path, capacity, near_full_fraction):
    from analyze_discovery_staging import summarize_case
    (tmp_path/'config.yaml').write_text(f'extra_config:\n  daosgds.gpu_buffer_gb: {capacity}\n')
    (tmp_path/'phases.json').write_text(json.dumps([
        dict(index=0,concurrency=4,start_ns=1_000_000_000,end_ns=11_000_000_000)]))
    events = [dict(event='occupancy_sample',time_ns=t*1_000_000_000,
                   used_bytes=int(used*2**30),ready_bytes=0,cpu_hot_bytes=0)
              for t,used in [(1,0),(2,7.5),(7,0),(11,0)]]
    (tmp_path/'trace.test.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    (tmp_path/'server.log').write_text('')
    stats, _, _ = summarize_case(tmp_path,8)
    assert stats['staging_gib'] == capacity
    assert stats['peak_staging_gib'] == 7.5
    assert stats['occupancy_time_fraction'].get('at_least_90pct',0) == pytest.approx(near_full_fraction)
