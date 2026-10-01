from concurrent.futures import ThreadPoolExecutor

import pytest
from eqbench_longform_matrix import cases_for, cases_from_plan, conversations, config_for, FirstWave, prefetch_counters


def test_36_cases_and_isolated_cancel():
    cases=cases_for()
    assert len(cases)==len({s['name'] for s in cases})==36
    for c in (8,16):
        for d in (8,4,2):
            for g in (8,4):
                rows={s['mode']:s for s in cases if (s['concurrency'],s['cpu_gib'],s['staging_gib'])==(c,d,g)}
                assert set(rows)=={'off','wait','cancel'}
                configs={k:config_for(v,'test-namespace') for k,v in rows.items()}
                for mode in ('wait','cancel'):
                    ec=configs[mode]['extra_config']
                    assert ec['daosgds.dram_prefetch_early_ready'] is True
                    assert ec.pop('daosgds.dram_prefetch_cancel_queued') is (mode=='cancel')
                    assert ec['daosgds.dram_prefetch_policy']=='capacity'
                assert configs['wait']==configs['cancel']
                assert configs['off']['extra_config']['daosgds.dram_prefetch'] is False


def test_workload_identical_across_conditions():
    stories={str(i):{'writing_prompt':str(i)} for i in range(1,5)}
    jobs=conversations(stories)
    assert len(jobs)==16
    assert list(jobs)==[str(i) for i in range(1,17)]
    for i in range(1,5):
        assert sum(j['source_story_id']==str(i) for j in jobs.values())==4
    assert len(jobs)*13*len(cases_for())==7488


def test_only_first_wave_waits():
    barrier=FirstWave(8)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(barrier.wait,2) for _ in range(19)]
        for f in futures:f.result(timeout=3)
    assert barrier.remaining==0


def test_off_has_no_prefetch_stats_but_on_requires_them():
    assert prefetch_counters({}, {'cpu_prefetch':None},False)=={}
    with pytest.raises(AssertionError,match='Missing'):
        prefetch_counters({}, {},True)
    assert prefetch_counters({'cpu_prefetch':{'staged_bytes':20}},
        {'cpu_prefetch':{'staged_bytes':120,'copy_errors':0,'deferred_pending_batches':0}},True)['staged_bytes']==100


def test_new_capacity_grid_and_resume_use_saved_plan():
    rows=cases_for([8,16,32],[4,8])
    assert len(rows)==36 and rows[0]['name']=='c8_d8_s4_off'
    assert {s['cpu_gib'] for s in rows}=={8,16,32}
    assert {s['staging_gib'] for s in rows}=={4,8}
    assert {s['concurrency'] for s in rows}=={8,16}
    assert cases_from_plan({'cases':rows})==rows
    assert cases_from_plan({'cases':cases_for()})==cases_for()
    bad=[dict(r) for r in rows];bad[0]['cpu_gib']=2
    with pytest.raises(AssertionError):cases_from_plan({'cases':bad})


def test_short_workload_saved_and_context_preflight():
    from eqbench_longform_matrix import workload_from_plan, context_preflight
    from eqbench_longform_pilot import load_templates, UPSTREAM
    workload=dict(chapters=4,max_tokens=2500,max_model_len=32768)
    plan=dict(workload=workload,requests_per_case=144)
    assert workload_from_plan(plan)==workload
    assert 144*len(cases_for([8,16,32],[4,8]))==5184
    assert workload_from_plan(dict(requests_per_case=208))['chapters']==8
    with pytest.raises(AssertionError):workload_from_plan(dict(plan,requests_per_case=208))
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):return [1]*1000
    jobs=conversations({str(i):{'writing_prompt':str(i)} for i in range(1,5)})
    result=context_preflight(Tokenizer(),jobs,load_templates(UPSTREAM/'data',4),workload)
    assert result['passed'] and result['minimum_remaining_tokens']==9268
    assert len(result['requests'])==144
    with pytest.raises(AssertionError,match='headroom'):
        context_preflight(Tokenizer(),jobs,load_templates(UPSTREAM/'data',4),workload,reserve=10000)
