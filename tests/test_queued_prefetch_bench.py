from contextlib import contextmanager
import json
from types import SimpleNamespace as NS

import yaml

import compare_queued_prefetch as bench
import cold_warm_prefetch as cold


def test_plan_has_balanced_repetitions_and_optional_readiness_control():
    cases = bench.cases_for([8,16], 3, True)
    assert len(cases) == 18
    assert len({s['name'] for s in cases}) == 18
    for c in (8,16):
        for r in (1,2,3):
            assert {s['mode'] for s in cases if s['concurrency']==c and s['repeat']==r} == set(bench.MODES)
    assert len(bench.cases_for([8],1,False)) == 2


def test_runner_keeps_prefetch_on_and_changes_only_policy(tmp_path, monkeypatch):
    empty=dict(used_bytes=0,cpu_hot_bytes=0,daos_puts=0,dram_mirror=dict(pending_bytes=0,errors=0))
    filled=dict(empty,cpu_hot_bytes=32,daos_puts=1)
    client=NS(get=lambda path:NS(text='metrics'))
    @contextmanager
    def server(a, config, folder):
        (folder/'server.log').write_text('healthy')
        yield client
    monkeypatch.setattr(cold,'server',server)
    monkeypatch.setattr(cold,'await_empty',lambda path:empty)
    monkeypatch.setattr(cold,'drain',lambda *args:filled)
    monkeypatch.setattr(cold,'rolling_requests',lambda records,c,invoke,save,health:save(dict(index=0)))
    configs=[]
    for mode, flags in bench.MODES.items():
        folder=tmp_path/mode; folder.mkdir()
        cold.run_case(NS(**flags),folder,[dict(index=0)],8,True)
        config=yaml.safe_load((folder/'config.yaml').read_text()); ec=config['extra_config']
        assert ec['daosgds.dram_prefetch'] is True
        assert ec.pop('daosgds.dram_prefetch_cancel_queued') == flags['cancel_queued']
        assert ec.pop('daosgds.dram_prefetch_early_ready') == flags['early_ready']
        ec.pop('daosgds.object_namespace'); ec.pop('daosgds.root'); configs.append(config)
        assert json.loads((folder/'warm/initial_sample.json').read_text()) == filled
    assert configs[0] == configs[1] == configs[2]


def test_report_end_to_end_with_synthetic_records(tmp_path, monkeypatch):
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    cases=bench.cases_for([8],1,False)
    put(tmp_path/'plan.json',dict(cases=cases,repeats=1,concurrency=[8],source_sha256={}))
    put(tmp_path/'status.json',dict(status='completed'))
    put(tmp_path/'requests.json',[dict(index=0,prompt_sha256='same')])
    events_by_case={}
    for spec in cases:
        case=tmp_path/spec['name']; case.mkdir()
        flags=bench.MODES[spec['mode']]
        cfg=dict(extra_config={'daosgds.dram_prefetch':True,
             'daosgds.dram_prefetch_cancel_queued':flags['cancel_queued'],
             'daosgds.dram_prefetch_early_ready':flags['early_ready'],
             'daosgds.object_namespace':spec['name'], 'daosgds.root':'/'+spec['name']})
        (case/'config.yaml').write_text(yaml.safe_dump(cfg))
        put(case/'status.json',dict(status='completed')); put(case/'native_maps.json',[])
        counters=dict(copy_errors=0,deferred_pending_batches=0,retrieve_cancelled_queued=0,
                      retrieve_ready_gpu=0,retrieve_waited_gpu=0,retrieve_capacity_cpu=0)
        initial=dict(used_bytes=0,cpu_hot_bytes=0,daos_puts=0,
                     dram_mirror=dict(pending_bytes=0,errors=0),cpu_prefetch=counters)
        put(case/'initial_sample.json',initial)
        events=[]; previous=initial
        for i,phase in enumerate(('cold','warm')):
            folder=case/phase; folder.mkdir()
            rid=spec['name']+'-'+phase
            put(folder/'replay_calls.json',[dict(index=0,prompt_sha256='same',server_request_id=rid,
                 ttft_ms=10,prompt_tokens=129,cached_tokens=128,completion_tokens=10)])
            put(folder/'phase.json',dict(start_ns=100*i,end_ns=100*i+90))
            put(folder/'initial_sample.json',previous)
            after=dict(previous,cpu_hot_bytes=32,daos_puts=1,cpu_prefetch=dict(counters,
                       retrieve_cancelled_queued=i+1 if flags['cancel_queued'] else 0))
            put(folder/'final_sample.json',after); previous=after
            events.append(dict(event='tier_lookup',pid=1,time_ns=100*i+1,used_bytes=0,
                               request_id=rid,tier='dram',queried_chunks=1,hit_chunks=1))
            if flags['cancel_queued']:
                events.append(dict(event='cpu_prefetch_retrieve_decision',pid=1,time_ns=100*i+2,
                     used_bytes=0,request_id=rid,decision='cancelled_queued',resolve_start_ns=1,resolve_end_ns=2))
        events_by_case[spec['name']]=events
    monkeypatch.setattr(bench,'read_events',lambda case:events_by_case[case.name])
    monkeypatch.setattr(bench,'join',lambda calls,events:[dict(calls[0],other_failed_chunks=0,
                       dram_lookup_chunks=1,daos_lookup_chunks=0,computed_prompt_tokens=1,
                       capacity_recomputed_tokens=0)])
    bench.report(tmp_path)
    assert len(json.loads((tmp_path/'summary.json').read_text()))==4
    assert all(r['same_cached_requests']==1 for r in json.loads((tmp_path/'paired_checks.json').read_text()))
    assert 'cancel' in (tmp_path/'RESULT_KO.md').read_text()
