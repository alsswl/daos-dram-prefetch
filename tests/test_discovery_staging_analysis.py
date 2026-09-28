"""Small synthetic accounting checks; no GPU, server, or benchmark execution."""
import json
from pathlib import Path

import pytest

import analyze_discovery_staging as analysis


def make_case(tmp_path: Path, monkeypatch):
    case = tmp_path / 'prefetch_off'
    case.mkdir()
    phases = [dict(index=0, concurrency=4, start_ns=0, end_ns=5_000_000_000),
              dict(index=1, concurrency=8, start_ns=5_000_000_000, end_ns=10_000_000_000)]
    (case/'phases.json').write_text(json.dumps(phases))
    (case/'server.log').write_text('Reqid: x, Total tokens 1408, Inference Engine computed tokens: 256, LMCache hit tokens: 1152, need to load: 1152\n')
    job=case/'agents'/'job_00000'; job.mkdir(parents=True)
    (case/'jobs.json').write_text(json.dumps([dict(index=0, phase=0, folder='agents/job_00000',
        end_ns=8_000_000_000, status='complete', forced=False, python_calls=2)]))
    (job/'llm_calls.json').write_text(json.dumps({'x':dict(start_ns=1_000_000_000,
        end_ns=8_000_000_000, ttft_ms=20, messages=[['shared prompt']])}))
    (job/'agent.log').write_text("ImportError: cannot import name '_lazywhere'\n")
    def event(t,name,used=0,ready=0,**fields):
        return dict(time_ns=int(t*1e9),event=name,used_bytes=int(used*2**30),
                    ready_bytes=int(ready*2**30),**fields)
    events=[event(0,'occupancy_sample',cpu_hot_bytes=0),
        event(1,'tier_lookup',request_id='a',tier='dram',queried_chunks=10,hit_chunks=4),
        event(1.1,'tier_lookup',request_id='a',tier='daos',queried_chunks=6,hit_chunks=5),
        event(2,'allocate',used=2,failed=False),
        event(4,'occupancy_sample',used=2,ready=1,cpu_hot_bytes=2**30),
        event(6,'free'),
        event(7,'tier_lookup',request_id='b',tier='dram',queried_chunks=20,hit_chunks=0),
        event(7.1,'tier_lookup',request_id='b',tier='daos',queried_chunks=20,hit_chunks=20),
        event(10,'occupancy_sample',cpu_hot_bytes=2**30)]
    monkeypatch.setattr(analysis,'read_events',lambda _:events)
    return case


def test_event_time_weighted_occupancy(tmp_path,monkeypatch):
    case=make_case(tmp_path,monkeypatch)
    stats,bins,_=analysis.summarize_case(case,8)
    assert stats['peak_staging_gib']==2
    assert stats['mean_staging_gib']==pytest.approx(.8)
    assert stats['mean_cpu_gib']==pytest.approx(.6)
    assert stats['occupancy_time_fraction']['empty']==pytest.approx(.6)
    assert bins[0]['mean_staging_gib']==pytest.approx(1.2)
    assert bins[1]['mean_staging_gib']==pytest.approx(.4)
    assert bins[1]['peak_staging_gib']==2
    assert stats['phases'][1]['peak_staging_gib']==2
    assert len(bins)==2
    assert stats['mean_llm_calls']==pytest.approx(.7)
    assert stats['peak_llm_calls']==1


def test_tier_denominator_and_failures_are_preserved(tmp_path,monkeypatch):
    case=make_case(tmp_path,monkeypatch)
    stats,_,_=analysis.summarize_case(case,8)
    assert stats['lookup_chunks']==30
    assert stats['dram_chunks']==4
    assert stats['daos_chunks']==25
    assert stats['miss_chunks']==1
    assert stats['dram_lookup_ratio']==pytest.approx(4/30)
    assert stats['phases'][0]['dram_pct']==40
    assert stats['phases'][1]['daos_pct']==100
    assert stats['last_dram_hit_elapsed_seconds']==1
    assert stats['agent_issue_jobs']['scipy_statsmodels_compatibility_jobs']==1
    assert stats['workflow_status']['complete']==1
    assert stats['python_calls']==2
    assert (case/'timeline.csv').exists()
